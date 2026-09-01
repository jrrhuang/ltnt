"""CREATE MAP — FastAPI router.

Usage in server.py (one-block wiring):

    from explore_map.routes import make_router as _create_map_router
    app.include_router(_create_map_router(jobs, GENERATED_DIR))

Images are written to {GENERATED_DIR}/{job_id}/images/ and served by
the existing `/images` static mount in server.py, so no new mounts.
"""
from __future__ import annotations

import json
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class CreateMapRequest(BaseModel):
    prompt: str
    model: str = "flux"   # "flux" | "sana" | "fluxfm"
    num_particles: int = 6        # roots = distinct LLM interpretations (comprehensive tree)
    num_clones: int = 3
    num_resampling_steps: int = 3
    augment: bool = True          # seed roots with prompt-augmentation variations
    num_steps: int = 28
    guidance_scale: float = 3.5
    rho: float = 0.0
    seed: int = 42
    height: int = 512
    width: int = 512
    r0: float = 0.12
    decay: float = 0.5


# Process-wide GPU lock. CREATE MAP and GENERATE both acquire this so they
# never load FLUX concurrently on a card that can only hold one copy.
GPU_LOCK = threading.Lock()


def make_router(jobs: dict, generated_dir: Path, free_previous=None) -> APIRouter:
    router = APIRouter(prefix="/api/create_map", tags=["create_map"])
    _gpu_lock = GPU_LOCK

    @router.post("/generate")
    def start(req: CreateMapRequest):
        job_id = "cm_" + uuid.uuid4().hex[:8]
        jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "stage": "Queued…",
            "result": None,
            "error": None,
        }

        def _cb(frac: float, stage: str):
            j = jobs.get(job_id)
            if j is None:
                return
            # Monotonic within a run
            j["progress"] = max(float(j.get("progress", 0.0)), 100.0 * float(frac))
            j["stage"] = stage

        def _run():
            # Acquire the GPU lock first; if another CREATE MAP is in flight,
            # this thread waits its turn instead of trying to load FLUX in parallel.
            acquired = _gpu_lock.acquire(timeout=1800)
            if not acquired:
                jobs[job_id].update({"status": "error",
                                     "error": "Timed out waiting for GPU"})
                return
            # ALSO take the interactive server's model lock: generates and maps
            # share ONE GPU + ONE resident model. Without this, a generate's
            # cold FLUX load races an in-flight map's model -> CUDA OOM
            # (observed live: two OOMs in job 8907057 with 43.6GB resident).
            import sys as _sys
            _main = _sys.modules.get("__main__")
            _model_lock = getattr(_main, "_model_lock", None)
            _got_model_lock = False
            if _model_lock is not None:
                # Session takeover (same dance as a new generate): a session
                # parked at waiting_selection holds the model lock for up to
                # 5 min — signal it to unwind instead of queueing silently.
                jobs[job_id].update({"status": "pending", "progress": 0.5,
                                     "stage": "Waiting for the current session to wind down…"})
                _cur = getattr(_main, "_current_job_id", None)
                if _cur and _cur in jobs:
                    jobs[_cur]["_cancelled"] = True
                    _ev = jobs[_cur].get("_selection_event")
                    if _ev is not None:
                        _ev.set()
                _got_model_lock = _model_lock.acquire(timeout=600)
                if not _got_model_lock:
                    _gpu_lock.release()
                    jobs[job_id].update({
                        "status": "error",
                        "error": "Another generation is still running — wait for it to finish, then try the map again.",
                        "progress": 0, "stage": "Busy"})
                    return
            try:
                jobs[job_id].update({"status": "running", "progress": 1.0, "stage": "Starting…"})
                import gc as _gc, torch as _torch
                def _mem(label):
                    if _torch.cuda.is_available():
                        a = _torch.cuda.memory_allocated() / 1e9
                        print(f"[create_map] {label}: allocated={a:.2f}GB")
                _mem("start")

                # ── REUSE the resident sampler when it matches the requested
                # model (the common case) — never load a second copy of a
                # model that is already on the GPU.
                _TYPE_TO_MODEL = {
                    "FluxDevInteractive": "flux",
                    "SanaInteractive": "sana",
                    "FluxFMWDMInteractive": "fluxfm",
                }
                _resident = (getattr(_main, "_warm_flux", None)
                             or getattr(_main, "_active_sampler", None))
                _reuse = None
                if _resident is not None:
                    _res_model = _TYPE_TO_MODEL.get(type(_resident).__name__)
                    # Reuse is FLUX-only: FLUX keeps its text encoders resident
                    # (so the map prompt can be re-embedded); SANA/FluxFM drop
                    # theirs after init and cannot encode a new prompt.
                    _intact = (getattr(_resident, "pipeline", None) is not None
                               and getattr(_resident, "transformer", None) is not None
                               and getattr(getattr(_resident, "pipeline", None),
                                           "text_encoder", None) is not None)
                    if _intact and _res_model == req.model == "flux":
                        print(f"[create_map] reusing resident {type(_resident).__name__} "
                              f"(no second model load)")
                        _reuse = _resident
                    else:
                        # Different (or gutted) model: switch properly — free
                        # via the server's own thorough helper, then the
                        # precompute loads fresh (with its VRAM guard).
                        print(f"[create_map] resident={_res_model or 'unknown'} "
                              f"requested={req.model} -> freeing before load")
                        _free = getattr(_main, "_free_previous_sampler", None)
                        if callable(_free):
                            try:
                                _main._active_sampler = _resident
                            except Exception:
                                pass
                            _free()
                        _gc.collect()
                        if _torch.cuda.is_available():
                            _torch.cuda.synchronize()
                            _torch.cuda.empty_cache()
                _mem("after reuse/free decision")

                from explore_map.precompute import precompute_radial_map
                out_dir = generated_dir / job_id
                manifest_path = precompute_radial_map(
                    prompt=req.prompt,
                    model=req.model,
                    num_particles=req.num_particles,
                    num_clones=req.num_clones,
                    num_resampling_steps=req.num_resampling_steps,
                    augment=req.augment,
                    num_steps=req.num_steps,
                    guidance_scale=req.guidance_scale,
                    rho=req.rho,
                    height=req.height,
                    width=req.width,
                    seed=req.seed,
                    r0=req.r0,
                    decay=req.decay,
                    output_dir=str(out_dir),
                    progress_cb=_cb,
                    sampler=_reuse,
                )
                with open(manifest_path) as f:
                    manifest = json.load(f)
                for img in manifest["images"]:
                    img["url"] = f"/images/{job_id}/{img['path']}"
                jobs[job_id].update({
                    "status": "done",
                    "progress": 100.0,
                    "stage": "Done",
                    "result": manifest,
                })
            except Exception as exc:
                traceback.print_exc()
                # RuntimeErrors raised by the precompute carry plain-words,
                # user-facing messages (e.g. the VRAM guard) — pass verbatim.
                _msg = (str(exc) if isinstance(exc, RuntimeError)
                        else f"{type(exc).__name__}: {exc}")
                jobs[job_id].update({
                    "status": "error",
                    "error": _msg,
                    "progress": 0,
                    "stage": "Error",
                })
            finally:
                if _got_model_lock:
                    try: _model_lock.release()
                    except Exception: pass
                _gpu_lock.release()

        threading.Thread(target=_run, daemon=True).start()
        return {"job_id": job_id}

    @router.get("/{job_id}")
    def status(job_id: str):
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        j = jobs[job_id]
        return {
            "status": j["status"],
            "progress": j.get("progress", 0),
            "stage": j.get("stage", ""),
            "result": j.get("result"),
            "error": j.get("error"),
        }

    return router
