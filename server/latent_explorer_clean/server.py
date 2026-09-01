"""
FastAPI backend for latent-explorer.
Wraps sd35_fmtt.py (generation) and cluster_images.py (UMAP/MDS clustering).

Interactive flow:
  1. POST /api/generate       → starts job, returns job_id
  2. GET  /api/jobs/{id}       → poll; status cycles through:
       pending → running → waiting_selection → running → ... → done
  3. POST /api/jobs/{id}/select → send selected particle indices, unblocks the job

Start with:
    conda activate /home/jerryhua/conda-envs/DPS311
    python server.py
"""

import os
# Force HuggingFace cache to scratch — must happen before any `transformers`
# / `diffusers` import so they see it when initializing.
# Respect an env-provided HF_HOME (e.g. a persistent shared cache); fall back
# to node-local scratch. Must be set before transformers/diffusers import.
os.environ.setdefault("HF_HOME", "/scratch/jerryhua/.cache/huggingface")
_HF_HUB = os.path.join(os.environ["HF_HOME"], "hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _HF_HUB)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_HUB)

import matplotlib
matplotlib.use('Agg')  # must be set before any other matplotlib import

import asyncio
import base64
import builtins
import io
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import gc
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure local modules are importable regardless of the cwd Python was
# launched from — must happen before the eager pre-imports below.
sys.path.insert(0, str(Path(__file__).parent))

# Eagerly import the heavy model / clustering modules at server startup.
# Otherwise the first Generate click pays a 1–3 min cold import tax while
# umap → pynndescent → sklearn JIT-compile via numba. Doing it here surfaces
# the cost in the backend terminal on launch instead of stalling at 5%.
# Heavy pre-imports + the eager FLUX load now run in a BACKGROUND boot thread
# (see _boot_worker) so uvicorn binds within seconds and /api/status can report
# REAL boot progress the whole time — no dead air while numba JITs or FLUX loads.
from pydantic import BaseModel

# ── BOOT / MODEL-LOAD STATUS ─────────────────────────────────────────────────
# Live, REAL progress for every "the model isn't ready yet" wait:
#   * boot: library imports -> eager FLUX load -> prompt-LLM warmup -> ready
#   * model switch: during any cold sampler load inside a generate job
# Weight-load fractions are measured, not simulated: GPU bytes allocated so far
# divided by the model's expected resident footprint.
_boot = {"ready": False, "phase": "starting", "detail": "", "fraction": 0.0}
_EXPECTED_BYTES = {   # approx resident GPU footprint per sampler (bf16)
    "flux": 33e9, "krea2": 37e9, "sd35": 8e9,
}


# When the job script ran a staging boot-beacon before this server (see
# boot_beacon.py), staging owned the [0, LTNT_BOOT_BASE) slice of the boot
# bar; our own boot phases are offset above it so the bar never jumps back.
_BOOT_BASE = float(os.environ.get("LTNT_BOOT_BASE", "0") or 0)


def _set_boot(phase, fraction, detail="", ready=None):
    _boot["phase"] = phase
    _boot["detail"] = detail
    frac = _BOOT_BASE + (1.0 - _BOOT_BASE) * min(max(float(fraction), 0.0), 1.0)
    # Never regress: the boot bar is monotone by construction.
    _boot["fraction"] = max(_boot["fraction"], round(frac, 3))
    if ready is not None:
        _boot["ready"] = bool(ready)
        if ready:
            _boot["fraction"] = 1.0


def _watch_cuda_mem(phase, expected_bytes, stop_evt, base, span, on_frac=None):
    """Sample GPU bytes allocated while a model loads -> real linear progress."""
    base_alloc = 0
    try:
        if torch.cuda.is_available():
            base_alloc = torch.cuda.memory_allocated()
    except Exception:
        pass
    while not stop_evt.wait(0.5):
        try:
            alloc = max(torch.cuda.memory_allocated() - base_alloc, 0)
        except Exception:
            alloc = 0
        frac = min(alloc / max(expected_bytes, 1), 0.99)
        _set_boot(phase, base + span * frac,
                  f"{alloc / 2**30:.1f} / {expected_bytes / 2**30:.0f} GB on GPU")
        if on_frac is not None:
            try:
                on_frac(frac)
            except Exception:
                pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.environ.get("CORS_ALLOW_ALL") else [
        "http://localhost:5173", "http://localhost:5174", "http://localhost:5175",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

GENERATED_DIR = Path(__file__).parent / "generated"
GENERATED_DIR.mkdir(exist_ok=True)
app.mount("/images", StaticFiles(directory=str(GENERATED_DIR)), name="images")

# OpenAI image-edit provider (optional; isolated in server_openai.py so it
# can be removed without touching the Reve path). Fails silently if the
# module can't be imported.
try:
    from server_openai import router as openai_router
    app.include_router(openai_router)
except Exception as _openai_exc:
    print(f"[server] OpenAI router not loaded: {_openai_exc}")

# Google Gemini / Nano Banana image-edit provider (optional, isolated).
try:
    from server_nanobanana import router as nanobanana_router
    app.include_router(nanobanana_router)
except Exception as _nb_exc:
    print(f"[server] Nano Banana router not loaded: {_nb_exc}")

# Serve built frontend (for Docker / production). LTNT_FRONTEND_DIST lets an
# isolated test instance serve its own build (e.g. dist_spantest) without
# touching the production dist/ the live server serves.
FRONTEND_DIST = (Path(__file__).parent / "ltnt_frontend"
                 / os.environ.get("LTNT_FRONTEND_DIST", "dist"))
if FRONTEND_DIST.exists():
    # Serve static assets (JS, CSS)
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend_assets")

# In-memory job store
jobs: dict = {}

# === SESSION REHYDRATION (additive) =========================================
# session_id -> job_id index so the frontend can rebuild the canvas after any
# navigation/refresh via GET /api/sessions/{id}/state. The session's images +
# lineage already persist on disk under generated/<session_id>/; this index
# just links the durable id to the live in-memory job (layout, status, etc.).
_session_index: dict = {}
# === END SESSION REHYDRATION ================================================

# === CREATE MAP (delete block to remove radial explore-map feature) =========
try:
    from explore_map.routes import make_router as _create_map_router
    app.include_router(_create_map_router(jobs, GENERATED_DIR))
except Exception as _cm_exc:
    print(f"[server] Create Map router not loaded: {_cm_exc}")
# === END CREATE MAP =========================================================

# === BOARD (AI-Pinterest pin store) — additive; persists pinned images ===
# Copies pinned images into board_store/ (home NFS, reliably server-writable;
# NOT /data autofs) and serves them at /board_images. See board_api.py.
try:
    import board_api
    board_api.mount(app, GENERATED_DIR)
except Exception as _board_exc:
    print(f"[server] Board router not loaded: {_board_exc}")
# === END BOARD ==========================================================

executor = ThreadPoolExecutor(max_workers=1)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Request models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "flux"          # "flux" | "sd35" | "krea2"
    n_images: int = 6
    num_steps: int = 28
    num_clones: int = 3
    num_resampling_steps: int = 3
    guidance_scale: float = 3.5
    rho: float = 0.4
    schedule_mode: str | None = "aggressive"  # "aggressive" (default), "linear", or None (even)
    lookahead_steps: int = 4   # Euler steps for the per-checkpoint preview — the
                               # dominant between-round cost (each step = 1 FLUX
                               # forward pass, ~1s). 4 halves the old cost (8) with
                               # previews still clearly recognizable for selection;
                               # 2 is usable-but-blurry, 8 is sharp/slow.
    lookahead_steps_first: int = 10  # Richer preview for ROUND 1 ONLY: round-1
                               # states sit at σ≈0.9 (largely uncommitted), so
                               # few-step previews read as blurry blobs. More
                               # steps make the first impression legible; later
                               # intervals keep the cheap lookahead_steps.
    augment: bool = True       # seed initial particles from LLM-expanded prompt
                               # variations for semantic diversity (FLUX only).
    num_variations: int = 6    # K distinct prompt variations to spread particles over (1 per particle for wide round-1 range).
    tree_mode: bool = False               # tree layout instead of clustering
    seed: int | None = None
    height: int = 512
    width: int = 512
    # Only used when model="fluxfm": "glass" (exact bridge) or "flow_map"
    # (Weighted-Diamond-Maps-style amortized clone jump).
    clone_mode: str = "glass"
    # Only used when model="flux": how children are bred from selected images.
    # "glass" (default, current behavior) = path-consistent in-trajectory GLASS
    # branching from the live latent. "sdedit" = renoise each clone UP to a
    # higher noise level (SDEdit) then plain-Euler denoise -> more diverse
    # children, less path consistency. rho is the knob in both modes (high rho =
    # stay closer to parent / less divergence; low rho = more divergence).
    spawn_mode: str = "glass"
    # ── EXPLORE FROM THIS PIN (image-seeded SDEdit exploration, FLUX only) ──
    # When set, the session starts from an existing image instead of pure noise:
    # the image is VAE-encoded to a clean latent, re-noised partway to keep its
    # structure, and the normal interactive exploration runs from there. The url
    # is a served path: '/board_images/<f>' (a saved pin) or '/images/<...>' (a
    # generated image). init_strength in [0,1]: higher = re-noise further =
    # explore more / keep less of the original (clamped 0.30–0.95 in the sampler).
    init_image_url: str | None = None
    init_strength: float = 0.55
    # ── SPAN MODE round 1 (seamless first run) ──────────────────────────────
    # When true (and the sampler supports it — currently SANA), round 1 maps
    # the prompt's space with INDEPENDENT fast full images STREAMED in small
    # batches: each batch lands on the canvas as it completes, a novelty meter
    # tracks how much new ground each batch covers, and the round auto-stops
    # when the spread saturates (or the user hits "spread looks good" -> POST
    # /api/jobs/{id}/finish_round). Selected span images then become breeding
    # parents for the classic GLASS rounds (they carry seeded recipes).
    span_round1: bool = False
    span_batch_size: int = 8
    span_max_images: int = 64
    span_num_steps: int = 14      # fast full solve per span image
    span_stop_ratio: float = 0.25  # auto-stop: novelty < ratio × batch-1 novelty
    span_breed_strength: float = 0.55  # SDEdit re-noise level for breeding round 2

class SelectRequest(BaseModel):
    indices: List[int]


class RegionModel(BaseModel):
    x: float
    y: float
    w: float
    h: float


class EditRequest(BaseModel):
    image_b64: str
    operation: str = "edit"
    region: RegionModel | None = None
    prompt: str = ""
    effect_name: str | None = None
    upscale_factor: int = 2


# ── Clustering helper ────────────────────────────────────────────────────────

def cluster_images(pil_images, device):
    """Return normalized [0.05, 0.95] 2D coords for a list of PIL images."""
    from cluster_images import extract_dinov2, l2_normalize

    n = len(pil_images)
    if n == 1:
        return np.array([[0.5, 0.5]])

    features = extract_dinov2(pil_images, device=device)
    features = l2_normalize(features)

    if n < 15:
        from sklearn.metrics.pairwise import cosine_distances
        from sklearn.manifold import MDS
        dist_matrix = cosine_distances(features)
        mds = MDS(n_components=2, dissimilarity='precomputed',
                  random_state=42, normalized_stress='auto')
        coords = mds.fit_transform(dist_matrix)
    else:
        import umap as umap_lib
        n_neighbors = min(15, max(2, n - 1))
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                                min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(features)

    coords -= coords.min(axis=0)
    span = coords.max(axis=0)
    span[span == 0] = 1
    coords = coords / span * 0.90 + 0.05
    return coords


def build_image_list(image_paths, coords, url_prefix, sizes=None):
    """Build JSON-serializable image list from paths + 2D coordinates."""
    images = []
    for i, (img_path, (x, y)) in enumerate(zip(image_paths, coords)):
        entry = {
            "id": i,
            "url": f"{url_prefix}/{img_path.name}",
            "x": float(x),
            "y": float(y),
        }
        if sizes is not None:
            entry["size"] = float(sizes[i])
        images.append(entry)
    return images


def compute_tree_positions(n_total, num_clones, interval_num):
    """Compute tree-layout positions and sizes for particles.

    Layout: particles are grouped as [anchor, clone, clone, ...] x N groups.
    Anchors go on a circle; clones orbit their anchor at a smaller radius.
    Deeper intervals get progressively smaller images and tighter orbits.

    Args:
        n_total: total number of particles
        num_clones: C (group size: 1 anchor + C-1 clones)
        interval_num: which resampling interval (1-indexed), controls depth sizing

    Returns:
        coords: [n_total, 2] positions in [0, 1]
        sizes: [n_total] relative sizes (1.0 = largest)
        is_anchor: [n_total] bool array
    """
    C = max(num_clones, 1)
    N = n_total // C  # number of groups
    if N == 0:
        N = n_total
        C = 1

    coords = np.zeros((n_total, 2))
    sizes = np.zeros(n_total)
    is_anchor = np.zeros(n_total, dtype=bool)

    # Progressive sizing: anchors at depth N match clones from depth N-1
    # Interval 1: anchors=1.0, clones=0.65
    # Interval 2: anchors=0.65, clones=0.42
    # Interval 3: anchors=0.42, clones=0.27
    depth = interval_num - 1
    anchor_size = max(1.0 * (0.65 ** depth), 0.2)
    clone_size = max(anchor_size * 0.65, 0.15)

    # Anchor spread adapts to count — more anchors = wider circle
    anchor_radius = min(0.35, 0.15 + 0.05 * N) if N > 1 else 0.0
    # Clone orbit shrinks with depth so deeper clones cluster tighter
    clone_radius = max(0.12 * (0.7 ** depth), 0.03)

    for g in range(N):
        # Anchor position on a circle
        angle = 2 * np.pi * g / N - np.pi / 2  # start from top
        ax = 0.5 + anchor_radius * np.cos(angle)
        ay = 0.5 + anchor_radius * np.sin(angle)

        anchor_idx = g * C
        if anchor_idx < n_total:
            coords[anchor_idx] = [ax, ay]
            sizes[anchor_idx] = anchor_size
            is_anchor[anchor_idx] = True

        # Clone positions — fan out from anchor, offset away from center
        n_clones_in_group = C - 1
        for j in range(1, C):
            clone_idx = anchor_idx + j
            if clone_idx >= n_total:
                break
            # Spread clones in an arc on the outward side of the anchor
            if n_clones_in_group == 1:
                clone_angle = angle  # same direction as anchor from center
            else:
                spread = min(np.pi * 0.8, np.pi / max(N, 1) * 2)
                clone_angle = angle - spread / 2 + spread * (j - 1) / (n_clones_in_group - 1)
            cx = ax + clone_radius * np.cos(clone_angle)
            cy = ay + clone_radius * np.sin(clone_angle)
            coords[clone_idx] = [cx, cy]
            sizes[clone_idx] = clone_size
            is_anchor[clone_idx] = False

    # Clamp to [0.05, 0.95]
    coords = np.clip(coords, 0.05, 0.95)

    return coords, sizes, is_anchor


# ── Generation worker (interactive) ──────────────────────────────────────────

class GenerateCancelled(Exception):
    """Raised when a session is superseded by a newer generate (takeover)."""


# Which model currently holds the GPU (set on every sampler load, cleared on
# free) — /api/models reports it so the frontend can default to the model that
# will respond FASTEST, and never to one this instance cannot actually load.
_resident_model = {"key": None}

_MODEL_REPOS = {
    "flux":  ("FLUX",   "models--black-forest-labs--FLUX.1-dev",  "FLUX_LOCAL_PATH"),
    "sd35":  ("SD3.5",  "models--stabilityai--stable-diffusion-3.5-medium", None),
    "krea2": ("KREA-2", "models--krea--Krea-2-Raw", "KREA_LOCAL_PATH"),
}


def _model_loadable(key) -> bool:
    """TRUTHFUL loadability: mirrors the loaders' actual resolution order —
    (1) the node-local staged copy (X_LOCAL_PATH), then (2) a complete
    snapshot under the CURRENT HF hub (what from_pretrained will really read).
    The old check hardcoded the /data hub, which this long-running offline
    process often CANNOT read (autofs) — it said available:true for models
    that then failed to load."""
    import glob as _glob
    label, repo, env_var = _MODEL_REPOS[key]
    try:
        if env_var:
            p = os.environ.get(env_var, "")
            # COMPLETE staged copy only: a preempted staging job can leave
            # model_index.json without the component dirs — require the
            # scheduler config + vae dir too (they land late in the copy).
            if (p and os.path.isfile(os.path.join(p, "model_index.json"))
                    and os.path.isfile(os.path.join(
                        p, "scheduler", "scheduler_config.json"))
                    and os.path.isdir(os.path.join(p, "vae"))):
                return True
        hub = os.environ.get("HUGGINGFACE_HUB_CACHE",
                             os.path.join(os.environ.get("HF_HOME", ""), "hub"))
        base = os.path.join(hub, repo)
        snaps = [s for s in _glob.glob(os.path.join(base, "snapshots", "*"))
                 if os.path.isfile(os.path.join(s, "model_index.json"))
                 or _glob.glob(os.path.join(s, "*.safetensors"))]
        if not snaps:
            return False
        if _glob.glob(os.path.join(base, "**", "*.incomplete"), recursive=True):
            return False
        return any(
            _glob.glob(os.path.join(sn, "*.safetensors"))
            or _glob.glob(os.path.join(sn, "**", "*.safetensors"), recursive=True)
            for sn in snaps
        )
    except Exception:
        return False


@app.get("/api/models")
def list_models():
    """Honest availability of each headline model on THIS instance.

    Ordered: the resident (already-on-GPU) model first, then models this
    process can actually load, then the honestly-unavailable rest. The
    frontend defaults to models[0] when available."""
    resident = _resident_model["key"]
    out = []
    for key, (label, repo, env_var) in _MODEL_REPOS.items():
        out.append({
            "key": key, "label": label,
            # The resident model is already on the GPU — it serves without any
            # disk read, so it is available even when this process has lost
            # sight of the hub (autofs); _model_loadable covers the rest.
            "available": _model_loadable(key) or key == resident,
            "resident": key == resident,
        })
    out.sort(key=lambda m: (not m["resident"], not m["available"]))
    return {"models": out,
            "default": next((m["key"] for m in out if m["available"]), None)}


def _friendly_error(exc) -> str:
    """User-facing message for a generation failure — NEVER leak raw traces/paths."""
    sl = str(exc).lower()
    if "out of memory" in sl or "cuda" in sl or "oom" in sl:
        return "The GPU is busy right now (another session may be generating). Please wait a few seconds and try again."
    if "no such file" in sl or ".safetensors" in sl or "not a directory" in sl or "cannot find" in sl or "couldn't connect" in sl or "connection" in sl:
        return "That model isn\'t available right now \u2014 try a different model."
    if "not waiting for selection" in sl or "lock" in sl or "another" in sl:
        return "Another session is currently generating \u2014 please wait a moment and try again."
    return "Sorry \u2014 we couldn\'t generate that. Please try again."


def run_generate(job_id: str, req: dict) -> None:
    """Blocking worker: runs sample_interactive with event-based user selection.

    Generates are SERIALIZED on the single resident model via `_model_lock` —
    they share one warm FLUX sampler, so two at once would corrupt each other.
    A new generate first signals any in-flight session to cancel (so it unwinds
    and releases the lock promptly), then takes the model = session takeover.
    """
    global _current_job_id
    # Take over any in-flight session.
    prev = _current_job_id
    if prev is not None and prev != job_id and prev in jobs:
        jobs[prev]["_cancelled"] = True
        ev = jobs[prev].get("_selection_event")
        if ev is not None:
            ev.set()  # unblock its mock_input so it can see the cancel and unwind

    if not _model_lock.acquire(timeout=6):
        jobs[job_id].update({
            "status": "error", "result": None,
            "error": "Another session is still generating \u2014 wait for it to finish or cancel it, then try again.",
            "progress": 0, "stage": "Busy",
        })
        return
    _current_job_id = job_id
    try:
        jobs[job_id]["status"] = "running"
        result = _do_generate(job_id, req)
        jobs[job_id].update({
            "status": "done", "result": result, "error": None,
            "progress": 100, "stage": "Done",
        })
    except GenerateCancelled:
        jobs[job_id].update({
            "status": "cancelled", "result": None, "error": None,
            "progress": 0, "stage": "Superseded by a newer prompt",
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        jobs[job_id].update({
            "status": "error", "result": None, "error": _friendly_error(exc),
            "progress": 0, "stage": "Error",
        })
        # A failure may have half-mutated the warm sampler — drop it so the next
        # generate reloads cleanly.
        try:
            _free_previous_sampler()
        except Exception:
            pass
    finally:
        if _current_job_id == job_id:
            _current_job_id = None
        _model_lock.release()


_model_lock = threading.Lock()  # serialize generates on the one resident model
_current_job_id = None          # job_id currently holding the model (for takeover)
_active_sampler = None  # track current sampler for cleanup
_warm_flux = None       # FLUX sampler kept resident across generations (warm reuse)


def _warm_augment_llm():
    """Fully warm the prompt-augment LLM (Qwen) on the device the span round's
    free-VRAM guard would pick, so the FIRST augment call of a session runs at
    warm speed and variations condition by batch 2-3 (critic blocker #1).

    Next to a resident FLUX the guard picks CPU (<12GB free VRAM) — warm the
    CPU path: load + float32 cast + a tiny generation. Never touches the GPU,
    never raises, never delays /api/status readiness (runs in a daemon thread).
    """
    try:
        import prompt_augment
        aug_dev = "cpu"
        try:
            if torch.cuda.is_available():
                free_b, _tot = torch.cuda.mem_get_info()
                if free_b >= 12 * (2 ** 30):
                    # Plenty of VRAM free NOW, but generation will consume it;
                    # the CPU path is the one the in-session guard uses next to
                    # a loaded image model — warm that. (CUDA warm would also
                    # hold ~7GB hostage.) Keep CPU either way.
                    aug_dev = "cpu"
        except Exception:
            pass
        print(f"[ltnt] warming prompt-augment LLM (Qwen) on {aug_dev}...", flush=True)
        prompt_augment.warm(aug_dev)
        print("[ltnt] prompt-augment LLM warm and resident.", flush=True)
    except Exception as e:
        print(f"[ltnt] augment-LLM warm failed (will lazy-load): {e}", flush=True)


def _boot_worker():
    """Background boot: pre-imports + eager FLUX + prompt-LLM, with REAL progress.

    Runs off the event loop so uvicorn binds immediately; /api/status reports
    each named phase. Holds _model_lock during the eager FLUX load so a racing
    generate cannot double-load the model. Same semantics as the old blocking
    startup hook otherwise (lazy fallbacks are all preserved).
    """
    global _warm_flux
    try:
        _set_boot("importing libraries", 0.02, "starting the engine")
        try:
            import cluster_images  # noqa: F401 — side-effect import (umap/numba JIT)
            _set_boot("importing libraries", 0.10, "clustering stack ready")
            import flux_fmtt_dev   # noqa: F401 — FluxDevInteractive + IncrementalLayout
            _set_boot("importing libraries", 0.16, "sampler stack ready")
        except Exception as _e:
            import traceback
            print(f"[server] WARNING: pre-import failed — first Generate will be slow: {type(_e).__name__}: {_e}")
            traceback.print_exc()
        try:
            import sd35_fmtt   # noqa: F401 — optional, only used for model='sd35'
        except Exception as _e:
            print(f"[server] (sd35_fmtt not imported up front: {_e})")

        # Escape hatch for secondary/test instances sharing a GPU with a
        # resident production server: skip the eager FLUX/Qwen loads.
        # The prompt-LLM warm is CPU/RAM-only, so it still runs (in its own
        # low-priority thread) unless explicitly disabled — otherwise the
        # first span round's variations land ~4 batches late (critic
        # blocker: 48 near-duplicate images before the first variation).
        if os.environ.get("LTNT_SKIP_EAGER_LOAD"):
            print("[ltnt] LTNT_SKIP_EAGER_LOAD set — skipping eager model loads.", flush=True)
            if not os.environ.get("LTNT_SKIP_QWEN_WARM"):
                threading.Thread(target=_warm_augment_llm, daemon=True).start()
            _set_boot("ready", 1.0, "", ready=True)
            return

        got_lock = _model_lock.acquire(timeout=600)
        try:
            if _warm_flux is None:
                from flux_fmtt_dev import FluxDevInteractive
                device = get_device()
                print("[ltnt] eager-loading FLUX at startup (while /data is fresh)...", flush=True)
                _stop = threading.Event()
                threading.Thread(
                    target=_watch_cuda_mem,
                    args=("loading the image model", _EXPECTED_BYTES["flux"], _stop, 0.16, 0.74),
                    daemon=True).start()
                try:
                    _warm_flux = FluxDevInteractive(prompt="a lighthouse on a cliff",
                                                    device=device, keep_text_encoders=True)
                finally:
                    _stop.set()
                print("[ltnt] FLUX eager-load complete; model resident.", flush=True)
                _resident_model["key"] = "flux"
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ltnt] eager FLUX load failed (will fall back to lazy load): {e}", flush=True)
        finally:
            if got_lock:
                _model_lock.release()

        _set_boot("warming up", 0.93, "loading the prompt assistant")
        # Warm the augment LLM in the background so "ready" is never delayed
        # by it — /api/status readiness only ever waited on FLUX.
        threading.Thread(target=_warm_augment_llm, daemon=True).start()
        _set_boot("ready", 1.0, "", ready=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        # Never leave the app gated on a failed boot — mark ready so the lazy
        # load paths can still serve.
        _set_boot("ready", 1.0, f"boot warning: {type(e).__name__}", ready=True)


@app.on_event("startup")
def _start_boot_worker():
    threading.Thread(target=_boot_worker, daemon=True).start()


def _load_sampler_with_progress(job_id, label, model_key, build_fn):
    """Run a cold sampler load with REAL progress on the job (5 -> 45%).

    The fraction is GPU bytes allocated / the model's expected footprint —
    measured, not simulated. Also mirrors into /api/status ("switching model")
    and widens the job's segment-0 progress base so the generation phase that
    follows starts at 45% instead of 10% (the load really was ~half the wait).
    """
    expected = _EXPECTED_BYTES.get(model_key, 20e9)
    stop = threading.Event()

    def _on_frac(frac):
        j = jobs.get(job_id)
        if j is None:
            return
        pct = 5 + int(40 * frac)
        if j.get("progress", 0) < pct:
            j["progress"] = pct
            j["stage"] = label

    threading.Thread(
        target=_watch_cuda_mem,
        args=(f"switching model ({model_key})", 	expected, stop, 0.0, 1.0),
        kwargs={"on_frac": _on_frac}, daemon=True).start()
    try:
        sampler = build_fn()
    finally:
        stop.set()
        _set_boot("ready", 1.0, "", ready=True)
    j = jobs.get(job_id)
    if j is not None:
        j["_seg0_base"] = 45   # generation now owns 45 -> 100%
        if j.get("progress", 0) < 45:
            j["progress"] = 45
    return sampler


def _free_previous_sampler():
    """Aggressively release GPU memory held by the previous sampler.

    A simple `del _active_sampler` isn't enough — the sampler holds several
    large submodules (pipeline, transformer, VAE, DINO) and closures from the
    finished generation thread may still reference them. Moving each submodule
    to CPU, then nulling the attributes, then collecting, reliably reclaims
    GPU memory on the next load.
    """
    global _active_sampler, _warm_flux
    _warm_flux = None  # freeing the model invalidates any warm cache
    if _active_sampler is None:
        return

    def _to_cpu(obj):
        try: obj.to('cpu')
        except Exception: pass

    for attr in ('pipeline', 'transformer', 'vae', 'text_encoder', 'text_encoder_2',
                 'dino_model'):
        sub = getattr(_active_sampler, attr, None)
        if sub is not None:
            _to_cpu(sub)

    # Drop tensor / module attributes so no refs linger on the instance.
    for attr in ('pipeline', 'transformer', 'vae', 'text_encoder', 'text_encoder_2',
                 'tokenizer', 'tokenizer_2', 'dino_model', 'dino_processor',
                 'cached_prompt_embeds', 'cached_pooled_prompt_embeds',
                 'cached_text_ids'):
        try: setattr(_active_sampler, attr, None)
        except Exception: pass

    _active_sampler = None
    _resident_model["key"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try: torch.cuda.ipc_collect()
        except Exception: pass
    print("[server] Freed previous model from GPU")


def _resolve_init_image(url: str):
    """Map a served image url to a file on disk for EXPLORE-FROM-THIS-PIN.

    Accepts:
      '/board_images/<file>'  -> board_store/images/<file>  (a saved pin)
      '/images/<...>'         -> GENERATED_DIR/<...>          (a generated image)
    Path-traversal-safe: the resolved path MUST stay inside the mount root, else
    None is returned (caller falls back to normal generation). Returns a Path or
    None (never raises) so a bad url can NEVER 500 the session.
    """
    try:
        if not url or not isinstance(url, str):
            return None
        clean = url.split("?")[0].split("#")[0]
        if clean.startswith("/board_images/"):
            from board_api import BOARD_IMAGES_DIR as _root
            rel = clean[len("/board_images/"):]
        elif clean.startswith("/images/"):
            _root = GENERATED_DIR
            rel = clean[len("/images/"):]
        else:
            return None
        root = Path(_root).resolve()
        src = (root / rel).resolve()
        if root != src and root not in src.parents:
            return None  # escaped the mount root
        if not src.is_file():
            return None
        return src
    except Exception as e:
        print(f"[image-seed] resolve failed for {url!r}: {e}")
        return None


# === SPAN MODE (round 1) =====================================================
# Round 1 of a fresh session = "map the prompt's space": independent fast FULL
# images streamed to the canvas batch-by-batch, a novelty meter that says in
# plain words whether new directions are still appearing, auto-stop when the
# spread saturates, and a user early-stop (POST /api/jobs/{id}/finish_round).
# The selected images then breed the classic GLASS rounds — every span image
# carries a seeded recipe (lineage_span.json) so it is exactly reproducible.

def _span_place_batch(new_feats, prev_feats, prev_coords, rng):
    """Canvas coords for a new batch — INCREMENTAL: already-shown images never
    move (no jarring re-shuffles). First batch: MDS over cosine distances.
    Later batches: each image lands near its most similar already-shown
    images (weighted average of the top-3 neighbours + a small outward jitter),
    then a light de-overlap nudge on the NEW points only."""
    if prev_feats is None or prev_coords is None or len(prev_coords) == 0:
        from sklearn.metrics.pairwise import cosine_distances
        from sklearn.manifold import MDS
        n = new_feats.shape[0]
        if n == 1:
            return np.array([[0.5, 0.5]])
        dist = cosine_distances(new_feats)
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, normalized_stress="auto")
        c = mds.fit_transform(dist)
        c -= c.min(axis=0)
        span = c.max(axis=0); span[span == 0] = 1
        return c / span * 0.80 + 0.10
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(new_feats, prev_feats)      # [B, N_prev]
    out = np.zeros((new_feats.shape[0], 2))
    for i in range(new_feats.shape[0]):
        k = min(3, sims.shape[1])
        top = np.argsort(sims[i])[::-1][:k]
        w = np.clip(sims[i][top], 1e-3, None)
        w = w / w.sum()
        base = (prev_coords[top] * w[:, None]).sum(axis=0)
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.04, 0.09)
        out[i] = base + r * np.array([np.cos(ang), np.sin(ang)])
    # Light de-overlap: push each NEW point off any point closer than min_sep.
    allc = np.vstack([prev_coords, out])
    n_prev = len(prev_coords)
    min_sep = 0.05
    for _ in range(12):
        for i in range(len(out)):
            p = out[i]
            d = allc - p
            dist = np.linalg.norm(d, axis=1)
            dist[n_prev + i] = np.inf
            j = int(np.argmin(dist))
            if dist[j] < min_sep:
                away = p - allc[j]
                nrm = np.linalg.norm(away)
                away = (away / nrm) if nrm > 1e-8 else \
                    np.array([np.cos(rng.uniform(0, 6.28)),
                              np.sin(rng.uniform(0, 6.28))])
                out[i] = p + away * (min_sep - dist[j]) * 0.7
                allc[n_prev + i] = out[i]
    return np.clip(out, 0.03, 0.97)


def _run_span_round1(job_id, req, sampler, output_dir, session_id, device,
                     master_seed):
    """Run the streaming span round + its selection wait. Returns a dict with
    the selected parents (clean latents + prompt/lineage info) for the
    breeding rounds, or raises GenerateCancelled. Blocking (worker thread)."""
    import time as _time
    from sklearn.metrics.pairwise import cosine_distances
    from PIL import Image as PILImage

    job = jobs[job_id]
    prompt = req["prompt"]
    batch_size = max(2, min(int(req.get("span_batch_size", 8) or 8), 16))
    max_images = max(batch_size, min(int(req.get("span_max_images", 64) or 64), 96))
    span_steps = max(6, min(int(req.get("span_num_steps", 14) or 14), 28))
    stop_ratio = float(req.get("span_stop_ratio", 0.25) or 0.25)
    _MIN_BATCHES = 3   # never auto-stop before the meter has real evidence
    guidance = float(req.get("guidance_scale", 4.5) or 4.5)
    height = int(req.get("height", 512) or 512)
    width = int(req.get("width", 512) or 512)

    url_prefix = f"/images/{session_id}/interval_1"
    rng = np.random.default_rng(int(master_seed) % (2 ** 31))
    state = {"features": None, "coords": None, "urls": [], "pils": [],
             "novelty": [], "stop_reason": "cap", "t_first_batch": None,
             "t0": _time.perf_counter(),
             # Auto-stop must not fire while the prompt-interpretation pass is
             # still pending — otherwise a slow (CPU) augment means the whole
             # round runs on the base prompt and gets called "repeating"
             # before a single interpretation was ever tried (seen on FLUX).
             "aug_done": False, "variations_used": False}
    _augment_requested = bool(req.get("augment", True)
                              and hasattr(sampler, "add_prompts"))

    # ── Background prompt augmentation: batch 1 starts IMMEDIATELY on the
    # artist's own prompt; the local LLM expands variations in parallel and
    # later batches spread over them as they register (append-only, no race).
    if _augment_requested:
        def _augment_bg():
            try:
                import prompt_augment
                K = max(1, int(req.get("num_variations", 6) or 6))
                # The augment LLM (~7GB on GPU) runs CONCURRENTLY with span
                # batches. Next to SANA (~9GB resident) that's fine; next to
                # FLUX + kept T5 (~34GB resident) it OOMs the card. Pick the
                # device by real free VRAM — CPU is slower (variations join a
                # few batches later) but can never kill the stream.
                aug_dev = device
                try:
                    if torch.cuda.is_available():
                        free_b, _tot = torch.cuda.mem_get_info()
                        if free_b < 12 * (2 ** 30):
                            aug_dev = "cpu"
                            print(f"[span] augment on CPU (only "
                                  f"{free_b / 2**30:.1f}GB VRAM free)")
                except Exception:
                    pass
                # TWO-STAGE augment so the FIRST variations land fast: a
                # small n=2 call finishes in ~25-30s on warm CPU (the model
                # stops at EOS after 2 lines), letting variations condition
                # by batch 2-3; the remaining K-2 arrive a batch later.
                seen = {prompt.strip().lower()}
                stages = [2, K - 2] if K > 2 else [K]
                for stage_n in stages:
                    if stage_n <= 0:
                        continue
                    extra = prompt_augment.augment_prompt(
                        prompt, n=stage_n, device=aug_dev)
                    fresh = []
                    for v in extra:
                        if v and v.strip().lower() not in seen:
                            fresh.append(v); seen.add(v.strip().lower())
                    if fresh:
                        sampler.add_prompts(fresh)
                        job["_span_prompts"] = list(getattr(sampler, "prompts", []))
                        print(f"[span] augment ready: +{len(fresh)} variations "
                              f"({getattr(sampler, 'num_prompts', 1)} prompts total)")
            except Exception as exc:
                print(f"[span] augment failed ({exc}); staying on base prompt.")
            finally:
                state["aug_done"] = True
        threading.Thread(target=_augment_bg, daemon=True).start()

    def _span_progress(batch_idx, n_done, k, nsteps):
        # Honest per-batch progress: each batch is what the user is waiting on.
        bsz = min(batch_size, max_images - n_done)
        job["progress"] = min(99, int(100.0 * k / max(nsteps, 1)))
        job["stage"] = (f"painting images {n_done + 1}–{n_done + bsz} "
                        f"(solve step {k + 1}/{nsteps})")

    def _prompt_plan(batch_idx, size):
        n = getattr(sampler, "num_prompts", 1)
        if batch_idx == 0 or n <= 1:
            return [0] * size   # batch 1 = the artist's exact prompt
        return [(batch_idx * size + j) % n for j in range(size)]

    def _on_batch(info):
        feats = info["features"]
        prev = state["features"]
        if prev is None:
            d = cosine_distances(feats, feats)
            np.fill_diagonal(d, np.inf)
            nov = float(np.mean(np.min(d, axis=1)))
            state["t_first_batch"] = _time.perf_counter() - state["t0"]
        else:
            nov = float(np.mean(np.min(cosine_distances(feats, prev), axis=1)))
        state["novelty"].append(nov)
        new_coords = _span_place_batch(feats, prev, state["coords"], rng)
        state["coords"] = (np.vstack([state["coords"], new_coords])
                           if state["coords"] is not None else new_coords)
        state["features"] = (np.vstack([prev, feats])
                             if prev is not None else feats.copy())
        state["urls"].extend(f"{url_prefix}/{Path(p).name}" for p in info["paths"])
        state["pils"].extend(info["pil_images"])

        total = len(state["urls"])
        base_nov = max(state["novelty"][0], 1e-6)
        # AUTO-STOP RULE (critic-validated): stop when novelty has been FLAT
        # for 2 consecutive batches — i.e. the last two batch-to-batch changes
        # are both smaller than a tolerance scaled to this prompt's own
        # baseline. (The old rule, novelty < stop_ratio x baseline, almost
        # never fired: with prompt variations the novelty floor stays high.)
        # The hard image cap remains the backstop.
        series = state["novelty"]
        flat_eps = max(0.015, 0.10 * base_nov)
        flat2 = (len(series) >= _MIN_BATCHES
                 and abs(series[-1] - series[-2]) < flat_eps
                 and abs(series[-2] - series[-3]) < flat_eps)
        if any(int(i) > 0 for i in info["prompt_idx"]):
            state["variations_used"] = True
        # Defer the auto-stop while the interpretation pass is pending: the
        # base prompt repeating is EXPECTED until variations join the stream.
        aug_pending = (_augment_requested and not state["aug_done"]
                       and not state["variations_used"])
        saturating = flat2 and not aug_pending
        images = [{"id": i, "url": u,
                   "x": float(state["coords"][i, 0]),
                   "y": float(state["coords"][i, 1])}
                  for i, u in enumerate(state["urls"])]
        # HONEST trend phrase: only claim "new directions" when the novelty
        # series actually supports it — either comfortably above an absolute
        # floor (ground-truth: near-dupes ≈ 0.09, real spread ≈ 0.31-0.50
        # DINO novelty) AND ≥ 80% of this prompt's own baseline, or visibly
        # rising batch-over-batch. A uniform stream must never read as
        # "STILL FINDING NEW DIRECTIONS" (critic blocker #1b).
        _NOV_FLOOR = 0.15
        rising = len(series) >= 2 and series[-1] > series[-2] + flat_eps
        supports_new = (nov >= max(_NOV_FLOOR, 0.8 * base_nov)) or rising
        if saturating:
            trend = "directions repeating — good time to pick"
        elif not supports_new and aug_pending:
            trend = ("very similar so far — adding new interpretations "
                     "of your prompt…")
        elif not supports_new:
            trend = ("staying close to one look — pick what stands out, "
                     "or stop here")
        else:
            trend = "still finding new directions"
        stage = (f"mapping your prompt's space — {total} images on the canvas "
                 f"(up to {max_images}) · {trend}")
        job.update({
            "images": images, "interval": 1,
            "stage": stage, "progress": 100,
            "span": {
                "active": True,
                "batch_n": len(state["novelty"]),
                "n_images": total, "max_images": max_images,
                "batch_size": batch_size,
                "novelty": round(nov, 4),
                "novelty_baseline": round(base_nov, 4),
                "span_series": [round(v, 4) for v in state["novelty"]],
                "saturating": bool(saturating),
                "message": trend,
                "n_prompt_variations": int(getattr(sampler, "num_prompts", 1)),
            },
        })
        if job.get("_cancelled"):
            state["stop_reason"] = "cancelled"
            return False
        if job.pop("_finish_span", False):
            state["stop_reason"] = "user"
            return False
        if saturating:
            state["stop_reason"] = "auto"
            return False
        return True

    job["stage"] = "mapping your prompt's space — first images in a few seconds"
    result = sampler.sample_span_round(
        master_seed=int(master_seed),
        output_dir=output_dir,
        batch_size=batch_size, max_images=max_images,
        num_steps=span_steps, guidance_scale=guidance,
        img_shape=(height, width),
        prompt_plan=_prompt_plan, on_batch=_on_batch,
        progress_callback=_span_progress,
    )
    if job.get("_cancelled"):
        raise GenerateCancelled()

    n_total = result["n_images"]
    stop_reason = state["stop_reason"] if state["stop_reason"] != "cap" else \
        result.get("stop_reason", "cap")
    span_pub = dict(job.get("span") or {})
    span_pub.update({"active": False, "stop_reason": stop_reason,
                     "first_batch_seconds": (round(state["t_first_batch"], 1)
                                             if state["t_first_batch"] else None)})
    _stop_words = {"auto": "the spread stopped finding new directions",
                   "user": "you called it — good spread",
                   "cap": f"hit the {max_images}-image cap"}
    job.update({
        "status": "waiting_selection",
        "stage": (f"round 1 mapped ({n_total} images — "
                  f"{_stop_words.get(stop_reason, stop_reason)}). "
                  f"Pick the directions you like."),
        "progress": 100,
        "interval": 1,
        "total_intervals": int(req.get("num_resampling_steps", 3)) + 1,
        "span": span_pub,
    })

    # ── Wait for the artist's picks (same contract as mock_input) ──────────
    selection_event = job["_selection_event"]
    selection_event.clear()
    print(f"  [span] waiting for selection over {n_total} span images...")
    _waited, _MAX_WAIT = 0.0, 300.0
    selected = None
    while True:
        while not selection_event.wait(timeout=1):
            _waited += 1.0
            if _waited >= _MAX_WAIT:
                print("  [span] selection wait timed out -> releasing model")
                job["_cancelled"] = True
                job["stage"] = "Session timed out — start a new prompt"
                break
        if job.get("_cancelled"):
            raise GenerateCancelled()
        # Refine during span selection is an instant no-op: span images are
        # already FULL fast solves, so "full quality" = the same file.
        refine_req = job.pop("_refine_request", None)
        if refine_req is not None:
            refined = dict(job.get("refined", {}))
            for i in refine_req:
                if 0 <= int(i) < n_total:
                    refined[str(int(i))] = state["urls"][int(i)]
            job["refined"] = refined
            job["status"] = "waiting_selection"
            selection_event.clear()
            continue
        selected = job.pop("_user_selection", None)
        break
    if not selected:
        selected = list(range(min(n_total, 6)))   # fallback: first few
    selected = sorted({int(i) for i in selected if 0 <= int(i) < n_total})
    if len(selected) > 8:
        print(f"[span] clamping {len(selected)} picks -> 8 breeding parents")
        selected = selected[:8]

    job["status"] = "running"
    job["progress"] = 0
    job["stage"] = (f"breeding new variations from your {len(selected)} "
                    f"pick{'s' if len(selected) != 1 else ''}...")
    print(f"  [span] selected parents: {selected}")

    lat = result["latents"]
    return {
        "selected": selected,
        "init_latents": lat[selected].clone(),
        "parent_prompt_idx": [int(result["prompt_idx"][i]) for i in selected],
        "parent_ids": [int(i) for i in selected],
        "features": state["features"],
        "coords": state["coords"],
        "urls": list(state["urls"]),
        "pils": list(state["pils"]),
        "n_images": n_total,
    }
# === END SPAN MODE ===========================================================


def _do_generate(job_id: str, req: dict) -> dict:
    global _active_sampler, _warm_flux
    from PIL import Image as PILImage
    import types

    # NB: freeing is now done per-branch below — the FLUX path keeps the model
    # warm across generations instead of always freeing + reloading.

    prompt = req["prompt"]
    model = req.get("model", "flux")
    _KNOWN_MODELS = {"flux", "krea2", "sd35"}
    if model not in _KNOWN_MODELS:
        raise ValueError("Unknown model " + repr(model) + "; expected one of " + str(sorted(_KNOWN_MODELS)))
    n_images = max(1, min(int(req["n_images"]), 24))  # robustness: cap particles (was unbounded -> OOM)
    _req_n = req.get("n_images")
    if n_images != _req_n:
        print("[robustness] clamped n_images " + str(_req_n) + " -> " + str(n_images))
    num_steps = req["num_steps"]
    num_clones = req["num_clones"]
    num_resampling_steps = req["num_resampling_steps"]
    tree_mode = req.get("tree_mode", False)
    guidance_scale = req["guidance_scale"]
    rho = req["rho"]
    schedule_mode = req["schedule_mode"]
    seed = req["seed"]
    height = req["height"]
    width = req["width"]

    device = get_device()
    # session_id is minted up front in start_generate (durable, in the job
    # record + _session_index from t=0 so rehydration works mid-generation).
    # Fallback mint keeps any non-endpoint caller working.
    session_id = jobs[job_id].get("session_id") or str(uuid.uuid4())[:8]
    jobs[job_id]["session_id"] = session_id
    _session_index.setdefault(session_id, job_id)
    output_dir = str(GENERATED_DIR / session_id)

    num_particles = n_images

    def set_progress(pct, stage):
        jobs[job_id]["progress"] = pct
        jobs[job_id]["stage"] = stage

    model_label = "FLUX" if model == "flux" else "SD3.5"
    set_progress(2, "Starting up...")
    set_progress(5, "Loading image model...")

    # ── Segment-level progress callback used by the sampler ───────────────────
    # Each generation segment (between resamples) is its own 0 → 100% cycle —
    # that matches what the user actually waits on: a single Generate or Branch.
    # During the first segment, we reserve the first 10% for model+DINO loading
    # so the bar doesn't reset to 0 after showing the loading captions.
    # The bar is monotonic within a session: explicit resets (between branches)
    # are done via direct assignment in mock_input, which this function does
    # not clobber.
    step_progress_active = [False]
    def sampler_progress(frac, phase, seg_idx=0, n_segments=1):
        step_progress_active[0] = True
        frac = min(max(frac, 0.0), 1.0)
        if seg_idx == 0:
            # After a cold model load the load phase already consumed 5-45%
            # of the bar (real, measured); generation owns the rest.
            base = jobs[job_id].get("_seg0_base", 10)
            pct = base + int(frac * (100 - base))
        else:
            pct = int(frac * 100)
        pct = min(pct, 100)
        current = jobs[job_id].get("progress", 0)
        jobs[job_id]["progress"] = max(pct, current)
        jobs[job_id]["stage"] = phase

    prompt_assignment = None
    # EXPLORE FROM THIS PIN: clean seed latent for image-seeded exploration.
    # Stays None unless model=="flux" AND init_image_url resolves + encodes ok;
    # any failure leaves it None -> normal pure-noise generation (never 500s).
    init_latent = None
    init_strength = float(req.get("init_strength", 0.55))
    if model == "flux":
        from flux_fmtt_dev import FluxDevInteractive
        # Reuse the resident model ONLY if it is still intact. A prior CREATE MAP
        # (or any heavy backend) frees the warm sampler's pipeline/text-encoders
        # to reclaim VRAM but cannot see this module's _warm_flux pointer, so the
        # object may be gutted. Validate before reuse; otherwise reconstruct fresh
        # (this also fixes the "set_prompts() needs the text encoders" crash on the
        # CREATE MAP -> GENERATE flow).
        warm_ok = (_warm_flux is not None
                   and getattr(_warm_flux, "pipeline", None) is not None
                   and getattr(_warm_flux.pipeline, "text_encoder", None) is not None)
        if warm_ok:
            # Warm path: reuse the resident model (no ~17s reload).
            sampler = _warm_flux
            set_progress(8, "Reusing warm model...")
        else:
            _free_previous_sampler()
            sampler = _load_sampler_with_progress(
                job_id, "Loading the FLUX image model (one-time, ~1 min)...",
                "flux",
                lambda: FluxDevInteractive(prompt=prompt, device=device,
                                           keep_text_encoders=True))
            _warm_flux = sampler

        # ── Prompt augmentation for diversity (local LLM on Babel, no spend) ──
        # The artist's ORIGINAL prompt is always kept as a seed (a pure latent
        # anchor); the rest are LIGHT nearby variations (added adjectives/mood/
        # lighting). So some anchors stay on the exact prompt, others drift —
        # exploring the neighborhood rather than teleporting to new concepts.
        variations = [prompt]
        # SPAN sessions must NOT pay the 10-30s upfront LLM wait — batch 1
        # starts immediately on the artist's exact prompt and the span
        # producer's background thread add_prompts() the variations as they
        # arrive (later batches spread over them).
        _span_flux = bool(req.get("span_round1")
                          and hasattr(sampler, "sample_span_round"))
        if req.get("augment", True) and not _span_flux:
            set_progress(6, "Exploring different ways to interpret your prompt...")
            try:
                import prompt_augment
                K = max(1, int(req.get("num_variations", 3)))
                # Per-token ticks (6 -> 9%): the LLM interpretation pass is a
                # 10-30s wait that used to freeze the bar at 6%.
                extra = (prompt_augment.augment_prompt(
                            prompt, n=K - 1, device=device,
                            token_cb=lambda done, total: set_progress(
                                6 + int(3 * min(done / max(total, 1), 1.0)),
                                "Exploring different ways to interpret your prompt..."))
                         if K > 1 else [])
                # de-dup + keep the original first
                seen = {prompt.strip().lower()}
                for v in extra:
                    if v and v.strip().lower() not in seen:
                        variations.append(v); seen.add(v.strip().lower())
                print(f"[augment] {len(variations)} seeds (orig + {len(variations)-1}): {variations}")
            except Exception as exc:
                print(f"[augment] failed ({exc}); using base prompt.")
                variations = [prompt]
        sampler.set_prompts(variations)   # warm re-embed of all variations
        K = len(variations)
        # Spread the initial particles round-robin across the seeds (variation 0
        # is the original, so some particles are seeded from the pure prompt).
        prompt_assignment = [i % K for i in range(num_particles)]
        _active_sampler = sampler

        # ── EXPLORE FROM THIS PIN: encode the seed image -> clean latent ──
        # Resolve url -> file, load with PIL, VAE-encode to z0 at the session's
        # resolution. Wrapped so a bad/missing image silently falls back to
        # normal pure-noise generation rather than failing the job.
        init_url = req.get("init_image_url")
        if init_url:
            try:
                src = _resolve_init_image(init_url)
                if src is None:
                    print(f"[image-seed] {init_url!r} did not resolve; normal generation.")
                else:
                    pil_seed = PILImage.open(src)
                    init_latent = sampler.encode_image_to_latent(
                        pil_seed, img_shape=(height, width))
                    set_progress(7, "Seeding from your image...")
                    print(f"[image-seed] encoded {src} -> latent {tuple(init_latent.shape)} "
                          f"(strength={init_strength}).")
            except Exception as exc:
                print(f"[image-seed] encode failed ({exc}); normal generation.")
                init_latent = None
    elif model == "krea2":
        # Krea-2-Raw is heavy (~37GB resident) and 512-ONLY: 1024 OOMs on L40S.
        # Force the session to 512x512 regardless of what the frontend requested.
        # Goes through _free_previous_sampler (like sana) so any warm FLUX is
        # dropped before this single big model loads.
        #
        # DIVERSITY: krea2 now supports prompt-augmented round-1 diversity
        # (set_prompts + per-particle prompt_idx). When augment is on we keep the
        # Gemma text encoder resident just long enough to embed the LLM prompt
        # variations, then free it (free_text_encoder) to reclaim VRAM before the
        # heavy solve. Without this, all round-1 particles share ONE prompt embed
        # and come out near-identical (the "not diverse enough" complaint).
        height = width = 512
        _free_previous_sampler()   # also drops any warm FLUX
        from krea2_fmtt import Krea2Interactive
        _krea_augment = bool(req.get("augment", True))
        sampler = _load_sampler_with_progress(
            job_id, "Loading the KREA-2 image model (one-time, ~2 min)...",
            "krea2",
            lambda: Krea2Interactive(prompt=prompt, device=device,
                                     dtype=torch.bfloat16,
                                     keep_text_encoder=_krea_augment))
        _active_sampler = sampler

        # ── Prompt augmentation for diversity (local LLM on Babel, no spend) ──
        # Mirror the FLUX non-span path: original prompt stays as anchor 0, the
        # LLM adds K-1 nearby interpretations, particles round-robin across them.
        variations = [prompt]
        if _krea_augment:
            set_progress(6, "Exploring different ways to interpret your prompt...")
            try:
                import prompt_augment
                K = max(1, int(req.get("num_variations", 6) or 6))
                extra = (prompt_augment.augment_prompt(
                            prompt, n=K - 1, device=device,
                            token_cb=lambda done, total: set_progress(
                                6 + int(3 * min(done / max(total, 1), 1.0)),
                                "Exploring different ways to interpret your prompt..."))
                         if K > 1 else [])
                seen = {prompt.strip().lower()}
                for v in extra:
                    if v and v.strip().lower() not in seen:
                        variations.append(v); seen.add(v.strip().lower())
                print(f"[augment/krea2] {len(variations)} seeds "
                      f"(orig + {len(variations)-1}): {variations}")
            except Exception as exc:
                print(f"[augment/krea2] failed ({exc}); using base prompt.")
                variations = [prompt]
            try:
                sampler.set_prompts(variations)   # warm re-embed of all variations
            except Exception as exc:
                print(f"[augment/krea2] set_prompts failed ({exc}); base prompt.")
                variations = [prompt]
            finally:
                # Variations are embedded; free the Gemma encoder before the
                # heavy 37GB solve (no mid-stream re-prompting in classic flow).
                try:
                    sampler.free_text_encoder()
                except Exception:
                    pass
        K = max(1, len(variations))
        # Spread the initial particles round-robin across the variations
        # (variation 0 is the artist's exact prompt, so some anchors stay pure).
        prompt_assignment = [i % K for i in range(num_particles)]
    else:
        _free_previous_sampler()
        from sd35_fmtt import SD35FMTT
        sampler = _load_sampler_with_progress(
            job_id, "Loading the SD3.5 image model (one-time, ~1 min)...",
            "sd35", lambda: SD35FMTT(prompt=prompt, device=device, skip_t5=True))
        _active_sampler = sampler

    # Non-regressing: after a cold model load the bar is already at ~45%
    # (real load progress) — never yank it back down to 12.
    jobs[job_id]["progress"] = max(jobs[job_id].get("progress", 0), 12)
    jobs[job_id]["stage"] = "Image model ready..."
    _resident_model["key"] = model  # honest /api/models ordering

    # Monkey-patch _load_dino to use DINOv2 (public) instead of DINOv3 (gated).
    # SanaInteractive and FluxFMWDMInteractive already use DINOv2 internally
    # with the right load flags; only patch the older flux/sd35 samplers.
    if model not in ("krea2",):
        def _load_dino_v2(self):
            # Idempotent: a warm-reused sampler already has DINO resident.
            if getattr(self, "dino_model", None) is not None:
                return
            from transformers import AutoImageProcessor, AutoModel
            dino_id = "facebook/dinov2-base"
            print(f"Loading DINOv2-B ({dino_id}) in place of DINOv3...")
            self.dino_processor = AutoImageProcessor.from_pretrained(dino_id)
            self.dino_model = AutoModel.from_pretrained(dino_id).to(self.device).eval()
            print("DINOv2-B loaded.")
        sampler._load_dino = types.MethodType(_load_dino_v2, sampler)

    # ── Interactive mock_input: blocks until frontend sends selection ─────────
    # At each resampling point, sample_interactive calls input() to get user
    # selection. We intercept this to:
    #   1. Read the preview images that were just saved to disk
    #   2. Cluster them and set job status to "waiting_selection" with image URLs
    #   3. Block on a threading.Event until POST /api/jobs/{id}/select arrives
    #   4. Return the user's selected indices as a comma-separated string

    selection_event = threading.Event()
    jobs[job_id]["_selection_event"] = selection_event
    interval_count = [0]
    prev_selected = [None]  # track previous selection for incremental layout
    _orig_input = builtins.input

    # Incremental layout for stable positioning across rounds
    from flux_fmtt_dev import IncrementalLayout, derive_seed
    inc_layout = IncrementalLayout(n_neighbors=15, min_dist=0.2)
    jobs[job_id]["_inc_layout"] = inc_layout  # store for adjust_layout endpoint
    jobs[job_id]["_dino_features"] = None

    # ===== SPAN MODE round 1 (seamless first run) ===========================
    # One master seed for the whole session: span roots derive from it, and
    # the breeding rounds get their own derived master so no draw collides.
    _session_master_seed = req.get("master_seed")
    if _session_master_seed is None:
        import secrets
        _session_master_seed = secrets.randbits(31)
    _session_master_seed = int(_session_master_seed)

    span_ctx = None
    if req.get("span_round1") and hasattr(sampler, "sample_span_round"):
        print(f"  [span] SPAN ROUND 1 active "
              f"(master_seed={_session_master_seed})", flush=True)
        span_ctx = _run_span_round1(job_id, req, sampler, output_dir,
                                    session_id, device, _session_master_seed)
        # Seed the incremental layout + canvas history with the span round so
        # the breeding rounds continue the SAME canvas: span coords stay
        # frozen where the artist saw them; clones land near their parents.
        interval_count[0] = 1          # span consumed round 1
        prev_selected[0] = span_ctx["selected"]
        inc_layout.coords = np.array(span_ctx["coords"], dtype=float)
        inc_layout.images = list(span_ctx["pils"])
        inc_layout.active = np.ones(len(inc_layout.coords), dtype=bool)
        inc_layout.round_ids = np.zeros(len(inc_layout.coords), dtype=int)
        inc_layout.mobility = np.ones(len(inc_layout.coords))
        jobs[job_id]["_all_features"] = span_ctx["features"].copy()
        jobs[job_id]["_all_urls"] = list(span_ctx["urls"])
        jobs[job_id]["_dino_features"] = span_ctx["features"]
        jobs[job_id]["_original_all_coords"] = inc_layout.coords.copy()
    elif req.get("span_round1"):
        print("  [span] requested but this sampler doesn't support "
              "sample_span_round — falling back to the classic flow.")
    # ===== END SPAN MODE =====================================================

    def mock_input(prompt_text):
        # REFINE re-entry: the sampler looped back to input() after rendering
        # full-quality images (see _prompt_user_selection). SAME round — keep
        # the published layout, just fold in the refined URLs and wait again.
        refine_reentry = jobs[job_id].pop("_refine_reentry", False)
        if not refine_reentry:
            interval_count[0] += 1
            jobs[job_id]["refined"] = {}  # refined URLs are per-round
        interval_dir = os.path.join(output_dir, f"interval_{interval_count[0]}")

        # Read preview images that sample_interactive just saved
        preview_paths = sorted(Path(interval_dir).glob("particle_*.png"))
        if refine_reentry:
            images = jobs[job_id].get("images", [])
            url_prefix = f"/images/{session_id}/interval_{interval_count[0]}"
            refined = dict(jobs[job_id].get("refined", {}))
            for p in sorted(Path(interval_dir).glob("refined_particle_*.png")):
                refined[p.stem.rsplit("_", 1)[-1]] = f"{url_prefix}/{p.name}"
            jobs[job_id]["refined"] = refined
        elif not preview_paths:
            images = []
        elif tree_mode:
            n_total = len(preview_paths)
            coords, sizes, is_anchor_arr = compute_tree_positions(
                n_total, num_clones, interval_count[0])
            url_prefix = f"/images/{session_id}/interval_{interval_count[0]}"

            if interval_count[0] == 1:
                images = []
                for i, (p, (x, y)) in enumerate(zip(preview_paths, coords)):
                    if is_anchor_arr[i]:
                        images.append({
                            "id": i,
                            "url": f"{url_prefix}/{p.name}",
                            "x": float(x), "y": float(y),
                            "size": 1.0,
                        })
            else:
                images = build_image_list(preview_paths, coords, url_prefix, sizes)
        else:
            pil_images = [PILImage.open(p).convert("RGB") for p in preview_paths]

            # Incremental layout: initialize on first round, update on subsequent
            from cluster_images import extract_dinov2, l2_normalize
            features = extract_dinov2(pil_images, device=device)
            features = l2_normalize(features)

            url_prefix_now = f"/images/{session_id}/interval_{interval_count[0]}"
            current_urls = [f"{url_prefix_now}/{p.name}" for p in preview_paths]

            if interval_count[0] == 1:
                inc_layout.initialize(features, pil_images)
                jobs[job_id]["_all_features"] = features.copy()
                jobs[job_id]["_all_urls"] = list(current_urls)
            else:
                inc_layout.update(
                    prev_selected[0], features, pil_images,
                    num_clones=num_clones, round_num=interval_count[0])
                old_features = jobs[job_id].get("_all_features")
                if old_features is not None:
                    jobs[job_id]["_all_features"] = np.vstack([old_features, features])
                else:
                    jobs[job_id]["_all_features"] = features.copy()
                old_urls = jobs[job_id].get("_all_urls", [])
                jobs[job_id]["_all_urls"] = old_urls + list(current_urls)
            jobs[job_id]["_dino_features"] = features

            # Capture original coords NOW (before any adjust call can modify them)
            jobs[job_id]["_original_all_coords"] = inc_layout.coords.copy()

            # Normalize using ALL coords as reference frame
            # (so clones stay near parents relative to full layout)
            active_mask = inc_layout.active
            all_coords = inc_layout.coords
            all_min = all_coords.min(axis=0)
            all_span = all_coords.max(axis=0) - all_min
            all_span[all_span == 0] = 1
            normed_all = (all_coords - all_min) / all_span * 0.90 + 0.05
            active_coords = normed_all[active_mask]

            url_prefix = f"/images/{session_id}/interval_{interval_count[0]}"
            images = build_image_list(preview_paths, active_coords, url_prefix)

        # Segment done — pin bar to 100% while user makes selection.
        selection_event.clear()
        # Span sessions have R+1 total rounds (span round 1 + R breeding
        # rounds); the job's total_intervals was set accordingly up front.
        _tot_rounds = jobs[job_id].get("total_intervals") or num_resampling_steps
        jobs[job_id].update({
            "status": "waiting_selection",
            "stage": f"Ready — pick the ones you like (round {interval_count[0]}/{_tot_rounds})",
            "progress": 100,
            "images": images,
            "interval": interval_count[0],
            "total_intervals": _tot_rounds,
        })

        # Block until the frontend sends a selection (poll so Ctrl+C works)
        print(f"  [server] Waiting for user selection at interval {interval_count[0]}...")
        # Timeout so an ABANDONED session cannot hold the model lock forever
        # (which wedges ALL other generation, with no recovery). After the
        # cap, treat the session as abandoned -> cancel -> release the lock.
        _sel_waited = 0.0
        _MAX_SELECTION_WAIT = 300.0  # 5 min: generous for a human picking, bounded for recovery
        while not selection_event.wait(timeout=1):
            _sel_waited += 1.0
            if _sel_waited >= _MAX_SELECTION_WAIT:
                print(f"  [server] selection wait timed out after {int(_MAX_SELECTION_WAIT)}s -> releasing model (stale session)")
                jobs[job_id]["_cancelled"] = True
                jobs[job_id]["stage"] = "Session timed out — start a new prompt"
                break

        # A newer generate may have taken over while we were blocked — unwind.
        if jobs[job_id].get("_cancelled"):
            raise GenerateCancelled()

        # REFINE takes precedence over a selection: hand the request to the
        # sampler, which renders full-quality images and re-enters this hook
        # (see refine_reentry above) — the round does NOT advance.
        refine_req = jobs[job_id].pop("_refine_request", None)
        if refine_req is not None:
            jobs[job_id]["_refine_reentry"] = True
            jobs[job_id]["status"] = "running"
            jobs[job_id]["stage"] = "Refining to full quality..."
            req_str = "refine:" + ",".join(str(int(i)) for i in refine_req)
            print(f"  [server] Refine requested: {req_str}")
            return req_str

        # Retrieve the selection the frontend posted
        selected = jobs[job_id].pop("_user_selection", None)
        if selected is None:
            # Fallback: keep all
            total = len(preview_paths)
            selected = list(range(total))

        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["stage"] = (
            f"Breeding new variations near your {len(selected)} pick"
            f"{'s' if len(selected) != 1 else ''}..."
        )

        prev_selected[0] = selected  # save for next incremental layout update

        result_str = ",".join(str(i) for i in selected)
        print(f"  [server] User selected: {result_str}")
        return result_str

    builtins.input = mock_input

    # Only pass progress_callback to samplers that accept it (FLUX does).
    import inspect
    sample_kwargs = dict(
        num_particles=num_particles,
        num_clones=num_clones,
        num_steps=num_steps,
        num_resampling_steps=num_resampling_steps,
        guidance_scale=guidance_scale,
        rho=rho,
        schedule_mode=schedule_mode,
        lookahead_steps=req.get("lookahead_steps", 8),
        output_dir=output_dir,
        img_shape=(height, width),
        seed=seed,
    )
    if "progress_callback" in inspect.signature(sampler.sample_interactive).parameters:
        sample_kwargs["progress_callback"] = sampler_progress
    # Pass clone_mode only to samplers that accept it (currently FluxFMWDMInteractive).
    if "clone_mode" in inspect.signature(sampler.sample_interactive).parameters:
        sample_kwargs["clone_mode"] = req.get("clone_mode", "glass")
    # Pass spawn_mode only to samplers that accept it (currently FluxDevInteractive):
    # "glass" (default, in-trajectory) or "sdedit" (renoise-up then Euler).
    if "spawn_mode" in inspect.signature(sampler.sample_interactive).parameters:
        sample_kwargs["spawn_mode"] = req.get("spawn_mode", "glass")
    # FluxFMWDMInteractive doesn't use a lookahead_steps argument; drop it.
    if "lookahead_steps" not in inspect.signature(sampler.sample_interactive).parameters:
        sample_kwargs.pop("lookahead_steps", None)
    # Round-1 preview boost — richer lookahead for the FIRST interval only
    # (signature-gated like the other optional args).
    if "lookahead_steps_first" in inspect.signature(sampler.sample_interactive).parameters:
        sample_kwargs["lookahead_steps_first"] = req.get("lookahead_steps_first", 10)
    # Per-particle prompt variations (FLUX prompt-augmented diversity), only for
    # samplers that accept it.
    if (prompt_assignment is not None
            and "prompt_idx" in inspect.signature(sampler.sample_interactive).parameters):
        sample_kwargs["prompt_idx"] = prompt_assignment
    # EXPLORE FROM THIS PIN: pass the seed latent only to samplers that accept it
    # (currently FluxDevInteractive). init_latent is None unless an image seeded
    # the session, so the default pure-noise path is untouched for every model.
    if (init_latent is not None
            and "init_latent" in inspect.signature(sampler.sample_interactive).parameters):
        sample_kwargs["init_latent"] = init_latent
        sample_kwargs["init_strength"] = init_strength
    # SAVE-THE-NOISE: per-session master seed. Every root/clone noise draw in
    # the sampler is derived from it and the full creation lineage is written
    # to lineage.json in the session output dir — a KB-sized recipe that makes
    # every node exactly reproducible. Signature-gated like the other optional
    # args; the request may pin one (int "master_seed") for exact reproduction.
    if "master_seed" in inspect.signature(sampler.sample_interactive).parameters:
        # Span sessions: breeding rounds get a DERIVED master so their root/
        # clone draws can never collide with the span images' root draws.
        _ms = (derive_seed(_session_master_seed, 909090)
               if span_ctx is not None else _session_master_seed)
        sample_kwargs["master_seed"] = int(_ms)
        print(f"  [server] session master_seed={int(_ms)} "
              f"(lineage.json -> {output_dir})", flush=True)

    # ── SPAN MODE: breeding rounds are SEEDED from the selected span images
    # (multi-parent SDEdit re-noise; anchors round-robin over the parents,
    # each conditioned on its parent's prompt variation). Signature-gated.
    if span_ctx is not None:
        _sig = inspect.signature(sampler.sample_interactive).parameters
        P = max(len(span_ctx["selected"]), 1)
        n2 = min(max(num_particles, P), 16)
        if "init_latents" in _sig:
            sample_kwargs.update(
                num_particles=n2,
                init_latents=span_ctx["init_latents"],
                init_strength=float(req.get("span_breed_strength", 0.55) or 0.55),
                init_parent_ids=span_ctx["parent_ids"],
                interval_start=1,
                prompt_idx=[span_ctx["parent_prompt_idx"][i % P]
                            for i in range(n2)],
            )
            print(f"  [span] breeding rounds: {n2} anchors from "
                  f"{P} parents (strength "
                  f"{sample_kwargs['init_strength']})", flush=True)

    try:
        sampler.sample_interactive(**sample_kwargs)
    finally:
        builtins.input = _orig_input

    # ── Collect final images ─────────────────────────────────────────────────
    set_progress(96, "Collecting final images...")
    final_dir = Path(output_dir) / "final"
    image_paths = sorted(final_dir.glob("particle_*.png"))

    if not image_paths:
        raise RuntimeError(f"No particle images found in {final_dir}.")

    url_prefix = f"/images/{session_id}/final"
    if tree_mode:
        n_total = len(image_paths)
        coords, sizes, _ = compute_tree_positions(
            n_total, num_clones, num_resampling_steps + 1)
        images_out = build_image_list(image_paths, coords, url_prefix, sizes)
    else:
        pil_images = [PILImage.open(p).convert("RGB") for p in image_paths]

        # Use the incremental layout (same as intermediate rounds) so final
        # clones sit near their parents. Without this we'd fall back to a
        # fresh UMAP and the frontend's applyParentPositions would compute
        # large parent→clone offsets, looking like auto-repulsion.
        use_incremental = (
            inc_layout is not None
            and inc_layout.coords is not None
            and prev_selected[0] is not None
        )
        if use_incremental:
            from cluster_images import extract_dinov2, l2_normalize
            features = extract_dinov2(pil_images, device=device)
            features = l2_normalize(features)
            inc_layout.update(
                prev_selected[0], features, pil_images,
                num_clones=num_clones, round_num=num_resampling_steps + 1,
            )
            jobs[job_id]["_original_all_coords"] = inc_layout.coords.copy()
            old_features = jobs[job_id].get("_all_features")
            jobs[job_id]["_all_features"] = (
                np.vstack([old_features, features]) if old_features is not None else features.copy()
            )
            jobs[job_id]["_all_urls"] = (
                list(jobs[job_id].get("_all_urls", [])) +
                [f"{url_prefix}/{p.name}" for p in image_paths]
            )
            active_mask = inc_layout.active
            all_coords = inc_layout.coords
            all_min = all_coords.min(axis=0)
            all_span = all_coords.max(axis=0) - all_min
            all_span[all_span == 0] = 1
            normed_all = (all_coords - all_min) / all_span * 0.90 + 0.05
            coords = normed_all[active_mask]
        else:
            coords = cluster_images(pil_images, device)
        images_out = build_image_list(image_paths, coords, url_prefix)

    set_progress(98, "Arranging layout...")
    return {"images": images_out, "session_id": session_id}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "device": get_device()}


@app.get("/api/status")
def server_status():
    """Boot / model-readiness status with REAL progress (see _boot above)."""
    st = dict(_boot)
    try:
        st["device"] = get_device()
    except Exception:
        st["device"] = "unknown"
    return st


@app.post("/api/generate")
def start_generate(req: GenerateRequest):
    """Kick off an interactive generation job. Returns job_id immediately."""
    job_id = str(uuid.uuid4())[:8]
    # Mint the durable session_id HERE (not deep in the worker) so the very
    # first poll already carries it and a mid-generation refresh can rehydrate.
    session_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "pending", "result": None, "error": None,
        "progress": 0, "stage": "Queued...",
        # total_intervals = round 1 (span/sketch) + N breeding rounds. Set the
        # +1 HERE so a mid-stream refresh (which reads this value before the
        # span round re-sets it) shows the SAME "OF N" as the live session
        # (critic-5 #1: restore said "OF 3" while live said "OF 4").
        "images": [], "interval": 0,
        "total_intervals": int(req.num_resampling_steps) + 1,
        "session_id": session_id, "prompt": req.prompt,
    }
    _session_index[session_id] = job_id

    import threading
    t = threading.Thread(target=run_generate, args=(job_id, req.model_dump()), daemon=True)
    t.start()

    return {"job_id": job_id, "session_id": session_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """Poll job status. When status is 'waiting_selection', images[] has previews."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    # Don't leak internal fields
    job = jobs[job_id]
    return {
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
        "progress": job.get("progress", 0),
        "stage": job.get("stage", ""),
        "images": job.get("images", []),
        "interval": job.get("interval", 0),
        "total_intervals": job.get("total_intervals", 0),
        "refined": job.get("refined", {}),
        "session_id": job.get("session_id"),
        "prompt": job.get("prompt", ""),
        "span": job.get("span"),
    }


# === SESSION REHYDRATION (additive endpoints) ================================

import re as _re


def _session_frozen(job):
    """Frozen (previous-round, inactive) images with layout coords.

    Mirrors adjust_layout's frozen_updates: normalize ALL layout coords, then
    emit the inactive entries matched to their URLs. Best-effort — any failure
    returns [] so rehydration never 500s.
    """
    out = []
    try:
        inc = job.get("_inc_layout")
        if inc is None or getattr(inc, "coords", None) is None:
            return out
        normed = _normalize_coords(inc.coords)
        active = inc.active
        urls = job.get("_all_urls", [])
        for i in range(len(normed)):
            if i < len(active) and not active[i] and i < len(urls) and urls[i]:
                out.append({"url": urls[i],
                            "x": float(normed[i, 0]), "y": float(normed[i, 1])})
    except Exception as exc:
        print(f"[session-state] frozen rebuild failed: {exc}")
        return []
    return out


def _grid_coords(n):
    """Deterministic fallback layout (disk-only rehydration has no UMAP coords)."""
    import math
    if n <= 0:
        return []
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        x = 0.05 + 0.90 * ((c + 0.5) / cols)
        y = 0.05 + 0.90 * ((r + 0.5) / rows)
        out.append((x, y))
    return out


def _disk_rounds(session_dir: Path):
    """Round-by-round image URLs rebuilt from the on-disk session dir."""
    sid = session_dir.name
    rounds = []
    for d in sorted(session_dir.glob("interval_*")):
        try:
            num = int(d.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        urls = [f"/images/{sid}/{d.name}/{p.name}"
                for p in sorted(d.glob("particle_*.png"))]
        refined = {p.stem.rsplit("_", 1)[-1]: f"/images/{sid}/{d.name}/{p.name}"
                   for p in sorted(d.glob("refined_particle_*.png"))}
        if urls or refined:
            rounds.append({"interval": num, "image_urls": urls, "refined": refined})
    rounds.sort(key=lambda r: r["interval"])
    fin = sorted((session_dir / "final").glob("particle_*.png"))
    if fin:
        rounds.append({"interval": "final",
                       "image_urls": [f"/images/{sid}/final/{p.name}" for p in fin],
                       "refined": {}})
    return rounds


@app.get("/api/sessions/{session_id}/state")
def get_session_state(session_id: str):
    """Everything the frontend needs to rebuild the canvas after navigation.

    Live path (job still in memory): current-round images WITH layout coords,
    status (incl. in-flight running/waiting_selection), frozen history, refined
    map, prompt, plus round-by-round URLs from disk.
    Disk fallback (server restarted / job evicted): rounds rebuilt from the
    session dir with deterministic grid coords; status 'done' if final/ exists,
    else 'stale' (images viewable, no live worker to select against).
    """
    if not _re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")

    session_dir = (GENERATED_DIR / session_id)
    try:
        session_dir = session_dir.resolve()
        if GENERATED_DIR.resolve() not in session_dir.parents:
            raise HTTPException(status_code=400, detail="Invalid session id")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session id")

    rounds = _disk_rounds(session_dir) if session_dir.is_dir() else []

    job_id = _session_index.get(session_id)
    job = jobs.get(job_id) if job_id else None
    if job is not None:
        return {
            "session_id": session_id,
            "job_id": job_id,
            "live": True,
            "status": job.get("status"),
            "progress": job.get("progress", 0),
            "stage": job.get("stage", ""),
            "images": job.get("images", []),
            "interval": job.get("interval", 0),
            "total_intervals": job.get("total_intervals", 0),
            "refined": job.get("refined", {}),
            "result": job.get("result"),
            "error": job.get("error"),
            "prompt": job.get("prompt", ""),
            "span": job.get("span"),
            "frozen": _session_frozen(job),
            "rounds": rounds,
        }

    if not rounds:
        raise HTTPException(status_code=404, detail="Session not found")

    # Disk-only fallback: no live job. Prefer final round, else latest interval.
    last = rounds[-1]
    is_done = last["interval"] == "final"
    urls = last["image_urls"]
    coords = _grid_coords(len(urls))
    images = [{"id": i, "url": u, "x": float(x), "y": float(y)}
              for i, (u, (x, y)) in enumerate(zip(urls, coords))]
    n_intervals = sum(1 for r in rounds if r["interval"] != "final")
    return {
        "session_id": session_id,
        "job_id": None,
        "live": False,
        "status": "done" if is_done else "stale",
        "progress": 100,
        "stage": "Restored from disk",
        "images": images,
        "interval": n_intervals,
        "total_intervals": n_intervals,
        "refined": last.get("refined", {}),
        "result": {"images": images, "session_id": session_id} if is_done else None,
        "error": None,
        "prompt": "",
        "frozen": [],
        "rounds": rounds,
    }
# === END SESSION REHYDRATION =================================================


@app.post("/api/jobs/{job_id}/select")
def submit_selection(job_id: str, req: SelectRequest):
    """Submit user's selected particle indices. Unblocks the generation thread."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "waiting_selection":
        raise HTTPException(status_code=400, detail=f"Job not waiting for selection (status: {job['status']})")

    # Store selection and unblock the worker thread
    job["_user_selection"] = req.indices
    event = job.get("_selection_event")
    if event:
        event.set()
    else:
        raise HTTPException(status_code=500, detail="No selection event found")

    return {"status": "ok", "selected": req.indices}


# === SPAN MODE: user early-stop ==============================================
@app.post("/api/jobs/{job_id}/finish_round")
def finish_round(job_id: str):
    """Gracefully end the streaming SPAN round: the in-flight batch finishes,
    then the round is presented for selection ("spread looks good — start
    picking"). Idempotent; 400 if the job has no streaming round to finish."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job.get("status") == "waiting_selection":
        return {"status": "ok", "note": "round already presented for selection"}
    span = job.get("span") or {}
    if not span.get("active"):
        raise HTTPException(
            status_code=400,
            detail="No streaming round to finish (span mode not active)")
    job["_finish_span"] = True
    return {"status": "ok", "stopping_after": "the batch currently painting"}
# === END SPAN MODE ===========================================================


class RefineRequest(BaseModel):
    indices: List[int]


@app.post("/api/jobs/{job_id}/refine")
def request_refine(job_id: str, req: RefineRequest):
    """Request FULL-QUALITY renders of specific particles while waiting_selection.

    The worker integrates the full remaining fine schedule from each stored
    latent (deterministic — no fresh noise), decodes with the full VAE, then
    returns to waiting_selection. Poll GET /api/jobs/{id}: refined URLs appear
    under `refined` ({index: url}); the round then still accepts /select.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "waiting_selection":
        raise HTTPException(status_code=400, detail=f"Job not waiting for selection (status: {job['status']})")
    if not req.indices:
        raise HTTPException(status_code=400, detail="indices must be non-empty")

    # Store the request and unblock the worker (same plumbing as /select; the
    # worker distinguishes refine-vs-select and loops back to waiting after).
    job["_refine_request"] = [int(i) for i in req.indices]
    event = job.get("_selection_event")
    if event:
        event.set()
    else:
        raise HTTPException(status_code=500, detail="No selection event found")

    return {"status": "ok", "indices": req.indices}


class AdjustRequest(BaseModel):
    repulsion: float = 0.5     # 0-1: Coulomb charge strength
    cohesion: float = 0.0      # 0-1: centroid-based cohesion (relative similarity)
    cohesion_spring: float = 0.0  # 0-1: pairwise spring cohesion (all similar pairs)


def _normalize_coords(coords):
    """Normalize coords to [0.05, 0.95]."""
    c = coords.copy()
    c -= c.min(axis=0)
    span = c.max(axis=0)
    span[span == 0] = 1
    return c / span * 0.90 + 0.05


@app.post("/api/jobs/{job_id}/adjust")
def adjust_layout(job_id: str, req: AdjustRequest):
    """Adjust layout with repulsion/cohesion. Always computed from ORIGINAL positions.

    Idempotent: same slider values always produce the same result.
    Setting both to 0 restores original positions exactly.
    ALL images (active + inactive) are affected.
    No elastic band — images expand freely. Cohesion counters repulsion.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    inc_layout = job.get("_inc_layout")
    if inc_layout is None or inc_layout.coords is None:
        raise HTTPException(status_code=400, detail="No layout available")

    from sklearn.metrics.pairwise import cosine_similarity

    # Store original positions on first adjust call
    if "_original_all_coords" not in job:
        job["_original_all_coords"] = inc_layout.coords.copy()

    original = job["_original_all_coords"]
    n = len(original)
    active_mask = inc_layout.active
    n_active = int(active_mask.sum())

    def _build_response(coords_all):
        """Normalize all coords and return active images with updated positions.

        For frozen images, we can't reliably match backend inactive indices
        to frontend frozenImages order. Instead, return all coords so the
        frontend can match by URL or re-render.
        """
        normed = _normalize_coords(coords_all)

        # Update active images
        active_normed = normed[active_mask]
        imgs = job.get("images", [])
        if len(imgs) == n_active:
            for i, img in enumerate(imgs):
                img["x"] = float(active_normed[i, 0])
                img["y"] = float(active_normed[i, 1])

        # Build a flat list of ALL inactive image positions with URLs
        # so frontend can match frozen images by URL
        all_urls = job.get("_all_urls", [])
        inactive_with_urls = []
        inactive_idx = 0
        for i in range(len(normed)):
            if not active_mask[i]:
                url = all_urls[i] if i < len(all_urls) else None
                inactive_with_urls.append({
                    "x": float(normed[i, 0]),
                    "y": float(normed[i, 1]),
                    "url": url,
                })

        return {"images": imgs, "frozen_updates": inactive_with_urls}

    # If all sliders 0, restore original
    if req.repulsion == 0 and req.cohesion == 0 and req.cohesion_spring == 0:
        inc_layout.coords = original.copy()
        return _build_response(original)

    # Always start from original positions (idempotent)
    coords = original.copy()

    # DINO similarity — use ALL accumulated features
    cos_sim_all = None
    all_features = job.get("_all_features")
    if (req.cohesion > 0 or req.cohesion_spring > 0) and all_features is not None:
        if len(all_features) == n:
            cos_sim_all = cosine_similarity(all_features)
            np.fill_diagonal(cos_sim_all, 0.0)

    # Compute scale from ACTIVE images (current round's density, not global extent)
    active_coords_orig = original[active_mask]
    if len(active_coords_orig) > 1:
        init_range = max(active_coords_orig[:, 0].max() - active_coords_orig[:, 0].min(),
                         active_coords_orig[:, 1].max() - active_coords_orig[:, 1].min(), 1e-6)
    else:
        init_range = max(original[:, 0].max() - original[:, 0].min(),
                         original[:, 1].max() - original[:, 1].min(), 1e-6)

    max_step = init_range * 0.008
    min_dist = init_range * 0.10 * req.repulsion

    # ── Phase 1: Cohesion FIRST (settle similar images together) ──
    if (req.cohesion > 0 or req.cohesion_spring > 0) and cos_sim_all is not None:
        for iteration in range(100):
            forces = np.zeros_like(coords)
            alpha = max(0.01, 1.0 - iteration / 100)

            # Cohesion A (centroid): pull toward cluster of similar neighbors
            if req.cohesion > 0:
                for i in range(n):
                    sims = cos_sim_all[i]
                    mean_sim = np.mean(sims[sims > 0]) if np.any(sims > 0) else 0
                    weights = np.maximum(sims - mean_sim, 0)
                    total_w = weights.sum()
                    if total_w < 1e-8:
                        continue
                    centroid = (weights[:, None] * coords).sum(axis=0) / total_w
                    dx = centroid - coords[i]
                    d = max(np.linalg.norm(dx), 1e-8)
                    # Constant magnitude pull (NOT proportional to distance)
                    pull_mag = req.cohesion * 0.02 * init_range
                    forces[i] += min(pull_mag, d) * dx / d

            # Cohesion B (spring): constant-magnitude pull between similar pairs
            if req.cohesion_spring > 0:
                for i in range(n):
                    for j in range(i + 1, n):
                        sim = max(cos_sim_all[i, j], 0.0)
                        if sim < 0.2:
                            continue
                        dx = coords[j] - coords[i]
                        d = max(np.linalg.norm(dx), 1e-8)
                        # Constant pull scaled by similarity (NOT by distance)
                        pull_mag = req.cohesion_spring * 0.01 * sim * init_range
                        f = min(pull_mag, d * 0.3) * dx / d
                        forces[i] += f
                        forces[j] -= f

            # Light centering
            forces -= 0.02 * (coords.mean(axis=0) - original.mean(axis=0))
            coords += alpha * forces

    # ── Phase 2: Coulomb repulsion (has final say over spacing) ──
    if req.repulsion > 0:
        # Inverse-linear Coulomb (1/d, not 1/d²) — gentler, no explosions
        charge = req.repulsion * req.repulsion * init_range * 0.008
        velocity = np.zeros_like(coords)

        for iteration in range(200):
            forces = np.zeros_like(coords)
            alpha = max(0.01, 1.0 - iteration / 200)

            # Coulomb: ALL pairs (active + inactive), everyone pushes everyone
            for i in range(n):
                for j in range(i + 1, n):
                    dx = coords[j] - coords[i]
                    d = max(np.linalg.norm(dx), 1e-8)
                    force_mag = min(charge / d, max_step)  # 1/d not 1/d²
                    f = force_mag * dx / d
                    forces[i] -= f
                    forces[j] += f

            # Centering
            forces -= 0.02 * (coords.mean(axis=0) - original.mean(axis=0))

            velocity = 0.8 * velocity + alpha * forces
            speeds = np.linalg.norm(velocity, axis=1, keepdims=True)
            speeds = np.maximum(speeds, 1e-8)
            velocity = velocity * np.minimum(speeds, max_step) / speeds
            coords += velocity

    # ── Phase 3: Hard collision (guarantee no overlaps) ──
    if min_dist > 0:
        for _ in range(100):
            resolved = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = coords[j] - coords[i]
                    d = np.linalg.norm(dx)
                    if d < 1e-8:
                        angle = np.random.uniform(0, 2 * np.pi)
                        coords[i] -= min_dist * 0.5 * np.array([np.cos(angle), np.sin(angle)])
                        coords[j] += min_dist * 0.5 * np.array([np.cos(angle), np.sin(angle)])
                        resolved = False
                    elif d < min_dist:
                        overlap = (min_dist - d) / 2.0
                        coords[i] -= overlap * dx / d
                        coords[j] += overlap * dx / d
                        resolved = False
            if resolved:
                break

    # Re-center
    coords -= coords.mean(axis=0) - original.mean(axis=0)

    inc_layout.coords = coords.copy()
    return _build_response(coords)


# ── Explore mode: serve pre-computed map ─────────────────────────────────────

class ExploreRequest(BaseModel):
    prompt: str
    model: str = "flux"
    num_particles: int = 6
    num_clones: int = 3
    num_resampling_steps: int = 3        # 162-img tree (R=4=486 is impractically slow)
    num_steps: int = 16
    guidance_scale: float = 3.5
    rho: float = 0.0
    seed: int = 42
    height: int = 512
    width: int = 512
    augment: bool = True


@app.post("/api/explore/generate")
def start_explore(req: ExploreRequest):
    """Pre-compute an explore map. Returns job_id to poll."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "pending", "result": None, "error": None,
        "progress": 0, "stage": "Queued...",
    }

    def run():
        if not _model_lock.acquire(timeout=2):
            jobs[job_id].update({"status": "error", "result": None,
                "error": "Another generation is in progress \u2014 please wait a moment and try again.",
                "progress": 0, "stage": "Busy"})
            return
        try:
            jobs[job_id].update({"status": "running", "progress": 5,
                                 "stage": "Pre-computing explore map..."})
            # Free resident FLUX models before building the precompute FLUX
            # (else two ~24GB FLUX coexist -> CUDA OOM). Warm-reuse guard rebuilds.
            global _warm_flux, _active_sampler
            for _samp in (_warm_flux, _active_sampler):
                if _samp is None:
                    continue
                for _a in ('pipeline', 'transformer', 'vae', 'text_encoder',
                           'text_encoder_2', 'dino_model', 'dino_processor',
                           'cached_prompt_embeds', 'cached_pooled_prompt_embeds',
                           'cached_text_ids'):
                    _o = getattr(_samp, _a, None)
                    if _o is not None and hasattr(_o, 'to'):
                        try: _o.to('cpu')
                        except Exception: pass
                    try: setattr(_samp, _a, None)
                    except Exception: pass
            _warm_flux = None
            _active_sampler = None
            import gc as _gc; _gc.collect()
            try:
                import torch as _t
                if _t.cuda.is_available(): _t.cuda.empty_cache()
            except Exception: pass
            from flux_explore_precompute import precompute_explore_map
            manifest_path = precompute_explore_map(
                prompt=req.prompt, model=req.model,
                num_particles=req.num_particles, num_clones=req.num_clones,
                augment=req.augment,
                num_resampling_steps=req.num_resampling_steps,
                num_steps=req.num_steps, guidance_scale=req.guidance_scale,
                rho=req.rho, height=req.height, width=req.width,
                seed=req.seed,
                output_dir=str(GENERATED_DIR / job_id),
            )
            # Load manifest
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)

            # Build image URLs
            session_id = job_id
            for img in manifest["images"]:
                img["url"] = f"/images/{session_id}/{img['path']}"

            jobs[job_id].update({
                "status": "done", "progress": 100, "stage": "Done",
                "result": manifest,
            })
        except Exception as exc:
            import traceback
            traceback.print_exc()
            jobs[job_id].update({
                "status": "error", "error": _friendly_error(exc),
                "progress": 0, "stage": "Error",
            })
        finally:
            try: _model_lock.release()
            except Exception: pass

    import threading
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/api/explore/{job_id}")
def get_explore(job_id: str):
    """Get explore map status/result."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "status": job["status"],
        "progress": job.get("progress", 0),
        "stage": job.get("stage", ""),
        "result": job.get("result"),
        "error": job.get("error"),
    }


# ── Reve API proxy (Stage 2 — image editing) ────────────────────────────────

@app.get("/api/effects")
async def list_effects():
    """Return the list of named effects available from the Reve API."""
    import httpx, traceback
    api_key = os.getenv("REVE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="REVE_API_KEY is not configured on the server")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://api.reve.com/v1/image/effect",
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Reve request failed: {type(exc).__name__}: {exc}")
    if not resp.is_success:
        raise HTTPException(status_code=resp.status_code, detail=f"Reve /effect error {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Reve /effect returned non-JSON: {resp.text[:200]}")


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    """Proxy for Reve image operations (edit, vary, upscale, remove_bg, effect)."""
    import httpx, traceback
    from PIL import Image as PILImage

    api_key = os.getenv("REVE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="REVE_API_KEY is not configured on the server")

    try:
        # base64 may arrive with a `data:image/png;base64,` prefix — strip it.
        raw_b64 = req.image_b64 or ""
        if raw_b64.startswith("data:"):
            raw_b64 = raw_b64.split(",", 1)[1] if "," in raw_b64 else ""
        img_bytes = base64.b64decode(raw_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64")

    try:
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {type(exc).__name__}: {exc}")
    W, H = img.size

    reve_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build the Reve request body up front (inputs don't change across retries).
    reve_url = None
    reve_body = None
    if req.operation == "edit":
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required for edit operation")
        # Reve's /v1/image/edit is prompt-only at the API level — no mask, no
        # region. To give the user a sense of "edit here", we crop the
        # selected region client-rendered box and send only that crop to
        # Reve; the frontend then pastes the edited crop back at the region
        # coordinates. Trade-off: localization works, but Reve doesn't see
        # surrounding context, so the edited patch can look disconnected.
        reve_url = "https://api.reve.com/v1/image/edit"
        if req.region:
            from PIL import Image as _PILImage
            region = req.region
            rx = max(0.0, min(1.0, region.x))
            ry = max(0.0, min(1.0, region.y))
            rw = max(0.0, min(1.0 - rx, region.w))
            rh = max(0.0, min(1.0 - ry, region.h))
            px, py = int(rx * W), int(ry * H)
            pw, ph = max(1, int(rw * W)), max(1, int(rh * H))
            cropped = img.crop((px, py, px + pw, py + ph))
            # Upscale small crops so the model has enough pixels to work
            # with. The frontend will squash the result back into the
            # region on composite, so going bigger here loses nothing.
            MIN_EDGE = 256
            cw, ch = cropped.size
            if min(cw, ch) < MIN_EDGE:
                scale = MIN_EDGE / min(cw, ch)
                cropped = cropped.resize(
                    (int(cw * scale), int(ch * scale)), _PILImage.LANCZOS
                )
            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            reve_body = {
                "edit_instruction": req.prompt,
                "reference_image": base64.b64encode(buf.getvalue()).decode(),
            }
        else:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            reve_body = {
                "edit_instruction": req.prompt,
                "reference_image": base64.b64encode(buf.getvalue()).decode(),
            }
    elif req.operation in ("vary", "upscale", "remove_bg", "effect"):
        if req.operation == "effect" and not req.effect_name:
            raise HTTPException(status_code=400, detail="effect_name is required")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        full_b64 = base64.b64encode(buf.getvalue()).decode()
        postprocessing = []
        if req.operation == "upscale":
            postprocessing.append({"process": "upscale", "upscale_factor": req.upscale_factor})
        elif req.operation == "remove_bg":
            postprocessing.append({"process": "remove_background"})
        elif req.operation == "effect":
            postprocessing.append({"process": "effect", "effect_name": req.effect_name})
        reve_body = {
            "prompt": req.prompt.strip() or "<img0>",
            "reference_images": [full_b64],
        }
        if postprocessing:
            reve_body["postprocessing"] = postprocessing
        reve_url = "https://api.reve.com/v1/image/remix"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown operation: {req.operation!r}")

    # Retry transient failures (RemoteProtocolError, ReadError, ConnectError, etc.)
    # Reve occasionally drops the connection mid-response body.
    resp = None
    last_exc = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(reve_url, headers=reve_headers, json=reve_body)
            break
        except Exception as exc:
            last_exc = exc
            print(f"[reve] attempt {attempt + 1}/3 failed: {type(exc).__name__}: {exc}")
            if attempt == 2:
                break
            await asyncio.sleep(1.0 + attempt)  # 1s, 2s backoff

    if resp is None:
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail=f"Reve request failed after 3 attempts: {type(last_exc).__name__}: {last_exc}",
        )

    if not resp.is_success:
        print(f"[reve] {req.operation} failed: {resp.status_code} {resp.text[:500]}")
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Reve API error {resp.status_code}: {resp.text[:300]}",
        )

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Reve returned non-JSON: {resp.text[:200]}")

    return {
        "edited_b64": data.get("image"),
        "credits_remaining": data.get("credits_remaining"),
        "operation": req.operation,
    }


# Serve frontend index.html for all non-API routes (SPA catch-all)
if FRONTEND_DIST.exists():
    from fastapi.responses import FileResponse

    @app.get("/{path:path}")
    def serve_frontend(path: str):
        # Try serving a static file first, fall back to index.html
        file_path = FRONTEND_DIST / path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"), headers={"Cache-Control": "no-store, must-revalidate"})


if __name__ == "__main__":
    import uvicorn
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    srv_args = p.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=srv_args.port, reload=False)
