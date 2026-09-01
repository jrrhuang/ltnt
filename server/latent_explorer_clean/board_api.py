"""board_api — persistent "board" (pin) store for LTNT — the AI-Pinterest feature.

A *pin* persists the IMAGE ITSELF (copied into a durable dir), not just a URL,
because generated images live in GENERATED_DIR and may be cleaned between
sessions. Each pin record: {id, image, prompt, savedAt}.

Storage location — ~/diffusion/model_inference/board_store/ (home NFS):
    /home/jerryhua is NFS-mounted (nas10:/srv/nfs/home), persistent across
    sessions AND visible on every compute node. The resident server already
    reads/writes this tree (GENERATED_DIR sits right next to board_store).
    We deliberately do NOT use /data/user_data — it's autofs-mounted and the
    resident server frequently can't see it on a fresh node (documented in
    job_ltnt_server.sh; that's why models get staged to /scratch). /scratch is
    node-local + ephemeral, so it's wrong for a persistent board. Home is the
    one location the server can RELIABLY write AND that survives sessions.

Writes to board.json are atomic-ish (temp file + os.replace) and a missing or
empty index is treated as an empty list.
"""

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Storage layout ─────────────────────────────────────────────────────────
BOARD_DIR = Path(__file__).parent / "board_store"
BOARD_IMAGES_DIR = BOARD_DIR / "images"
BOARD_INDEX = BOARD_DIR / "board.json"

# Created lazily/tolerantly — the dir may not exist yet, and we never want an
# import-time failure to take down the server.
try:
    BOARD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
except Exception as _e:  # pragma: no cover
    print(f"[board] WARNING: could not pre-create board dir {BOARD_IMAGES_DIR}: {_e}")

_LOCK = Lock()

router = APIRouter()


# ── Index helpers ──────────────────────────────────────────────────────────
def _load() -> list:
    """Read the pin index. Missing / empty / corrupt → []."""
    try:
        if not BOARD_INDEX.exists():
            return []
        raw = BOARD_INDEX.read_text()
        if not raw.strip():
            return []
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[board] WARNING: could not read board.json, treating as empty: {e}")
        return []


def _save(pins: list) -> None:
    """Atomically persist the pin index (temp file + os.replace)."""
    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    tmp = BOARD_INDEX.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pins, indent=2))
    os.replace(tmp, BOARD_INDEX)


def _public(pin: dict) -> dict:
    """Shape a stored record for the API: add the served /board_images url."""
    return {
        "id": pin["id"],
        "url": f"/board_images/{pin['image']}",
        "prompt": pin.get("prompt", ""),
        "savedAt": pin.get("savedAt", 0),
        "sourceUrl": pin.get("sourceUrl", ""),
    }


def _resolve_source(url: str, generated_dir: Path) -> Path:
    """Map a '/images/<...>' URL to its file under GENERATED_DIR. Reject anything
    that escapes the mount (path traversal)."""
    prefix = "/images/"
    if not url or not url.startswith(prefix):
        raise HTTPException(status_code=400, detail=f"Unsupported image url: {url!r}")
    rel = url[len(prefix):].split("?")[0].split("#")[0]
    src = (generated_dir / rel).resolve()
    gd = generated_dir.resolve()
    if gd not in src.parents and src != gd:
        raise HTTPException(status_code=400, detail="Image url resolves outside GENERATED_DIR")
    if not src.is_file():
        raise HTTPException(status_code=404, detail=f"Source image not found: {rel}")
    return src


# ── Request models ─────────────────────────────────────────────────────────
class PinRequest(BaseModel):
    url: str
    prompt: str = ""


class UnpinRequest(BaseModel):
    id: str


# ── Routes ─────────────────────────────────────────────────────────────────
def make_router(generated_dir: Path) -> APIRouter:
    """Build the board router, closing over GENERATED_DIR for source resolution."""

    @router.get("/api/board")
    def list_board():
        pins = _load()
        pins = sorted(pins, key=lambda p: p.get("savedAt", 0), reverse=True)
        return {"pins": [_public(p) for p in pins]}

    @router.post("/api/board/pin")
    def pin(req: PinRequest):
        src = _resolve_source(req.url, generated_dir)
        with _LOCK:
            pins = _load()
            # Dedup by source url — return the existing pin unchanged.
            for p in pins:
                if p.get("sourceUrl") == req.url:
                    return _public(p)
            pid = uuid.uuid4().hex[:12]
            ext = src.suffix or ".png"
            fname = f"{pid}{ext}"
            try:
                shutil.copy2(src, BOARD_IMAGES_DIR / fname)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to copy image to board: {e}")
            record = {
                "id": pid,
                "image": fname,
                "prompt": req.prompt or "",
                "savedAt": int(time.time() * 1000),
                "sourceUrl": req.url,
            }
            pins.append(record)
            try:
                _save(pins)
            except Exception as e:
                # Roll back the copied file so we don't leak orphans.
                try:
                    (BOARD_IMAGES_DIR / fname).unlink(missing_ok=True)
                except Exception:
                    pass
                raise HTTPException(status_code=500, detail=f"Failed to persist board index: {e}")
            return _public(record)

    @router.delete("/api/board/{pin_id}")
    def unpin(pin_id: str):
        return _do_unpin(pin_id)

    @router.post("/api/board/unpin")
    def unpin_post(req: UnpinRequest):
        return _do_unpin(req.id)

    def _do_unpin(pin_id: str):
        with _LOCK:
            pins = _load()
            kept, removed = [], None
            for p in pins:
                if p.get("id") == pin_id:
                    removed = p
                else:
                    kept.append(p)
            if removed is None:
                raise HTTPException(status_code=404, detail=f"Pin not found: {pin_id}")
            _save(kept)
            try:
                (BOARD_IMAGES_DIR / removed["image"]).unlink(missing_ok=True)
            except Exception as e:
                print(f"[board] WARNING: pin removed from index but image unlink failed: {e}")
            return {"ok": True, "id": pin_id}

    return router


def mount(app, generated_dir: Path) -> None:
    """Register the board static-image mount + API routes on the FastAPI app."""
    try:
        app.mount(
            "/board_images",
            StaticFiles(directory=str(BOARD_IMAGES_DIR)),
            name="board_images",
        )
    except Exception as e:
        print(f"[board] WARNING: could not mount /board_images: {e}")
    app.include_router(make_router(generated_dir))
    print(f"[board] Board store ready at {BOARD_DIR}")
