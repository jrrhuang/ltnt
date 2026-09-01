"""
OpenAI Images API proxy (gpt-image-1 / gpt-image-2).

Isolated in its own file + router so it can be added / removed without
touching the Reve code path. To enable:
    from server_openai import router as openai_router
    app.include_router(openai_router)

Env vars:
    OPENAI_API_KEY       required
    OPENAI_IMAGE_MODEL   optional, default "gpt-image-2"
"""

from __future__ import annotations

import base64
import io
import os
import traceback
from typing import Optional, Tuple

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image as PILImage

router = APIRouter()

OPENAI_EDIT_URL = "https://api.openai.com/v1/images/edits"


class OpenAIEditRequest(BaseModel):
    image_b64: str                # full PNG, base64 (may include "data:..." prefix)
    prompt: str
    mask_b64: Optional[str] = None  # optional PNG; transparent = edit area
    n: int = 1
    size: str = "auto"            # "auto" | "1024x1024" | "1024x1536" | "1536x1024"
    quality: str = "auto"         # "auto" | "low" | "medium" | "high"
    model: Optional[str] = None


def _strip_data_url(b64: str) -> str:
    if not b64:
        return ""
    if b64.startswith("data:"):
        return b64.split(",", 1)[1] if "," in b64 else ""
    return b64


# gpt-image-2 input constraints (from the public docs)
_MIN_PIXELS = 655_360
_MAX_PIXELS = 8_294_400
_MAX_EDGE   = 3840
_EDGE_MULT  = 16


def _normalize_for_openai(
    img_bytes: bytes, mask_bytes: Optional[bytes] = None,
) -> Tuple[bytes, Optional[bytes]]:
    """Resize image (+ mask if given) to a size gpt-image-2 will accept.

    Rules enforced:
      - total pixels in [655_360, 8_294_400]
      - both edges ≤ 3840 px
      - both edges are multiples of 16 px
      - mask and image end up the same size
      - mask is saved as RGBA PNG (transparent = edit area)
    A 512×512 FLUX output gets bicubic-upscaled to ~816×816, which meets the
    minimum-pixel gate. No detail is added — this is just pad-to-spec.
    """
    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    mask = (
        PILImage.open(io.BytesIO(mask_bytes)).convert("RGBA")
        if mask_bytes is not None else None
    )

    W, H = img.size

    # 1. Scale up if below minimum pixel count.
    if W * H < _MIN_PIXELS:
        scale = (_MIN_PIXELS / (W * H)) ** 0.5 * 1.05  # 5 % margin
        W, H = int(W * scale), int(H * scale)

    # 2. Scale down if above maximum pixel count.
    if W * H > _MAX_PIXELS:
        scale = (_MAX_PIXELS / (W * H)) ** 0.5 * 0.98
        W, H = int(W * scale), int(H * scale)

    # 3. Cap each edge at MAX_EDGE.
    if max(W, H) > _MAX_EDGE:
        scale = _MAX_EDGE / max(W, H)
        W, H = int(W * scale), int(H * scale)

    # 4. Round to multiples of 16 (up, then re-check).
    def _round16(x): return ((x + _EDGE_MULT - 1) // _EDGE_MULT) * _EDGE_MULT
    W, H = _round16(W), _round16(H)
    # After rounding, we might have dropped below the minimum pixel count; grow
    # the shorter edge in 16-px steps until we clear the floor again.
    while W * H < _MIN_PIXELS:
        if W <= H: W += _EDGE_MULT
        else:      H += _EDGE_MULT

    img_resized = img.resize((W, H), PILImage.LANCZOS)
    out_img = io.BytesIO()
    img_resized.save(out_img, format="PNG")

    out_mask = None
    if mask is not None:
        # NEAREST (not Lanczos) to keep alpha strictly binary. Interpolated
        # mask edges produce fractional alpha values, which gpt-image-2
        # treats as undefined — in practice it paints black into the
        # blurred-edge ring around an edit. Threshold after resize as a
        # belt-and-braces guard.
        mask_resized = mask.resize((W, H), PILImage.NEAREST)
        r, g, b, a = mask_resized.split()
        # Binary-threshold alpha: anything <128 becomes fully transparent
        # (edit area), otherwise fully opaque (keep area).
        a = a.point(lambda v: 0 if v < 128 else 255)
        mask_resized = PILImage.merge("RGBA", (r, g, b, a))
        buf = io.BytesIO()
        mask_resized.save(buf, format="PNG")
        out_mask = buf.getvalue()

    return out_img.getvalue(), out_mask


@router.post("/api/openai/edit")
async def openai_edit(req: OpenAIEditRequest):
    """Proxy for OpenAI /v1/images/edits.

    Accepts a full image + prompt (+ optional mask). Returns one or more
    base64-encoded edited images in the shape the frontend expects.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server")

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        img_bytes = base64.b64decode(_strip_data_url(req.image_b64))
    except Exception:
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64")

    mask_bytes = None
    if req.mask_b64:
        try:
            mask_bytes = base64.b64decode(_strip_data_url(req.mask_b64))
        except Exception:
            raise HTTPException(status_code=400, detail="mask_b64 is not valid base64")

    model = req.model or os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")

    # Reshape image + mask to satisfy gpt-image-2's input constraints (min
    # pixel count, multiples of 16, matched dimensions). No-op if already
    # within spec.
    try:
        img_bytes, mask_bytes = _normalize_for_openai(img_bytes, mask_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not prepare image for OpenAI: {type(exc).__name__}: {exc}")

    # Guard against degenerate masks:
    #   - 0 transparent pixels → user didn't select anything; fall back to a
    #     full-image edit instead of erroring.
    #   - Very few transparent pixels (< 256 total) → user's selection is too
    #     small for the model to act on; reject early with a clear message
    #     rather than letting OpenAI return something useless.
    if mask_bytes is not None:
        try:
            mask_img = PILImage.open(io.BytesIO(mask_bytes)).convert("RGBA")
            alpha = mask_img.split()[-1]
            # count pixels with alpha == 0 (the "edit here" region)
            transparent_px = sum(1 for v in alpha.getdata() if v == 0)
            total_px = mask_img.size[0] * mask_img.size[1]
            if transparent_px == 0:
                print("[openai] mask has no transparent pixels — treating as full-image edit")
                mask_bytes = None
            elif transparent_px < 256:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Selected region is too small ({transparent_px} px). "
                        f"Draw a larger box/lasso, or deselect to edit the whole image."
                    ),
                )
            elif transparent_px >= total_px * 0.98:
                # Essentially the whole canvas → equivalent to no mask; skip it
                # so OpenAI doesn't spend time on a near-useless guidance pass.
                print("[openai] mask covers ~all of image — treating as full-image edit")
                mask_bytes = None
        except HTTPException:
            raise
        except Exception as exc:
            print(f"[openai] mask inspection failed, forwarding anyway: {exc}")

    files = [("image[]", ("image.png", img_bytes, "image/png"))]
    if mask_bytes is not None:
        files.append(("mask", ("mask.png", mask_bytes, "image/png")))

    data = {
        "model": model,
        "prompt": req.prompt,
        "n": str(max(1, min(int(req.n or 1), 4))),
        "size": req.size or "auto",
        "quality": req.quality or "auto",
    }

    headers = {"Authorization": f"Bearer {api_key}"}

    # Retry transient failures — OpenAI occasionally drops connections.
    resp = None
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(OPENAI_EDIT_URL, headers=headers, data=data, files=files)
            break
        except Exception as exc:
            last_exc = exc
            print(f"[openai] attempt {attempt + 1}/3 failed: {type(exc).__name__}: {exc}")

    if resp is None:
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI request failed after 3 attempts: {type(last_exc).__name__}: {last_exc}",
        )

    if not resp.is_success:
        print(f"[openai] edit failed: {resp.status_code} {resp.text[:500]}")
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"OpenAI API error {resp.status_code}: {resp.text[:400]}",
        )

    try:
        payload = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"OpenAI returned non-JSON: {resp.text[:200]}")

    data_items = payload.get("data") or []
    if not data_items:
        raise HTTPException(status_code=502, detail="OpenAI returned no images")

    return {
        "images_b64": [it.get("b64_json") for it in data_items if it.get("b64_json")],
        "model": payload.get("model") or model,
        "usage": payload.get("usage"),
    }
