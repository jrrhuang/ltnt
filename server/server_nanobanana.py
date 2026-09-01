"""
Google Gemini / Nano Banana image-edit proxy.

Mirrors the structure of server_openai.py. Isolated in its own file + router
so it can be added or removed without touching the other providers.

Env vars:
    GEMINI_API_KEY              required (Google AI Studio key)
    NANOBANANA_MODEL            optional, default "gemini-3.1-flash-image-preview"
                                (Flash tier — best speed/quality tradeoff).
                                Set to "nano-banana-pro-preview" for the
                                slower, higher-quality Pro model, or
                                "gemini-2.5-flash-image" for the older Flash.

To enable:
    from server_nanobanana import router as nanobanana_router
    app.include_router(nanobanana_router)
"""

from __future__ import annotations

import base64
import io
import os
import traceback
from typing import List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


def _strip_data_url(b64: str) -> str:
    if not b64:
        return ""
    if b64.startswith("data:"):
        return b64.split(",", 1)[1] if "," in b64 else ""
    return b64


class NanoBananaRegion(BaseModel):
    x: float
    y: float
    w: float
    h: float


class NanoBananaEditRequest(BaseModel):
    image_b64: str                # full PNG, base64 (may include "data:..." prefix)
    prompt: str
    mask_b64: Optional[str] = None  # ignored — Gemini has no mask API
    # Optional rect (normalized 0..1). When present the server crops to this
    # region before sending to Gemini, so the model sees only the selected
    # area and the edit is spatially anchored to the user's box. The
    # frontend then pastes the result back at the same coordinates.
    region: Optional[NanoBananaRegion] = None
    n: int = 1
    model: Optional[str] = None


@router.post("/api/nanobanana/edit")
async def nanobanana_edit(req: NanoBananaEditRequest):
    """Proxy for Google's gemini-*-image models.

    Returns one or more base64 PNGs in the same shape as the OpenAI proxy
    (`images_b64`) so the frontend can treat both providers uniformly.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server")

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        img_bytes = base64.b64decode(_strip_data_url(req.image_b64))
    except Exception:
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64")

    if not img_bytes:
        raise HTTPException(status_code=400, detail="image_b64 is empty")

    # If a region was provided, crop to it so Gemini only sees the selected
    # area. The frontend pastes the returned crop back at the region
    # coordinates. This gives the user spatial localization even though
    # Gemini's API has no mask/region parameter.
    if req.region:
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            W, H = img.size
            rx = max(0.0, min(1.0, req.region.x))
            ry = max(0.0, min(1.0, req.region.y))
            rw = max(0.0, min(1.0 - rx, req.region.w))
            rh = max(0.0, min(1.0 - ry, req.region.h))
            px, py = int(rx * W), int(ry * H)
            pw, ph = max(1, int(rw * W)), max(1, int(rh * H))
            cropped = img.crop((px, py, px + pw, py + ph))
            # Upscale tiny crops so Gemini has enough pixels to edit; the
            # frontend will squash the result back into the region on
            # composite, so we don't lose anything by going bigger here.
            MIN_EDGE = 256
            cw, ch = cropped.size
            if min(cw, ch) < MIN_EDGE:
                scale = MIN_EDGE / min(cw, ch)
                cropped = cropped.resize(
                    (int(cw * scale), int(ch * scale)), PILImage.LANCZOS
                )
            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not crop region: {type(exc).__name__}: {exc}")

    model = req.model or os.getenv("NANOBANANA_MODEL", "gemini-3.1-flash-image-preview")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    image_part = {
        "inline_data": {
            "mime_type": "image/png",
            "data": base64.b64encode(img_bytes).decode(),
        }
    }
    body = {
        "contents": [{
            "parts": [
                {"text": req.prompt},
                image_part,
            ]
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "candidateCount": max(1, min(int(req.n or 1), 4)),
        },
    }

    # Retry transient failures.
    resp = None
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(url, headers=headers, json=body)
            break
        except Exception as exc:
            last_exc = exc
            print(f"[nanobanana] attempt {attempt + 1}/3 failed: {type(exc).__name__}: {exc}")

    if resp is None:
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail=f"Gemini request failed after 3 attempts: {type(last_exc).__name__}: {last_exc}",
        )

    if not resp.is_success:
        print(f"[nanobanana] edit failed: {resp.status_code} {resp.text[:500]}")
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Gemini API error {resp.status_code}: {resp.text[:400]}",
        )

    try:
        payload = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Gemini returned non-JSON: {resp.text[:200]}")

    # Dig out image parts from candidates[].content.parts[].inline_data.data
    images_b64: List[str] = []
    for cand in payload.get("candidates", []) or []:
        parts = (cand.get("content") or {}).get("parts", []) or []
        for p in parts:
            inline = p.get("inline_data") or p.get("inlineData")  # either casing
            if inline and inline.get("data"):
                images_b64.append(inline["data"])

    if not images_b64:
        # Sometimes Gemini returns a text-only refusal; surface it.
        text_parts = []
        for cand in payload.get("candidates", []) or []:
            for p in (cand.get("content") or {}).get("parts", []) or []:
                if p.get("text"):
                    text_parts.append(p["text"])
        msg = " | ".join(text_parts)[:400] or "no image returned"
        raise HTTPException(status_code=502, detail=f"Gemini returned no image ({msg})")

    return {
        "images_b64": images_b64,
        "model": model,
    }
