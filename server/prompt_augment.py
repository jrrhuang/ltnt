"""Local-LLM prompt augmentation for LTNT diversity.

Given a base prompt, expand it into N distinct image-generation prompts that
explore genuinely different visual directions (style / mood / composition /
medium), so the initial particle population spans semantically diverse regions
of the model's latent space rather than one "confident default".

Runs a small, ungated, Apache-2.0 instruct model (Qwen2.5-3B-Instruct) entirely
locally — no paid API, ships with the open-source repo. Loaded lazily and cached.
"""
import os
import re
from typing import List

_LLM = {"model": None, "tok": None}
_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def _load():
    """Load the LLM once, onto CPU. augment_prompt() moves it to GPU only for the
    brief generation and offloads it again, so it doesn't hold ~6GB of GPU while
    the (much larger, resident) FLUX model generates."""
    if _LLM["model"] is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Resolve the explicit local snapshot dir (same robustness pattern as FLUX:
    # avoids huggingface's flaky offline Hub resolution on cluster nodes).
    target = _MODEL_ID
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    repo_dir = os.path.join(hf_home, "hub", "models--" + _MODEL_ID.replace("/", "--"))
    ref = os.path.join(repo_dir, "refs", "main")
    if os.path.exists(ref):
        snap = os.path.join(repo_dir, "snapshots", open(ref).read().strip())
        if os.path.isdir(snap):
            target = snap
    tok = AutoTokenizer.from_pretrained(target)
    model = AutoModelForCausalLM.from_pretrained(target, torch_dtype=torch.bfloat16).eval()
    _LLM["model"], _LLM["tok"] = model, tok


def warm(device: str = "cpu"):
    """Fully warm the augment LLM for `device` so the FIRST real augment call
    runs at warm speed (~35s on CPU) instead of paying disk->RAM load +
    bf16->f32 cast + cold kernels (~60-90s) inside the first span round.

    For CPU (the device the free-VRAM guard picks next to a resident FLUX):
    load, cast to float32 (the CPU matmul dtype augment_prompt uses), and run a
    tiny generation to JIT/warm the CPU kernels. Safe to call from a background
    thread; never raises.
    """
    try:
        import torch
        _load()
        model, tok = _LLM["model"], _LLM["tok"]
        if device != "cuda":
            if model.dtype != torch.float32:
                model.to(torch.float32)
            inputs = tok("hello", return_tensors="pt")
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=4, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        print(f"[prompt_augment] warm({device}) complete — first augment call "
              "will run at warm speed.")
    except Exception as e:
        print(f"[prompt_augment] warm failed (lazy load will cover): "
              f"{type(e).__name__}: {e}")


# Explicit style/medium markers the ARTIST may have pinned in their prompt.
# If the base prompt names one of these, every variation must keep it (the
# LLM is instructed to; this list is the programmatic safety net that
# re-appends a dropped constraint). Lowercase substring match.
_STYLE_MARKERS = [
    "ukiyo-e", "ukiyoe", "woodblock", "woodcut", "linocut",
    "oil painting", "oil on canvas", "watercolor", "watercolour", "gouache",
    "acrylic", "pastel drawing", "charcoal", "pencil sketch", "pen and ink",
    "ink line art", "ink wash", "sumi-e", "etching", "engraving", "lithograph",
    "pixel art", "8-bit", "voxel", "low poly", "3d render", "claymation",
    "stop motion", "papercraft", "paper cutout", "origami", "collage",
    "stained glass", "mosaic", "tapestry", "embroidery", "cross-stitch",
    "photorealistic", "photoreal", "polaroid", "daguerreotype", "cyanotype",
    "film noir", "anime", "manga", "cel shaded", "cartoon", "comic book",
    "art nouveau", "art deco", "bauhaus", "cubist", "cubism", "impressionist",
    "impressionism", "expressionist", "surrealist", "pointillism", "fauvist",
    "minimalist poster", "vaporwave", "synthwave", "steampunk", "cyberpunk",
    "isometric", "blueprint", "technical drawing", "chalkboard",
    "graffiti", "street art", "fresco", "byzantine icon", "pop art",
    "in the style of",
]


def _pinned_styles(prompt: str) -> List[str]:
    """Style/medium constraints the artist explicitly wrote into the prompt."""
    pl = prompt.lower()
    return [m for m in _STYLE_MARKERS if m in pl]


def augment_prompt(prompt: str, n: int = 4, device: str = "cuda",
                   temperature: float = 0.8, token_cb=None,
                   mode: str = "wild") -> List[str]:
    """Return up to n diverse image-prompt variations of `prompt`.

    Explicit style/medium constraints in the prompt (e.g. "ukiyo-e woodblock
    print") are PRESERVED in every variation: the LLM is instructed to keep
    them verbatim and vary everything else, and any variation that still drops
    a pinned style gets it re-appended. token_cb(done, total) is an optional
    per-token progress hook (the ~10-30s LLM generation is a visible wait).

    On any failure, degrades gracefully to [prompt] * n so generation never
    breaks just because augmentation hiccuped.
    """
    moved = False
    try:
        import torch
        _load()
        model, tok = _LLM["model"], _LLM["tok"]
        if device == "cuda" and torch.cuda.is_available():
            # GPU only for the brief generation; explicit bf16 restores fast
            # math in case a previous CPU call cast the weights to float32.
            model.to("cuda", dtype=torch.bfloat16); moved = True
        elif model.dtype != torch.float32:
            # CPU path: bf16 weights vs float32 activations break the CPU
            # matmul ("mat1 and mat2 must have the same dtype"). Cast once;
            # the dtype check makes this a cached no-op on later calls.
            model.to(torch.float32)
        pinned = _pinned_styles(prompt)
        if mode == "restyle":
            # Composition-preserving variations (Jerry 2026-08-28): used
            # when children spawn SHALLOW (structure band frozen) — only
            # words that act at mid-noise can show up in the image, so
            # scene/framing/camera vocabulary is banned here.
            sys_msg = (
                "You are a prompt designer for an AI image tool. Given a "
                "prompt, produce RESTYLINGS of the same picture: keep the "
                "SUBJECT, POSE, COMPOSITION and FRAMING exactly as implied "
                "— do NOT change the scene, background, camera angle, or "
                "what is depicted. Vary ONLY the art MEDIUM (oil painting, "
                "watercolor, woodcut, ink, 3D render, photoreal), the "
                "LIGHTING and COLOR PALETTE, the surface TEXTURE, and the "
                "MOOD. Each variation should read as the SAME image "
                "rendered by a different artist. Concise concrete image "
                "prompts in natural English, no commentary. Output ONLY a "
                "numbered list.")
            user_msg = (f'Prompt: "{prompt}"\nGive {n} restylings of the '
                        f'same picture (same subject, pose, composition; '
                        f'different medium/lighting/palette/mood).')
        else:
            sys_msg = None  # replaced below
        if sys_msg is None:
            sys_msg = (
            "You are a prompt designer for an AI image tool artists use to explore "
            "WILDLY DIFFERENT interpretations of ONE idea. Given a prompt, produce "
            "variations that keep the SAME core SUBJECT, but each takes a genuinely "
            "DIFFERENT visual approach: vary the art MEDIUM/STYLE (photoreal, oil "
            "painting, woodcut, watercolor, 3D render, ink line art, cinematic), the "
            "COMPOSITION and FRAMING (sweeping aerial, intimate macro close-up, wide "
            "minimalist, dramatic low angle), the LIGHTING/TIME/SEASON, and the MOOD. "
            "The goal is BROAD interpretive range, NOT subtle tweaks: an artist should "
            "feel each variation is a distinct creative DIRECTION on the same subject. "
            "Keep them concise concrete image prompts in natural English with spaces "
            "The original SUBJECT must remain the CLEAR DOMINANT focus of every "
            "variation — vary the treatment, never let the style or setting REPLACE "
            "the subject. "
            "HARD CONSTRAINT: if the artist's prompt SPECIFIES an art style, medium, "
            "technique, or artist reference (e.g. 'ukiyo-e woodblock print', 'oil "
            "painting', 'in the style of Moebius'), that choice is FIXED — repeat "
            "those exact style words VERBATIM in EVERY variation and NEVER substitute "
            "a different medium or style. In that case vary ONLY the composition, "
            "framing, subject treatment, palette, lighting, season, and mood WITHIN "
            "the artist's stated style. "
            "(NEVER underscores), no commentary. Output ONLY a numbered list."
        )
        if mode != "restyle":
            user_msg = (f'Prompt: "{prompt}"\nGive {n} boldly DIFFERENT interpretations that '
                        f'keep the same subject but vary the style, medium, composition, framing, '
                        f'lighting, and mood as much as possible.')
        if pinned:
            user_msg = (
                f'Prompt: "{prompt}"\n'
                f'The artist has FIXED the style: {", ".join(pinned)}. '
                f'Give {n} boldly DIFFERENT interpretations that ALL keep those exact '
                f'style words verbatim, varying only the composition, framing, subject '
                f'treatment, palette, lighting, and mood within that style.'
            )
        text = tok.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        _MAX_NEW = 256
        _gen_kwargs = dict(max_new_tokens=_MAX_NEW, do_sample=True,
                           temperature=temperature, top_p=0.95,
                           pad_token_id=tok.eos_token_id)
        if token_cb is not None:
            # Per-token progress hook via StoppingCriteria (never stops early;
            # just ticks). Failures in the hook must not break generation.
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _Tick(StoppingCriteria):
                def __init__(self, start_len):
                    self.start_len = start_len

                def __call__(self, input_ids, scores, **kw):
                    try:
                        token_cb(int(input_ids.shape[1] - self.start_len), _MAX_NEW)
                    except Exception:
                        pass
                    return False

            _gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [_Tick(inputs.input_ids.shape[1])])
        with torch.no_grad():
            out = model.generate(**inputs, **_gen_kwargs)
        gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        def _clean(s: str) -> str:
            # Small models sometimes emit snake_case "tags" despite instructions;
            # normalize to natural spaces (FLUX's encoder prefers them).
            return re.sub(r"\s+", " ", s.replace("_", " ")).strip().strip('"').strip()

        # Parse "1. ...", "2) ...", "- ..." lines first.
        variants = []
        for line in gen.splitlines():
            m = re.match(r"\s*(?:\d+[.)]|[-*])\s+(.*\S)", line)
            if m:
                v = _clean(m.group(1))
                if v:
                    variants.append(v)
        # Fallback: model didn't number them — take substantive non-preamble lines.
        if not variants:
            for line in gen.splitlines():
                s = _clean(line)
                if len(s) >= 15 and not s.lower().startswith(("here", "sure", "certainly", "prompt")):
                    variants.append(s)
        variants = variants[:n]
        if not variants:
            return [prompt] * n
        # Safety net: re-attach any pinned style the LLM still dropped, so an
        # explicit "ukiyo-e woodblock print" can NEVER be paraphrased away.
        if pinned:
            fixed = []
            for v in variants:
                missing = [m for m in pinned if m not in v.lower()]
                if missing:
                    v = v.rstrip(" .") + ", " + ", ".join(missing)
                    print(f"[prompt_augment] re-pinned style {missing} on: {v!r}")
                fixed.append(v)
            variants = fixed
        # pad to n by cycling if the model returned fewer
        while len(variants) < n:
            variants.append(variants[len(variants) % len(variants)])
        return variants
    except Exception as e:
        print(f"[prompt_augment] failed ({type(e).__name__}: {e}); using base prompt.")
        return [prompt] * n
    finally:
        # Always offload the LLM back to CPU so it doesn't hold GPU during the
        # (large) FLUX generation that follows.
        if moved:
            try:
                import torch
                _LLM["model"].to("cpu")
                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    import sys
    dev = "cuda" if "--cpu" not in sys.argv else "cpu"
    for p in ["possibilities", "a lonely lighthouse", "the feeling of nostalgia"]:
        print(f"\n=== base: {p!r}  (device={dev}) ===")
        for i, v in enumerate(augment_prompt(p, n=4, device=dev), 1):
            print(f"  {i}. {v}")
