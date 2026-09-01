"""CREATE MAP — radial precompute.

Runs the FLUX tree expansion, decodes every particle, extracts DINO
features, and builds a radial/similarity-aware layout. Writes images
to {output_dir}/images/ and manifest.json to {output_dir}/.

This module is a self-contained alternative to flux_explore_precompute.py.
It does not modify the original.

`progress_cb(frac: float in [0,1], stage: str)` is called throughout so
the server can forward progress to the client.
"""
from __future__ import annotations

import os
import json
import math
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = str(Path(__file__).resolve().parent.parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from explore_map.layout import build_radial_layout


def _linear_resample_steps(num_steps: int, R: int) -> list:
    if R == 0:
        return []
    max_step = num_steps * 0.4
    out = []
    for k in range(R):
        frac = ((k + 1) / (R + 1)) ** 1.5
        step = max(1, round(1 + (max_step - 1) * frac))
        out.append(step)
    out = sorted(set(out))
    while len(out) < R:
        for i in range(len(out) - 1):
            if out[i + 1] - out[i] > 1:
                out.insert(i + 1, out[i] + 1)
                break
        else:
            out.append(out[-1] + 1)
        out = sorted(set(out))
    return out[:R]


def _particle_depth(i: int, C: int, R: int) -> int:
    """Depth = number of non-zero digits in base-C representation of
    `i mod C^R`. Each non-zero digit corresponds to one divergence event
    (seg-k clone) in the particle's history."""
    if R == 0:
        return 0
    x = i % (C ** R)
    depth = 0
    while x > 0:
        if x % C != 0:
            depth += 1
        x //= C
    return depth


def _parent_index(i: int, depth: int, C: int, R: int) -> Optional[int]:
    """Parent = zero out the lowest non-zero base-C digit of `i mod C^R`.
    That undoes the most recent divergence event."""
    if depth == 0 or R == 0:
        return None
    period = C ** R
    base = (i // period) * period
    x = i % period
    multiplier = 1
    while x > 0:
        d = x % C
        if d != 0:
            return base + (i % period) - d * multiplier
        x //= C
        multiplier *= C
    return None


def precompute_radial_map(
    prompt: str,
    *,
    model: str = "flux",
    num_particles: int = 4,
    num_clones: int = 3,
    num_resampling_steps: int = 3,
    augment: bool = True,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    rho: float = 0.0,
    height: int = 512,
    width: int = 512,
    seed: Optional[int] = 42,
    output_dir: str = "outputs/create_map",
    r0: float = 0.12,
    decay: float = 0.5,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    sampler=None,
) -> str:
    """Produces manifest.json at {output_dir}/manifest.json. Returns its path."""
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    def _prog(f, s):
        if progress_cb is not None:
            try:
                progress_cb(float(max(0.0, min(1.0, f))), str(s))
            except Exception:
                pass

    device = "cuda"

    # ── Model: reuse or load with REAL progress ─────────────────────────────
    # When the caller passes the server's resident `sampler` (same model
    # already on the GPU), we REUSE it — never instantiate a second copy of a
    # ~33GB model next to the resident one (observed CUDA OOM in prod).
    # A fresh load reports a measured fraction (GPU bytes allocated / expected
    # footprint) so the bar moves the whole ~45s instead of freezing at 2%.
    _EXPECTED_BYTES = {"flux": 33e9, "sana": 9e9, "fluxfm": 33e9, "krea2": 30e9}
    # Krea-2 is 512-only on an L40S (1024 OOMs); force the square 512 shape it
    # was validated at, regardless of caller-supplied height/width.
    if model == "krea2":
        height, width = 512, 512
    reused_sampler = sampler is not None
    if reused_sampler:
        _prog(0.14, "Reusing the loaded image model…")
        if model == "sana" and height == 512 and width == 512:
            height, width = 1024, 1024
    else:
        _prog(0.02, "Loading the image model…")
        # VRAM guard: fail FAST with a plain-words message instead of OOMing
        # minutes into a doomed duplicate load.
        if torch.cuda.is_available():
            try:
                _free_b, _total_b = torch.cuda.mem_get_info()
            except Exception:
                _free_b = None
            _need = _EXPECTED_BYTES.get(model, 30e9)
            if _free_b is not None and _free_b < _need * 0.95:
                raise RuntimeError(
                    f"Not enough GPU memory to load the {model.upper()} model for the map "
                    f"({_free_b / 2**30:.1f} GB free, ~{_need / 2**30:.0f} GB needed). "
                    "Another model is still loaded — finish or cancel the current "
                    "generation, then try the map again."
                )
        import threading as _threading
        _stop_watch = _threading.Event()

        def _watch_load():
            try:
                base = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            except Exception:
                base = 0
            exp = _EXPECTED_BYTES.get(model, 30e9)
            while not _stop_watch.wait(0.5):
                try:
                    alloc = max(torch.cuda.memory_allocated() - base, 0)
                except Exception:
                    alloc = 0
                f = min(alloc / exp, 0.99)
                _prog(0.02 + 0.12 * f,
                      f"Loading the image model ({alloc / 2**30:.1f} / {exp / 2**30:.0f} GB)…")

        _threading.Thread(target=_watch_load, daemon=True).start()
        try:
            if model == "sana":
                from sana_fmtt import SanaInteractive
                sampler = SanaInteractive(prompt=prompt, device=device)
                # SANA's native res is 1024; defaults that originated from FLUX (512)
                # would tile awkwardly through the DC-AE — bump to 1024 if at FLUX default.
                if height == 512 and width == 512:
                    height, width = 1024, 1024
            elif model == "fluxfm":
                from flux_fmwdm_interactive import FluxFMWDMInteractive
                sampler = FluxFMWDMInteractive(prompt=prompt, device=device)
            elif model == "krea2":
                import types
                from krea2_fmtt import Krea2Interactive
                sampler = Krea2Interactive(prompt=prompt, device=device)
                # Krea-2's native predict_velocity/decode work in PACKED token
                # space [B, tokens, 64], but the generic tree loop below drives
                # every sampler in FLUX's UNPACKED 4D convention
                # ([B, C, lat_h, lat_w]) — anchors are randn'd 4D, cloned via
                # 5D expand, etc. Wrap the two call-sites so they accept the
                # 4D latent and pack it internally, exactly like FLUX does.
                # _setup_geometry(512,512) must run first so self.latent_h/
                # latent_w/position_ids/_mu are populated for _pack/decode.
                sampler._setup_geometry(int(height), int(width))
                _krea2_pv = sampler.predict_velocity
                _krea2_dec = sampler.decode

                def _pv_unpacked(self, z, t_cur, guidance_scale=4.5,
                                 prompt_idx=None):
                    # z: [B, C, lat_h, lat_w] (unpacked) -> pack -> velocity ->
                    # unpack back to 4D so the loop's Euler update stays 4D.
                    z_packed = self._pack(z.to(self.dtype))
                    v_packed = _krea2_pv(z_packed, t_cur, guidance_scale,
                                         prompt_idx=None)
                    imgH = self.latent_h * self.vae_scale_factor
                    imgW = self.latent_w * self.vae_scale_factor
                    v = self.pipeline._unpack_latents(v_packed, imgH, imgW)
                    # _unpack_latents yields [B, C, 1, lh, lw] for Krea-2's
                    # 5D VAE convention; drop the frame axis if present.
                    if v.dim() == 5:
                        v = v[:, :, 0]
                    return v.to(z.dtype)

                def _dec_unpacked(self, z):
                    # z: [B, C, lat_h, lat_w] (unpacked) -> pack -> native decode
                    return _krea2_dec(self._pack(z.to(self.dtype)))

                sampler.predict_velocity = types.MethodType(_pv_unpacked, sampler)
                sampler.decode = types.MethodType(_dec_unpacked, sampler)
            else:
                from flux_fmtt_dev import FluxDevInteractive
                sampler = FluxDevInteractive(prompt=prompt, device=device,
                                             keep_text_encoders=bool(augment))
        finally:
            _stop_watch.set()
        _prog(0.14, "Image model ready…")

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # Swap DINO for public DINOv2 (same trick used in the legacy precompute).
    # SanaInteractive and FluxFMWDMInteractive already use DINOv2 with the right
    # load flags; only patch the FLUX sampler which defaults to gated DINOv3.
    if model not in ("sana", "fluxfm", "krea2") and not reused_sampler:
        # (A reused server sampler already carries the server's idempotent
        # DINOv2 loader — don't overwrite the resident instance's method.)
        import types
        def _load_dino_v2(self):
            if getattr(self, "dino_model", None) is not None:
                return  # idempotent: already resident
            from transformers import AutoImageProcessor, AutoModel
            dino_id = "facebook/dinov2-base"
            self.dino_processor = AutoImageProcessor.from_pretrained(dino_id)
            self.dino_model = AutoModel.from_pretrained(
                dino_id, low_cpu_mem_usage=False
            ).to(self.device).eval()
        sampler._load_dino = types.MethodType(_load_dino_v2, sampler)
    _prog(0.145, "Loading the similarity model…")
    sampler._load_dino()

    # A REUSED sampler still carries the previous session's cached prompt
    # embeddings — re-embed THIS map's prompt (augment block below overrides
    # with the full variation set when it runs).
    if reused_sampler and model == "flux":
        sampler.set_prompts([prompt])

    # Augmentation-seeded roots: each root gets a DIFFERENT interpretation of the
    # prompt so the precomputed tree is a COMPREHENSIVE set spanning distinct
    # regions, not just latent variations of one reading. FLUX-only (set_prompts).
    _aug_K = 0
    if augment and model == "flux":
        try:
            import prompt_augment
            _prog(0.16, "Exploring prompt interpretations…")
            # Per-token ticks: the LLM generation is a 10-30s wait.
            _vars = [prompt] + prompt_augment.augment_prompt(
                prompt, n=max(0, num_particles - 1), device=device,
                token_cb=lambda done, total: _prog(
                    0.16 + 0.08 * min(done / max(total, 1), 1.0),
                    "Exploring prompt interpretations…"))
            _vars = _vars[:num_particles] or [prompt]
            sampler.set_prompts(_vars)
            _aug_K = len(_vars)
            print(f"[create_map] augmented roots: {_aug_K} interpretations")
            # Embeddings are cached -> free the text encoders to reclaim VRAM
            # for the (large) tree generation. predict_velocity uses the cached
            # embeds via prompt_idx, not the encoders. NEVER gut a REUSED
            # (server-resident) sampler — its warm reuse needs the encoders.
            if not reused_sampler:
                for _attr in ("text_encoder", "text_encoder_2"):
                    _te = getattr(sampler.pipeline, _attr, None)
                    if _te is not None:
                        try: _te.to("cpu")
                        except Exception: pass
                        setattr(sampler.pipeline, _attr, None)
                import gc as _gc; _gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as _e:
            print(f"[create_map] augment failed ({_e}); single-prompt tree")
            _aug_K = 0
            if reused_sampler:
                # Make sure the tree isn't silently generated against the
                # PREVIOUS session's cached prompt embeddings.
                try: sampler.set_prompts([prompt])
                except Exception: pass

    C = num_clones
    R = num_resampling_steps

    # Model-dependent latent geometry + time schedule.
    if model == "krea2":
        # Krea-2: work in FLUX-style UNPACKED 4D anchors [P, C, lat_h, lat_w];
        # the wrapped predict_velocity/decode (installed at load) pack per call.
        # _setup_geometry(512,512) already ran at load, so latent_h/latent_w/mu
        # are populated. Use Krea-2's OWN exponential-shifted sigma schedule.
        latent_channels = sampler.num_channels_latents      # 16 @ 512
        latent_h = sampler.latent_h                          # 64 @ 512
        latent_w = sampler.latent_w                          # 64 @ 512
        time_steps = sampler.get_sigma_schedule(num_steps).to(device)
    elif model == "sana":
        # SANA: no FLUX-style packing. Latent is [B, 32, H/32, W/32].
        latent_channels = sampler.latent_channels
        latent_h = int(height) // sampler.vae_scale_factor
        latent_w = int(width) // sampler.vae_scale_factor
        # Linear σ from 1 → 1/N then a trailing 0 — matches sana_fmtt.sample_interactive.
        sigmas_arr = np.linspace(1.0, 1.0 / num_steps, num_steps)
        time_steps = torch.tensor(
            list(sigmas_arr) + [0.0], device=device, dtype=torch.float32)
    else:
        from diffusers.pipelines.flux.pipeline_flux import (
            calculate_shift, retrieve_timesteps)
        latent_channels = 16
        latent_h = 2 * (int(height) // (sampler.vae_scale_factor * 2))
        latent_w = 2 * (int(width) // (sampler.vae_scale_factor * 2))
        image_seq_len = (latent_h // 2) * (latent_w // 2)
        mu = calculate_shift(
            image_seq_len,
            sampler.pipeline.scheduler.config.get("base_image_seq_len", 256),
            sampler.pipeline.scheduler.config.get("max_image_seq_len", 4096),
            sampler.pipeline.scheduler.config.get("base_shift", 0.5),
            sampler.pipeline.scheduler.config.get("max_shift", 1.15),
        )
        sigmas_arr = np.linspace(1.0, 1 / num_steps, num_steps)
        timesteps, _ = retrieve_timesteps(
            sampler.pipeline.scheduler, num_steps, device, sigmas=sigmas_arr, mu=mu)
        time_steps = torch.cat([timesteps / 1000.0, torch.zeros(1, device=device)])

    resample_steps = _linear_resample_steps(num_steps, R)
    resample_at = set(resample_steps)
    segment_bounds = [0] + resample_steps + [num_steps]

    anchors = torch.randn(num_particles, latent_channels, latent_h, latent_w,
                          device=device, dtype=sampler.dtype)
    total_particles = num_particles * C
    # Maps each particle -> its root's prompt-variation index; grows with the tree.
    _root_of = (torch.arange(num_particles) % _aug_K).repeat_interleave(C) if _aug_K else None
    X_all = anchors.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
    X_all = X_all.reshape(total_particles, latent_channels, latent_h, latent_w)

    is_anchor = torch.zeros(total_particles, dtype=torch.bool)
    for i in range(num_particles):
        is_anchor[i * C] = True

    glass_params = None
    interval_count = 0

    # Tree generation uses 25-62% of the bar. Late steps carry a big
    # per-particle loop (the tree has grown to N*C^R particles), so we also
    # tick INSIDE each step per particle-update — the % must keep moving.
    _prog(0.25, "Growing the tree…")

    def _tree_prog(k_done_frac, n_imgs):
        _prog(0.25 + 0.37 * (k_done_frac / num_steps),
              f"Growing the tree — step {min(int(k_done_frac) + 1, num_steps)}"
              f"/{num_steps} ({n_imgs} images)…")

    for k in range(num_steps):
        t_cur = time_steps[k].item()
        t_next = time_steps[k + 1].item()
        dt = t_next - t_cur

        if glass_params is None:
            seg_end_step = next((b for b in segment_bounds if b > k), num_steps)
            sigma_seg_start = time_steps[k].item()
            sigma_seg_end = time_steps[seg_end_step].item()
            M_inner = seg_end_step - k

            clone_indices = (~is_anchor).nonzero(as_tuple=True)[0]
            if len(clone_indices) > 0:
                parent_states = torch.zeros(len(clone_indices), latent_channels, latent_h, latent_w,
                                            device=device, dtype=sampler.dtype)
                for ci, idx in enumerate(clone_indices):
                    group = idx.item() // C
                    parent_states[ci] = X_all[group * C]

                clip = 1e-8
                alpha_start = 1.0 - sigma_seg_start
                alpha_end = 1.0 - sigma_seg_end
                bar_gamma = rho * sigma_seg_end / max(sigma_seg_start, clip)
                bar_alpha = alpha_end - bar_gamma * alpha_start
                bar_sigma = math.sqrt(max(sigma_seg_end ** 2 * (1.0 - rho ** 2), 0.0))
                bar_sigma_0 = 1.0
                eps = torch.randn_like(parent_states)
                clone_inner = bar_gamma * parent_states + bar_sigma_0 * eps
                gp = dict(
                    x_t=parent_states.clone(),
                    alpha_start=alpha_start,
                    sigma_start=sigma_seg_start,
                    bar_gamma=bar_gamma,
                    bar_alpha=bar_alpha,
                    bar_sigma=bar_sigma,
                    bar_sigma_0=bar_sigma_0,
                )
                X_all[clone_indices] = clone_inner
                glass_params = gp
                glass_params["seg_start_step"] = k
                glass_params["M_inner"] = M_inner
                glass_params["clone_indices"] = clone_indices
            else:
                glass_params = {"seg_start_step": k, "M_inner": M_inner,
                                "clone_indices": torch.tensor([], dtype=torch.long)}

        m = k - glass_params["seg_start_step"]
        M = glass_params["M_inner"]
        s = m / M
        ds = 1.0 / M

        anchor_indices = is_anchor.nonzero(as_tuple=True)[0]
        _step_work = max(len(anchor_indices) + len(glass_params["clone_indices"]), 1)
        _done_in_step = 0
        with torch.no_grad():
            for ai in anchor_indices:
                z_p = X_all[ai:ai+1]
                _pi = _root_of[ai:ai+1] if _root_of is not None else None
                v = sampler.predict_velocity(z_p, t_cur, guidance_scale, prompt_idx=_pi)
                X_all[ai:ai+1] = z_p + dt * v
                _done_in_step += 1
                _tree_prog(k + _done_in_step / _step_work, total_particles)

        clone_indices = glass_params["clone_indices"]
        if len(clone_indices) > 0:
            gp = glass_params
            x_t = gp["x_t"]
            clip = 1e-8
            bar_alpha_s = s * gp["bar_alpha"]
            bar_sigma_s = (1.0 - s) * gp["bar_sigma_0"] + s * gp["bar_sigma"]
            dot_bar_sigma_s = gp["bar_sigma"] - gp["bar_sigma_0"]
            dot_bar_alpha_s = gp["bar_alpha"]
            w1 = dot_bar_sigma_s / max(bar_sigma_s, clip)
            w2 = dot_bar_alpha_s - bar_alpha_s * w1
            w3 = -gp["bar_gamma"] * w1
            mu1 = gp["alpha_start"]
            mu2 = bar_alpha_s + gp["bar_gamma"] * gp["alpha_start"]
            S11 = gp["sigma_start"] ** 2
            S12 = gp["bar_gamma"] * S11
            S22 = bar_sigma_s ** 2 + gp["bar_gamma"] ** 2 * S11
            det = max(S11 * S22 - S12 * S12, clip)
            inv11 = S22 / det
            inv12 = -S12 / det
            inv22 = S11 / det
            bproduct = max(mu1*mu1*inv11 + 2*mu1*mu2*inv12 + mu2*mu2*inv22, clip)
            t_star = max(min(1.0/(1.0+math.sqrt(max(1.0/bproduct, 0.0))), 0.999), 0.001)
            sigma_star = 1.0 - t_star
            alpha_star = t_star
            w_xt = alpha_star * (mu1*inv11 + mu2*inv12) / bproduct
            w_xs = alpha_star * (mu1*inv12 + mu2*inv22) / bproduct

            with torch.no_grad():
                for ci, idx in enumerate(clone_indices):
                    x_parent = x_t[ci:ci+1]
                    x_clone = X_all[idx:idx+1]
                    S_input = w_xt * x_parent + w_xs * x_clone
                    _pi = _root_of[idx:idx+1] if _root_of is not None else None
                    v_star = sampler.predict_velocity(S_input, sigma_star, guidance_scale, prompt_idx=_pi)
                    denoiser = S_input - sigma_star * v_star
                    velocity = w1 * x_clone + w2 * denoiser + w3 * x_parent
                    X_all[idx:idx+1] = x_clone + ds * velocity
                    _done_in_step += 1
                    _tree_prog(k + _done_in_step / _step_work, total_particles)

        if k in resample_at:
            glass_params = None
            interval_count += 1
            is_last = (interval_count == R)
            if is_last:
                num_particles = total_particles
                is_anchor = torch.ones(total_particles, dtype=torch.bool)
            else:
                kept = X_all.clone()
                num_particles = total_particles
                total_particles = num_particles * C
                X_all = kept.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
                X_all = X_all.reshape(total_particles, latent_channels, latent_h, latent_w)
                if _root_of is not None:
                    _root_of = _root_of.repeat_interleave(C)
                is_anchor = torch.zeros(total_particles, dtype=torch.bool)
                for i in range(num_particles):
                    is_anchor[i * C] = True

        _tree_prog(float(k + 1), total_particles)

    # Decode — 62% → 88%
    pil_images: list[Image.Image] = []
    with torch.no_grad():
        for p_idx in range(total_particles):
            img_t = sampler.decode(X_all[p_idx:p_idx+1])
            arr = (img_t[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            pil.save(os.path.join(images_dir, f"img_{p_idx:04d}.png"))
            pil_images.append(pil)
            _prog(0.62 + 0.26 * ((p_idx + 1) / total_particles),
                  f"Rendering image {p_idx+1}/{total_particles}…")

    # DINO features — 88% → 95%, chunked so the % keeps moving on big maps.
    _prog(0.88, "Reading the map…")
    _feat_chunks = []
    _CHUNK = 16
    for _s0 in range(0, len(pil_images), _CHUNK):
        _feat_chunks.append(
            sampler._extract_dino_features(pil_images[_s0:_s0 + _CHUNK]))
        _end = min(_s0 + _CHUNK, len(pil_images))
        _prog(0.88 + 0.07 * (_end / len(pil_images)),
              f"Analyzing similarity {_end}/{len(pil_images)}…")
    features = np.concatenate(_feat_chunks, axis=0)
    features = np.asarray(features, dtype=np.float32)

    # Build tree metadata
    parents: list = []
    depths: list = []
    for i in range(total_particles):
        d = _particle_depth(i, C, R)
        depths.append(d)
        parents.append(_parent_index(i, d, C, R))

    _prog(0.95, "Arranging by similarity…")
    coords = build_radial_layout(features, parents, depths, r0=r0, decay=decay,
                                 seed=(seed if seed is not None else 42))

    manifest = {
        "prompt": prompt,
        "model": model,
        "layout": "radial-v1",
        "num_particles": num_particles,
        "num_clones": C,
        "num_resampling_steps": R,
        "total_images": total_particles,
        "max_depth": R,
        "r0": r0,
        "decay": decay,
        "images": [
            {
                "id": i,
                "path": f"images/img_{i:04d}.png",
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "depth": depths[i],
                "parent": parents[i],
            }
            for i in range(total_particles)
        ],
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # A REUSED (server-resident) sampler is left fully intact — the whole point
    # is that the next generate keeps its warm model. Only a sampler WE loaded
    # gets torn down so the next request can load fresh.
    if not reused_sampler:
        try:
            sampler._unload_dino()
        except Exception:
            pass

        # Release the model from GPU so the next request (GENERATE or another
        # CREATE MAP) can load fresh.
        try:
            import gc
            for attr in ('pipeline', 'transformer', 'vae', 'text_encoder', 'text_encoder_2',
                         'tokenizer', 'tokenizer_2', 'dino_model', 'dino_processor'):
                sub = getattr(sampler, attr, None)
                if sub is not None and hasattr(sub, 'to'):
                    try: sub.to('cpu')
                    except Exception: pass
                try: setattr(sampler, attr, None)
                except Exception: pass
            del sampler
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    _prog(1.0, "Done")
    return manifest_path
