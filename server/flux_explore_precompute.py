"""
Pre-compute a hierarchical latent space map for the Explore mode.

Generates all images at all zoom levels upfront (tree expansion with all
images kept). Computes DINO features and UMAP positions for all images.
Saves a JSON manifest with tree structure, positions, depths, and image paths.

Usage:
    python flux_explore_precompute.py --prompt "possibilities" --output_dir outputs/explore_map
"""

import os
import json
import gc

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional, Tuple, List
from PIL import Image
import sys
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def compute_linear_diversity_schedule(num_steps: int, num_resampling_steps: int) -> list:
    R = num_resampling_steps
    if R == 0:
        return []
    max_step = num_steps * 0.4
    steps = []
    for k in range(R):
        frac = ((k + 1) / (R + 1)) ** 1.5
        step = max(1, round(1 + (max_step - 1) * frac))
        steps.append(step)
    steps = sorted(set(steps))
    while len(steps) < R:
        for i in range(len(steps) - 1):
            if steps[i + 1] - steps[i] > 1:
                steps.insert(i + 1, steps[i] + 1)
                break
        else:
            steps.append(steps[-1] + 1)
        steps = sorted(set(steps))
    return steps[:R]


def get_particle_depth(i, num_clones, num_resampling_steps):
    C = num_clones
    R = num_resampling_steps
    for k in range(R):
        group_size = C ** (R - k)
        if i % group_size == 0:
            return k
    return R


def precompute_explore_map(
    prompt: str,
    num_particles: int = 6,
    num_clones: int = 3,
    num_resampling_steps: int = 4,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    rho: float = 0.0,
    height: int = 512,
    width: int = 512,
    seed: Optional[int] = 42,
    output_dir: str = "outputs/explore_map",
    model: str = "flux",
    augment: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda"

    # Load model
    if model == "flux":
        from flux_fmtt_dev import FluxDevInteractive
        sampler = FluxDevInteractive(prompt=prompt, device=device,
                                     keep_text_encoders=bool(augment))
    elif model == "krea2":
        from krea2_fmtt import Krea2Interactive
        sampler = Krea2Interactive(prompt=prompt, device=device, dtype=torch.bfloat16)
    else:
        from sd35_fmtt import SD35FMTT
        sampler = SD35FMTT(prompt=prompt, device=device, skip_t5=True)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # Monkey-patch DINO to use DINOv2 (public)
    import types
    def _load_dino_v2(self):
        from transformers import AutoImageProcessor, AutoModel
        dino_id = "facebook/dinov2-base"
        print(f"Loading DINOv2-B ({dino_id})...")
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_id)
        self.dino_model = AutoModel.from_pretrained(dino_id).to(self.device).eval()
    sampler._load_dino = types.MethodType(_load_dino_v2, sampler)
    sampler._load_dino()

    # Augmentation-seeded roots: each root = a different interpretation so the
    # precomputed tree is a COMPREHENSIVE set spanning distinct regions (FLUX-only).
    _aug_K = 0
    if augment and model == "flux":
        try:
            import prompt_augment
            _vars = [prompt] + prompt_augment.augment_prompt(
                prompt, n=max(0, num_particles - 1), device=device)
            _vars = _vars[:num_particles] or [prompt]
            sampler.set_prompts(_vars)
            _aug_K = len(_vars)
            print(f"[explore] augmented roots: {_aug_K} interpretations")
            for _attr in ("text_encoder", "text_encoder_2"):
                _te = getattr(sampler.pipeline, _attr, None)
                if _te is not None:
                    try: _te.to("cpu")
                    except Exception: pass
                    setattr(sampler.pipeline, _attr, None)
            import gc as _gc; _gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as _e:
            print(f"[explore] augment failed ({_e}); single-prompt tree")
            _aug_K = 0

    C = num_clones
    R = num_resampling_steps

    if model == "flux":
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
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
    elif model == "krea2":
        sampler._setup_geometry(height, width)   # sets _mu, position_ids, grid (needed by predict_velocity)
        latent_h = height // sampler.vae_scale_factor
        latent_w = width // sampler.vae_scale_factor
        time_steps = sampler.get_sigma_schedule(num_steps)
    else:
        latent_h = height // sampler.vae_scale_factor
        latent_w = width // sampler.vae_scale_factor
        time_steps = sampler.get_sigma_schedule(num_steps)

    # Resampling schedule
    resample_steps = compute_linear_diversity_schedule(num_steps, R)
    resample_at = set(resample_steps)
    segment_bounds = [0] + resample_steps + [num_steps]
    print(f"Resampling at steps: {resample_steps}")

    # Initialize particles
    anchors = torch.randn(num_particles, 16, latent_h, latent_w,
                          device=device, dtype=sampler.dtype)
    total_particles = num_particles * C
    _root_of = (torch.arange(num_particles) % _aug_K).repeat_interleave(C) if _aug_K else None
    X_all = anchors.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
    X_all = X_all.reshape(total_particles, 16, latent_h, latent_w)

    is_anchor = torch.zeros(total_particles, dtype=torch.bool)
    for i in range(num_particles):
        is_anchor[i * C] = True

    glass_params = None
    interval_count = 0

    # Track all particles across all levels for the manifest
    # Each entry: {path, depth, parent_idx, level_born}
    all_images_info = []

    pbar = tqdm(range(num_steps), total=num_steps, desc="Explore-Precompute")

    for k in pbar:
        t_cur = time_steps[k].item()
        t_next = time_steps[k + 1].item()
        dt = t_next - t_cur

        # GLASS init at segment start
        if glass_params is None:
            # BRIDGE-BOUNDARY FIX (root cause of the observed distribution
            # drift; evidence: outputs/glass_sigma_sweep/{cascade,exact_d4}_
            # results.json). The resample event at fine step b fires AFTER
            # stepping b — states are at time_steps[b+1] — but the bridge
            # used to target time_steps[b] (first bound > k). So segments
            # longer than 1 step OVERSHOT s past 1 (extra inner step at k=b),
            # and a segment starting ON a resample step (adjacent events) was
            # INTERRUPTED mid-bridge, expanding OFF-MARGINAL states. Fix:
            # target the ACTUAL expansion boundary time_steps[b+1] with
            # M_inner = b+1-k inner steps: s runs 0..(M-1)/M and the bridge
            # lands EXACTLY on-marginal (s=1) at expansion. Adjacent events
            # give one EXACT step (exactness beats step count); wider spacing
            # automatically gives >= 2 inner steps.
            seg_end_step = None
            for b in resample_steps:
                if b >= k:
                    seg_end_step = b + 1
                    break
            if seg_end_step is None:
                # FINAL-ROUND BRIDGE CAP — mirror of flux_fmtt_dev (validated:
                # outputs/glass_sigma_sweep/followup_results.json). Bridging a
                # final segment to sigma=0 reduces per-child novelty
                # (parent-child DINO cosdist 0.444 vs 0.518 for Euler
                # completion). After the last resample this tree is all
                # anchors, so this cap matters for R=0 / degenerate schedules
                # where clones would otherwise bridge the whole descent. Cap
                # at a short handoff (M <= 3, on-marginal at s=1), then plain
                # Euler to 0 (handoff switch in the step block below).
                seg_end_step = min(k + 3, num_steps)

            sigma_seg_start = time_steps[k].item()
            sigma_seg_end = time_steps[seg_end_step].item()
            M_inner = seg_end_step - k

            clone_indices = (~is_anchor).nonzero(as_tuple=True)[0]
            if len(clone_indices) > 0:
                parent_states = torch.zeros(len(clone_indices), 16, latent_h, latent_w,
                                            device=device, dtype=sampler.dtype)
                for ci, idx in enumerate(clone_indices):
                    group = idx.item() // C
                    parent_states[ci] = X_all[group * C]

                if model == "flux":
                    from flux_fmtt_dev import FluxDevInteractive
                    # Use the _glass_init from flux_fmtt_dev
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
                else:
                    clone_inner, gp = sampler._glass_init(
                        parent_states, sigma_seg_start, sigma_seg_end, rho)

                X_all[clone_indices] = clone_inner
                glass_params = gp
                glass_params["seg_start_step"] = k
                glass_params["M_inner"] = M_inner
                glass_params["clone_indices"] = clone_indices
            else:
                glass_params = {"seg_start_step": k, "M_inner": M_inner,
                                "clone_indices": torch.tensor([], dtype=torch.long)}

        # Step all particles
        m = k - glass_params["seg_start_step"]
        M = glass_params["M_inner"]
        if m >= M and len(glass_params["clone_indices"]) > 0:
            # FINAL-ROUND HANDOFF: capped final-segment bridge landed
            # on-marginal (s=1); clones complete by plain Euler to sigma=0.
            # Only reachable in the final segment.
            glass_params["handoff_clone_indices"] = glass_params["clone_indices"]
            glass_params["clone_indices"] = torch.tensor([], dtype=torch.long)
            print(f"  Final-segment handoff at step {k}: "
                  f"{len(glass_params['handoff_clone_indices'])} clones "
                  f"-> plain Euler to sigma=0")
        s = m / M
        ds = 1.0 / M

        pbar.set_postfix(particles=total_particles, t=f"{t_cur:.3f}")

        with torch.no_grad():
            anchor_indices = is_anchor.nonzero(as_tuple=True)[0]
            handoff_clone_indices = glass_params.get("handoff_clone_indices")
            if handoff_clone_indices is not None and len(handoff_clone_indices) > 0:
                anchor_indices = torch.cat([anchor_indices, handoff_clone_indices])
            _BCH = 64  # chunk so a huge batch doesn't OOM the transformer
            for _s0 in range(0, len(anchor_indices), _BCH):
                _ix = anchor_indices[_s0:_s0 + _BCH]
                z_p = X_all[_ix]                              # BATCHED [chunk, ...]
                if model == "flux":
                    _pi = _root_of[_ix] if _root_of is not None else None
                    v = sampler.predict_velocity(z_p, t_cur, guidance_scale, prompt_idx=_pi)
                elif model == "krea2":
                    v = sampler.predict_velocity(z_p, t_cur, guidance_scale)
                else:
                    v = sampler.predict_velocity_cfg(z_p, t_cur, guidance_scale)
                X_all[_ix] = z_p + dt * v

        # Clones: GLASS inner ODE
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
                _BCH = 64
                x_parent_all = x_t                            # [N_clone, ...] ci-aligned parents
                x_clone_all = X_all[clone_indices]            # [N_clone, ...] clones
                S_input_all = w_xt * x_parent_all + w_xs * x_clone_all
                _vstar = []
                for _s0 in range(0, len(clone_indices), _BCH):
                    _S = S_input_all[_s0:_s0 + _BCH]
                    if model == "flux":
                        _pi = _root_of[clone_indices[_s0:_s0 + _BCH]] if _root_of is not None else None
                        _vstar.append(sampler.predict_velocity(_S, sigma_star, guidance_scale, prompt_idx=_pi))
                    elif model == "krea2":
                        _vstar.append(sampler.predict_velocity(_S, sigma_star, guidance_scale))
                    else:
                        _vstar.append(sampler.predict_velocity_cfg(_S, sigma_star, guidance_scale))
                v_star = torch.cat(_vstar, dim=0)
                denoiser = S_input_all - sigma_star * v_star
                velocity = w1 * x_clone_all + w2 * denoiser + w3 * x_parent_all
                X_all[clone_indices] = x_clone_all + ds * velocity

        # Auto-expand at resampling
        if k in resample_at:
            glass_params = None
            interval_count += 1

            is_last_resample = (interval_count == R)

            if is_last_resample:
                num_particles = total_particles
                is_anchor = torch.ones(total_particles, dtype=torch.bool)
            else:
                kept_latents = X_all.clone()
                num_particles = total_particles
                total_particles = num_particles * C
                X_all = kept_latents.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
                X_all = X_all.reshape(total_particles, 16, latent_h, latent_w)
                if _root_of is not None:
                    _root_of = _root_of.repeat_interleave(C)
                is_anchor = torch.zeros(total_particles, dtype=torch.bool)
                for i in range(num_particles):
                    is_anchor[i * C] = True

            print(f"  After resampling {interval_count}: {total_particles} particles")

    # ═══ Decode all particles ═══
    print(f"\nDecoding {total_particles} images...")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    pil_images = []
    _DCH = 32  # batch the VAE decode in chunks (was one image at a time)
    with torch.no_grad():
        for _s0 in tqdm(range(0, total_particles, _DCH), desc="  Decoding"):
            _e = min(_s0 + _DCH, total_particles)
            imgs_t = sampler.decode(X_all[_s0:_e])          # BATCHED [chunk, C, H, W]
            arr_all = (imgs_t.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)
            for j in range(arr_all.shape[0]):
                pil_img = Image.fromarray(arr_all[j])
                pil_img.save(os.path.join(images_dir, f"img_{_s0 + j:04d}.png"))
                pil_images.append(pil_img)

    # ═══ DINO features + UMAP ═══
    print("Extracting DINO features...")
    features = sampler._extract_dino_features(pil_images)

    print("Computing UMAP layout...")
    import umap
    n = len(pil_images)
    reducer = umap.UMAP(
        n_neighbors=min(15, n - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(features)

    # Normalize to [0.05, 0.95]
    coords -= coords.min(axis=0)
    span = coords.max(axis=0)
    span[span == 0] = 1
    coords = coords / span * 0.90 + 0.05

    # ═══ Build manifest ═══
    print("Building manifest...")
    manifest = {
        "prompt": prompt,
        "model": model,
        "num_particles": num_particles,
        "num_clones": C,
        "num_resampling_steps": R,
        "total_images": total_particles,
        "max_depth": R,
        "images": [],
    }

    for i in range(total_particles):
        depth = get_particle_depth(i, C, R)

        # Find parent index
        if depth == 0:
            parent = None
        else:
            group_size = C ** (R - depth + 1)
            parent = (i // group_size) * group_size

        manifest["images"].append({
            "id": i,
            "path": f"images/img_{i:04d}.png",
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "depth": depth,
            "parent": parent,
        })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest: {manifest_path}")
    print(f"  Images: {images_dir}/")
    print(f"  Total: {total_particles} images across {R+1} depth levels")
    print(f"  Depth distribution: " +
          ", ".join(f"d{d}={sum(1 for img in manifest['images'] if img['depth']==d)}"
                   for d in range(R+1)))

    sampler._unload_dino()
    return manifest_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-compute explore map")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default="flux", choices=["flux", "sd35"])
    parser.add_argument("--num_particles", type=int, default=6)
    parser.add_argument("--num_clones", type=int, default=3)
    parser.add_argument("--num_resampling_steps", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/explore_map")
    args = parser.parse_args()

    precompute_explore_map(**vars(args))
