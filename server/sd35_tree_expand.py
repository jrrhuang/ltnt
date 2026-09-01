"""
SD3.5 Tree Expansion — auto-expand all particles, hierarchical visualization.
Adapted from flux_tree_expand.py for SD3.5 (CFG, Euler ODE, no flow map).
"""

import os

SCRATCH_HF_HOME = "/scratch/jerryhua/.cache/huggingface"
if os.path.exists("/scratch/jerryhua"):
    os.makedirs(SCRATCH_HF_HOME, exist_ok=True)
    os.environ["HF_HOME"] = SCRATCH_HF_HOME
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(SCRATCH_HF_HOME, "hub")

import gc
import torch
import math
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List
from PIL import Image
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from sd35_fmtt import SD35FMTT, compute_linear_diversity_schedule
from cluster_images import plot_image_cluster
from flux_tree_expand import get_tree_edges, get_particle_depth, plot_tree_cluster


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SD3.5 Tree Expansion")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_clones", type=int, default=2)
    parser.add_argument("--num_resampling_steps", type=int, default=3)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--lookahead_steps", type=int, default=8)
    parser.add_argument("--linear_diversity", action="store_true")
    parser.add_argument("--efficient", action="store_true")
    parser.add_argument("--resample_schedule", type=int, nargs="+", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip_t5", action="store_true")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/tree_sd35")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    init_kwargs = dict(prompt=args.prompt, device="cuda", skip_t5=args.skip_t5)
    if args.model_id:
        init_kwargs["model_id"] = args.model_id
    sampler = SD35FMTT(**init_kwargs)

    num_steps = args.num_steps
    num_particles = args.num_particles
    C = args.num_clones
    num_resampling_steps = args.num_resampling_steps
    guidance_scale = args.guidance_scale
    rho = args.rho
    lookahead_steps = args.lookahead_steps
    imgH, imgW = args.height, args.width
    device = "cuda"

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    sampler._load_dino()

    latH = imgH // sampler.vae_scale_factor
    latW = imgW // sampler.vae_scale_factor

    # Initialize anchors
    anchors = torch.randn(num_particles, 16, latH, latW, device=device, dtype=sampler.dtype)
    total_particles = num_particles * C

    # Sigma schedule
    sigmas = sampler.get_sigma_schedule(num_steps)

    # Resampling schedule
    if args.resample_schedule is not None:
        resample_steps = sorted(args.resample_schedule)
        num_resampling_steps = len(resample_steps)
    elif args.linear_diversity:
        resample_steps = compute_linear_diversity_schedule(num_steps, num_resampling_steps)
        print(f"Linear diversity schedule: steps {resample_steps}")
    else:
        resample_steps = sorted(
            int(round((i + 1) * num_steps / (num_resampling_steps + 1)))
            for i in range(num_resampling_steps)
        )
    resample_at = set(resample_steps)
    segment_bounds = [0] + resample_steps + [num_steps]

    print(f"Resampling at steps: {resample_steps}")
    print(f"Sigmas at those steps: {[f'{sigmas[s].item():.4f}' for s in resample_steps]}")

    # Build particle tensor
    X_all = anchors.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
    X_all = X_all.reshape(total_particles, 16, latH, latW)

    is_anchor = torch.zeros(total_particles, dtype=torch.bool)
    for i in range(num_particles):
        is_anchor[i * C] = True

    glass_params = None
    interval_count = 0

    pbar = tqdm(range(num_steps), total=num_steps, desc="SD3.5-TreeExpand")

    for k in pbar:
        sigma_cur = sigmas[k].item()
        sigma_next = sigmas[k + 1].item()
        d_sigma = sigma_next - sigma_cur

        # GLASS init at segment start
        if glass_params is None:
            seg_end_step = None
            for b in segment_bounds:
                if b > k:
                    seg_end_step = b
                    break
            if seg_end_step is None:
                seg_end_step = num_steps

            sigma_seg_start = sigmas[k].item()
            sigma_seg_end = sigmas[seg_end_step].item()
            M_inner = seg_end_step - k

            print(f"\n  Segment: steps {k}-{seg_end_step}, "
                  f"σ={sigma_seg_start:.4f}→{sigma_seg_end:.4f}, "
                  f"M={M_inner}, particles={total_particles}")

            clone_indices = (~is_anchor).nonzero(as_tuple=True)[0]
            if len(clone_indices) > 0:
                parent_states = torch.zeros(len(clone_indices), 16, latH, latW,
                                            device=device, dtype=sampler.dtype)
                for ci, idx in enumerate(clone_indices):
                    group = idx.item() // C
                    parent_states[ci] = X_all[group * C]

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
        s = m / M
        ds = 1.0 / M

        pbar.set_postfix(particles=total_particles, sigma=f"{sigma_cur:.3f}")

        with torch.no_grad():
            # Anchors: standard Euler ODE
            anchor_indices = is_anchor.nonzero(as_tuple=True)[0]
            for ai in anchor_indices:
                z_p = X_all[ai:ai+1]
                v = sampler.predict_velocity_cfg(z_p, sigma_cur, guidance_scale)
                X_all[ai:ai+1] = z_p + d_sigma * v

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
                for ci, idx in enumerate(clone_indices):
                    x_parent = x_t[ci:ci+1]
                    x_clone = X_all[idx:idx+1]
                    S_input = w_xt * x_parent + w_xs * x_clone
                    v_star = sampler.predict_velocity_cfg(S_input, sigma_star, guidance_scale)
                    denoiser = S_input - sigma_star * v_star
                    velocity = w1 * x_clone + w2 * denoiser + w3 * x_parent
                    X_all[idx:idx+1] = x_clone + ds * velocity

        # Auto-expand at resampling
        if k in resample_at:
            glass_params = None
            interval_count += 1

            is_last_resample = (interval_count == num_resampling_steps)

            if is_last_resample:
                num_particles = total_particles
                is_anchor = torch.ones(total_particles, dtype=torch.bool)
                print(f"  Kept all {num_particles} (final segment, no cloning)")
            else:
                kept_latents = X_all.clone()
                num_particles = total_particles
                total_particles = num_particles * C
                X_all = kept_latents.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
                X_all = X_all.reshape(total_particles, 16, latH, latW)
                is_anchor = torch.zeros(total_particles, dtype=torch.bool)
                for i in range(num_particles):
                    is_anchor[i * C] = True
                print(f"  Kept all {num_particles} → {num_particles} × {C} = "
                      f"{total_particles} total")

    # Final: decode all
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    print(f"\n--- Final ({total_particles} particles) ---")

    pil_images = []
    with torch.no_grad():
        for p_idx in tqdm(range(total_particles), desc="  Decoding"):
            img_t = sampler.decode(X_all[p_idx:p_idx+1])
            pil_img = SD35FMTT._tensor_to_pil(img_t)
            pil_img.save(os.path.join(final_dir, f"particle_{p_idx:03d}.png"))
            pil_images.append(pil_img)

    # Cluster + tree
    print("  Extracting DINOv3-B features...")
    features = sampler._extract_dino_features(pil_images)
    thumb = max(20, 80 - total_particles // 2)

    plot_image_cluster(
        features, pil_images,
        title=f"SD3.5 — {total_particles} particles",
        save_path=os.path.join(final_dir, "cluster_dinov3.png"),
        n_neighbors=min(15, total_particles - 1),
        min_dist=0.1, thumb_size=thumb,
    )

    edges = get_tree_edges(total_particles, C, num_resampling_steps)
    plot_tree_cluster(
        features, pil_images, edges, num_resampling_steps,
        title=f"SD3.5 — {total_particles} particles",
        save_path=os.path.join(final_dir, "tree_dinov3.png"),
        thumb_size=thumb, num_clones=C,
    )

    labels = [str(i) for i in range(total_particles)]
    SD35FMTT._save_grid(pil_images, os.path.join(final_dir, "grid.png"), labels)

    sampler._unload_dino()

    print(f"\n  Grid: {final_dir}/grid.png")
    print(f"  Tree: {final_dir}/tree_dinov3.png")
    print(f"  Cluster: {final_dir}/cluster_dinov3.png")


if __name__ == "__main__":
    main()
