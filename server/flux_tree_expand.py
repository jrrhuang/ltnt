"""
FLUX FlowMap Tree Expansion — auto-expand all particles at every resampling step.

Runs the same anchor-clone GLASS sampling as flux_fmtt_interactive_flowmap.py,
but instead of prompting the user, keeps ALL particles at every resampling step.
This causes exponential tree growth: N×C → N×C² → N×C³ → ...

Outputs cluster plots and grids at each interval and at the final step.

--efficient flag: scales down lookahead steps proportionally to remaining time,
since less denoising is needed closer to t=0.
"""

import os

# Set HuggingFace cache to scratch — must be set BEFORE importing diffusers/transformers
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
_VLM_DIR = os.path.join(SCRIPT_DIR, "fluxfm_guidance", "vlm")
if _VLM_DIR not in sys.path:
    sys.path.insert(0, _VLM_DIR)

from flux_fmtt_interactive_flowmap import FluxTTInteractive
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from cluster_images import plot_image_cluster


def get_tree_edges(total_particles: int, num_clones: int, num_resampling_steps: int):
    """
    Compute tree edges from the final particle indices.

    Each non-root particle has exactly ONE edge to its immediate parent.
    The parent is the anchor of the smallest group containing this particle
    at the level where it was cloned.

    For particle i with C clones and R resampling steps:
    - If i % C != 0: parent = (i // C) * C, level = R-1 (leaf, thinnest)
    - If i % C == 0 but i % C² != 0: parent = (i // C²) * C², level = R-2
    - If i % C^k == 0 but i % C^(k+1) != 0: parent = (i // C^(k+1)) * C^(k+1), level = R-k-1
    - If i % C^R == 0: root, no edge

    Returns list of (parent_idx, child_idx, level) where level 0 = root
    (thickest) and level R-1 = leaves (thinnest).
    """
    edges = []
    R = num_resampling_steps
    C = num_clones

    for i in range(total_particles):
        # Find the level at which this particle was cloned
        for k in range(R):
            group_size = C ** (k + 1)
            if i % group_size != 0:
                # Cloned at level R-1-k, parent is the anchor of this group
                parent = (i // group_size) * group_size
                level = R - 1 - k  # 0 = root (thickest), R-1 = leaf (thinnest)
                edges.append((parent, i, level))
                break
        # If i % C^R == 0, it's a root — no edge

    return edges


def get_particle_depth(i, num_clones, num_resampling_steps):
    """Return depth of particle i. 0=root anchor, R=leaf clone."""
    C = num_clones
    R = num_resampling_steps
    for k in range(R):
        group_size = C ** (R - k)
        if i % group_size == 0:
            return k  # survived as anchor through level k
    return R  # leaf clone


def compute_2d_coords(embeddings, layout="mds", n_neighbors=15, min_dist=0.1):
    """Compute 2D coordinates from embeddings using the specified layout method.

    Args:
        embeddings: [N, D] L2-normalized feature vectors.
        layout: One of 'mds', 'umap', 'tsne', 'pacmap', 'pca'.
        n_neighbors: Neighbor count for UMAP/t-SNE.
        min_dist: Min distance for UMAP.

    Returns:
        coords: [N, 2] array, method_name: str
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA

    n_samples = embeddings.shape[0]

    if layout == "mds":
        from sklearn.manifold import MDS
        cos_dist = np.maximum(1.0 - cosine_similarity(embeddings), 0.0)
        np.fill_diagonal(cos_dist, 0.0)
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                  normalized_stress="auto")
        return mds.fit_transform(cos_dist), "MDS"

    elif layout == "umap":
        import umap
        reducer = umap.UMAP(n_neighbors=min(n_neighbors, n_samples - 1),
                            min_dist=min_dist, metric="cosine", random_state=42)
        return reducer.fit_transform(embeddings), "UMAP"

    elif layout == "tsne":
        from sklearn.manifold import TSNE
        # PCA to 50 dims first (standard practice)
        if embeddings.shape[1] > 50:
            embeddings = PCA(n_components=50, random_state=42).fit_transform(embeddings)
        tsne = TSNE(n_components=2, metric="cosine", random_state=42,
                    perplexity=min(30, n_samples - 1))
        return tsne.fit_transform(embeddings), "t-SNE"

    elif layout == "pacmap":
        import pacmap
        pm = pacmap.PaCMAP(n_components=2, n_neighbors=min(n_neighbors, n_samples - 1),
                           random_state=42)
        return pm.fit_transform(embeddings), "PaCMAP"

    elif layout == "pca":
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(embeddings), "PCA"

    else:
        raise ValueError(f"Unknown layout: {layout}. Use mds/umap/tsne/pacmap/pca.")


def plot_tree_cluster(embeddings, images, edges, num_resampling_steps,
                      title, save_path, n_neighbors=15, min_dist=0.1,
                      thumb_size=40, num_clones=2, layout="mds"):
    """
    Plot cluster map with tree edges connecting anchors to clones.

    Supports multiple layout methods: mds, umap, tsne, pacmap, pca.
    Lines are thicker and more opaque for edges closer to the root.
    Edges colored by root ancestor.
    Thumbnail size decreases with depth: root=largest, leaf=smallest.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_samples = embeddings.shape[0]
    R = num_resampling_steps
    C = num_clones

    # Compute 2D layout
    coords, method_name = compute_2d_coords(embeddings, layout=layout,
                                            n_neighbors=n_neighbors, min_dist=min_dist)

    # Spring-force refinement: pull parent-child pairs closer
    init_range = max(coords[:, 0].max() - coords[:, 0].min(),
                     coords[:, 1].max() - coords[:, 1].min(), 1e-6)

    for iteration in range(300):
        forces = np.zeros_like(coords)

        # Attraction: parent-child pairs
        for parent_idx, child_idx, level in edges:
            depth_frac = level / max(R - 1, 1)
            target_dist = init_range * (0.25 - 0.20 * depth_frac)
            dx = coords[child_idx] - coords[parent_idx]
            d = max(np.linalg.norm(dx), 1e-8)
            if d > target_dist:
                strength = 0.02 * (d - target_dist) / d
                forces[parent_idx] += strength * dx
                forces[child_idx] -= strength * dx

        # Repulsion: prevent overlap
        min_sep = init_range * 0.12
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dx = coords[j] - coords[i]
                d = max(np.linalg.norm(dx), 1e-8)
                if d < min_sep:
                    strength = 0.3 * (min_sep - d) / d
                    forces[i] -= strength * dx
                    forces[j] += strength * dx

        coords += forces
        coords -= coords.mean(axis=0)

    fig_size = 22 + max(0, (n_samples - 32)) * 0.2
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Root ancestor colors
    root_stride = C ** R
    n_roots = max(1, (n_samples + root_stride - 1) // root_stride)
    root_colors = plt.cm.Set1(np.linspace(0, 1, max(n_roots, 2)))

    # Draw tree edges — only closest 2 levels
    min_draw_level = max(0, R - 2)
    for parent_idx, child_idx, level in edges:
        if level < min_draw_level:
            continue
        depth_frac = (level - min_draw_level) / max(R - 1 - min_draw_level, 1)
        linewidth = 3.0 - 1.5 * depth_frac
        alpha = 0.75 - 0.35 * depth_frac

        root = parent_idx // root_stride
        color = root_colors[root % len(root_colors)]

        x = [coords[parent_idx, 0], coords[child_idx, 0]]
        y = [coords[parent_idx, 1], coords[child_idx, 1]]
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha,
                zorder=1, solid_capstyle="round")

    # Compute depth for each particle (0=root, R=leaf)
    depths = [get_particle_depth(i, C, R) for i in range(n_samples)]

    # Draw thumbnails — size decreases with depth
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    hw_base = max(min(x_range, y_range) * 0.065, 1e-6)

    for i, (x, y) in enumerate(coords):
        depth = depths[i]
        scale = 1.0 - 0.7 * (depth / max(R, 1))
        hw = hw_base * scale
        ts = max(int(thumb_size * scale), 16)
        thumb = images[i].resize((ts, ts), Image.LANCZOS)
        ax.imshow(np.array(thumb),
                  extent=(x - hw, x + hw, y - hw, y + hw),
                  zorder=2 + depth)

    pad = hw_base * 4
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(f"{title} ({method_name} + tree, {n_roots} roots)", fontsize=18)

    fig.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved tree cluster ({method_name}): {save_path}")


class FluxTreeExpand(FluxTTInteractive):
    """Subclass that auto-selects all particles (full tree expansion)."""

    def sample_tree(self,
                    num_steps: int = 8,
                    num_particles: int = 4,
                    num_clones: int = 3,
                    num_resampling_steps: int = 3,
                    guidance_scale: float = 3.5,
                    rho: float = 0.4,
                    lookahead_steps: int = 4,
                    efficient: bool = False,
                    sigma0_scale: float = 1.0,
                    glass_substeps: int = 1,
                    resample_schedule: List[int] = None,
                    img_shape: Optional[Tuple[int, int]] = None,
                    seed: Optional[int] = None,
                    output_dir: str = "tree_expand") -> List[Image.Image]:
        """
        Full tree expansion with GLASS Flows.

        Same as sample_interactive but auto-keeps ALL particles at every
        resampling step, causing exponential growth.

        Particle counts (N=4, C=3, R=3):
          Start:      4 × 3 = 12
          Resample 1: 12 × 3 = 36
          Resample 2: 36 × 3 = 108
          Resample 3 (last, no clone): 108
          Final: 108 images

        Args:
            efficient: If True, scale lookahead_steps down proportionally to
                remaining time t. E.g. at t=0.25 with base lookahead=4, use 1.
            sigma0_scale: Multiplier for bar_sigma_0 (default 1.0). Values > 1
                inject more initial noise into GLASS inner ODE, boosting clone
                diversity at the cost of exact posterior sampling. E.g. 3.0
                triples the init noise.
            glass_substeps: Number of GLASS inner sub-steps per outer ODE step
                (default 1). Increases effective M without changing the anchor
                schedule. Use with sigma0_scale > 1 to give the inner ODE enough
                steps to converge from the extra noise.
        """
        imgH, imgW = img_shape if img_shape is not None else (512, 512)
        device = self.device
        os.makedirs(output_dir, exist_ok=True)

        prompt_embeds = self.cached_prompt_embeds
        pooled_prompt_embeds = self.cached_pooled_prompt_embeds

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self._load_dino()

        latent_h = 2 * (int(imgH) // (self.vae_scale_factor * 2))
        latent_w = 2 * (int(imgW) // (self.vae_scale_factor * 2))

        # ===== Initialize N anchor particles =====
        anchors = torch.randn(num_particles, 16, latent_h, latent_w,
                              device=device, dtype=self.dtype)
        total_particles = num_particles * num_clones
        C = num_clones

        # Time schedule
        image_seq_len = (latent_h // 2) * (latent_w // 2)
        mu = calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.get("base_image_seq_len", 256),
            self.pipeline.scheduler.config.get("max_image_seq_len", 4096),
            self.pipeline.scheduler.config.get("base_shift", 0.5),
            self.pipeline.scheduler.config.get("max_shift", 1.15),
        )
        sigmas_arr = np.linspace(1.0, 1 / num_steps, num_steps)
        timesteps, _ = retrieve_timesteps(
            self.pipeline.scheduler, num_steps, device, sigmas=sigmas_arr, mu=mu)
        time_steps = torch.cat([timesteps / 1000.0, torch.zeros(1, device=device)])

        # Resampling points
        if resample_schedule is not None:
            resample_steps = sorted(resample_schedule)
            num_resampling_steps = len(resample_steps)
        elif num_resampling_steps > 0:
            resample_steps = sorted(
                int(round((i + 1) * num_steps / (num_resampling_steps + 1)))
                for i in range(num_resampling_steps)
            )
        else:
            resample_steps = []
        resample_at = set(resample_steps)
        segment_bounds = [0] + resample_steps + [num_steps]

        # Build initial particle tensor
        X_all = anchors.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
        X_all = X_all.reshape(total_particles, 16, latent_h, latent_w)

        is_anchor = torch.zeros(total_particles, dtype=torch.bool)
        for i in range(num_particles):
            is_anchor[i * C] = True

        glass_params = None
        interval_count = 0

        # Predict final particle count
        final_count = num_particles * num_clones
        for r in range(num_resampling_steps):
            if r < num_resampling_steps - 1:
                final_count *= num_clones  # expand
            # last resample: no cloning, count stays
        print(f"\nTree expansion: {num_particles}×{num_clones} initial, "
              f"{num_resampling_steps} resampling steps → {final_count} final particles")

        pbar = tqdm(range(num_steps), total=num_steps, desc="FLUX-TreeExpand")

        for k in pbar:
            t_cur = time_steps[k].item()
            t_next = time_steps[k + 1].item()
            dt = t_next - t_cur

            # ===== At segment start: GLASS init for clones =====
            if glass_params is None:
                seg_end_step = None
                for b in segment_bounds:
                    if b > k:
                        seg_end_step = b
                        break
                if seg_end_step is None:
                    seg_end_step = num_steps

                sigma_seg_start = time_steps[k].item()
                sigma_seg_end = time_steps[seg_end_step].item()
                M_inner = seg_end_step - k

                print(f"\n  Segment: steps {k}-{seg_end_step}, "
                      f"t={sigma_seg_start:.4f}→{sigma_seg_end:.4f}, "
                      f"M={M_inner}, particles={total_particles}")

                clone_indices = (~is_anchor).nonzero(as_tuple=True)[0]
                if len(clone_indices) > 0:
                    parent_states = torch.zeros(len(clone_indices), 16, latent_h, latent_w,
                                                device=device, dtype=self.dtype)
                    for ci, idx in enumerate(clone_indices):
                        group = idx.item() // C
                        parent_states[ci] = X_all[group * C]

                    clone_inner, gp = self._glass_init_flux(
                        parent_states, sigma_seg_start, sigma_seg_end, rho)

                    # Override bar_sigma_0 for diversity boost
                    if sigma0_scale != 1.0:
                        old_s0 = gp["bar_sigma_0"]
                        new_s0 = old_s0 * sigma0_scale
                        # Re-init with scaled noise
                        eps = torch.randn_like(parent_states)
                        clone_inner = gp["bar_gamma"] * parent_states + new_s0 * eps
                        gp["bar_sigma_0"] = new_s0
                        print(f"    sigma0_scale={sigma0_scale}: "
                              f"bar_sigma_0 {old_s0:.2f} → {new_s0:.2f}")

                    X_all[clone_indices] = clone_inner

                    glass_params = gp
                    glass_params["seg_start_step"] = k
                    glass_params["M_inner"] = M_inner
                    glass_params["clone_indices"] = clone_indices
                else:
                    glass_params = {"seg_start_step": k, "M_inner": M_inner,
                                    "clone_indices": torch.tensor([], dtype=torch.long)}

            # ===== Step all particles =====
            m = k - glass_params["seg_start_step"]
            M = glass_params["M_inner"]
            M_eff = M * glass_substeps  # effective inner steps
            ds_sub = 1.0 / M_eff

            pbar.set_postfix(particles=total_particles, t=f"{t_cur:.3f}")

            with torch.no_grad():
                # Anchors: flow map ODE step (one step per outer step, unchanged)
                anchor_indices = is_anchor.nonzero(as_tuple=True)[0]
                for ai in anchor_indices:
                    z_p = X_all[ai:ai+1]
                    v = self.predict_vector(z_p, t_cur,
                                            prompt_embeds=prompt_embeds[:1],
                                            pooled_prompt_embeds=pooled_prompt_embeds[:1],
                                            guidance_scale=guidance_scale,
                                            t_next=t_next)
                    X_all[ai:ai+1] = z_p + dt * v

            # Clones: GLASS inner ODE sub-steps
            clone_indices = glass_params["clone_indices"]
            if len(clone_indices) > 0:
                gp = glass_params
                x_t = gp["x_t"]
                clip = 1e-8

                for sub in range(glass_substeps):
                    s = (m * glass_substeps + sub) / M_eff

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
                            v_star = self.predict_vector(
                                S_input, sigma_star,
                                prompt_embeds=prompt_embeds[:1],
                                pooled_prompt_embeds=pooled_prompt_embeds[:1],
                                guidance_scale=guidance_scale,
                                t_next=sigma_star)
                            denoiser = S_input - sigma_star * v_star
                            velocity = w1 * x_clone + w2 * denoiser + w3 * x_parent
                            X_all[idx:idx+1] = x_clone + ds_sub * velocity

            # ===== Auto-expand at resampling intervals =====
            if k in resample_at:
                glass_params = None
                interval_count += 1
                interval_dir = os.path.join(output_dir, f"interval_{interval_count}")
                os.makedirs(interval_dir, exist_ok=True)

                t_now = t_next

                # Efficient lookahead: scale with remaining time
                if efficient:
                    effective_lookahead = max(1, round(lookahead_steps * t_now))
                else:
                    effective_lookahead = lookahead_steps

                print(f"\n--- Interval {interval_count} (step {k}/{num_steps}, "
                      f"t={t_now:.4f}, particles={total_particles}, "
                      f"lookahead={effective_lookahead}) ---")

                # Labels
                labels = []
                for i in range(total_particles):
                    group = i // C
                    pos = i % C
                    if pos == 0:
                        labels.append(f"{i} (a g{group})")
                    else:
                        labels.append(f"{i} (c g{group})")

                # Flow map look-ahead + decode
                pil_images = []
                with torch.no_grad():
                    for p_idx in tqdm(range(total_particles),
                                     desc=f"  Lookahead (×{effective_lookahead})",
                                     leave=False):
                        z0 = self._flow_map_lookahead(
                            X_all[p_idx:p_idx+1], t_now,
                            prompt_embeds=prompt_embeds[:1],
                            pooled_prompt_embeds=pooled_prompt_embeds[:1],
                            guidance_scale=guidance_scale,
                            lookahead_steps=effective_lookahead,
                        )
                        img_t = self.decode(z0)
                        pil_img = self._tensor_to_pil(img_t)
                        pil_img.save(os.path.join(interval_dir, f"particle_{p_idx:03d}.png"))
                        pil_images.append(pil_img)

                # DINOv3-B cluster
                print("  Extracting DINOv3-B features...")
                features = self._extract_dino_features(pil_images)
                cluster_path = os.path.join(interval_dir, "cluster_dinov3.png")
                plot_image_cluster(
                    features, pil_images,
                    title=f"Interval {interval_count} (step {k}, t={t_now:.3f}, "
                          f"n={total_particles})",
                    save_path=cluster_path,
                    n_neighbors=min(15, total_particles - 1),
                    min_dist=0.1,
                    thumb_size=max(40, 80 - total_particles // 2),
                )
                self._save_grid(pil_images, os.path.join(interval_dir, "grid.png"), labels)

                # Auto-select ALL
                kept_indices = list(range(total_particles))
                kept_latents = X_all.clone()
                num_particles = total_particles

                is_last_resample = (interval_count == num_resampling_steps)

                if is_last_resample:
                    total_particles = num_particles
                    X_all = kept_latents
                    is_anchor = torch.ones(total_particles, dtype=torch.bool)
                    print(f"  Kept all {num_particles} (final segment, no cloning)")
                else:
                    total_particles = num_particles * C
                    X_all = kept_latents.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
                    X_all = X_all.reshape(total_particles, 16, latent_h, latent_w)
                    is_anchor = torch.zeros(total_particles, dtype=torch.bool)
                    for i in range(num_particles):
                        is_anchor[i * C] = True
                    print(f"  Kept all {num_particles} → {num_particles} × {C} = "
                          f"{total_particles} total")

        # ===== Final: decode all particles =====
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        print(f"\n--- Final (step {num_steps}/{num_steps}, {total_particles} particles) ---")

        pil_images = []
        with torch.no_grad():
            for p_idx in tqdm(range(total_particles), desc="  Decoding final"):
                img_t = self.decode(X_all[p_idx:p_idx+1])
                pil_img = self._tensor_to_pil(img_t)
                pil_img.save(os.path.join(final_dir, f"particle_{p_idx:03d}.png"))
                pil_images.append(pil_img)

        # Final cluster (plain)
        print("  Extracting DINOv3-B features for final cluster...")
        features = self._extract_dino_features(pil_images)
        thumb = max(20, 80 - total_particles // 2)
        cluster_path = os.path.join(final_dir, "cluster_dinov3.png")
        plot_image_cluster(
            features, pil_images,
            title=f"Final — {total_particles} particles",
            save_path=cluster_path,
            n_neighbors=min(15, total_particles - 1),
            min_dist=0.1,
            thumb_size=thumb,
        )

        # Final cluster with tree edges — generate for ALL layout methods
        edges = get_tree_edges(total_particles, C, num_resampling_steps)
        layout_methods = ["mds", "umap", "tsne", "pacmap", "pca"]
        for lm in layout_methods:
            tree_path = os.path.join(final_dir, f"tree_{lm}.png")
            try:
                plot_tree_cluster(
                    features, pil_images, edges, num_resampling_steps,
                    title=f"Final — {total_particles} particles",
                    save_path=tree_path,
                    thumb_size=thumb,
                    num_clones=C,
                    layout=lm,
                )
            except Exception as e:
                print(f"  Warning: {lm} layout failed: {e}")

        # Final grid
        labels = [str(i) for i in range(total_particles)]
        self._save_grid(pil_images, os.path.join(final_dir, "grid.png"), labels)

        print(f"\n  Final images: {final_dir}/")
        print(f"  Final grid:    {final_dir}/grid.png")
        for lm in layout_methods:
            print(f"  Tree ({lm}):   {final_dir}/tree_{lm}.png")
        for i in range(min(interval_count, num_resampling_steps)):
            d = os.path.join(output_dir, f"interval_{i+1}")
            print(f"  Interval {i+1}: {d}/")

        self._unload_dino()
        return pil_images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FLUX FlowMap Tree Expansion")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=8)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_clones", type=int, default=3)
    parser.add_argument("--num_resampling_steps", type=int, default=3)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--lookahead_steps", type=int, default=4)
    parser.add_argument("--efficient", action="store_true",
                        help="Scale down lookahead steps proportionally to remaining time")
    parser.add_argument("--sigma0_scale", type=float, default=1.0,
                        help="Multiplier for GLASS init noise (default 1.0). "
                             "Higher = more diversity, breaks exact posterior.")
    parser.add_argument("--glass_substeps", type=int, default=1,
                        help="GLASS inner sub-steps per outer step (default 1). "
                             "Increase with sigma0_scale to avoid oversmoothing.")
    parser.add_argument("--resample_schedule", type=int, nargs="+", default=None,
                        help="Custom resampling step indices (e.g. 3 7 17). "
                             "Overrides --num_resampling_steps.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/tree_expand")
    parser.add_argument("--lora_path", type=str,
                        default=os.path.join(SCRIPT_DIR, "fluxfm_guidance/checkpoints/flux-flowmap-lora-512"))
    args = parser.parse_args()

    sampler = FluxTreeExpand(
        prompt=args.prompt,
        device="cuda",
        vlm_device=None,
        lora_path=args.lora_path,
    )

    schedule_str = f"schedule={args.resample_schedule}" if args.resample_schedule else f"R={args.num_resampling_steps}"
    print(f"\nTree expansion with GLASS Flows (ρ={args.rho}, {schedule_str}, "
          f"σ0_scale={args.sigma0_scale}, substeps={args.glass_substeps}, efficient={args.efficient})")
    pil_images = sampler.sample_tree(
        num_steps=args.num_steps,
        num_particles=args.num_particles,
        num_clones=args.num_clones,
        num_resampling_steps=args.num_resampling_steps,
        guidance_scale=args.guidance_scale,
        rho=args.rho,
        lookahead_steps=args.lookahead_steps,
        efficient=args.efficient,
        sigma0_scale=args.sigma0_scale,
        glass_substeps=args.glass_substeps,
        resample_schedule=args.resample_schedule,
        img_shape=(args.height, args.width),
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"\nDone. {len(pil_images)} final images generated.")
