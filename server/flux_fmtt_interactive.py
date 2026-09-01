"""
FLUX FlowMap Interactive FMTT Sampler

Adapts the FMTT algorithm from "Test-time scaling of diffusions with flow maps"
(Sabour et al., 2025, arXiv:2511.22688) for interactive particle pruning.

Instead of reward-guided selection, the user manually selects which particles
to keep at each resampling interval, guided by flow map look-ahead previews
and DINOv3-B feature cluster visualization.

Uses the original FLUX FlowMap with dual timestep conditioning (NOT multi-step
Euler). Inherits from FluxFlowMapVLMSampler for model loading & predict_vector.

Reference: https://arxiv.org/abs/2511.22688
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

# Script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add script dir to path for imports
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Add fluxfm_guidance/vlm to path for parent class
_VLM_DIR = os.path.join(SCRIPT_DIR, "fluxfm_guidance", "vlm")
if _VLM_DIR not in sys.path:
    sys.path.insert(0, _VLM_DIR)

from fluxfm_vlm_sampler import FluxFlowMapVLMSampler
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from cluster_images import plot_image_cluster


def compute_linear_diversity_schedule(num_steps: int, num_resampling_steps: int) -> list:
    """Compute resampling step indices for approximately linear diversity decrease."""
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


class FluxTTInteractive(FluxFlowMapVLMSampler):
    """
    FLUX FlowMap sampler with interactive FMTT for test-time scaling.

    Same ODE/SDE update as the original FMTT (Algorithm 1), but replaces
    reward-based resampling with manual user selection at each interval.

    At each resampling interval:
    1. Flow map look-ahead previews are computed for all particles
    2. DINOv3-B features are extracted and a UMAP cluster plot is saved
    3. User selects which particles to keep via terminal input
    4. Kept particles are cloned (round-robin) to fill back to N*C total

    Time convention: FLUX uses t=1 (noise) → t=0 (clean).
    """

    # =========================================================================
    # Static helpers (same as original FluxTT)
    # =========================================================================

    @staticmethod
    def _get_epsilon_t(epsilon, t: float) -> float:
        """Get epsilon value at time t. Supports constant or paper schedule.
        Paper: epsilon_t = t (FLUX convention, = 1-τ in paper convention)."""
        if isinstance(epsilon, str) and epsilon == "schedule":
            return t
        return float(epsilon)

    # =========================================================================
    # DINOv3-B feature extraction (persistent model)
    # =========================================================================

    def _load_dino(self):
        """Load DINOv3-B model and processor, keep on GPU."""
        from transformers import AutoImageProcessor, AutoModel
        dino_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        print(f"Loading DINOv3-B ({dino_id})...")
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_id)
        self.dino_model = AutoModel.from_pretrained(dino_id).to(self.device).eval()
        print("DINOv3-B loaded and kept on GPU")

    def _unload_dino(self):
        """Free DINOv3-B from GPU."""
        if hasattr(self, "dino_model") and self.dino_model is not None:
            del self.dino_model
            del self.dino_processor
            self.dino_model = None
            self.dino_processor = None
            gc.collect()
            torch.cuda.empty_cache()

    def _extract_dino_features(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Extract L2-normalized DINOv3-B features from a list of PIL images."""
        embeddings = []
        with torch.no_grad():
            for img in pil_images:
                inputs = self.dino_processor(images=img, return_tensors="pt").to(self.device)
                outputs = self.dino_model(**inputs)
                emb = outputs.pooler_output  # [1, 768]
                embeddings.append(emb.cpu().numpy())
        features = np.concatenate(embeddings, axis=0)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norms + 1e-8)

    # =========================================================================
    # Interactive helpers
    # =========================================================================

    @staticmethod
    def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
        """Convert [1, 3, H, W] tensor in [0,1] to PIL Image."""
        arr = (img_tensor[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _save_grid(pil_images: List[Image.Image], save_path: str, labels: List[str] = None):
        """Save a labeled grid of images."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(pil_images)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        for i, ax in enumerate(axes):
            if i < n:
                ax.imshow(np.array(pil_images[i]))
                ax.set_title(labels[i] if labels else str(i), fontsize=14)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _prompt_user_selection(self, total: int, interval_idx: int,
                               interval_dir: str) -> List[int]:
        """Print paths and prompt user for particle indices to keep."""
        print("\n" + "=" * 60)
        print(f"  INTERACTIVE PRUNING — Interval {interval_idx}")
        print("=" * 60)
        print(f"  Particle previews:  {interval_dir}/particle_*.png")
        print(f"  Labeled grid:       {interval_dir}/grid.png")
        print(f"  DINOv3 cluster:     {interval_dir}/cluster_dinov3.png")
        print(f"  Total particles:    {total}")
        print("=" * 60)

        while True:
            raw = input(f"  Enter indices to KEEP (0-{total-1}), comma-separated: ").strip()
            if not raw:
                print("  No input. Please enter at least one index.")
                continue
            try:
                indices = [int(x.strip()) for x in raw.split(",")]
                if not indices:
                    raise ValueError("empty")
                if any(i < 0 or i >= total for i in indices):
                    print(f"  Invalid index. Must be in 0-{total-1}.")
                    continue
                indices = sorted(set(indices))
                print(f"  Keeping particles: {indices}")
                return indices
            except ValueError:
                print("  Invalid input. Enter comma-separated integers (e.g. 0,2,5).")

    # =========================================================================
    # Flow map look-ahead (deterministic, from original)
    # =========================================================================

    def _flow_map_lookahead(self, z: torch.Tensor, t_cur: float,
                            prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor,
                            guidance_scale: float, lookahead_steps: int = 1) -> torch.Tensor:
        """
        Predict clean image z0 from z at time t_cur using flow map.

        Single step: predict velocity with dual timestep (t_cur, 0), then z0 = z - t*u.
        Multi-step: chain flow map calls from t_cur → 0 in lookahead_steps steps.

        Deterministic — no noise, no score function.
        """
        if lookahead_steps == 1:
            u = self.predict_vector(z, t_cur, prompt_embeds=prompt_embeds,
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   guidance_scale=guidance_scale, t_next=0.0)
            return z - t_cur * u

        # Multi-step flow map: t_cur → t_mid → ... → 0
        z_cur = z
        t = t_cur
        dt_look = t_cur / lookahead_steps
        for s in range(lookahead_steps):
            t_next_look = t - dt_look
            if t_next_look < 1e-6:
                t_next_look = 0.0
            u = self.predict_vector(z_cur, t, prompt_embeds=prompt_embeds,
                                   pooled_prompt_embeds=pooled_prompt_embeds,
                                   guidance_scale=guidance_scale, t_next=t_next_look)
            z_cur = z_cur + (t_next_look - t) * u
            t = t_next_look
        return z_cur

    # =========================================================================
    # Interactive FMTT sampling
    # =========================================================================

    # =========================================================================
    # GLASS Flows for FLUX (same math as SD3.5, different velocity prediction)
    # =========================================================================

    def _glass_init_flux(self, X_bar: torch.Tensor, sigma_start: float,
                         sigma_end: float, rho: float) -> Tuple[torch.Tensor, dict]:
        """GLASS stochastic init. sigma = t_flux (both 1=noise, 0=clean)."""
        clip = 1e-8
        alpha_start = 1.0 - sigma_start
        alpha_end = 1.0 - sigma_end

        bar_gamma = rho * sigma_end / max(sigma_start, clip)
        bar_alpha = alpha_end - bar_gamma * alpha_start
        bar_sigma = math.sqrt(max(sigma_end ** 2 * (1.0 - rho ** 2), 0.0))
        bar_sigma_0 = 1.0

        eps = torch.randn_like(X_bar)
        X_inner = bar_gamma * X_bar + bar_sigma_0 * eps

        glass_params = dict(
            x_t=X_bar.clone(),
            alpha_start=alpha_start,
            sigma_start=sigma_start,
            bar_gamma=bar_gamma,
            bar_alpha=bar_alpha,
            bar_sigma=bar_sigma,
            bar_sigma_0=bar_sigma_0,
        )
        return X_inner, glass_params

    def _glass_velocity_step_flux(self, X_inner: torch.Tensor, s: float,
                                   ds: float, glass_params: dict,
                                   prompt_embeds: torch.Tensor,
                                   pooled_prompt_embeds: torch.Tensor,
                                   guidance_scale: float):
        """One GLASS inner ODE step for FLUX."""
        clip = 1e-8
        gp = glass_params
        x_t = gp["x_t"]

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
            P = X_inner.shape[0]
            for p in range(P):
                S_input = w_xt * x_t[p:p+1] + w_xs * X_inner[p:p+1]
                # FLUX denoiser: D(x, sigma) = x - sigma * v(x, sigma, t_next=sigma)
                v_star = self.predict_vector(S_input, sigma_star,
                                             prompt_embeds=prompt_embeds[:1],
                                             pooled_prompt_embeds=pooled_prompt_embeds[:1],
                                             guidance_scale=guidance_scale,
                                             t_next=sigma_star)
                denoiser = S_input - sigma_star * v_star
                velocity = w1 * X_inner[p:p+1] + w2 * denoiser + w3 * x_t[p:p+1]
                X_inner[p:p+1] = X_inner[p:p+1] + ds * velocity

        return X_inner

    def sample_interactive(self,
                           num_steps: int = 28,
                           num_particles: int = 4,
                           num_clones: int = 2,
                           num_resampling_steps: int = 4,
                           guidance_scale: float = 3.5,
                           rho: float = 0.4,
                           lookahead_steps: int = 1,
                           schedule_mode: Optional[str] = "linear",
                           img_shape: Optional[Tuple[int, int]] = None,
                           seed: Optional[int] = None,
                           output_dir: str = "interactive_fmtt") -> torch.Tensor:
        """
        Interactive FMTT with GLASS Flows for FLUX.

        Anchor-clone design (same as SD3.5 version):
        - N anchor particles run standard deterministic ODE
        - Each anchor has (C-1) clones that diverge via GLASS stochastic init
        - At resampling: user selects, selected become anchors, clones get GLASS
        - N shrinks to match user selection count

        FLUX-specific: uses predict_vector with dual timestep, flow map lookahead.
        Time convention: t=1 (noise) → t=0 (clean), same role as SD3.5's sigma.

        Args:
            num_steps: Total ODE steps K.
            num_particles: Initial number of anchor particles N.
            num_clones: Clones per anchor C (total = N*C).
            num_resampling_steps: Number of pruning intervals R.
            guidance_scale: FLUX guidance scale.
            rho: GLASS correlation (0=max diversity, 1=no diversity). Default 0.4.
            lookahead_steps: Flow map steps for preview look-ahead (default 1).
            img_shape: Output image (H, W). Default (256, 256).
            seed: Random seed.
            output_dir: Directory to save interval previews and final outputs.

        Returns:
            Final selected image [1, 3, H, W] in [0, 1].
        """
        imgH, imgW = img_shape if img_shape is not None else (256, 256)
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
        if schedule_mode == "linear" and num_resampling_steps > 0:
            resample_steps = compute_linear_diversity_schedule(num_steps, num_resampling_steps)
            print(f"  Linear diversity schedule: steps {resample_steps}")
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

        pbar = tqdm(range(num_steps), total=num_steps, desc="FLUX-Interactive-GLASS")

        for k in pbar:
            t_cur = time_steps[k].item()
            t_next = time_steps[k + 1].item()
            dt = t_next - t_cur  # negative

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
                      f"t={sigma_seg_start:.4f}→{sigma_seg_end:.4f}, M={M_inner}")

                clone_indices = (~is_anchor).nonzero(as_tuple=True)[0]
                if len(clone_indices) > 0:
                    parent_states = torch.zeros(len(clone_indices), 16, latent_h, latent_w,
                                                device=device, dtype=self.dtype)
                    for ci, idx in enumerate(clone_indices):
                        group = idx.item() // C
                        parent_states[ci] = X_all[group * C]

                    clone_inner, gp = self._glass_init_flux(
                        parent_states, sigma_seg_start, sigma_seg_end, rho)
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
            s = m / M
            ds = 1.0 / M

            with torch.no_grad():
                # Anchors: standard deterministic Euler ODE
                anchor_indices = is_anchor.nonzero(as_tuple=True)[0]
                for ai in anchor_indices:
                    z_p = X_all[ai:ai+1]
                    v = self.predict_vector(z_p, t_cur,
                                            prompt_embeds=prompt_embeds[:1],
                                            pooled_prompt_embeds=pooled_prompt_embeds[:1],
                                            guidance_scale=guidance_scale,
                                            t_next=t_cur)
                    X_all[ai:ai+1] = z_p + dt * v

            # Clones: GLASS inner ODE step
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
                        v_star = self.predict_vector(
                            S_input, sigma_star,
                            prompt_embeds=prompt_embeds[:1],
                            pooled_prompt_embeds=pooled_prompt_embeds[:1],
                            guidance_scale=guidance_scale,
                            t_next=sigma_star)
                        denoiser = S_input - sigma_star * v_star
                        velocity = w1 * x_clone + w2 * denoiser + w3 * x_parent
                        X_all[idx:idx+1] = x_clone + ds * velocity

            # ===== Interactive pruning at resampling intervals =====
            if k in resample_at:
                glass_params = None

                interval_count += 1
                interval_dir = os.path.join(output_dir, f"interval_{interval_count}")
                os.makedirs(interval_dir, exist_ok=True)

                t_now = t_next
                print(f"\n--- Interval {interval_count} (step {k}/{num_steps}, "
                      f"t={t_now:.4f}) ---")
                print(f"Computing flow map look-ahead previews for {total_particles} particles...")

                # Labels
                labels = []
                for i in range(total_particles):
                    group = i // C
                    pos = i % C
                    if pos == 0:
                        labels.append(f"{i} (anchor g{group})")
                    else:
                        labels.append(f"{i} (clone g{group})")

                # a) Flow map look-ahead + decode
                pil_images = []
                with torch.no_grad():
                    for p_idx in range(total_particles):
                        z0 = self._flow_map_lookahead(
                            X_all[p_idx:p_idx+1], t_now,
                            prompt_embeds=prompt_embeds[:1],
                            pooled_prompt_embeds=pooled_prompt_embeds[:1],
                            guidance_scale=guidance_scale,
                            lookahead_steps=lookahead_steps,
                        )
                        img_t = self.decode(z0)
                        pil_img = self._tensor_to_pil(img_t)
                        pil_img.save(os.path.join(interval_dir, f"particle_{p_idx:03d}.png"))
                        pil_images.append(pil_img)

                # b) DINOv3-B cluster
                print("Extracting DINOv3-B features and generating cluster plot...")
                features = self._extract_dino_features(pil_images)
                cluster_path = os.path.join(interval_dir, "cluster_dinov3.png")
                plot_image_cluster(
                    features, pil_images,
                    title=f"DINOv3-B Cluster — Interval {interval_count} "
                          f"(step {k}, t={t_now:.3f})",
                    save_path=cluster_path,
                    n_neighbors=min(15, total_particles - 1),
                    min_dist=0.1,
                    thumb_size=80,
                )

                # c) Save labeled grid
                self._save_grid(pil_images, os.path.join(interval_dir, "grid.png"), labels)

                # d) Prompt user
                kept_indices = self._prompt_user_selection(
                    total_particles, interval_count, interval_dir
                )

                # e) Selected → new anchors, shrink N
                kept_latents = X_all[kept_indices].clone()
                num_particles = len(kept_indices)
                total_particles = num_particles * C

                X_all = kept_latents.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
                X_all = X_all.reshape(total_particles, 16, latent_h, latent_w)

                is_anchor = torch.zeros(total_particles, dtype=torch.bool)
                for i in range(num_particles):
                    is_anchor[i * C] = True

                print(f"  Selected {num_particles} → {num_particles} anchors × {C} = "
                      f"{total_particles} total (GLASS init at next segment)")

        # ===== Final: decode all, let user pick =====
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        print(f"\n--- Final output (step {num_steps}/{num_steps}) ---")

        labels = []
        for i in range(total_particles):
            group = i // C
            pos = i % C
            labels.append(f"{i} ({'A' if pos == 0 else 'C'}g{group})")

        pil_images = []
        with torch.no_grad():
            for p_idx in range(total_particles):
                img_t = self.decode(X_all[p_idx:p_idx+1])
                pil_img = self._tensor_to_pil(img_t)
                pil_img.save(os.path.join(final_dir, f"particle_{p_idx:03d}.png"))
                pil_images.append(pil_img)

        features = self._extract_dino_features(pil_images)
        plot_image_cluster(
            features, pil_images,
            title="DINOv3-B Cluster — Final",
            save_path=os.path.join(final_dir, "cluster_dinov3.png"),
            n_neighbors=min(15, total_particles - 1),
            min_dist=0.1, thumb_size=80,
        )
        self._save_grid(pil_images, os.path.join(final_dir, "grid.png"), labels)

        print(f"\n  Final images saved to: {final_dir}/")
        print(f"  Grid: {final_dir}/grid.png")
        print(f"  Cluster: {final_dir}/cluster_dinov3.png")

        self._unload_dino()

        # Return all final decoded images [N*C, 3, H, W]
        with torch.no_grad():
            return torch.cat([self.decode(X_all[i:i+1]).float()
                              for i in range(total_particles)], dim=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FLUX FlowMap Interactive FMTT")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default="output.png", help="Final output path")
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_clones", type=int, default=2)
    parser.add_argument("--num_resampling_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--rho", type=float, default=0.4,
                        help="GLASS correlation (0=max diversity, 1=no diversity). Default: 0.4")
    parser.add_argument("--lookahead_steps", type=int, default=1,
                        help="Flow map look-ahead steps (1=single call, 4=paper's 4-step)")
    parser.add_argument("--schedule_mode", type=str, default="linear", choices=["linear", "none"],
                        help="Front-loaded resampling schedule for linear diversity decrease")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="interactive_fmtt",
                        help="Output directory for interval previews")
    # Model paths
    parser.add_argument("--model_id", type=str, default=None,
                        help="FLUX model ID (default: black-forest-labs/FLUX.1-dev)")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to FlowMap LoRA weights")
    args = parser.parse_args()

    # Build kwargs for model init
    init_kwargs = dict(
        prompt=args.prompt,
        device="cuda",
        vlm_device=None,
    )
    if args.model_id:
        init_kwargs["model_id"] = args.model_id
    if args.lora_path:
        init_kwargs["lora_path"] = args.lora_path

    sampler = FluxTTInteractive(**init_kwargs)

    print(f"\nStarting interactive FMTT with GLASS Flows (ρ={args.rho}, schedule_mode={args.schedule_mode})...")
    best_image = sampler.sample_interactive(
        num_steps=args.num_steps,
        num_particles=args.num_particles,
        num_clones=args.num_clones,
        num_resampling_steps=args.num_resampling_steps,
        guidance_scale=args.guidance_scale,
        rho=args.rho,
        lookahead_steps=args.lookahead_steps,
        schedule_mode=args.schedule_mode if args.schedule_mode != "none" else None,
        img_shape=(args.height, args.width),
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Save final output
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    pil_final = FluxTTInteractive._tensor_to_pil(best_image)
    pil_final.save(args.output)
    print(f"\nFinal image saved to: {args.output}")
