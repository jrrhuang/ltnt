"""
Stable Diffusion 3.5 Medium FMTT (Flow Map Trajectory Tilting) Sampler

Adapts the FMTT algorithm from "Test-time scaling of diffusions with flow maps"
(Sabour et al., 2025, arXiv:2511.22688) to SD3.5 Medium.

Key differences from the FLUX FlowMap version:
- No flow map LoRA: look-ahead uses multi-step Euler forward instead
- Standard CFG: two forward passes (cond + uncond) per velocity prediction
- Generic pluggable reward function (not VLM-specific)

Usage:
    from sd35_fmtt import SD35FMTT

    sampler = SD35FMTT(prompt="a cat sitting on a mat", device="cuda")

    def my_reward(images):
        # images: [B, 3, H, W] in [0, 1]
        return torch.ones(images.shape[0])  # placeholder

    best_image = sampler.sample_fmtt(reward_fn=my_reward, num_particles=4, num_clones=2)

Reference: https://arxiv.org/abs/2511.22688
"""

import os

# Set HuggingFace cache to scratch — must be set BEFORE importing diffusers
SCRATCH_HF_HOME = "/scratch/jerryhua/.cache/huggingface"
if os.path.exists("/scratch/jerryhua"):
    os.makedirs(SCRATCH_HF_HOME, exist_ok=True)
    os.environ["HF_HOME"] = SCRATCH_HF_HOME
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(SCRATCH_HF_HOME, "hub")

import gc
import math
import torch
import torch.utils.checkpoint as cp
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Callable, List
from PIL import Image
import warnings
warnings.filterwarnings("ignore", message=".*Flax.*deprecated.*")
from diffusers import StableDiffusion3Pipeline

# Import DINO clustering utilities from existing code
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from cluster_images import plot_image_cluster

# Import IncrementalLayout if available (from flux_fmtt_dev)
try:
    from flux_fmtt_dev import IncrementalLayout
except ImportError:
    IncrementalLayout = None


DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"


def compute_aggressive_diversity_schedule(num_steps: int, num_resampling_steps: int) -> list:
    """Even more front-loaded schedule for maximum diversity.

    Places all resampling in the first ~25% of steps with steeper power-law (exponent 2.0).
    """
    R = num_resampling_steps
    if R == 0:
        return []
    max_step = num_steps * 0.25
    steps = []
    for k in range(R):
        frac = ((k + 1) / (R + 1)) ** 2.0
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


def compute_linear_diversity_schedule(num_steps: int, num_resampling_steps: int) -> list:
    """Compute resampling step indices for approximately linear diversity decrease.

    Places all resampling points in the first ~40% of steps (high-diversity region)
    with power-law spacing.
    """
    R = num_resampling_steps
    if R == 0:
        return []
    # Place in first 40% of steps with power-law spacing (exponent 1.5)
    max_step = num_steps * 0.4
    steps = []
    for k in range(R):
        frac = ((k + 1) / (R + 1)) ** 1.5
        step = max(1, round(1 + (max_step - 1) * frac))
        steps.append(step)
    # Deduplicate and sort
    steps = sorted(set(steps))
    # If dedup removed steps, fill gaps
    while len(steps) < R:
        for i in range(len(steps) - 1):
            if steps[i + 1] - steps[i] > 1:
                steps.insert(i + 1, steps[i] + 1)
                break
        else:
            steps.append(steps[-1] + 1)
        steps = sorted(set(steps))
    return steps[:R]


class SD35FMTT:
    """
    SD3.5 Medium sampler with Flow Map Trajectory Tilting (FMTT) for test-time scaling.

    Implements Algorithm 1 from "Test-time scaling of diffusions with flow maps":
    - Particle-based sampling with N particles and C clones each
    - Multi-step Euler look-ahead for reward evaluation (replaces flow map)
    - Two modes:
        - Sampling: importance-weighted resampling using softmax(A)
        - Searching: top-N selection based on reward r_t(x)
    - Position update with velocity, reward gradient, and optional noise

    SD3.5 uses flow matching with sigma convention:
    - sigma=1 is pure noise, sigma=0 is clean data
    - x_t = sigma * noise + (1 - sigma) * x_0
    - Euler step: x_{t+1} = x_t + (sigma_next - sigma) * v
    """

    def __init__(self, prompt: str, model_id: str = DEFAULT_MODEL_ID,
                 device: str = "cuda", reward_device: Optional[str] = None,
                 dtype=torch.bfloat16, negative_prompt: str = "",
                 max_sequence_length: int = 512,
                 additional_prompts: list = None,
                 skip_t5: bool = False):
        """
        Args:
            prompt: Text prompt for generation.
            model_id: SD3.5 HuggingFace model ID.
            device: Device for transformer and VAE.
            reward_device: Device for reward model (default: same as device).
            dtype: Model dtype.
            negative_prompt: Negative prompt for CFG.
            max_sequence_length: Max token length for T5 encoder.
            additional_prompts: Extra prompts to precompute embeddings for (sweep support).
            skip_t5: If True, skip T5-XXL text encoder to save ~10GB RAM.
        """
        self.device = device
        self.reward_device = reward_device
        self.dtype = dtype
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.skip_t5 = skip_t5
        self.model_id = model_id

        torch.backends.cuda.matmul.allow_tf32 = True

        self._load_model(prompt, negative_prompt, max_sequence_length, additional_prompts)

    def _load_model(self, prompt: str, negative_prompt: str,
                    max_sequence_length: int, additional_prompts: list = None):
        """Load SD3.5 pipeline, precompute embeddings, delete text encoders."""
        print(f"Loading SD3.5 from: {self.model_id}")
        load_kwargs = dict(torch_dtype=self.dtype)
        if self.skip_t5:
            print("  Skipping T5-XXL text encoder to save ~10GB RAM.")
            load_kwargs["text_encoder_3"] = None
            load_kwargs["tokenizer_3"] = None
        try:
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.model_id, local_files_only=True, **load_kwargs
            ).to(self.device)
        except Exception:
            print("Local cache miss, downloading from HuggingFace...")
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.model_id, **load_kwargs
            ).to(self.device)

        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.scheduler = self.pipeline.scheduler
        self.vae_scale_factor = self.pipeline.vae_scale_factor  # 8

        # Precompute text embeddings
        print("Precomputing text embeddings...")
        self.prompt_cache = {}
        with torch.no_grad():
            pe, npe, ppe, nppe = self.pipeline.encode_prompt(
                prompt=prompt, prompt_2=prompt, prompt_3=prompt,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                negative_prompt_3=negative_prompt,
                do_classifier_free_guidance=True,
                device=self.device,
                max_sequence_length=max_sequence_length,
            )
            self.cached_prompt_embeds = pe          # [1, seq, 4096]
            self.cached_neg_prompt_embeds = npe     # [1, seq, 4096]
            self.cached_pooled_embeds = ppe         # [1, 2048]
            self.cached_neg_pooled_embeds = nppe    # [1, 2048]
            self.prompt_cache[prompt] = (pe.cpu(), npe.cpu(), ppe.cpu(), nppe.cpu())

            if additional_prompts:
                for p in additional_prompts:
                    if p not in self.prompt_cache:
                        pe2, npe2, ppe2, nppe2 = self.pipeline.encode_prompt(
                            prompt=p, prompt_2=p, prompt_3=p,
                            negative_prompt=negative_prompt,
                            negative_prompt_2=negative_prompt,
                            negative_prompt_3=negative_prompt,
                            do_classifier_free_guidance=True,
                            device=self.device,
                            max_sequence_length=max_sequence_length,
                        )
                        self.prompt_cache[p] = (pe2.cpu(), npe2.cpu(), ppe2.cpu(), nppe2.cpu())
                print(f"Precomputed embeddings for {len(self.prompt_cache)} prompts")

        # Delete text encoders to free memory
        print("Removing text encoders to free memory...")
        del self.pipeline.text_encoder
        del self.pipeline.text_encoder_2
        del self.pipeline.text_encoder_3
        del self.pipeline.tokenizer
        del self.pipeline.tokenizer_2
        del self.pipeline.tokenizer_3
        self.pipeline.text_encoder = None
        self.pipeline.text_encoder_2 = None
        self.pipeline.text_encoder_3 = None
        self.pipeline.tokenizer = None
        self.pipeline.tokenizer_2 = None
        self.pipeline.tokenizer_3 = None
        gc.collect()
        torch.cuda.empty_cache()
        print("SD3.5 loaded (memory-efficient mode)")

    def set_prompt_embeddings(self, prompt: str):
        """Switch to precomputed embeddings for a different prompt (for sweeps)."""
        if prompt not in self.prompt_cache:
            raise ValueError(f"Prompt not precomputed: '{prompt[:50]}...'. "
                             "Pass it in additional_prompts at init time.")
        pe, npe, ppe, nppe = self.prompt_cache[prompt]
        self.cached_prompt_embeds = pe.to(self.device)
        self.cached_neg_prompt_embeds = npe.to(self.device)
        self.cached_pooled_embeds = ppe.to(self.device)
        self.cached_neg_pooled_embeds = nppe.to(self.device)
        self.prompt = prompt

    def predict_velocity_cfg(self, z: torch.Tensor, sigma: float,
                             guidance_scale: float) -> torch.Tensor:
        """
        Predict velocity with classifier-free guidance.

        Two forward passes: unconditional + conditional, then CFG combination.

        Args:
            z: Latent state [B, 16, H, W].
            sigma: Current sigma value in [0, 1].
            guidance_scale: CFG scale.

        Returns:
            CFG-combined velocity [B, 16, H, W].
        """
        B = z.shape[0]

        # Duplicate latents for CFG: [uncond, cond]
        z_input = torch.cat([z, z], dim=0)

        # Timestep: sigma * 1000 (scheduler convention)
        timestep = torch.tensor([sigma * 1000.0], device=z.device, dtype=z.dtype)
        timestep = timestep.expand(2 * B)

        # Expand embeddings for batch
        neg_embeds = self.cached_neg_prompt_embeds.expand(B, -1, -1)
        pos_embeds = self.cached_prompt_embeds.expand(B, -1, -1)
        encoder_hs = torch.cat([neg_embeds, pos_embeds], dim=0)

        neg_pooled = self.cached_neg_pooled_embeds.expand(B, -1)
        pos_pooled = self.cached_pooled_embeds.expand(B, -1)
        pooled = torch.cat([neg_pooled, pos_pooled], dim=0)

        # Forward pass
        v = self.transformer(
            hidden_states=z_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hs,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

        # CFG: v = v_uncond + scale * (v_cond - v_uncond)
        v_uncond, v_cond = v.chunk(2)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    def _predict_velocity_cfg_wrapper(self, z: torch.Tensor, sigma_t: torch.Tensor,
                                      guidance_scale_t: torch.Tensor) -> torch.Tensor:
        """Wrapper taking all tensor args for torch.checkpoint compatibility."""
        return self.predict_velocity_cfg(z, sigma_t.item(), guidance_scale_t.item())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            z: Latent tensor [B, 16, H, W].

        Returns:
            Images [B, 3, H*8, W*8] in [0, 1] range.
        """
        z_scaled = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(z_scaled, return_dict=False)[0]
        image = (image + 1.0) / 2.0
        return image.clamp(0, 1)

    def get_sigma_schedule(self, num_steps: int) -> torch.Tensor:
        """
        Get the sigma schedule from the scheduler.

        Returns:
            Tensor of sigmas [num_steps + 1], from ~1 (noise) to 0 (clean).
        """
        self.scheduler.set_timesteps(num_steps, device=self.device)
        return self.scheduler.sigmas  # already has trailing 0 appended

    def euler_lookahead(self, z: torch.Tensor, sigma_cur: float,
                        guidance_scale: float, num_steps: int = 8,
                        grad_checkpoint: bool = False) -> torch.Tensor:
        """
        Predict clean image z0 from z at sigma_cur using multi-step Euler forward.

        Replaces the flow map look-ahead in FLUX FMTT. Takes num_steps Euler steps
        from sigma_cur down to sigma=0, each with a CFG velocity prediction.

        All steps preserve gradients (no torch.no_grad) so reward gradients
        can flow back through the look-ahead.

        Args:
            z: Latent at sigma_cur [1, 16, H, W], may require grad.
            sigma_cur: Current sigma in (0, 1].
            guidance_scale: CFG scale.
            num_steps: Number of Euler sub-steps (default 8).
            grad_checkpoint: If True, use gradient checkpointing per sub-step.

        Returns:
            z0_pred: Predicted clean latent [1, 16, H, W].
        """
        z_cur = z
        d_sigma = sigma_cur / num_steps

        for s in range(num_steps):
            sigma = sigma_cur - s * d_sigma
            sigma_next = max(sigma - d_sigma, 0.0)
            dt = sigma_next - sigma  # negative

            if grad_checkpoint:
                sigma_t = torch.tensor(sigma, device=z.device)
                gs_t = torch.tensor(guidance_scale, device=z.device)
                v = cp.checkpoint(
                    self._predict_velocity_cfg_wrapper,
                    z_cur, sigma_t, gs_t,
                    use_reentrant=False,
                )
            else:
                v = self.predict_velocity_cfg(z_cur, sigma, guidance_scale)

            z_cur = z_cur + dt * v

        return z_cur

    @staticmethod
    def _get_chi_t(chi, sigma: float) -> float:
        """Get chi value at current sigma. Supports constant or paper schedule.

        Paper schedule (adapted to sigma convention):
            chi_t = 1.05 * sigma / (1 - sigma + 0.05)
        """
        if isinstance(chi, str) and chi == "schedule":
            return 1.05 * sigma / (1.0 - sigma + 0.05)
        return float(chi)

    @staticmethod
    def _get_epsilon_t(epsilon, sigma: float) -> float:
        """Get epsilon value at current sigma. Supports constant or paper schedule.

        Paper schedule: epsilon_t = sigma
        """
        if isinstance(epsilon, str) and epsilon == "schedule":
            return sigma
        return float(epsilon)

    # =========================================================================
    # DINOv3-B feature extraction (persistent model for interactive mode)
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
        """Extract L2-normalized DINOv3-B features from a list of PIL images.

        Args:
            pil_images: List of PIL RGB images.

        Returns:
            np.ndarray of shape [N, 768], L2-normalized.
        """
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

    @staticmethod
    def _in_notebook():
        """Check if running inside a Jupyter/Colab notebook."""
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    def _show_grid_inline(self, pil_images, labels=None):
        """Display image grid inline in notebook, or skip in terminal."""
        if not self._in_notebook():
            return
        import matplotlib
        matplotlib.use("module://matplotlib_inline.backend_inline")
        import matplotlib.pyplot as plt
        n = len(pil_images)
        cols = min(n, 4)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        for i, ax in enumerate(axes):
            if i < n:
                ax.imshow(np.array(pil_images[i]))
                ax.set_title(labels[i] if labels else str(i), fontsize=12)
            ax.axis("off")
        fig.tight_layout()
        plt.show()

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
    # Interactive FMTT sampling
    # =========================================================================

    # =========================================================================
    # GLASS Flows transition (Algorithm 1 from Holderrieth et al., ICLR 2026)
    # =========================================================================

    def _glass_init(self, X_bar: torch.Tensor, sigma_start: float,
                    sigma_end: float, rho: float) -> Tuple[torch.Tensor, dict]:
        """
        GLASS stochastic initialization for a segment.

        Computes the conditional distribution parameters and initializes
        X̄_0 = γ̄·x_t + σ̄_0·ε for each particle. Returns the initialized
        inner state and the GLASS parameters needed for inner ODE steps.

        Args:
            X_bar: particle states [P, 16, H, W] (these become the fixed x_t)
            sigma_start: sigma at start of segment
            sigma_end: sigma at end of segment
            rho: GLASS correlation parameter

        Returns:
            X_inner: initialized inner state [P, 16, H, W]
            glass_params: dict with parameters for inner ODE steps
        """
        clip = 1e-8

        # Paper convention: alpha = 1-sigma, beta = sigma
        alpha_start = 1.0 - sigma_start
        alpha_end = 1.0 - sigma_end

        # Conditional distribution parameters (Eq 17)
        bar_gamma = rho * sigma_end / max(sigma_start, clip)
        bar_alpha = alpha_end - bar_gamma * alpha_start
        bar_sigma_sq = sigma_end ** 2 * max(1.0 - rho ** 2, 0.0)
        bar_sigma = math.sqrt(max(bar_sigma_sq, 0.0))
        bar_sigma_0 = 1.0  # initial inner noise scale

        # Stochastic init: X̄_0 = γ̄·x_t + σ̄_0·ε
        eps = torch.randn_like(X_bar)
        X_inner = bar_gamma * X_bar + bar_sigma_0 * eps

        glass_params = dict(
            x_t=X_bar.clone(),  # fixed conditioning point for this segment
            alpha_start=alpha_start,
            sigma_start=sigma_start,
            bar_gamma=bar_gamma,
            bar_alpha=bar_alpha,
            bar_sigma=bar_sigma,
            bar_sigma_0=bar_sigma_0,
        )
        return X_inner, glass_params

    def _glass_velocity_step(self, X_inner: torch.Tensor, s: float,
                             ds: float, glass_params: dict,
                             guidance_scale: float):
        """
        One inner ODE step of the GLASS velocity field.

        Args:
            X_inner: current inner state [P, 16, H, W]
            s: current inner time in [0, 1]
            ds: inner time step size
            glass_params: from _glass_init
            guidance_scale: CFG scale for velocity prediction

        Returns:
            X_inner: updated inner state
        """
        clip = 1e-8
        gp = glass_params
        x_t = gp["x_t"]

        bar_alpha_s = s * gp["bar_alpha"]
        bar_sigma_s = (1.0 - s) * gp["bar_sigma_0"] + s * gp["bar_sigma"]

        # Weight coefficients (Eq 21)
        dot_bar_sigma_s = gp["bar_sigma"] - gp["bar_sigma_0"]
        dot_bar_alpha_s = gp["bar_alpha"]
        w1 = dot_bar_sigma_s / max(bar_sigma_s, clip)
        w2 = dot_bar_alpha_s - bar_alpha_s * w1
        w3 = -gp["bar_gamma"] * w1

        # Sufficient statistic (Proposition 2) — per-particle
        mu1 = gp["alpha_start"]
        mu2 = bar_alpha_s + gp["bar_gamma"] * gp["alpha_start"]
        S11 = gp["sigma_start"] ** 2
        S12 = gp["bar_gamma"] * S11
        S22 = bar_sigma_s ** 2 + gp["bar_gamma"] ** 2 * S11

        det = S11 * S22 - S12 * S12
        det = max(det, clip)
        inv11 = S22 / det
        inv12 = -S12 / det
        inv22 = S11 / det

        bproduct = mu1 * mu1 * inv11 + 2 * mu1 * mu2 * inv12 + mu2 * mu2 * inv22
        bproduct = max(bproduct, clip)

        # t* and alpha_star
        g_inv = 1.0 / (1.0 + math.sqrt(max(1.0 / bproduct, 0.0)))
        t_star = max(min(g_inv, 0.999), 0.001)
        alpha_star = t_star
        sigma_star = 1.0 - t_star

        # Sufficient statistic weights
        w_xt = alpha_star * (mu1 * inv11 + mu2 * inv12) / bproduct
        w_xs = alpha_star * (mu1 * inv12 + mu2 * inv22) / bproduct

        # Process each particle
        with torch.no_grad():
            P = X_inner.shape[0]
            for p in range(P):
                S_input = w_xt * x_t[p:p+1] + w_xs * X_inner[p:p+1]

                # GLASS denoiser: D_{t*}(S_input, sigma_star) = S_input - sigma_star * v
                v_star = self.predict_velocity_cfg(S_input, sigma_star, guidance_scale)
                denoiser = S_input - sigma_star * v_star

                # GLASS velocity: w1*X̄_s + w2*D + w3*x_t
                velocity = w1 * X_inner[p:p+1] + w2 * denoiser + w3 * x_t[p:p+1]
                X_inner[p:p+1] = X_inner[p:p+1] + ds * velocity

        return X_inner

    def sample_interactive(self,
                           num_steps: int = 28,
                           num_particles: int = 4,
                           num_clones: int = 2,
                           num_resampling_steps: int = 4,
                           guidance_scale: float = 4.5,
                           rho: float = 0.4,
                           lookahead_steps: int = 8,
                           schedule_mode: Optional[str] = "linear",
                           img_shape: Optional[Tuple[int, int]] = None,
                           seed: Optional[int] = None,
                           output_dir: str = "interactive_fmtt") -> torch.Tensor:
        """
        Interactive FMTT with GLASS Flows for stochastic clone divergence.

        Anchor-clone design:
        - N anchor particles run standard deterministic ODE (no GLASS)
        - Each anchor has (C-1) clones that diverge via GLASS stochastic init
        - Total particles = N * C (N anchors + N*(C-1) clones)

        At each resampling point:
        1. Deterministic Euler lookahead to σ=0 for all particles (preview)
        2. User selects which particles to keep
        3. Selected particles become new anchors (deterministic ODE)
        4. Clones are created from anchors with GLASS stochastic init

        Particle layout: [anchor_0, clone_0_1, ..., clone_0_{C-1},
                          anchor_1, clone_1_1, ..., clone_1_{C-1}, ...]
        Within each group of C, index 0 is the anchor.

        Args:
            num_steps: Total ODE steps K.
            num_particles: Number of anchor particles N.
            num_clones: Clones per anchor C (total particles = N*C).
            num_resampling_steps: Number of pruning intervals R.
            guidance_scale: SD3.5 CFG scale.
            rho: GLASS correlation (0=max diversity, 1=no diversity). Default 0.4.
            lookahead_steps: Euler sub-steps for preview look-ahead.
            img_shape: Output image (H, W). Default (1024, 1024).
            seed: Random seed.
            output_dir: Directory to save interval previews and final outputs.

        Returns:
            Final selected image [1, 3, H, W] in [0, 1].
        """
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)
        device = self.device
        os.makedirs(output_dir, exist_ok=True)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self._load_dino()

        latH = imgH // self.vae_scale_factor
        latW = imgW // self.vae_scale_factor

        # ===== Initialize N anchor particles =====
        anchors = torch.randn(num_particles, 16, latH, latW, device=device, dtype=self.dtype)

        # Total particles: N groups of C (1 anchor + C-1 clones each)
        total_particles = num_particles * num_clones
        C = num_clones

        # Sigma schedule
        sigmas = self.get_sigma_schedule(num_steps)

        # Resampling points
        if schedule_mode == "aggressive" and num_resampling_steps > 0:
            resample_steps = compute_aggressive_diversity_schedule(num_steps, num_resampling_steps)
            print(f"  Aggressive diversity schedule: steps {resample_steps}")
        elif schedule_mode == "linear" and num_resampling_steps > 0:
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

        # Build initial particle tensor: anchors + clones
        # Layout: [a0, c0_1, ..., c0_{C-1}, a1, c1_1, ..., c1_{C-1}, ...]
        X_all = anchors.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
        X_all = X_all.reshape(total_particles, 16, latH, latW)

        # Track which are anchors vs clones
        is_anchor = torch.zeros(total_particles, dtype=torch.bool)
        for i in range(num_particles):
            is_anchor[i * C] = True  # first in each group is anchor

        # GLASS state for clones
        glass_params = None
        interval_count = 0
        inc_layout = IncrementalLayout() if IncrementalLayout is not None else None
        prev_kept_indices = None

        pbar = tqdm(range(num_steps), total=num_steps, desc="SD3.5-Interactive")

        for k in pbar:
            sigma_cur = sigmas[k].item()
            sigma_next = sigmas[k + 1].item()
            d_sigma = sigma_next - sigma_cur

            # ===== At segment start: GLASS init for clones =====
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
                      f"σ={sigma_seg_start:.4f}→{sigma_seg_end:.4f}, M={M_inner}")

                # GLASS init only for clones, using their parent anchor's state
                clone_indices = (~is_anchor).nonzero(as_tuple=True)[0]
                if len(clone_indices) > 0:
                    # Each clone's parent anchor: clone at index i*C+j has anchor at i*C
                    clone_parents = X_all.clone()  # will extract per-clone
                    clone_states = X_all[clone_indices]

                    # Build parent states for each clone
                    parent_states = torch.zeros_like(clone_states)
                    for ci, idx in enumerate(clone_indices):
                        group = idx.item() // C
                        anchor_idx = group * C
                        parent_states[ci] = X_all[anchor_idx]

                    # GLASS init: X̄_0 = γ̄·x_parent + σ̄_0·ε
                    clone_inner, gp = self._glass_init(
                        parent_states, sigma_seg_start, sigma_seg_end, rho)
                    X_all[clone_indices] = clone_inner

                    glass_params = gp
                    glass_params["seg_start_step"] = k
                    glass_params["M_inner"] = M_inner
                    glass_params["clone_indices"] = clone_indices
                    glass_params["parent_states"] = parent_states
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
                    v = self.predict_velocity_cfg(z_p, sigma_cur, guidance_scale)
                    X_all[ai:ai+1] = z_p + d_sigma * v

                # Clones: GLASS inner ODE velocity step
                clone_indices = glass_params["clone_indices"]
                if len(clone_indices) > 0:
                    gp = glass_params
                    x_t = gp["x_t"]  # parent states at segment start
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

                    for ci, idx in enumerate(clone_indices):
                        x_parent = x_t[ci:ci+1]
                        x_clone = X_all[idx:idx+1]
                        S_input = w_xt * x_parent + w_xs * x_clone
                        v_star = self.predict_velocity_cfg(S_input, sigma_star, guidance_scale)
                        denoiser = S_input - sigma_star * v_star
                        velocity = w1 * x_clone + w2 * denoiser + w3 * x_parent
                        X_all[idx:idx+1] = x_clone + ds * velocity

            # ===== Interactive pruning at resampling intervals =====
            if k in resample_at:
                glass_params = None

                interval_count += 1
                interval_dir = os.path.join(output_dir, f"interval_{interval_count}")
                os.makedirs(interval_dir, exist_ok=True)

                print(f"\n--- Interval {interval_count} (step {k}/{num_steps}, "
                      f"sigma={sigma_next:.4f}) ---")
                print(f"Computing look-ahead previews for {total_particles} particles...")

                # Labels: mark anchors vs clones
                labels = []
                for i in range(total_particles):
                    group = i // C
                    pos = i % C
                    if pos == 0:
                        labels.append(f"{i} (anchor g{group})")
                    else:
                        labels.append(f"{i} (clone g{group})")

                # a) Deterministic Euler look-ahead + decode
                pil_images = []
                with torch.no_grad():
                    for p_idx in range(total_particles):
                        z0 = self.euler_lookahead(
                            X_all[p_idx:p_idx+1], sigma_next,
                            guidance_scale, num_steps=lookahead_steps
                        )
                        img_t = self.decode(z0)
                        pil_img = self._tensor_to_pil(img_t)
                        pil_img.save(os.path.join(interval_dir, f"particle_{p_idx:03d}.png"))
                        pil_images.append(pil_img)

                # b) DINOv3-B features + layout
                print("Extracting DINOv3-B features...")
                features = self._extract_dino_features(pil_images)

                # Incremental layout
                if inc_layout is not None:
                    if interval_count == 1:
                        inc_layout.initialize(features, pil_images)
                    else:
                        inc_layout.update(prev_kept_indices, features, pil_images, num_clones=C)
                    inc_layout.plot(
                        title=f"Interval {interval_count} (step {k}, sigma={sigma_next:.3f})",
                        save_path=os.path.join(interval_dir, "layout_incremental.png"),
                        thumb_size=80,
                    )

                # Standard cluster for comparison
                cluster_path = os.path.join(interval_dir, "cluster_dinov3.png")
                plot_image_cluster(
                    features, pil_images,
                    title=f"DINOv3-B Cluster — Interval {interval_count} "
                          f"(step {k}, sigma={sigma_next:.3f})",
                    save_path=cluster_path,
                    n_neighbors=min(15, total_particles - 1),
                    min_dist=0.1,
                    thumb_size=80,
                )

                # c) Save labeled grid + show inline in notebook
                self._save_grid(pil_images, os.path.join(interval_dir, "grid.png"), labels)
                self._show_grid_inline(pil_images, labels)

                # d) Prompt user
                kept_indices = self._prompt_user_selection(
                    total_particles, interval_count, interval_dir
                )
                prev_kept_indices = kept_indices

                # e) Selected → new anchors
                kept_latents = X_all[kept_indices].clone()
                num_particles = len(kept_indices)

                # Last resampling? No cloning — just run anchors to the end
                is_last_resample = (interval_count == num_resampling_steps)

                if is_last_resample:
                    total_particles = num_particles
                    X_all = kept_latents
                    is_anchor = torch.ones(total_particles, dtype=torch.bool)
                    print(f"  Selected {num_particles} particles (final segment, no cloning)")
                else:
                    total_particles = num_particles * C
                    X_all = kept_latents.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
                    X_all = X_all.reshape(total_particles, 16, latH, latW)
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

        if inc_layout is not None and inc_layout.coords is not None:
            inc_layout.update(list(range(total_particles)), features, pil_images)
            inc_layout.plot(
                title="Final — Incremental Layout",
                save_path=os.path.join(final_dir, "layout_incremental.png"),
                thumb_size=80,
            )

        self._save_grid(pil_images, os.path.join(final_dir, "grid.png"), labels)
        self._show_grid_inline(pil_images, labels)

        print(f"\n  Final images saved to: {final_dir}/")
        print(f"  Grid: {final_dir}/grid.png")
        print(f"  Cluster: {final_dir}/cluster_dinov3.png")
        print(f"  Incremental: {final_dir}/layout_incremental.png")

        self._unload_dino()

        # Return all final decoded images [N*C, 3, H, W]
        with torch.no_grad():
            return torch.cat([self.decode(X_all[i:i+1]).float()
                              for i in range(total_particles)], dim=0)

    def compute_reward_gradient(self, z: torch.Tensor, sigma_cur: float,
                                reward_fn: Callable, guidance_scale: float,
                                lookahead_steps: int = 8,
                                grad_checkpoint: bool = False
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reward and its gradient via multi-step Euler look-ahead.

        Reward is scaled by (1 - sigma): r_sigma(x) = (1 - sigma) * r(x0_pred)

        Args:
            z: Current latent [1, 16, H, W], will be cloned with requires_grad.
            sigma_cur: Current sigma.
            reward_fn: r(images) -> [B] rewards. images are [B, 3, H, W] in [0,1].
            guidance_scale: CFG scale.
            lookahead_steps: Number of Euler sub-steps for look-ahead.
            grad_checkpoint: Use gradient checkpointing in look-ahead.

        Returns:
            reward: Scalar reward (detached).
            grad_r: Gradient of reward w.r.t. z [1, 16, H, W] (detached).
        """
        z_grad = z.clone().requires_grad_(True)

        z0_pred = self.euler_lookahead(z_grad, sigma_cur, guidance_scale,
                                       num_steps=lookahead_steps,
                                       grad_checkpoint=grad_checkpoint)
        x0_pred = self.decode(z0_pred).float()

        if self.reward_device:
            x0_pred = x0_pred.to(self.reward_device)

        reward = (1.0 - sigma_cur) * reward_fn(x0_pred)
        grad_r = torch.autograd.grad(reward.sum(), z_grad)[0]

        return reward.detach(), grad_r.detach()

    def compute_reward_batch(self, z_batch: torch.Tensor, sigma_cur: float,
                             reward_fn: Callable, guidance_scale: float,
                             lookahead_steps: int = 8) -> torch.Tensor:
        """
        Compute rewards for a batch of particles (no gradients).

        Args:
            z_batch: [B, 16, H, W] batch of latents.
            sigma_cur: Current sigma.
            reward_fn: Reward function.
            guidance_scale: CFG scale.
            lookahead_steps: Euler sub-steps for look-ahead.

        Returns:
            rewards: [B] tensor of rewards.
        """
        with torch.no_grad():
            # Process one at a time to avoid OOM
            rewards = []
            for i in range(z_batch.shape[0]):
                z_i = z_batch[i:i+1]
                z0_pred = self.euler_lookahead(z_i, sigma_cur, guidance_scale,
                                              num_steps=lookahead_steps)
                x0_pred = self.decode(z0_pred).float()
                if self.reward_device:
                    x0_pred = x0_pred.to(self.reward_device)
                r_i = (1.0 - sigma_cur) * reward_fn(x0_pred)
                rewards.append(r_i.reshape(1) if r_i.dim() == 0 else r_i)
            return torch.cat(rewards)

    def sample_fmtt(self,
                    reward_fn: Callable,
                    num_steps: int = 28,
                    num_particles: int = 4,
                    num_clones: int = 2,
                    num_resampling_steps: int = 4,
                    guidance_scale: float = 4.5,
                    chi=0.0,
                    epsilon=0.0,
                    mode: str = "searching",
                    lookahead_steps: int = 8,
                    grad_checkpoint: bool = False,
                    img_shape: Optional[Tuple[int, int]] = None,
                    seed: Optional[int] = None,
                    callback: Optional[Callable] = None) -> torch.Tensor:
        """
        FMTT sampling with multi-step Euler look-ahead (Algorithm 1).

        Args:
            reward_fn: Reward function r(x). Takes [B, 3, H, W] images in [0,1],
                       returns [B] scalar rewards. Higher is better.
            num_steps: Number of simulation steps K.
            num_particles: Number of particles N.
            num_clones: Number of clones per particle C.
            num_resampling_steps: Number of resampling events R (evenly spaced).
            guidance_scale: SD3.5 CFG guidance scale.
            chi: Reward gradient scaling. Float (constant) or "schedule" for
                 time-varying: chi_t = 1.05*sigma/(1-sigma+0.05).
            epsilon: Noise level. Float (constant) or "schedule" for
                 time-varying: epsilon_t = sigma.
            mode: "sampling" (importance-weighted) or "searching" (top-N).
            lookahead_steps: Number of Euler sub-steps for reward look-ahead (default 8).
            grad_checkpoint: Use gradient checkpointing in look-ahead.
            img_shape: Output image (H, W). Default (1024, 1024).
            seed: Random seed.
            callback: Optional callback(step, best_image, sigma, best_reward).

        Returns:
            Best generated image [1, 3, H, W] in [0, 1].
        """
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)
        device = self.device

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # Latent shape: spatial dims / vae_scale_factor
        latH = imgH // self.vae_scale_factor
        latW = imgW // self.vae_scale_factor
        latent_shape = (num_particles, 16, latH, latW)

        # ===== Initialize particles (Algorithm 1, Line 2-5) =====
        X = torch.randn(latent_shape, device=device, dtype=self.dtype)  # [N, 16, H, W]

        # Clone: [N, C, 16, H, W] -> [N*C, 16, H, W]
        X_bar = X.unsqueeze(1).expand(-1, num_clones, -1, -1, -1).clone()
        X_bar = X_bar.reshape(num_particles * num_clones, 16, latH, latW)

        # Importance weights (sampling mode)
        if mode == "sampling":
            A = torch.zeros(num_particles * num_clones, device=device, dtype=torch.float32)

        # Sigma schedule from scheduler
        sigmas = self.get_sigma_schedule(num_steps)

        # Compute resampling steps: R evenly-spaced points within [1, K-1]
        if num_resampling_steps > 0:
            resample_at = set(
                int(round((i + 1) * num_steps / (num_resampling_steps + 1)))
                for i in range(num_resampling_steps)
            )
        else:
            resample_at = set()

        total_particles = num_particles * num_clones

        pbar = tqdm(range(num_steps), total=num_steps, desc=f"SD3.5-FMTT ({mode})")

        for k in pbar:
            sigma_cur = sigmas[k].item()
            sigma_next = sigmas[k + 1].item()
            d_sigma = sigma_next - sigma_cur  # negative (sigma decreasing)

            chi_t = self._get_chi_t(chi, sigma_cur)
            eps_t = self._get_epsilon_t(epsilon, sigma_cur)
            need_grad = chi_t > 0 or eps_t > 0

            # ===== Update each particle (Algorithm 1, Line 8) =====
            for p in range(total_particles):
                z_p = X_bar[p:p+1]

                # Instantaneous velocity (no grad needed)
                with torch.no_grad():
                    v = self.predict_velocity_cfg(z_p, sigma_cur, guidance_scale)

                # Reward gradient (if chi or epsilon > 0)
                if need_grad:
                    reward_p, grad_r = self.compute_reward_gradient(
                        z_p, sigma_cur, reward_fn, guidance_scale,
                        lookahead_steps=lookahead_steps,
                        grad_checkpoint=grad_checkpoint,
                    )
                    if p == 0:
                        grad_norm = torch.norm(grad_r).item()
                        r_val = reward_p.item() if reward_p.dim() == 0 else reward_p.sum().item()
                        print(f"  [k={k} p=0] reward={r_val:.6f} "
                              f"grad_norm={grad_norm:.6e} chi_t={chi_t:.4f} eps_t={eps_t:.4f}")
                else:
                    grad_r = torch.zeros_like(z_p)

                # Update: x += [v + chi*grad_r + eps*(s + grad_r)] * d_sigma + noise
                update = v + chi_t * grad_r

                if eps_t > 0:
                    # Score function: s_sigma(x) = (-(1-sigma)*v - x) / sigma
                    if sigma_cur > 1e-6:
                        s_t = (-(1.0 - sigma_cur) * v - z_p) / sigma_cur
                    else:
                        s_t = torch.zeros_like(v)
                    update = update + eps_t * (s_t + grad_r)
                    noise = torch.randn_like(z_p)
                    X_bar[p:p+1] = z_p + update * d_sigma + math.sqrt(2 * eps_t * abs(d_sigma)) * noise
                else:
                    X_bar[p:p+1] = z_p + update * d_sigma

                # Importance weight update (sampling mode)
                # dA = [r_sigma / (1-sigma)] * |d_sigma| = r_raw * |d_sigma|
                if mode == "sampling":
                    with torch.no_grad():
                        r_for_A = self.compute_reward_batch(
                            X_bar[p:p+1], sigma_cur, reward_fn,
                            guidance_scale, lookahead_steps
                        )
                        if (1.0 - sigma_cur) > 0.01:
                            A[p] += (r_for_A / (1.0 - sigma_cur)).float().item() * abs(d_sigma)

            # ===== Resample / Select (Algorithm 1, Line 13-21) =====
            if k in resample_at:
                if mode == "sampling":
                    probs = torch.softmax(A, dim=0)
                    indices = torch.multinomial(probs, num_particles, replacement=True)
                    X = X_bar[indices].clone()
                    A.zero_()

                elif mode == "searching":
                    with torch.no_grad():
                        rewards_list = []
                        for p_idx in range(total_particles):
                            r_p = self.compute_reward_batch(
                                X_bar[p_idx:p_idx+1], sigma_cur, reward_fn,
                                guidance_scale, lookahead_steps
                            )
                            rewards_list.append(r_p.reshape(1) if r_p.dim() == 0 else r_p)
                        rewards_all = torch.cat(rewards_list)

                    _, top_indices = torch.topk(rewards_all, num_particles)
                    X = X_bar[top_indices].clone()

                # Re-clone
                X_bar = X.unsqueeze(1).expand(-1, num_clones, -1, -1, -1).clone()
                X_bar = X_bar.reshape(num_particles * num_clones, 16, latH, latW)

            # Callback
            if callback is not None and k % 5 == 0:
                with torch.no_grad():
                    rewards_list = []
                    for p_idx in range(min(num_particles, total_particles)):
                        r_p = self.compute_reward_batch(
                            X_bar[p_idx:p_idx+1], sigma_cur, reward_fn,
                            guidance_scale, lookahead_steps
                        )
                        rewards_list.append(r_p.reshape(1) if r_p.dim() == 0 else r_p)
                    rewards_all = torch.cat(rewards_list)
                    best_idx = rewards_all.argmax()

                    z0_best = self.euler_lookahead(
                        X_bar[best_idx:best_idx+1], sigma_cur,
                        guidance_scale, num_steps=lookahead_steps
                    )
                    img_best = self.decode(z0_best)
                callback(k, img_best, sigma_cur, rewards_all[best_idx].item())

        # ===== Final: decode all particles, return best =====
        with torch.no_grad():
            final_rewards = []
            for p_idx in range(total_particles):
                img_p = self.decode(X_bar[p_idx:p_idx+1]).float()
                if self.reward_device:
                    img_p = img_p.to(self.reward_device)
                r_p = reward_fn(img_p)
                final_rewards.append(r_p.reshape(1) if r_p.dim() == 0 else r_p)
            final_rewards = torch.cat(final_rewards)
            best_idx = final_rewards.argmax()
            img_final = self.decode(X_bar[best_idx:best_idx+1]).float()

        return img_final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SD3.5 FMTT test-time scaling")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="fmtt_output.png")
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--num_particles", type=int, default=6)
    parser.add_argument("--num_clones", type=int, default=3)
    parser.add_argument("--num_resampling_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--chi", type=str, default="0.0", help="Chi value or 'schedule'")
    parser.add_argument("--epsilon", type=str, default=None, help="Epsilon value or 'schedule'. Default: 0.0 for reward-guided.")
    parser.add_argument("--rho", type=float, default=0.4, help="GLASS correlation (interactive mode). 0=max diversity, 1=no diversity. Default: 0.4")
    parser.add_argument("--mode", type=str, default="searching", choices=["sampling", "searching"])
    parser.add_argument("--lookahead_steps", type=int, default=8)
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: manually prune particles at each interval")
    parser.add_argument("--schedule_mode", type=str, default="linear", choices=["linear", "aggressive", "none"],
                        help="Resampling schedule mode: 'linear' (front-loaded, default) or 'none' (even)")
    parser.add_argument("--skip_t5", action="store_true",
                        help="Skip T5-XXL text encoder to save ~10GB RAM")
    parser.add_argument("--model_id", type=str, default=None,
                        help="HuggingFace model ID (default: stabilityai/stable-diffusion-3.5-medium)")
    parser.add_argument("--output_dir", type=str, default="interactive_fmtt",
                        help="Output directory for interactive mode previews")
    args = parser.parse_args()

    # Parse epsilon for reward-guided mode
    if args.epsilon is None:
        epsilon = 0.0
    else:
        epsilon = args.epsilon if args.epsilon == "schedule" else float(args.epsilon)

    init_kwargs = dict(prompt=args.prompt, device="cuda", skip_t5=args.skip_t5)
    if args.model_id:
        init_kwargs["model_id"] = args.model_id
    sampler = SD35FMTT(**init_kwargs)

    if args.interactive:
        print(f"\nStarting interactive FMTT with GLASS Flows (ρ={args.rho}, schedule_mode={args.schedule_mode})...")
        best_image = sampler.sample_interactive(
            num_steps=args.num_steps,
            num_particles=args.num_particles,
            num_clones=args.num_clones,
            num_resampling_steps=args.num_resampling_steps,
            guidance_scale=args.guidance_scale,
            rho=args.rho,
            schedule_mode=args.schedule_mode if args.schedule_mode != "none" else None,
            lookahead_steps=args.lookahead_steps,
            img_shape=(args.height, args.width),
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        chi = args.chi if args.chi == "schedule" else float(args.chi)

        # Placeholder reward (replace with actual reward function)
        def dummy_reward(images):
            """Placeholder: returns mean pixel intensity as reward."""
            return images.mean(dim=(1, 2, 3))

        print("\nStarting FMTT sampling...")
        best_image = sampler.sample_fmtt(
            reward_fn=dummy_reward,
            num_steps=args.num_steps,
            num_particles=args.num_particles,
            num_clones=args.num_clones,
            num_resampling_steps=args.num_resampling_steps,
            guidance_scale=args.guidance_scale,
            chi=chi,
            epsilon=epsilon,
            mode=args.mode,
            lookahead_steps=args.lookahead_steps,
            grad_checkpoint=args.grad_checkpoint,
            img_shape=(args.height, args.width),
            seed=args.seed,
        )

    # Save final image(s)
    if best_image.shape[0] == 1:
        img_np = (best_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(args.output)
        print(f"Saved: {args.output}")
    else:
        # Interactive mode returns all final particles
        out_base, out_ext = os.path.splitext(args.output)
        for i in range(best_image.shape[0]):
            img_np = (best_image[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            path = f"{out_base}_{i}{out_ext}"
            Image.fromarray(img_np).save(path)
        print(f"Saved {best_image.shape[0]} images to {out_base}_*{out_ext}")
