"""
FLUX FlowMap Interactive Sampler — Weighted-Diamond-Maps clone mode (WDM).

Parallel sibling to flux_fmtt_interactive_flowmap.py. Same base class
(FluxFlowMapVLMSampler), same flow-map LoRA, same overall interactive
loop shape — but the clone divergence machinery is swappable:

  clone_mode='glass'      → existing GLASS Flows inner bridge (exact
                            samples from p_{t'|t} via the inner ODE,
                            costs M_inner velocity evaluations per clone).

  clone_mode='flow_map'   → Weighted-Diamond-Maps-style amortization
                            (Holderrieth et al. 2026 §5, extended to
                            intermediate-time endpoints): one renoise
                            step + one flow-map jump per clone. Biased
                            relative to p_{t'|t} (paper-validated only
                            for endpoint t=1; the intermediate-endpoint
                            extension is the SNIS-consistent variant
                            derived from the same Bayes decomposition).
                            ~M_inner× speedup; bias invisible for
                            human-in-the-loop selection.

Lookahead is always a single flow-map call (lookahead_steps=1) because
that's exactly what the flow map was distilled for — no theoretical
risk, ~lookahead_steps× speedup. This file does NOT modify
flux_fmtt_dev.py or flux_fmtt_interactive_flowmap.py.

Time convention: FLUX uses t=1 (noise) → t=0 (clean). In WDM the
renoise direction is "back toward noise," i.e. t_RN > t_seg_start.
"""

import os

# HF cache pin — must precede diffusers/transformers import.
SCRATCH_HF_HOME = "/scratch/jerryhua/.cache/huggingface"
if os.path.exists("/scratch/jerryhua"):
    os.makedirs(SCRATCH_HF_HOME, exist_ok=True)
    os.environ["HF_HOME"] = SCRATCH_HF_HOME
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(SCRATCH_HF_HOME, "hub")

import gc
import math
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Base class lives in fluxfm_guidance/vlm. Add to sys.path so we can import it.
_FLUXFM_VLM_DIR = os.path.join(SCRIPT_DIR, "fluxfm_guidance", "vlm")
if os.path.isdir(_FLUXFM_VLM_DIR) and _FLUXFM_VLM_DIR not in sys.path:
    sys.path.insert(0, _FLUXFM_VLM_DIR)

from fluxfm_vlm_sampler import FluxFlowMapVLMSampler  # noqa: E402

# Reuse small, model-agnostic utilities. These are pure helpers, not
# interactive-loop code, so importing them does not couple us to
# flux_fmtt_dev's GENERATE path.
from flux_fmtt_dev import (  # noqa: E402
    IncrementalLayout,
    compute_aggressive_diversity_schedule,
    compute_linear_diversity_schedule,
)
from cluster_images import plot_image_cluster  # noqa: E402
from spawn import legacy as spawn_legacy  # noqa: E402


# =============================================================================
# Renoising helpers (FLUX time convention: t=1 noise, t=0 clean)
# =============================================================================

def _flux_alpha(t: float) -> float:
    """FLUX schedule: α_t = 1 - t (1 at clean, 0 at noise)."""
    return 1.0 - t


def _flux_sigma(t: float) -> float:
    """FLUX schedule: σ_t = t (0 at clean, 1 at noise)."""
    return t


def _renoise_to_more_noise(x_t: torch.Tensor, t: float, t_RN: float,
                            eps: torch.Tensor) -> torch.Tensor:
    """Forward-kernel renoising from x at time t to a noisier state at t_RN
    (t_RN > t in FLUX convention). Equation (21) of Holderrieth et al. 2026
    translated to FLUX time:

        x_{t_RN} = (α_{t_RN}/α_t)·x_t  +  √(σ²_{t_RN} − (α²_{t_RN}/α²_t)·σ²_t)·ε

    The coefficient under the square root is nonneg iff σ²_{t_RN}/α²_{t_RN}
    > σ²_t/α²_t (monotone SNR), which holds whenever t_RN > t.
    """
    if t_RN <= t + 1e-9:
        return x_t.clone()
    a_t, a_RN = _flux_alpha(t), _flux_alpha(t_RN)
    s_t, s_RN = _flux_sigma(t), _flux_sigma(t_RN)
    mean_coef = a_RN / max(a_t, 1e-12)
    var = s_RN * s_RN - (a_RN * a_RN / max(a_t * a_t, 1e-24)) * s_t * s_t
    noise_coef = math.sqrt(max(var, 0.0))
    return mean_coef * x_t + noise_coef * eps


def _rho_to_t_RN(rho: float, t_seg_start: float) -> float:
    """Map the divergence knob ρ ∈ [0, 1] to renoising depth.

    ρ = 1 → t_RN = t_seg_start (no renoising, clone equals anchor).
    ρ = 0 → t_RN = 1 (full noise, clone is essentially independent).

    Linear interpolation in t-space matches the GLASS ρ semantics: small
    ρ = more divergence, large ρ = less divergence."""
    rho = max(0.0, min(1.0, rho))
    return t_seg_start + (1.0 - rho) * (1.0 - t_seg_start)


# =============================================================================
# The interactive sampler
# =============================================================================

class FluxFMWDMInteractive(FluxFlowMapVLMSampler):
    """FLUX-FlowMap interactive sampler with a toggle for clone divergence:

    - clone_mode='glass'     : GLASS inner-ODE bridge (exact, M_inner NFE).
    - clone_mode='flow_map'  : WDM-style renoise+flow-map-jump (biased, 1 NFE).

    Lookahead at resample points is always a single flow-map call.
    """

    # ----------------------------------------------------------- DINO ----
    def _load_dino(self):
        from transformers import AutoImageProcessor, AutoModel
        dino_id = "facebook/dinov2-base"
        print(f"Loading DINOv2-base ({dino_id})…")
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_id)
        self.dino_model = AutoModel.from_pretrained(
            dino_id, low_cpu_mem_usage=False
        ).to(self.device).eval()
        print("DINOv2 loaded.")

    def _unload_dino(self):
        if hasattr(self, "dino_model") and self.dino_model is not None:
            del self.dino_model
            del self.dino_processor
            self.dino_model = None
            self.dino_processor = None
            gc.collect()
            torch.cuda.empty_cache()

    def _extract_dino_features(self, pil_images: List[Image.Image]) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for img in pil_images:
                inputs = self.dino_processor(
                    images=img, return_tensors="pt"
                ).to(self.device)
                outputs = self.dino_model(**inputs)
                embeddings.append(outputs.pooler_output.cpu().numpy())
        features = np.concatenate(embeddings, axis=0)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norms + 1e-8)

    # ----------------------------------------------------- GLASS bridge --
    def _glass_init_flux(self, X_bar: torch.Tensor, sigma_start: float,
                          sigma_end: float, rho: float,
                          master_seed=None, stage: int = 0,
                          indices=None) -> Tuple[torch.Tensor, dict]:
        """GLASS stochastic init in FLUX time (σ ≡ t, 1=noise → 0=clean).
        Identical formula to the FluxTTInteractive version in
        flux_fmtt_interactive_flowmap.py — kept local so this file is
        self-contained."""
        clip = 1e-8
        # t->0 is DEGENERATE for a GLASS bridge: bar_gamma = rho*0/sigma
        # = 0 and bar_sigma = 0*sqrt(1-rho^2) = 0, so the parent coupling
        # AND the stochastic width both vanish and every clone collapses
        # Bridge to a floor sigma; the caller finishes deterministically.
        _floor = float(os.environ.get("GLASS_SIGMA_END_MIN", "0.35"))
        if sigma_end < _floor:
            sigma_end = _floor
        alpha_start = 1.0 - sigma_start
        alpha_end = 1.0 - sigma_end

        bar_gamma = rho * sigma_end / max(sigma_start, clip)
        bar_alpha = alpha_end - bar_gamma * alpha_start
        bar_sigma = math.sqrt(max(sigma_end ** 2 * (1.0 - rho ** 2), 0.0))
        bar_sigma_0 = 1.0

        if master_seed is None:
            eps = torch.randn_like(X_bar)
        else:
            rows = indices if indices is not None else range(X_bar.shape[0])
            eps = torch.cat([
                spawn_legacy.child_noise(X_bar[r:r + 1], master_seed,
                                         stage, int(i))
                for r, i in enumerate(rows)], 0)
        X_inner = bar_gamma * X_bar + bar_sigma_0 * eps

        gp = dict(
            x_t=X_bar.clone(),
            alpha_start=alpha_start,
            sigma_start=sigma_start,
            bar_gamma=bar_gamma,
            bar_alpha=bar_alpha,
            bar_sigma=bar_sigma,
            bar_sigma_0=bar_sigma_0,
        )
        return X_inner, gp

    # --------------------------------------------- WDM clone amortization --
    def _fmrg_cfg(self):
        """FMRG toggle+params from a flag file (hot-editable, no restart):
        {"iters":2,"step":0.35,"lmbda":0.15}. Missing file = OFF."""
        import json as _j
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fmrg_flag.json")
        if not os.path.exists(p):
            return None
        try:
            return _j.load(open(p))
        except Exception:
            return None

    def _dino_feat_tensor(self, img01: torch.Tensor) -> torch.Tensor:
        """Differentiable DINO pooler feature of an image in [0,1],
        [1,3,H,W] float32. Manual resize+normalize (processor is PIL-only)."""
        import torch.nn.functional as _F
        if getattr(self, "dino_model", None) is None:
            self._load_dino()
        x = _F.interpolate(img01, size=(224, 224), mode="bilinear",
                           align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        out = self.dino_model(pixel_values=x)
        f = out.pooler_output[0]
        return f / (f.norm() + 1e-8)

    def _fmrg_repel_one(self, x_RN, t_RN, pe, ppe, gs, cfg):
        """FMRG-J greedy sibling repulsion (port of fmrg
        reward_consistency_xt): reward = negative mean cosine to the
        ALREADY-SPAWNED siblings' endpoint features; lmbda anchors to the
        renoised state so parent-kinship is preserved. First child of a
        brood passes through untouched (nothing to repel from)."""
        import torch.nn.functional as _F
        feats = getattr(self, "_fmrg_feats", [])
        iters = int(cfg.get("iters", 2))
        step = float(cfg.get("step", 0.35))
        lam = float(cfg.get("lmbda", 0.15))
        x0_anchor = x_RN.detach().clone()
        x = x_RN.detach().clone()
        try:
            for _ in range(iters if feats else 0):
                with torch.enable_grad():
                    xl = x.clone().requires_grad_(True)
                    u = self.predict_vector(
                        xl, t_RN, prompt_embeds=pe[:1],
                        pooled_prompt_embeds=ppe[:1],
                        guidance_scale=gs, t_next=0.0)
                    z0 = xl - t_RN * u
                    img01 = ((self.decode(z0).float() + 1) / 2).clamp(0, 1)
                    f = self._dino_feat_tensor(img01)
                    sib = torch.stack([t.to(f.device) for t in feats])
                    loss = (f @ sib.T).mean() \
                        + lam * _F.mse_loss(xl.float(), x0_anchor.float())
                    g = torch.autograd.grad(loss, xl)[0]
                gn = g.norm()
                x = (x - step * g / (gn + 1e-8) * x.norm()
                     / x.numel() ** 0.5).detach().to(x_RN.dtype)
                del xl, u, z0, img01, f, g
                torch.cuda.empty_cache()
            # cache this child's endpoint feature (detached, 1 extra NFE)
            with torch.no_grad():
                u = self.predict_vector(
                    x, t_RN, prompt_embeds=pe[:1],
                    pooled_prompt_embeds=ppe[:1],
                    guidance_scale=gs, t_next=0.0)
                img01 = ((self.decode(x - t_RN * u).float() + 1)
                         / 2).clamp(0, 1)
                self._fmrg_feats = feats + [
                    self._dino_feat_tensor(img01).detach()]
            print(f"[fmrg] child {len(self._fmrg_feats)}: repelled from "
                  f"{len(feats)} sibs", flush=True)
            return x
        except Exception as _fe:
            print(f"[fmrg] repel failed ({type(_fe).__name__}: {_fe}); "
                  f"plain spawn", flush=True)
            torch.cuda.empty_cache()
            return x_RN

    def _wdm_clone_jump(self, x_anchor_seg_start: torch.Tensor,
                       t_seg_start: float, t_seg_end: float,
                       t_RN: float, prompt_embeds: torch.Tensor,
                       pooled_prompt_embeds: torch.Tensor,
                       guidance_scale: float,
                       master_seed=None, stage: int = 0,
                       child_index: int = 0) -> torch.Tensor:
        """One WDM clone jump:
          1. ε ~ N(0, I), renoise anchor from t_seg_start to noisier t_RN.
          2. Flow-map jump from t_RN to t_seg_end in one NFE.

        Returns the clone's state at t_seg_end. Stochasticity comes
        entirely from ε in the renoising step; the flow-map call itself
        is deterministic given input."""
        # Seeded from the master seed and the child's position, so a
        # brood replays regardless of what else the process drew first.
        eps = spawn_legacy.child_noise(x_anchor_seg_start, master_seed,
                                       stage, child_index)
        # boundary state upward; "lookahead" rebuilds the parent's clean
        # endpoint estimate and re-noises THAT to FLUXFM_LOOKAHEAD_TAU
        # (SDEdit on the predicted image — composition-preserving).
        _spawn_from = os.environ.get("FLUXFM_SPAWN_FROM", "xt")
        if _spawn_from != "lookahead" and t_RN > t_seg_start + 1e-6:
            # GLOBAL band: renoise the boundary state upward (structural
            # variation — the standard WDM hop).
            x_RN = _renoise_to_more_noise(
                x_anchor_seg_start, t_seg_start, t_RN, eps)
            _cfg = self._fmrg_cfg()
            if _cfg is not None:
                x_RN = self._fmrg_repel_one(
                    x_RN, t_RN, prompt_embeds, pooled_prompt_embeds,
                    guidance_scale, _cfg)
        else:
            # diversity — texture/detail variation, structure kept).
            # Boundary states sit at high sigma, so the local band
            # (t_RN < t_seg_start) is unreachable from them: re-anchor
            # from the parent's CLEAN endpoint estimate (one map call),
            # renoise THAT into the low-sigma band that sets detail,
            # then rejoin the trajectory below.
            v0 = self.predict_vector(
                x_anchor_seg_start, t_seg_start,
                prompt_embeds=prompt_embeds[:1],
                pooled_prompt_embeds=pooled_prompt_embeds[:1],
                guidance_scale=guidance_scale, t_next=0.0)
            z0p = x_anchor_seg_start - t_seg_start * v0
            # Exact interpolant renoise of a clean state to level t_RN
            # (alpha = 1 - t, sigma = t in FLUX convention).
            _lt = float(os.environ.get("FLUXFM_LOOKAHEAD_TAU", "0.75")) \
                if _spawn_from == "lookahead" else t_RN
            x_RN = (1.0 - _lt) * z0p + _lt * eps
            t_RN = _lt
            print(f"[wdm-local] endpoint re-anchor: t_RN={t_RN:.2f} "
                  f"(< seg {t_seg_start:.2f})", flush=True)
        if t_seg_end <= t_RN + 1e-6:
            # map call from t~1 to the segment end is a 1-NFE generation
            # while the parent gets 28 steps; lookahead_steps never
            # reaches this path. Chain N jumps.
            _ns = spawn_legacy.clone_steps()
            if _ns > 1:
                return spawn_legacy.descend_chain(
                    x_RN, t_RN, t_seg_end, _ns,
                    lambda _x, _t, _tn: self.predict_vector(
                        _x, _t,
                        prompt_embeds=prompt_embeds[:1],
                        pooled_prompt_embeds=pooled_prompt_embeds[:1],
                        guidance_scale=guidance_scale,
                        t_next=_tn,
                    ))
            v = self.predict_vector(
                x_RN, t_RN,
                prompt_embeds=prompt_embeds[:1],
                pooled_prompt_embeds=pooled_prompt_embeds[:1],
                guidance_scale=guidance_scale,
                t_next=t_seg_end,
            )
            # Flow-map "velocity" v at (t_RN, t_seg_end) integrates the
            # entire jump in one step: x_end = x_RN + (t_seg_end - t_RN)·v.
            return x_RN + (t_seg_end - t_RN) * v
        # Rare: rejoin level is noisier than the local band — exact
        # forward renoise up to it (no extra NFE).
        return _renoise_to_more_noise(
            x_RN, t_RN, t_seg_end,
            spawn_legacy.child_noise(x_RN, master_seed, stage + 1000,
                                     child_index))

    # ------------------------------------------------- predict_velocity shim --
    def predict_velocity(self, z: torch.Tensor, t_cur: float,
                          guidance_scale: float = 3.5) -> torch.Tensor:
        """Compatibility shim for callers (e.g. explore_map/precompute.py)
        that were written against base FLUX's `predict_velocity(z, t, guidance)`
        interface. Maps to the flow-map `predict_vector` with a tiny
        downstream timestep — close enough to instantaneous to be a drop-in
        for GLASS-bridge clone math, but avoids the degenerate t_next=t_cur
        case where the dual-timestep LoRA was never trained."""
        eps = 0.01
        t_next = max(t_cur - eps, 0.0)
        return self.predict_vector(
            z, t_cur,
            prompt_embeds=self.cached_prompt_embeds[:1],
            pooled_prompt_embeds=self.cached_pooled_prompt_embeds[:1],
            guidance_scale=guidance_scale,
            t_next=t_next,
        )

    # ------------------------------------------------------- Lookahead --
    def flowmap_lookahead(self, z: torch.Tensor, t_cur: float,
                          prompt_embeds: torch.Tensor,
                          pooled_prompt_embeds: torch.Tensor,
                          guidance_scale: float) -> torch.Tensor:
        """Single-step flow-map lookahead from t_cur to t=0. Deterministic,
        1 NFE. This is the *standard* use case the flow map was distilled
        for — no extrapolation."""
        # sharp). N chained (t -> t') map calls trade N NFEs (~0.15s each)
        # for visibly crisper decodes than the single full-range jump.
        # FLUXFM_LOOKAHEAD_STEPS=1 restores the one-jump behavior.
        n_steps = getattr(self, "_la_steps", None) or max(
            1, int(os.environ.get("FLUXFM_LOOKAHEAD_STEPS", "3")))
        z_cur, t = z, float(t_cur)
        for i in range(n_steps):
            t_next = t * (1.0 - (i + 1) / n_steps) if i < n_steps - 1 else 0.0
            u = self.predict_vector(
                z_cur, t,
                prompt_embeds=prompt_embeds[:1],
                pooled_prompt_embeds=pooled_prompt_embeds[:1],
                guidance_scale=guidance_scale,
                t_next=t_next,
            )
            z_cur = z_cur + (t_next - t) * u
            t = t_next
        return z_cur

    # ------------------------------------------------------- Utilities --
    @staticmethod
    def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
        arr = (img_tensor[0].permute(1, 2, 0).cpu().float().numpy() * 255
              ).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _save_grid(pil_images: List[Image.Image], save_path: str,
                   labels: List[str] = None):
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
        print("\n" + "=" * 60)
        print(f"  INTERACTIVE PRUNING — Interval {interval_idx}")
        print("=" * 60)
        while True:
            raw = input(f"  Keep indices (0-{total-1}), comma-sep: ").strip()
            if not raw:
                continue
            try:
                indices = sorted(set(int(x.strip()) for x in raw.split(",")))
                if not indices or any(i < 0 or i >= total for i in indices):
                    print(f"  Invalid. Must be in 0-{total-1}.")
                    continue
                return indices
            except ValueError:
                print("  Invalid input.")

    # ============================================== Interactive sampler --
    def sample_interactive(self,
                           num_steps: int = 28,
                           num_particles: int = 4,
                           num_clones: int = 2,
                           num_resampling_steps: int = 4,
                           guidance_scale: float = 3.5,
                           rho: float = 0.4,
                           lookahead_steps: int = 3,
                           schedule_mode: Optional[str] = "linear",
                           img_shape: Optional[Tuple[int, int]] = None,
                           seed: Optional[int] = None,
                           output_dir: str = "interactive_fmwdm",
                           clone_mode: str = "glass",
                           progress_callback=None) -> torch.Tensor:
        """clone_mode ∈ {'glass', 'flow_map'}. Lookahead is always 1 NFE."""
        assert clone_mode in ("glass", "flow_map"), \
            f"clone_mode must be 'glass' or 'flow_map', got {clone_mode!r}"

        from diffusers.pipelines.flux.pipeline_flux import (
            calculate_shift, retrieve_timesteps,
        )

        imgH, imgW = img_shape if img_shape is not None else (512, 512)
        device = self.device
        os.makedirs(output_dir, exist_ok=True)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        n_segments_total = num_resampling_steps + 1
        if progress_callback:
            progress_callback(0.02, "Loading image similarity model...",
                              0, n_segments_total)
        self._load_dino()
        if progress_callback:
            progress_callback(0.05, "Preparing...", 0, n_segments_total)

        latent_h = 2 * (int(imgH) // (self.vae_scale_factor * 2))
        latent_w = 2 * (int(imgW) // (self.vae_scale_factor * 2))

        prompt_embeds = self.cached_prompt_embeds
        pooled_prompt_embeds = self.cached_pooled_prompt_embeds

        # tree ablation: master_seed was accepted but ignored, so fluxfm
        # sessions could not be parent-matched or replayed). Round-1
        # anchors are now deterministic given seed; spawn noise remains
        # stochastic (full breed replay = future work, recorded in STATE).
        if seed is not None:
            _gen = torch.Generator(device="cpu").manual_seed(int(seed))
            anchors = torch.randn(num_particles, 16, latent_h, latent_w,
                                  generator=_gen).to(device, self.dtype)
            print(f"[fluxfm] round-1 anchors seeded ({int(seed)})",
                  flush=True)
        else:
            anchors = torch.randn(num_particles, 16, latent_h, latent_w,
                                  device=device, dtype=self.dtype)
        total_particles = num_particles
        C = num_clones

        # FLUX shifted schedule (same as flux_fmtt_dev / flux_fmtt_interactive_flowmap).
        image_seq_len = (latent_h // 2) * (latent_w // 2)
        mu = calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.get("base_image_seq_len", 256),
            self.pipeline.scheduler.config.get("max_image_seq_len", 4096),
            self.pipeline.scheduler.config.get("base_shift", 0.5),
            self.pipeline.scheduler.config.get("max_shift", 1.15),
        )
        sigmas_arr = np.linspace(1.0, 1.0 / num_steps, num_steps)
        timesteps, _ = retrieve_timesteps(
            self.pipeline.scheduler, num_steps, device,
            sigmas=sigmas_arr, mu=mu,
        )
        time_steps = torch.cat([timesteps / 1000.0,
                                 torch.zeros(1, device=device)])

        # Resampling schedule.
        # diversity should emerge AUTOMATICALLY from spawning at t closer
        # to 0 in later stages. The aggressive/linear schedules pack all
        # boundaries into the first 40% of steps (parents always at high
        # sigma -> local band unreachable). For flow_map clones, spread
        # boundaries across ~[15%, 80%] of the trajectory instead: last
        # spawn happens from a mostly-resolved parent = naturally local.
        # FLUXFM_SPREAD_BOUNDARIES=0 restores the packed schedule.
        if (num_resampling_steps > 0
                and os.environ.get("FLUXFM_FRONTLOAD", "1") == "1"):
            # decay REQUIRES front-loading, because measured D(tau) is
            # convex near noise). Stage i targets a linearly-decaying D,
            # inverted through the measured curve to a sigma, then snapped
            # to the nearest schedule step. Applies to ALL clone modes
            _D_pts = [(0.116, 0.5), (0.153, 0.7), (0.173, 0.9),
                      (0.261, 1.0)]
            _d_hi = float(os.environ.get("FLUXFM_D_HI", "0.24"))
            _d_lo = float(os.environ.get("FLUXFM_D_LO", "0.12"))
            _sig = [time_steps[i].item() for i in range(num_steps)]
            _st = []
            for _i in range(num_resampling_steps):
                _f = (_i / max(num_resampling_steps - 1, 1)
                      if num_resampling_steps > 1 else 0.0)
                _d = _d_hi - (_d_hi - _d_lo) * _f
                _tau = _D_pts[-1][1]
                for (_d0, _t0), (_d1, _t1) in zip(_D_pts, _D_pts[1:]):
                    if _d0 <= _d <= _d1:
                        _tau = _t0 + (_t1 - _t0) * (_d - _d0) / max(
                            _d1 - _d0, 1e-6)
                        break
                else:
                    _tau = (_D_pts[0][1] if _d < _D_pts[0][0]
                            else _D_pts[-1][1])
                _st.append(min(range(1, num_steps),
                               key=lambda k: abs(_sig[k] - _tau)))
            resample_steps = sorted(set(_st))
            print(f"[frontload] boundaries={resample_steps} sigmas="
                  f"{[round(_sig[k], 3) for k in resample_steps]}",
                  flush=True)
        elif (clone_mode == "flow_map" and num_resampling_steps > 0
                and os.environ.get("FLUXFM_SPREAD_BOUNDARIES", "1") == "1"):
            _lo, _hi = 0.15, 0.80
            resample_steps = sorted({int(round(num_steps * (
                _lo + (_hi - _lo) * i / max(num_resampling_steps - 1, 1))))
                for i in range(num_resampling_steps)})
        elif schedule_mode == "aggressive" and num_resampling_steps > 0:
            resample_steps = compute_aggressive_diversity_schedule(
                num_steps, num_resampling_steps)
        elif schedule_mode == "linear" and num_resampling_steps > 0:
            resample_steps = compute_linear_diversity_schedule(
                num_steps, num_resampling_steps)
        elif num_resampling_steps > 0:
            resample_steps = sorted(
                int(round((i + 1) * num_steps / (num_resampling_steps + 1)))
                for i in range(num_resampling_steps)
            )
        else:
            resample_steps = []
        resample_at = set(resample_steps)
        segment_bounds = [0] + resample_steps + [num_steps]

        X_all = anchors.clone()
        is_anchor = torch.ones(total_particles, dtype=torch.bool)
        glass_params = None
        # For flow_map clones: once we leap to segment end, mark them
        # "done" so the inner loop doesn't touch them again.
        wdm_clone_done = torch.zeros(total_particles, dtype=torch.bool)
        interval_count = 0
        # prompt; at spawn each clone gets an LLM-written variation of its
        # PARENT'S prompt (hierarchical: grandchildren mutate the child's
        # phrasing). Fixes children-too-similar: shared conditioning pulls
        # all clones into one semantic basin regardless of renoise depth.
        # FLUXFM_PROMPT_TREE=0 disables (single shared prompt).
        self._la_steps = max(1, int(lookahead_steps or 3))
        self._ptree_on = os.environ.get("FLUXFM_PROMPT_TREE", "1") == "1"
        self._p_prompts = [self.prompt] * num_particles
        self._p_embeds = {}   # idx -> (pe, ppe); absent = base embeds

        def _tree_embeds(idx):
            e = self._p_embeds.get(int(idx))
            return e if e is not None else (prompt_embeds,
                                            pooled_prompt_embeds)

        # Pool-based spawn: encoders are freed after init, so children draw
        # pre-encoded variations from self.prompt_cache (populated via the
        # additional_prompts pool at construction). No runtime encoding.
        self._ptree_pool = [p for p in getattr(self, "_ptree_pool", [])
                            if p in getattr(self, "prompt_cache", {})]
        self._ptree_next = 0

        def _tree_spawn(parent_idx, child_idx, mutate_frac=1.0,
                        brood_pos=0.0, force_base=False):
            if force_base:
                # on the artist's EXACT prompt — no LLM augmentation —
                # so pure-renoise diversity is always visible in-session.
                self._p_prompts[int(child_idx)] = self.prompt
                self._p_embeds.pop(int(child_idx), None)
                print(f"[ptree] {int(parent_idx)}->{int(child_idx)}: "
                      f"BASE (unaugmented)", flush=True)
                return "base"
            if not self._ptree_on or not self._ptree_pool:
                return
            # stages, and the tree dominates early diversity). The first
            # mutate_frac of each brood gets a fresh reading; the rest
            # inherit the parent's prompt verbatim. Stage 1: all mutate;
            # final stage: none (pure refinement, subject locked).
            if brood_pos >= mutate_frac:
                self._p_prompts[int(child_idx)] = \
                    self._p_prompts[int(parent_idx)]
                _pe_par = self._p_embeds.get(int(parent_idx))
                if _pe_par is not None:
                    self._p_embeds[int(child_idx)] = _pe_par
                else:
                    self._p_embeds.pop(int(child_idx), None)
                print(f"[ptree] {int(parent_idx)}->{int(child_idx)}: "
                      f"inherit", flush=True)
                return "inherit"
            try:
                # Stage-routed vocabulary: deep first-stage spawns take
                # WILD readings (scene words act at high sigma); later
                # shallow spawns take RESTYLE readings (medium/palette
                # words act at mid sigma). Falls back to the flat pool.
                _pool = self._ptree_pool
                if interval_count == 1:
                    _pool = getattr(self, "_ptree_pool_wild", None) or _pool
                else:
                    _pool = getattr(self, "_ptree_pool_restyle", None) \
                        or _pool
                new = _pool[self._ptree_next % len(_pool)]
                self._ptree_next += 1
                self._p_prompts[int(child_idx)] = new
                pe_c, ppe_c = self.prompt_cache[new]
                dev = prompt_embeds.device
                self._p_embeds[int(child_idx)] = (
                    pe_c.to(dev, prompt_embeds.dtype),
                    ppe_c.to(dev, pooled_prompt_embeds.dtype))
                print(f"[ptree] {int(parent_idx)}->{int(child_idx)}: "
                      f"{new[:70]!r}", flush=True)
                return "mutated"
            except Exception as _pe:
                print(f"[ptree] spawn failed ({_pe}); child inherits "
                      f"parent prompt verbatim", flush=True)
            return None
        inc_layout = IncrementalLayout(n_neighbors=15, min_dist=0.2)
        prev_kept_indices = None

        pbar = tqdm(range(num_steps), total=num_steps,
                    desc=f"FluxFM-WDM({clone_mode})")

        for k in pbar:
            if progress_callback:
                seg_idx = 0
                for b in resample_steps:
                    if k >= b:
                        seg_idx += 1
                seg_start_step = segment_bounds[seg_idx]
                seg_end_step_pb = segment_bounds[seg_idx + 1]
                seg_len = max(seg_end_step_pb - seg_start_step, 1)
                frac = 0.05 + 0.35 * ((k - seg_start_step) / seg_len)
                progress_callback(
                    frac,
                    f"Generating images — step {k+1-seg_start_step}/{seg_len}",
                    seg_idx, n_segments_total,
                )

            t_cur = time_steps[k].item()
            t_next = time_steps[k + 1].item()
            dt = t_next - t_cur  # negative; FLUX t descends

            # ===== Segment start: init clones =====
            if glass_params is None:
                seg_end_step = next(
                    (b for b in segment_bounds if b > k), num_steps)
                sigma_seg_start = time_steps[k].item()
                sigma_seg_end = time_steps[seg_end_step].item()
                M_inner = seg_end_step - k

                clone_indices = (~is_anchor).nonzero(as_tuple=True)[0]
                wdm_clone_done[:] = False

                if len(clone_indices) > 0:
                    parent_states = torch.zeros(
                        len(clone_indices), 16, latent_h, latent_w,
                        device=device, dtype=self.dtype)
                    for ci, idx in enumerate(clone_indices):
                        group = idx.item() // C
                        parent_states[ci] = X_all[group * C]

                    _fcfg = self._fmrg_cfg()
                    if _fcfg is not None and clone_mode == "flow_map":
                        # paper — guidance rides the DETERMINISTIC solve;
                        # no stochastic transition needed; grad normalized
                        # to ||u||). Clones start as EXACT parent copies
                        # and diverge via repulsion-guided velocities over
                        # the first K steps of the segment.
                        X_all[clone_indices] = parent_states
                        glass_params = {
                            "mode": "fmrg",
                            "clone_indices": clone_indices,
                            "seg_start_step": k,
                            "M_inner": M_inner,
                            "K": int(_fcfg.get("steps", 4)),
                            "step_size": float(_fcfg.get("step", 0.2)),
                        }
                        for ci, idx in enumerate(clone_indices):
                            _tree_spawn(
                                (int(idx) // C) * C, idx,
                                mutate_frac=1.0 - _prog
                                if "_prog" in dir() else 1.0,
                                brood_pos=ci / max(len(clone_indices), 1),
                                force_base=(
                                    ci == len(clone_indices) - 1))
                        print(f"[fmrg2] {len(clone_indices)} clones start "
                              f"as parent copies; guided K="
                              f"{glass_params['K']} steps", flush=True)
                    elif clone_mode == "glass":
                        clone_inner, gp = self._glass_init_flux(
                            parent_states, sigma_seg_start,
                            sigma_seg_end, rho,
                            master_seed=seed, stage=k,
                            indices=[int(i) for i in clone_indices])
                        X_all[clone_indices] = clone_inner
                        glass_params = gp
                        glass_params["seg_start_step"] = k
                        _gf = float(os.environ.get(
                            "GLASS_SIGMA_END_MIN", "0.35"))
                        _bend = next(
                            (bb for bb in range(k + 1, num_steps)
                             if time_steps[bb].item() <= _gf), num_steps)
                        glass_params["M_inner"] = max(_bend - k, 1)
                        print(f"[glass] bridge sigma "
                              f"{time_steps[k].item():.3f} -> "
                              f"{time_steps[min(_bend, num_steps-1)].item():.3f}"
                              f" over {max(_bend-k,1)} steps", flush=True)
                        glass_params["clone_indices"] = clone_indices
                    else:  # 'flow_map' — WDM amortization
                        # diversity at the beginning layers). Mirrors the
                        # krea2 rho schedule: first breed stage runs at
                        # 0.5x rho (deep renoise, wide variation), ramping
                        # to ~1.3x by the final stage (tight refinement).
                        # FLUXFM_RHO_SCHEDULE=0 restores flat rho.
                        _rho_eff = rho
                        if os.environ.get("FLUXFM_RHO_SCHEDULE", "1") == "1":
                            # diversity should decrease linearly per stage,
                            # and measured D(tau)=1-DINO_sim is CONVEX near
                            # noise: D(.5)=.116 D(.7)=.153 D(.9)=.173
                            # D(1)=.261 - so tau must be FRONT-LOADED).
                            # Stages are spaced linearly in D between D_HI
                            # and D_LO, then mapped through the inverted
                            # measured curve to a renoise depth tau, then to
                            # the rho the kernel expects (t_RN linear in
                            # rho). Curve is n=1-parent preliminary; knobs:
                            # FLUXFM_D_HI / FLUXFM_D_LO.
                            _D_pts = [(0.116, 0.5), (0.153, 0.7),
                                      (0.173, 0.9), (0.261, 1.0)]
                            _d_hi = float(os.environ.get(
                                "FLUXFM_D_HI", "0.24"))
                            _d_lo = float(os.environ.get(
                                "FLUXFM_D_LO", "0.10"))
                            _stages = max(num_resampling_steps - 1, 1)
                            _prog = min(max(interval_count - 1, 0),
                                        _stages) / _stages
                            _d_tgt = _d_hi - (_d_hi - _d_lo) * _prog
                            _tau = _D_pts[0][1]
                            for (_da, _ta), (_db, _tb) in zip(
                                    _D_pts, _D_pts[1:]):
                                if _d_tgt >= _da:
                                    _f = ((_d_tgt - _da)
                                          / max(_db - _da, 1e-6))
                                    _tau = _ta + min(max(_f, 0.0), 1.0)                                         * (_tb - _ta)
                            # tau -> rho: t_RN = t_seg+(1-rho)(1-t_seg)
                            # and tau plays t_RN at t_seg~=sigma_seg_start.
                            _rho_eff = min(max(
                                1.0 - (_tau - sigma_seg_start)
                                / max(1.0 - sigma_seg_start, 1e-6),
                                0.02), 0.9)
                            print(f"[fluxfm D-schedule] stage="
                                  f"{interval_count} D_tgt={_d_tgt:.3f} "
                                  f"tau={_tau:.3f} rho_eff={_rho_eff:.2f}",
                                  flush=True)
                            print(f"[fluxfm rho-schedule] stage="
                                  f"{interval_count} rho {rho:.2f}->"
                                  f"{_rho_eff:.2f}", flush=True)
                        t_RN = _rho_to_t_RN(_rho_eff, sigma_seg_start)
                        with torch.no_grad():
                            for ci, idx in enumerate(clone_indices):
                                _kind = _tree_spawn(
                                    (int(idx) // C) * C, idx,
                                    mutate_frac=1.0 - _prog,
                                    brood_pos=ci / max(
                                        len(clone_indices), 1),
                                    force_base=(
                                        ci == len(clone_indices) - 1))
                                # be valid per GLASS / Weighted-Diamond-Maps).
                                # WDM hop form: t_RN = t + rho*(1-t), rho in
                                # (0,1) the stochasticity dial. rho laddered
                                # across the brood for spread. This keeps
                                # x_t load-bearing (rho<1) and supersedes the
                                # D-schedule absolute depths, the deep clamp,
                                # and the endpoint re-anchor (t_RN >= t_seg
                                # always, so that branch is dead).
                                # Ladder centers on the REQUEST rho (the
                                # kinship dial), +/-0.2, env overridable.
                                _rho_lo = float(os.environ.get(
                                    "FLUXFM_RHO_LO",
                                    str(max(0.05, rho - 0.2))))
                                _rho_hi = float(os.environ.get(
                                    "FLUXFM_RHO_HI",
                                    str(min(0.97, rho + 0.2))))
                                _fr = ci / max(len(clone_indices) - 1, 1)
                                _rho_c = _rho_lo + (_rho_hi - _rho_lo) * _fr
                                _t_RN_c = (sigma_seg_start
                                           + _rho_c * (1.0 - sigma_seg_start))
                                _pe_c, _ppe_c = _tree_embeds(idx)
                                # "anything to make them more different").
                                # Low guidance samples a BROADER region of
                                # p(x|prompt) than the prompt-typical mode,
                                # so laddering CFG across the brood widens
                                # the basin the children are drawn from.
                                # FLUXFM_CFG_LO/HI; equal values = off.
                                _gl = float(os.environ.get(
                                    "FLUXFM_CFG_LO", str(guidance_scale)))
                                _gh = float(os.environ.get(
                                    "FLUXFM_CFG_HI", str(guidance_scale)))
                                _gs_c = _gl + (_gh - _gl) * (
                                    ci / max(len(clone_indices) - 1, 1))
                                x_end = self._wdm_clone_jump(
                                    parent_states[ci:ci+1],
                                    sigma_seg_start, sigma_seg_end, _t_RN_c,
                                    _pe_c, _ppe_c,
                                    _gs_c,
                                    master_seed=seed, stage=k,
                                    child_index=int(idx),
                                )
                                X_all[idx:idx+1] = x_end
                                wdm_clone_done[idx] = True
                        # Empty glass_params — clones won't be touched
                        # again in this segment.
                        glass_params = {
                            "seg_start_step": k,
                            "M_inner": M_inner,
                            "clone_indices": torch.tensor(
                                [], dtype=torch.long),
                            "mode": "flow_map",
                        }
                else:
                    glass_params = {"seg_start_step": k,
                                    "M_inner": M_inner,
                                    "clone_indices": torch.tensor(
                                        [], dtype=torch.long)}

            # ===== Anchor step (always Euler with flow-map velocity) =====
            with torch.no_grad():
                anchor_indices = is_anchor.nonzero(as_tuple=True)[0]
                for ai in anchor_indices:
                    z_p = X_all[ai:ai+1]
                    v = self.predict_vector(
                        z_p, t_cur,
                        prompt_embeds=prompt_embeds[:1],
                        pooled_prompt_embeds=pooled_prompt_embeds[:1],
                        guidance_scale=guidance_scale,
                        t_next=t_next,
                    )
                    X_all[ai:ai+1] = z_p + dt * v

            # ===== Clone step =====
            # In 'glass' mode: GLASS inner-ODE bridge step (M_inner times
            # per segment, matches flux_fmtt_interactive_flowmap).
            # In 'flow_map' mode: clones already at segment-end after the
            # init jump → nothing to do here.
            clone_indices = glass_params["clone_indices"]
            if (len(clone_indices) > 0
                    and glass_params.get("mode") == "fmrg"):
                _m = k - glass_params["seg_start_step"]
                _K = glass_params["K"]
                _ss = glass_params["step_size"]
                _guided = _m < _K
                _feats_det = []
                if _guided:
                    with torch.no_grad():
                        for idx in clone_indices:
                            _pe_c, _ppe_c = _tree_embeds(idx)
                            _z = X_all[idx:idx+1]
                            _u0 = self.predict_vector(
                                _z, t_cur, prompt_embeds=_pe_c,
                                pooled_prompt_embeds=_ppe_c,
                                guidance_scale=guidance_scale, t_next=0.0)
                            _img = ((self.decode(
                                _z - t_cur * _u0).float() + 1) / 2
                                ).clamp(0, 1)
                            _feats_det.append(
                                self._dino_feat_tensor(_img).detach())
                for ci, idx in enumerate(clone_indices):
                    _pe_c, _ppe_c = _tree_embeds(idx)
                    _z = X_all[idx:idx+1]
                    _u = self.predict_vector(
                        _z, t_cur, prompt_embeds=_pe_c,
                        pooled_prompt_embeds=_ppe_c,
                        guidance_scale=guidance_scale, t_next=t_next)
                    if _guided and len(clone_indices) > 1:
                        try:
                            with torch.enable_grad():
                                _zl = _z.detach().clone().requires_grad_(
                                    True)
                                _ug = self.predict_vector(
                                    _zl, t_cur, prompt_embeds=_pe_c,
                                    pooled_prompt_embeds=_ppe_c,
                                    guidance_scale=guidance_scale,
                                    t_next=0.0)
                                _img = ((self.decode(
                                    _zl - t_cur * _ug).float() + 1) / 2
                                    ).clamp(0, 1)
                                _f = self._dino_feat_tensor(_img)
                                _sib = torch.stack(
                                    [f for j, f in enumerate(_feats_det)
                                     if j != ci])
                                _loss = (_f @ _sib.T).mean()
                                _g = torch.autograd.grad(_loss, _zl)[0]
                            _un = torch.linalg.vector_norm(_u.detach())
                            _gh = _g / (torch.linalg.vector_norm(_g)
                                        + 1e-8) * _un
                            _u = _u.detach() + _ss * _gh.to(_u.dtype)
                            del _zl, _ug, _img, _f, _g
                        except Exception as _fe:
                            print(f"[fmrg2] guide fail "
                                  f"({type(_fe).__name__}); plain step",
                                  flush=True)
                            torch.cuda.empty_cache()
                    X_all[idx:idx+1] = _z + dt * _u
            elif (len(clone_indices) > 0 and clone_mode == "glass"
                  and (k - glass_params["seg_start_step"])
                  >= glass_params["M_inner"]):
                # Bridge done at the floor; finish the trajectory with the
                # ordinary deterministic solve.
                for idx in clone_indices:
                    _pe_c, _ppe_c = _tree_embeds(idx)
                    _z = X_all[idx:idx+1]
                    _u = self.predict_vector(
                        _z, t_cur, prompt_embeds=_pe_c,
                        pooled_prompt_embeds=_ppe_c,
                        guidance_scale=guidance_scale, t_next=t_next)
                    X_all[idx:idx+1] = _z + dt * _u
            elif len(clone_indices) > 0 and clone_mode == "glass":
                m = k - glass_params["seg_start_step"]
                M = glass_params["M_inner"]
                s = m / M
                ds = 1.0 / M
                gp = glass_params
                x_t_bar = gp["x_t"]
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
                bproduct = max(
                    mu1*mu1*inv11 + 2*mu1*mu2*inv12 + mu2*mu2*inv22, clip)
                t_star = max(
                    min(1.0/(1.0+math.sqrt(max(1.0/bproduct, 0.0))), 0.999),
                    0.001)
                sigma_star = 1.0 - t_star
                alpha_star = t_star
                w_xt = alpha_star * (mu1*inv11 + mu2*inv12) / bproduct
                w_xs = alpha_star * (mu1*inv12 + mu2*inv22) / bproduct

                with torch.no_grad():
                    for ci, idx in enumerate(clone_indices):
                        x_parent = x_t_bar[ci:ci+1]
                        x_clone = X_all[idx:idx+1]
                        S_input = w_xt * x_parent + w_xs * x_clone
                        v_star = self.predict_vector(
                            S_input, sigma_star,
                            prompt_embeds=prompt_embeds[:1],
                            pooled_prompt_embeds=pooled_prompt_embeds[:1],
                            guidance_scale=guidance_scale,
                            # NOT t_next=sigma_star: t_next==t_cur is the
                            # degenerate zero-length jump for a 2-timestep
                            # flow map (base FLUX/Krea ignore t_next, so
                            # the inherited GLASS code was fine there).
                            t_next=max(sigma_star - 1e-2, 0.0),
                        )
                        denoiser = S_input - sigma_star * v_star
                        velocity = (w1 * x_clone + w2 * denoiser
                                    + w3 * x_parent)
                        X_all[idx:idx+1] = x_clone + ds * velocity

            # ===== Resample interval =====
            if k in resample_at:
                glass_params = None
                interval_count += 1
                interval_dir = os.path.join(
                    output_dir, f"interval_{interval_count}")
                os.makedirs(interval_dir, exist_ok=True)
                t_now = t_next
                # self-labeling — parent/child roles, stage, noise level,
                # per-tile prompt). Written next to the tiles so any
                # renderer can caption without log archaeology.
                try:
                    import json as _bj
                    _meta = {
                        "stage": interval_count,
                        "t_now": float(t_now),
                        "spawn_t_RN": float(locals().get("t_RN") or 0.0)
                        if interval_count > 1 else None,
                        "particles": [
                            {"idx": int(_pi),
                             "role": ("anchor" if bool(is_anchor[_pi])
                                      else "child"),
                             "prompt": self._p_prompts[int(_pi)][:120]}
                            for _pi in range(total_particles)],
                    }
                    with open(os.path.join(interval_dir, "meta.json"),
                              "w") as _bf:
                        _bj.dump(_meta, _bf, indent=1)
                except Exception as _me:
                    print(f"[board-meta] write failed: {_me}", flush=True)
                seg_idx_done = interval_count - 1

                labels = []
                anchor_group = -1
                for i in range(total_particles):
                    if is_anchor[i]:
                        anchor_group += 1
                        labels.append(f"{i} (anchor g{anchor_group})")
                    else:
                        labels.append(f"{i} (clone g{anchor_group})")

                pil_images = []
                with torch.no_grad():
                    for p_idx in range(total_particles):
                        if progress_callback:
                            dec_frac = 0.40 + 0.50 * (
                                p_idx / total_particles)
                            progress_callback(
                                dec_frac,
                                f"Decoding image {p_idx+1}/{total_particles}…",
                                seg_idx_done, n_segments_total,
                            )
                        _pe_i, _ppe_i = _tree_embeds(p_idx)
                        z0 = self.flowmap_lookahead(
                            X_all[p_idx:p_idx+1], t_now,
                            _pe_i, _ppe_i,
                            guidance_scale,
                        )
                        img_t = self.decode(z0)
                        pil_img = self._tensor_to_pil(img_t)
                        pil_img.save(os.path.join(
                            interval_dir, f"particle_{p_idx:03d}.png"))
                        pil_images.append(pil_img)

                if progress_callback:
                    progress_callback(
                        0.95, "Clustering images…",
                        seg_idx_done, n_segments_total,
                    )

                features = self._extract_dino_features(pil_images)
                if inc_layout is not None:
                    if interval_count == 1:
                        inc_layout.initialize(features, pil_images)
                    else:
                        inc_layout.update(
                            prev_kept_indices, features, pil_images,
                            num_clones=C, round_num=interval_count)
                    inc_layout.plot(
                        title=f"Interval {interval_count} (k={k}, t={t_now:.3f})",
                        save_path=os.path.join(
                            interval_dir, "layout_incremental.png"),
                        thumb_size=80,
                    )

                cluster_path = os.path.join(
                    interval_dir, "cluster_dinov2.png")
                plot_image_cluster(
                    features, pil_images,
                    title=f"DINOv2 Cluster — Interval {interval_count}",
                    save_path=cluster_path,
                    n_neighbors=min(15, total_particles - 1),
                    min_dist=0.1, thumb_size=80,
                )
                self._save_grid(
                    pil_images,
                    os.path.join(interval_dir, "grid.png"), labels)

                kept_indices = self._prompt_user_selection(
                    total_particles, interval_count, interval_dir)
                prev_kept_indices = kept_indices

                kept_latents = X_all[kept_indices].clone()
                num_particles = len(kept_indices)
                total_particles = num_particles * C
                X_all = kept_latents.unsqueeze(1).expand(
                    -1, C, -1, -1, -1).clone()
                X_all = X_all.reshape(
                    total_particles, 16, latent_h, latent_w)
                is_anchor = torch.zeros(
                    total_particles, dtype=torch.bool)
                for i in range(num_particles):
                    is_anchor[i * C] = True
                wdm_clone_done = torch.zeros(
                    total_particles, dtype=torch.bool)

        # Final decode (1-NFE flow-map jump from current t to 0 — same as
        # the lookahead used at intervals; here it produces the kept final
        # images that the user takes home).
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        final_seg_idx = n_segments_total - 1
        labels = []
        anchor_group = -1
        for i in range(total_particles):
            if is_anchor[i]:
                anchor_group += 1
                labels.append(f"{i} (Ag{anchor_group})")
            else:
                labels.append(f"{i} (Cg{anchor_group})")

        pil_images = []
        with torch.no_grad():
            for p_idx in range(total_particles):
                if progress_callback:
                    dec_frac = 0.40 + 0.50 * (p_idx / total_particles)
                    progress_callback(
                        dec_frac,
                        f"Decoding final image {p_idx+1}/{total_particles}…",
                        final_seg_idx, n_segments_total,
                    )
                img_t = self.decode(X_all[p_idx:p_idx+1])
                pil_img = self._tensor_to_pil(img_t)
                pil_img.save(os.path.join(
                    final_dir, f"particle_{p_idx:03d}.png"))
                pil_images.append(pil_img)

        features = self._extract_dino_features(pil_images)
        plot_image_cluster(
            features, pil_images,
            title="DINOv2 Cluster — Final",
            save_path=os.path.join(final_dir, "cluster_dinov2.png"),
            n_neighbors=min(15, total_particles - 1),
            min_dist=0.1, thumb_size=80,
        )
        # NOTE: skipping IncrementalLayout.update at final — pre-existing
        # off-by-one with kept_indices=range(total_particles) when
        # total_particles grew via cloning since the last interval update
        # (same workaround as in sana_fmtt.py).
        self._save_grid(
            pil_images, os.path.join(final_dir, "grid.png"), labels)
        self._unload_dino()

        with torch.no_grad():
            return torch.cat([
                self.decode(X_all[i:i+1]).float()
                for i in range(total_particles)
            ], dim=0)


# =============================================================================
# CLI (for standalone smoke tests)
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="FLUX-FlowMap Interactive (WDM toggle)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_clones", type=int, default=2)
    parser.add_argument("--num_resampling_steps", type=int, default=3)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--schedule_mode", type=str, default="linear",
                        choices=["linear", "aggressive", "none"])
    parser.add_argument("--clone_mode", type=str, default="glass",
                        choices=["glass", "flow_map"])
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="interactive_fmwdm")
    args = parser.parse_args()

    sampler = FluxFMWDMInteractive(prompt=args.prompt, device="cuda")
    best = sampler.sample_interactive(
        num_steps=args.num_steps,
        num_particles=args.num_particles,
        num_clones=args.num_clones,
        num_resampling_steps=args.num_resampling_steps,
        guidance_scale=args.guidance_scale,
        rho=args.rho,
        schedule_mode=(args.schedule_mode
                       if args.schedule_mode != "none" else None),
        img_shape=(args.height, args.width),
        seed=args.seed,
        output_dir=args.output_dir,
        clone_mode=args.clone_mode,
    )
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    pil_final = FluxFMWDMInteractive._tensor_to_pil(best[0:1])
    pil_final.save(args.output)
    print(f"Final image saved to: {args.output}")
