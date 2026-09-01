"""
SANA Interactive FMTT Sampler (flow-matching with GLASS Flows).

Mirror of FluxDevInteractive — same public contract (`predict_velocity`,
`decode`, `_extract_dino_features`, `sample_interactive`) — but for the
SANA-1.6B-1024px family.

SANA is trained with a rectified-flow / flow-matching objective. The
canonical `SanaPipeline` ships with a `DPMSolverMultistepScheduler`, but the
transformer's raw output is a flow-matching velocity in the FLUX time
convention (t=1 noise → t=0 clean). We swap in `FlowMatchEulerDiscreteScheduler`
so the rest of the math (anchor Euler ODE, GLASS clone bridge) drops in
verbatim from FluxDevInteractive.

Key differences vs FLUX:
  * latent channels: 32 (FLUX is 16)
  * vae_scale_factor: 32 (FLUX is 16) — same image resolution → 32× smaller side
  * text encoder is Gemma-2 2B (much smaller than FLUX's T5-XXL)
  * encoder_attention_mask is required (FLUX uses dense embeddings)
  * no per-image position embeddings packed into a sequence — SANA keeps
    [B, C, H, W] throughout, no FLUX-style packing.
"""
from __future__ import annotations

import gc
import math
import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from diffusers import SanaPipeline, FlowMatchEulerDiscreteScheduler

# Reuse FLUX's incremental layout and schedule helpers — they are model-agnostic.
from flux_fmtt_dev import (
    IncrementalLayout,
    compute_aggressive_diversity_schedule,
    compute_linear_diversity_schedule,
    derive_seed,
    randn_from_seed,
)
from cluster_images import plot_image_cluster


DEFAULT_MODEL_ID = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"


class SanaInteractive:
    """SANA-1.6B interactive sampler with GLASS Flows."""

    def __init__(self, prompt: str, model_id: str = DEFAULT_MODEL_ID,
                 device: str = "cuda", dtype=torch.bfloat16,
                 keep_text_encoder: bool = False):
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.prompt = prompt
        # SPAN MODE needs warm re-prompting (prompt variations arrive after
        # load); keep_text_encoder=True keeps Gemma-2 resident (~5GB) so
        # set_prompts()/add_prompts() work. Default False = original behavior.
        self.keep_text_encoder = bool(keep_text_encoder)

        torch.backends.cuda.matmul.allow_tf32 = True
        self._load_model(prompt)

    # ------------------------------------------------------------------
    def _load_model(self, prompt: str):
        print(f"Loading SANA from: {self.model_id}")
        # Resolve the cached snapshot DIRECTORY off the filesystem and load from
        # that concrete path — avoids huggingface's flaky offline Hub resolution
        # inside the long-running server on fresh cluster nodes (same fix as FLUX).
        load_target = self.model_id
        # Prefer the node-local /scratch copy staged by the job script — the
        # long-running offline server can't read SANA's /data autofs subpath on
        # fresh nodes, but /scratch is a real local mount (no autofs).
        _local = os.environ.get("SANA_LOCAL_PATH")
        _resolved = False
        if _local and os.path.isfile(os.path.join(_local, "model_index.json")):
            load_target = _local
            _resolved = True
            print(f"Loading SANA from node-local: {_local}")
        try:
            import glob, time as _t
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            repo_dir = os.path.join(hf_home, "hub",
                                    "models--" + self.model_id.replace("/", "--"))
            # Resolve via snapshots/* directly (the path the job script warms and
            # that diffusers actually reads). Avoid refs/main — that subpath stays
            # un-warmed in the long-running server's autofs view and reads missing.
            snap_glob = os.path.join(repo_dir, "snapshots", "*")
            for _att in range(6):
                if _resolved:
                    break
                snaps = [s for s in glob.glob(snap_glob)
                         if os.path.isfile(os.path.join(s, "model_index.json"))]
                if snaps:
                    load_target = sorted(snaps)[-1]
                    print(f"Loading SANA from local snapshot dir: {load_target}")
                    break
                _t.sleep(1.0)  # let autofs settle
        except Exception as e:
            print(f"Could not resolve local snapshot dir ({type(e).__name__}); using id.")
        self.pipeline = SanaPipeline.from_pretrained(
            load_target, torch_dtype=self.dtype
        ).to(self.device)

        # Swap to flow-matching scheduler so GLASS bridge math (same time
        # convention as FLUX: t=1 noise → t=0 clean) applies verbatim.
        self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.vae_scale_factor = self.pipeline.vae_scale_factor  # 32 for SANA DC-AE
        self.latent_channels = self.transformer.config.in_channels  # 32
        self.timestep_scale = float(getattr(self.transformer.config, "timestep_scale", 1.0))

        # Precompute text embeddings (CFG enabled: keeps both cond + uncond).
        print("Precomputing text embeddings...")
        with torch.no_grad():
            (prompt_embeds, prompt_mask,
             neg_embeds, neg_mask) = self.pipeline.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=True,
                negative_prompt="",
                device=self.device,
            )
        # Cast to model dtype. Embeddings are stored as LISTS (one entry per
        # prompt variation) so different particles can be conditioned on
        # different prompts (see predict_velocity(prompt_idx=...)). Index 0 is
        # always the artist's original prompt.
        self.cached_prompt_embeds_list = [prompt_embeds.to(self.dtype)]
        self.cached_prompt_mask_list = [prompt_mask]
        self.prompts = [prompt]
        # Back-compat single-prompt aliases (used by the default code paths).
        self.cached_prompt_embeds = prompt_embeds.to(self.dtype)
        self.cached_prompt_mask = prompt_mask
        self.cached_neg_embeds = neg_embeds.to(self.dtype)
        self.cached_neg_mask = neg_mask

        if self.keep_text_encoder:
            print("Keeping text encoder resident (span mode / warm re-prompting).")
        else:
            # Drop text encoder + tokenizer to free memory.
            print("Removing text encoder to free memory...")
            for attr in ("text_encoder", "tokenizer"):
                sub = getattr(self.pipeline, attr, None)
                if sub is not None:
                    del sub
                try:
                    setattr(self.pipeline, attr, None)
                except Exception:
                    pass
            gc.collect()
            torch.cuda.empty_cache()
        print("SANA loaded (memory-efficient mode)")

    # ------------------------------------------------------------------
    # Prompt variations (span mode / prompt-augmented diversity)
    # ------------------------------------------------------------------

    @property
    def num_prompts(self) -> int:
        return len(getattr(self, "cached_prompt_embeds_list", []) or [1])

    def add_prompts(self, prompts: List[str]) -> int:
        """APPEND prompt variations to the cached embedding list.

        Append-only on purpose: a background thread can add variations while a
        generation thread reads indices < the old length — existing indices
        never move. Needs keep_text_encoder=True. Returns the new count.
        """
        if getattr(self.pipeline, "text_encoder", None) is None:
            raise RuntimeError(
                "add_prompts() needs the text encoder, but it was freed. "
                "Construct with keep_text_encoder=True for warm re-prompting.")
        if isinstance(prompts, str):
            prompts = [prompts]
        for p in prompts:
            with torch.no_grad():
                pe, pm, _, _ = self.pipeline.encode_prompt(
                    prompt=p, do_classifier_free_guidance=True,
                    negative_prompt="", device=self.device)
            # Single-item appends are effectively atomic under the GIL, so a
            # concurrent reader never sees a half-registered variation.
            self.cached_prompt_embeds_list.append(pe.to(self.dtype))
            self.cached_prompt_mask_list.append(pm)
            self.prompts.append(p)
        print(f"[sana] prompt variations now: {len(self.prompts)}")
        return len(self.prompts)

    # ------------------------------------------------------------------
    # Velocity prediction
    # ------------------------------------------------------------------

    def _gather_prompt_embeds(self, batch_size: int, prompt_idx=None):
        """Per-row conditioned embeddings [B, seq, dim] + mask [B, seq].

        prompt_idx: optional per-row iterable/LongTensor of variation indices
        into cached_prompt_embeds_list (clamped, so an index that raced ahead
        of add_prompts degrades to the base prompt instead of crashing).
        None -> everyone on variation 0 (original behavior).
        """
        if prompt_idx is None:
            return (self.cached_prompt_embeds.expand(batch_size, -1, -1),
                    self.cached_prompt_mask.expand(batch_size, -1))
        n = self.num_prompts
        idx = [min(max(int(i), 0), n - 1) for i in prompt_idx]
        embeds = torch.cat([self.cached_prompt_embeds_list[i] for i in idx], dim=0)
        mask = torch.cat([self.cached_prompt_mask_list[i] for i in idx], dim=0)
        return embeds, mask

    def predict_velocity(self, z: torch.Tensor, t_cur: float,
                         guidance_scale: float = 4.0,
                         prompt_idx=None) -> torch.Tensor:
        """Single-step instantaneous flow-matching velocity.

        SANA's transformer is trained to predict the rectified-flow velocity
        v(x_t, t) = ε - x_0 directly (per the SANA paper, eq. 5). We pass the
        latent + cached text embeddings, apply classifier-free guidance via a
        doubled batch (uncond + cond), then return the unpacked velocity.

        Time convention: t_cur ∈ [0, 1] with 1 = pure noise. SanaTransformer
        expects the timestep in the model's native scale, which is
        `t * 1000 * timestep_scale` (the pipeline does `timestep * timestep_scale`
        after the scheduler emits a 0..1000 value).
        """
        if isinstance(t_cur, torch.Tensor):
            t_cur = t_cur.item()
        batch_size = z.shape[0]
        device = z.device
        dtype = self.dtype

        # Build CFG-doubled batch. Per-row prompt variations via prompt_idx.
        cond_embeds, cond_mask = self._gather_prompt_embeds(batch_size, prompt_idx)
        do_cfg = guidance_scale is not None and guidance_scale > 1.0
        if do_cfg:
            latent_input = torch.cat([z, z], dim=0)
            embeds = torch.cat(
                [self.cached_neg_embeds.expand(batch_size, -1, -1),
                 cond_embeds], dim=0)
            mask = torch.cat(
                [self.cached_neg_mask.expand(batch_size, -1),
                 cond_mask], dim=0)
        else:
            latent_input = z
            embeds = cond_embeds
            mask = cond_mask

        # Scheduler emits timesteps in [0, 1000]; transformer scales by
        # `timestep_scale`. We hold t in [0, 1] so multiply explicitly here.
        timestep = torch.full(
            (latent_input.shape[0],), t_cur * 1000.0 * self.timestep_scale,
            device=device, dtype=dtype,
        )

        # Forward.
        out = self.transformer(
            hidden_states=latent_input.to(dtype),
            encoder_hidden_states=embeds,
            encoder_attention_mask=mask,
            timestep=timestep,
            return_dict=False,
        )[0]
        out = out.float()

        # Handle learned-sigma models (out_channels = 2 * in_channels).
        if self.transformer.config.out_channels // 2 == self.latent_channels:
            out = out.chunk(2, dim=1)[0]

        # CFG mix.
        if do_cfg:
            v_uncond, v_cond = out.chunk(2, dim=0)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = out

        return v.to(z.dtype)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode SANA latent → image [B, 3, H, W] in [0, 1].
        SANA's DC-AE uses a single `scaling_factor` (no shift_factor)."""
        z_scaled = z / self.vae.config.scaling_factor
        with torch.no_grad():
            samples = self.vae.decode(z_scaled.to(self.vae.dtype),
                                       return_dict=False)[0]
        samples = torch.clamp((samples + 1) / 2, 0, 1)
        return samples

    def euler_lookahead(self, z: torch.Tensor, t_cur: float,
                        guidance_scale: float, num_steps: int = 8,
                        step_cb=None, prompt_idx=None) -> torch.Tensor:
        """Deterministic Euler integration t_cur → 0 for preview.

        SANA's flow is stiff: uniform steps in t from a high-σ checkpoint diverge
        to noise. So place the lookahead steps on the SAME shifted sigma curve as
        the main schedule (uniform in *naive* σ, then re-shifted) — few steps, but
        correctly spaced.
        """
        z_cur = z.clone()
        shift = float(getattr(self, "_flow_shift", 3.0))
        # inverse-shift t_cur back to naive σ, take uniform naive steps to 0, reshift
        inv = t_cur / max(shift - (shift - 1.0) * t_cur, 1e-8)
        naive = np.linspace(inv, 0.0, num_steps + 1)
        pts = shift * naive / (1.0 + (shift - 1.0) * naive)
        for i in range(num_steps):
            t = float(pts[i]); t_next = float(pts[i + 1])
            v = self.predict_velocity(z_cur, t, guidance_scale,
                                      prompt_idx=prompt_idx)
            z_cur = z_cur + (t_next - t) * v
            if step_cb is not None:
                step_cb(i + 1, num_steps)
        return z_cur

    # ------------------------------------------------------------------
    # DINO features — reuse FLUX implementation verbatim
    # ------------------------------------------------------------------

    def _load_dino(self):
        from transformers import AutoImageProcessor, AutoModel
        dino_id = "facebook/dinov2-base"
        print(f"Loading DINOv2-base ({dino_id})...")
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_id)
        self.dino_model = AutoModel.from_pretrained(
            dino_id, low_cpu_mem_usage=False
        ).to(self.device).eval()
        print("DINOv2 loaded")

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

    # ------------------------------------------------------------------
    # GLASS bridge
    # ------------------------------------------------------------------

    def _glass_init(self, X_bar: torch.Tensor, sigma_start: float,
                    sigma_end: float, rho: float,
                    generator=None) -> Tuple[torch.Tensor, dict]:
        """Same math as FluxDevInteractive._glass_init — flow-matching time
        convention, σ ≡ t with σ=1 (noise) → σ=0 (clean).

        The single eps draw is the ONLY stochasticity in a GLASS clone. A CPU
        torch.Generator (or a list of them, one per batch row) makes the draw
        exactly reproducible; None keeps the original torch.randn_like path.
        """
        clip = 1e-8
        alpha_start = 1.0 - sigma_start
        alpha_end = 1.0 - sigma_end

        bar_gamma = rho * sigma_end / max(sigma_start, clip)
        bar_alpha = alpha_end - bar_gamma * alpha_start
        bar_sigma = math.sqrt(max(sigma_end ** 2 * (1.0 - rho ** 2), 0.0))
        bar_sigma_0 = 1.0

        if generator is None:
            eps = torch.randn_like(X_bar)
        elif isinstance(generator, (list, tuple)):
            eps = torch.stack([
                torch.randn(tuple(X_bar.shape[1:]), generator=g, device="cpu",
                            dtype=torch.float32)
                for g in generator
            ]).to(device=X_bar.device, dtype=X_bar.dtype)
        else:
            eps = torch.randn(tuple(X_bar.shape), generator=generator,
                              device="cpu", dtype=torch.float32
                              ).to(device=X_bar.device, dtype=X_bar.dtype)
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
                               interval_dir: str, refine_fn=None) -> List[int]:
        print("\n" + "=" * 60)
        print(f"  INTERACTIVE PRUNING — Interval {interval_idx}")
        print("=" * 60)
        print(f"  Previews: {interval_dir}/particle_*.png")
        print(f"  Grid:     {interval_dir}/grid.png")
        print("=" * 60)
        while True:
            raw = input(f"  Keep indices (0-{total-1}), comma-sep: ").strip()
            if not raw:
                continue
            # REFINE: "refine:i,j" renders full-quality versions (mirror of
            # flux_fmtt_dev), then loops back to waiting for the KEEP selection.
            if refine_fn is not None and raw.startswith("refine:"):
                try:
                    ridx = sorted(set(
                        int(x) for x in raw[len("refine:"):].split(",") if x.strip()))
                    ridx = [i for i in ridx if 0 <= i < total]
                    if ridx:
                        refine_fn(ridx)
                except Exception as e:
                    print(f"  Refine failed: {type(e).__name__}: {e}")
                continue
            try:
                indices = sorted(set(int(x.strip()) for x in raw.split(",")))
                if not indices or any(i < 0 or i >= total for i in indices):
                    print(f"  Invalid. Must be in 0-{total-1}.")
                    continue
                return indices
            except ValueError:
                print("  Invalid input.")

    # ------------------------------------------------------------------
    # SPAN MODE (round 1): stream INDEPENDENT full images in small batches
    # ------------------------------------------------------------------

    def _shifted_sigma_schedule(self, num_steps: int) -> np.ndarray:
        """The SANA flow-matching schedule with the standard resolution shift
        applied — shared by sample_interactive, span mode and lookahead."""
        sigmas_arr = np.linspace(1.0, 1.0 / num_steps, num_steps)
        _shift = float(getattr(self.pipeline.scheduler.config, "shift", 1.0) or 1.0)
        if _shift <= 1.0:
            _shift = 3.0   # SANA 1024px standard flow shift (default pipeline value)
        self._flow_shift = _shift
        return _shift * sigmas_arr / (1.0 + (_shift - 1.0) * sigmas_arr)

    def sample_span_round(self, *,
                          master_seed: int,
                          output_dir: str,
                          batch_size: int = 8,
                          max_images: int = 64,
                          num_steps: int = 14,
                          guidance_scale: float = 4.5,
                          img_shape: Optional[Tuple[int, int]] = (512, 512),
                          prompt_plan=None,
                          on_batch=None,
                          progress_callback=None) -> dict:
        """SPAN ROUND 1: map the prompt's space with INDEPENDENT full images,
        generated in small batches and handed to the caller as each batch
        completes — the caller streams them to the canvas immediately.

        Every image is a fast FULL solve (num_steps Euler steps on the shifted
        schedule), not a lookahead preview — round 1 is legible by construction.
        Image i's init noise comes from derive_seed(master_seed, i), identical
        to the root-particle convention, so every span image has a breedable /
        exactly-reproducible recipe (written to lineage_span.json).

        prompt_plan: optional callable (batch_idx, size) -> List[int] choosing,
            per image, which cached prompt variation conditions it (semantic
            spread across batches; variations may be added concurrently via
            add_prompts — indices are clamped defensively).
        on_batch: callable(info dict) -> bool. Called after each completed
            batch with pil_images / features / paths / prompt_idx / timing.
            Return False to stop gracefully (auto-stop or user early-stop).
        Returns dict with clean latents [N,C,h,w] (CPU), stacked features,
        paths, per-image prompt_idx + seeds, and the stop reason.
        """
        import json as _json
        import time as _time

        imgH, imgW = img_shape if img_shape is not None else (512, 512)
        master_seed = int(master_seed)
        interval_dir = os.path.join(output_dir, "interval_1")
        os.makedirs(interval_dir, exist_ok=True)

        if getattr(self, "dino_model", None) is None:
            self._load_dino()

        latent_h = int(imgH) // self.vae_scale_factor
        latent_w = int(imgW) // self.vae_scale_factor
        sigmas_arr = self._shifted_sigma_schedule(num_steps)
        pts = list(sigmas_arr) + [0.0]
        print(f"[span] flow shift={self._flow_shift} num_steps={num_steps} "
              f"batch={batch_size} cap={max_images}")

        lineage: List[dict] = []
        all_latents: List[torch.Tensor] = []
        all_paths: List[str] = []
        all_features: List[np.ndarray] = []
        all_prompt_idx: List[int] = []
        all_seeds: List[int] = []
        stop_reason = "cap"
        batch_idx = 0
        n_done = 0

        while n_done < max_images:
            bsz = min(batch_size, max_images - n_done)
            g0 = n_done
            t0 = _time.perf_counter()

            # Per-image prompt assignment for THIS batch (clamped in
            # _gather_prompt_embeds if a variation isn't registered yet).
            if prompt_plan is not None:
                try:
                    p_idx = list(prompt_plan(batch_idx, bsz))[:bsz]
                except Exception as _pe:
                    print(f"[span] prompt_plan failed ({_pe}); base prompt.")
                    p_idx = [0] * bsz
            else:
                p_idx = [0] * bsz
            p_idx = [min(max(int(i), 0), self.num_prompts - 1) for i in p_idx]

            seeds = [derive_seed(master_seed, g0 + j) for j in range(bsz)]
            z = torch.stack([
                randn_from_seed(s, (self.latent_channels, latent_h, latent_w),
                                self.device, self.dtype)
                for s in seeds
            ])

            with torch.no_grad():
                for k in range(num_steps):
                    if progress_callback:
                        progress_callback(batch_idx, n_done, k, num_steps)
                    t_cur, t_next = float(pts[k]), float(pts[k + 1])
                    v = self.predict_velocity(z, t_cur, guidance_scale,
                                              prompt_idx=p_idx)
                    z = z + (t_next - t_cur) * v

                pil_images, paths = [], []
                for j in range(bsz):
                    img_t = self.decode(z[j:j + 1])
                    pil = self._tensor_to_pil(img_t)
                    p = os.path.join(interval_dir,
                                     f"particle_{g0 + j:03d}.png")
                    pil.save(p)
                    pil_images.append(pil)
                    paths.append(p)

            features = self._extract_dino_features(pil_images)
            gen_s = _time.perf_counter() - t0

            for j in range(bsz):
                lineage.append(dict(
                    idx=g0 + j, kind="span_root", parent=None,
                    seed=seeds[j], rho=None, sigma_start=None, sigma_end=None,
                    spawn_step=None, prompt_idx=p_idx[j],
                    num_steps=int(num_steps)))
            all_latents.append(z.detach().float().cpu())
            all_paths.extend(paths)
            all_features.append(features)
            all_prompt_idx.extend(p_idx)
            all_seeds.extend(seeds)
            n_done += bsz
            print(f"[span] batch {batch_idx + 1} done: images {g0}..{n_done - 1} "
                  f"in {gen_s:.1f}s (prompts {sorted(set(p_idx))})", flush=True)

            cont = True
            if on_batch is not None:
                info = dict(batch_idx=batch_idx, start=g0, count=bsz,
                            total=n_done, pil_images=pil_images, paths=paths,
                            features=features, prompt_idx=p_idx,
                            seeds=seeds, gen_seconds=gen_s)
                try:
                    cont = bool(on_batch(info))
                except Exception as _obe:
                    import traceback; traceback.print_exc()
                    print(f"[span] on_batch failed ({_obe}); stopping stream.")
                    cont = False
            batch_idx += 1
            if not cont:
                stop_reason = "callback"
                break

        # ===== SAVE THE NOISE: span recipe (kept separate from the breeding
        # rounds' lineage.json so neither overwrites the other) =====
        _recipe = dict(
            master_seed=master_seed, model_id=self.model_id, mode="span_round1",
            num_steps=int(num_steps), guidance_scale=float(guidance_scale),
            img_shape=[int(imgH), int(imgW)],
            prompts=list(getattr(self, "prompts", [self.prompt])),
            sigmas=[float(t) for t in pts],
            n_images=int(n_done), stop_reason=stop_reason,
            lineage=lineage,
        )
        try:
            with open(os.path.join(output_dir, "lineage_span.json"), "w") as fh:
                _json.dump(_recipe, fh, indent=2)
            print(f"[span] recipe saved: {output_dir}/lineage_span.json")
        except Exception as _le:
            print(f"[span] WARNING: lineage_span.json write failed: {_le}")

        return dict(
            latents=torch.cat(all_latents, dim=0) if all_latents else None,
            features=(np.vstack(all_features) if all_features else None),
            paths=all_paths, prompt_idx=all_prompt_idx, seeds=all_seeds,
            n_images=n_done, n_batches=batch_idx, stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Interactive sampling — identical structure to FluxDevInteractive
    # ------------------------------------------------------------------

    def sample_interactive(self,
                           num_steps: int = 28,
                           num_particles: int = 4,
                           num_clones: int = 2,
                           num_resampling_steps: int = 4,
                           guidance_scale: float = 4.0,
                           rho: float = 0.4,
                           lookahead_steps: int = 8,
                           # Round-1 previews live at σ≈0.9 where the image is
                           # largely uncommitted — a few Euler steps give blurry
                           # near-identical blobs. Spend MORE lookahead steps on
                           # the FIRST interval only (later intervals are more
                           # committed and stay cheap). None -> use lookahead_steps.
                           lookahead_steps_first: Optional[int] = None,
                           schedule_mode: Optional[str] = "linear",
                           img_shape: Optional[Tuple[int, int]] = None,
                           seed: Optional[int] = None,
                           master_seed: Optional[int] = None,
                           output_dir: str = "interactive_sana",
                           progress_callback=None,
                           save_debug_plots: bool = False,
                           # Per-anchor prompt-variation indices (clones inherit
                           # their parent's). None -> all on variation 0.
                           prompt_idx=None,
                           # ── SPAN/IMAGE-SEEDED START (SDEdit) ─────────────
                           # init_latents: [P, C, h, w] CLEAN latents (e.g. the
                           # selected span-round parents). Each of the
                           # num_particles anchors is built by re-noising
                           # parent (i % P) to t'=init_strength with its own
                           # per-root eps — so several anchors per parent
                           # diverge. None -> pure-noise roots (original path).
                           init_latents=None,
                           init_strength: float = 0.55,
                           # Global ids of the parents (for lineage only).
                           init_parent_ids=None,
                           # Offset for interval_N dir names: a span round
                           # already wrote interval_1, so breeding rounds
                           # start at interval_{1+interval_start}.
                           interval_start: int = 0) -> torch.Tensor:
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)
        device = self.device
        os.makedirs(output_dir, exist_ok=True)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # ===== SAVE THE NOISE: session master seed + lineage recording =====
        # Root init noise and each GLASS clone's single eps draw are derived
        # deterministically from master_seed; every particle creation is
        # recorded and written to lineage.json (exact-reproduction recipe).
        if master_seed is None:
            master_seed = int(seed) if seed is not None else \
                random.SystemRandom().randrange(2 ** 31)
        master_seed = int(master_seed)
        print(f"  [save-noise] master_seed={master_seed}")
        lineage: List[dict] = []
        selections: List[dict] = []
        spawn_event_counter = 0

        n_segments_total = num_resampling_steps + 1
        if progress_callback:
            progress_callback(0.02, "Loading image similarity model...",
                              0, n_segments_total)
        self._load_dino()
        if progress_callback:
            progress_callback(0.05, "Preparing...", 0, n_segments_total)

        latent_h = int(imgH) // self.vae_scale_factor
        latent_w = int(imgW) // self.vae_scale_factor

        root_seeds = [derive_seed(master_seed, i) for i in range(num_particles)]
        anchors = torch.stack([
            randn_from_seed(rs, (self.latent_channels, latent_h, latent_w),
                            device, self.dtype)
            for rs in root_seeds
        ])
        total_particles = num_particles
        C = num_clones

        # Per-particle prompt-variation index (clones inherit their parent's).
        if prompt_idx is not None:
            self.particle_prompt_idx = torch.as_tensor(
                list(prompt_idx), dtype=torch.long)[:total_particles]
            if len(self.particle_prompt_idx) < total_particles:
                pad = torch.zeros(
                    total_particles - len(self.particle_prompt_idx),
                    dtype=torch.long)
                self.particle_prompt_idx = torch.cat(
                    [self.particle_prompt_idx, pad])
        else:
            self.particle_prompt_idx = torch.zeros(
                total_particles, dtype=torch.long)

        def _pidx(indices) -> List[int]:
            """Prompt-variation indices for a tensor of particle indices."""
            return [int(self.particle_prompt_idx[int(i)].item())
                    for i in indices]

        # Flow-matching timestep schedule. SANA needs a flow-matching time SHIFT:
        # the default SanaPipeline applies one (~3.0 at 1024px), and WITHOUT it the
        # transformer is queried at the wrong noise levels and the output is mostly
        # noise (verified: vanilla SanaPipeline is clean, this raw-linear schedule
        # was not). FLUX applies the equivalent via `mu` in retrieve_timesteps; here
        # we apply the standard resolution shift directly to the sigmas:
        #     σ' = shift·σ / (1 + (shift-1)·σ)
        sigmas_arr = self._shifted_sigma_schedule(num_steps)
        print(f"[sana] flow shift={self._flow_shift}")
        time_steps = torch.tensor(
            list(sigmas_arr) + [0.0], device=device, dtype=torch.float32
        )

        # ===== SPAN/IMAGE-SEEDED (SDEdit) start — only when init_latents given
        # (mirror of FluxDevInteractive's init_latent path, extended to MULTIPLE
        # parents). Re-noise each parent to t'=init_strength (snapped DOWN to a
        # real schedule step) with an INDEPENDENT per-anchor eps; anchors are
        # assigned to parents round-robin so each pick breeds several variants.
        # Default path (init_latents is None) leaves everything untouched.
        k_start = 0
        _seeded_root_meta = None   # (t_renoise, k_start) when seeded
        _anchor_parent = None      # per-anchor parent row in init_latents
        if init_latents is not None:
            try:
                t_prime = float(init_strength)
            except (TypeError, ValueError):
                t_prime = 0.55
            t_prime = max(0.30, min(0.95, t_prime))
            z0 = init_latents.to(device=device, dtype=self.dtype)
            if z0.dim() == 3:
                z0 = z0.unsqueeze(0)
            if tuple(z0.shape[1:]) == (self.latent_channels, latent_h, latent_w):
                for _ki in range(num_steps):
                    if time_steps[_ki].item() <= t_prime:
                        k_start = _ki
                        break
                else:
                    k_start = num_steps - 1
                t_renoise = time_steps[k_start].item()
                P = z0.shape[0]
                _anchor_parent = [i % P for i in range(num_particles)]
                eps = torch.stack([
                    randn_from_seed(rs, (self.latent_channels, latent_h, latent_w),
                                    device, self.dtype)
                    for rs in root_seeds
                ])
                parents = torch.stack([z0[p] for p in _anchor_parent])
                anchors = (1.0 - t_renoise) * parents + t_renoise * eps
                _seeded_root_meta = (float(t_renoise), int(k_start))
                print(f"  [image-seed] {P} parent latent(s), init_strength="
                      f"{init_strength} -> t'={t_prime:.3f} snapped to step "
                      f"{k_start} (t={t_renoise:.4f}); {num_particles} anchors "
                      f"round-robin over parents.")
            else:
                print(f"  [image-seed] WARNING: init_latents shape "
                      f"{tuple(z0.shape)} != (*,{self.latent_channels},"
                      f"{latent_h},{latent_w}); falling back to pure noise.")

        for _ri in range(num_particles):
            _e = dict(idx=_ri, kind="root", parent=None,
                      seed=root_seeds[_ri], rho=None,
                      sigma_start=None, sigma_end=None,
                      spawn_step=None,
                      prompt_idx=int(self.particle_prompt_idx[_ri].item()))
            if _seeded_root_meta is not None:
                _e.update(kind="sdedit", sigma_start=_seeded_root_meta[0],
                          spawn_step=_seeded_root_meta[1])
                if init_parent_ids is not None and _anchor_parent is not None:
                    try:
                        _e["span_parent"] = int(
                            init_parent_ids[_anchor_parent[_ri]])
                    except (IndexError, TypeError, ValueError):
                        pass
            lineage.append(_e)

        # Resampling schedule
        if schedule_mode == "aggressive" and num_resampling_steps > 0:
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

        # Seeded-session resample rebuild (mirror of flux_fmtt_dev): when the
        # trajectory starts at k_start > 0, the full-range checkpoints can all
        # fall BEFORE k_start and never fire -> zero interaction rounds.
        # Recompute them evenly inside (k_start, num_steps).
        if init_latents is not None and k_start > 0 and num_resampling_steps > 0:
            _rebuilt = sorted(set(
                k_start + int(round((i + 1) * (num_steps - k_start)
                                    / (num_resampling_steps + 1)))
                for i in range(num_resampling_steps)
            ))
            resample_steps = [s for s in _rebuilt if k_start < s < num_steps]
            resample_at = set(resample_steps)
            segment_bounds = [k_start] + resample_steps + [num_steps]
            n_segments_total = len(resample_steps) + 1
            print(f"  [image-seed] seeded resample schedule (k_start={k_start}): "
                  f"steps {resample_steps} -> {n_segments_total} segment(s)")

        X_all = anchors.clone()
        is_anchor = torch.ones(total_particles, dtype=torch.bool)
        glass_params = None
        interval_count = 0
        inc_layout = IncrementalLayout(n_neighbors=15, min_dist=0.2)
        prev_kept_indices = None

        # ── Honest progress accounting ──────────────────────────────────────
        # The reported fraction is (work done / total work) for the segment,
        # in "main ODE step" units, INCLUDING the preview lookahead + decode
        # tail that used to be hidden behind a 40→90% jump. Per segment:
        #   * each main step        = 1 unit (batched anchor + clone forwards)
        #   * preview, per particle = lookahead_steps batch-1 forwards
        #                             (~0.5 units each) + VAE decode (~0.15)
        #   * DINO features + layout tail ≈ 2 units
        # Monotone by construction: phases are cumulative within a segment and
        # the server clamps with max(prev, new).
        _la_first = lookahead_steps_first if lookahead_steps_first else lookahead_steps
        _LA_UNIT = 0.5
        _DEC_UNIT = 0.15
        _TAIL_UNITS = 2.0

        def _seg_progress_units(seg_idx, n_parts):
            """Return (seg_len, la_for_segment, total_units) for one segment."""
            seg_len = max(segment_bounds[seg_idx + 1] - segment_bounds[seg_idx], 1)
            la = _la_first if seg_idx == 0 else lookahead_steps
            if seg_idx == n_segments_total - 1:
                la = 0  # final segment decodes latents directly (no lookahead)
            prev_units = n_parts * (la * _LA_UNIT + _DEC_UNIT)
            return seg_len, la, seg_len + prev_units + _TAIL_UNITS

        from tqdm import tqdm
        # Seeded sessions begin integration at k_start (first step <= t');
        # default path has k_start=0 -> full schedule, unchanged.
        pbar = tqdm(range(k_start, num_steps), total=num_steps - k_start,
                    desc="SANA-Interactive")

        for k in pbar:
            if progress_callback:
                seg_idx = 0
                for b in resample_steps:
                    if k >= b:
                        seg_idx += 1
                seg_start = segment_bounds[seg_idx]
                seg_len, _la_seg, _total_units = _seg_progress_units(
                    seg_idx, total_particles)
                frac = (k - seg_start) / _total_units
                progress_callback(
                    frac,
                    f"Generating images — step {k + 1 - seg_start}/{seg_len}",
                    seg_idx, n_segments_total,
                )

            t_cur = time_steps[k].item()
            t_next = time_steps[k + 1].item()
            dt = t_next - t_cur

            # Segment start: GLASS init for clones
            if glass_params is None:
                # BRIDGE-BOUNDARY FIX — mirror of flux_fmtt_dev (evidence:
                # outputs/glass_sigma_sweep). The resample event at fine step
                # b fires AFTER stepping b (states at time_steps[b+1]); the
                # bridge must target that ACTUAL boundary with M_inner=b+1-k
                # inner steps so it lands EXACTLY on-marginal (s=1) at
                # expansion — never overshoots (extra step at k=b) and is
                # never interrupted mid-bridge by an adjacent event.
                seg_end_step = next(
                    (b + 1 for b in resample_steps if b >= k), None)
                if seg_end_step is None:
                    # FINAL-ROUND BRIDGE CAP — mirror of flux_fmtt_dev
                    # (validated: outputs/glass_sigma_sweep/followup_results
                    # .json). Bridging the last segment all the way to
                    # sigma=0 reduces per-child novelty (parent-child DINO
                    # cosdist 0.444 vs 0.518 for Euler completion) and makes
                    # lookahead previews of mid-descent clones systematically
                    # wrong. Cap at a short handoff (M <= 3, on-marginal at
                    # s=1), then plain Euler to 0 (handoff switch below).
                    # Lineage sigma_end records the handoff sigma.
                    seg_end_step = min(k + 3, num_steps)
                sigma_seg_start = time_steps[k].item()
                sigma_seg_end = time_steps[seg_end_step].item()
                M_inner = seg_end_step - k

                clone_indices = (~is_anchor).nonzero(as_tuple=True)[0]
                if len(clone_indices) > 0:
                    parent_states = torch.zeros(
                        len(clone_indices), self.latent_channels,
                        latent_h, latent_w,
                        device=device, dtype=self.dtype,
                    )
                    for ci, idx in enumerate(clone_indices):
                        group = idx.item() // C
                        parent_states[ci] = X_all[group * C]

                    # Per-clone seeds -> per-row CPU generators, so each
                    # clone's single eps draw is independently reproducible.
                    clone_gens = []
                    for idx in clone_indices:
                        _ci = int(idx.item())
                        _cs = derive_seed(master_seed, spawn_event_counter, _ci)
                        clone_gens.append(
                            torch.Generator(device="cpu").manual_seed(_cs))
                        lineage.append(dict(
                            idx=_ci, kind="glass", parent=(_ci // C) * C,
                            seed=_cs, rho=float(rho),
                            sigma_start=float(sigma_seg_start),
                            sigma_end=float(sigma_seg_end), spawn_step=int(k),
                            prompt_idx=int(
                                self.particle_prompt_idx[_ci].item())))
                    clone_inner, gp = self._glass_init(
                        parent_states, sigma_seg_start, sigma_seg_end, rho,
                        generator=clone_gens)
                    spawn_event_counter += 1
                    X_all[clone_indices] = clone_inner
                    glass_params = gp
                    glass_params["seg_start_step"] = k
                    glass_params["M_inner"] = M_inner
                    glass_params["clone_indices"] = clone_indices
                else:
                    glass_params = {"seg_start_step": k, "M_inner": M_inner,
                                    "clone_indices": torch.tensor([], dtype=torch.long)}

            # Anchor Euler step
            m = k - glass_params["seg_start_step"]
            M = glass_params["M_inner"]
            if m >= M and len(glass_params["clone_indices"]) > 0:
                # FINAL-ROUND HANDOFF: the capped final-segment bridge landed
                # on-marginal (s=1) at seg_start+M; its clones complete by
                # plain Euler to sigma=0 (per-child novelty + preview
                # truthfulness). Only reachable in the final segment —
                # bounded segments are cleared by their event exactly at m=M.
                glass_params["handoff_clone_indices"] = glass_params["clone_indices"]
                glass_params["clone_indices"] = torch.tensor([], dtype=torch.long)
                print(f"  Final-segment handoff at step {k}: "
                      f"{len(glass_params['handoff_clone_indices'])} clones "
                      f"-> plain Euler to sigma=0")
            s = m / M
            ds = 1.0 / M

            with torch.no_grad():
                anchor_indices = is_anchor.nonzero(as_tuple=True)[0]
                handoff_clone_indices = glass_params.get("handoff_clone_indices")
                if handoff_clone_indices is not None and len(handoff_clone_indices) > 0:
                    anchor_indices = torch.cat([anchor_indices,
                                                handoff_clone_indices])
                if len(anchor_indices) > 0:
                    # BATCHED anchor Euler step: one forward over all anchors
                    # (was a per-particle loop). Same MATH; bf16 batched GEMM
                    # is ~1e-2 different from single-row (cuBLAS reduction order).
                    idx_a = anchor_indices
                    z_a = X_all[idx_a]
                    v_a = self.predict_velocity(z_a, t_cur, guidance_scale,
                                                prompt_idx=_pidx(idx_a))
                    X_all[idx_a] = z_a + dt * v_a

            # Clone GLASS inner step
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
                bproduct = max(mu1*mu1*inv11 + 2*mu1*mu2*inv12
                               + mu2*mu2*inv22, clip)
                t_star = max(
                    min(1.0/(1.0+math.sqrt(max(1.0/bproduct, 0.0))), 0.999),
                    0.001)
                sigma_star = 1.0 - t_star
                alpha_star = t_star
                w_xt = alpha_star * (mu1*inv11 + mu2*inv12) / bproduct
                w_xs = alpha_star * (mu1*inv12 + mu2*inv22) / bproduct

                with torch.no_grad():
                    # BATCHED GLASS clone step: weights are per-segment scalars
                    # (same for all clones), so stack all clones into one
                    # forward. x_t is ci-aligned with clone_indices.
                    cidx = clone_indices
                    x_parent = x_t                       # [N_clone, ...] parents
                    x_clone = X_all[cidx]                # [N_clone, ...] clones
                    S_input = w_xt * x_parent + w_xs * x_clone
                    v_star = self.predict_velocity(
                        S_input, sigma_star, guidance_scale,
                        prompt_idx=_pidx(cidx))
                    denoiser = S_input - sigma_star * v_star
                    velocity = w1 * x_clone + w2 * denoiser + w3 * x_parent
                    X_all[cidx] = x_clone + ds * velocity

            # Resampling intervals — identical to FLUX's flow
            if k in resample_at:
                glass_params = None
                interval_count += 1
                # interval_start offsets dir names when a SPAN round already
                # owns interval_1 (breeding rounds then start at interval_2).
                interval_dir = os.path.join(
                    output_dir, f"interval_{interval_count + interval_start}")
                os.makedirs(interval_dir, exist_ok=True)
                t_now = t_next
                seg_idx_done = interval_count - 1

                labels = []
                anchor_group = -1
                for i in range(total_particles):
                    if is_anchor[i]:
                        anchor_group += 1
                        labels.append(f"{i} (anchor g{anchor_group})")
                    else:
                        labels.append(f"{i} (clone g{anchor_group})")

                # Round-1 preview boost: first interval gets the richer
                # lookahead (states at σ≈0.9 need more Euler steps to show
                # actual content); later intervals keep the cheap setting.
                la_this = _la_first if interval_count == 1 else lookahead_steps
                _seg_len_d, _la_d, _tot_units_d = _seg_progress_units(
                    seg_idx_done, total_particles)
                import time as _ptime
                _t_prev0 = _ptime.perf_counter()
                pil_images = []
                with torch.no_grad():
                    for p_idx in range(total_particles):
                        if progress_callback:
                            _done = _seg_len_d + p_idx * (la_this * _LA_UNIT + _DEC_UNIT)
                            progress_callback(
                                _done / _tot_units_d,
                                f"Sketching preview {p_idx + 1}/{total_particles}...",
                                seg_idx_done, n_segments_total,
                            )
                        # Per-lookahead-step ticks: keeps the bar moving through
                        # the whole preview integration (zero dead air).
                        _sketch_cb = None
                        if progress_callback:
                            def _sketch_cb(_done, _n, _p=p_idx):
                                progress_callback(
                                    (_seg_len_d
                                     + (_p * la_this + _done) * _LA_UNIT
                                     + _p * _DEC_UNIT) / _tot_units_d,
                                    f"Sketching preview {_p + 1}/{total_particles}"
                                    f" — step {_done}/{_n}",
                                    seg_idx_done, n_segments_total,
                                )
                        z0 = self.euler_lookahead(
                            X_all[p_idx:p_idx+1], t_now,
                            guidance_scale, num_steps=la_this,
                            step_cb=_sketch_cb,
                            prompt_idx=_pidx([p_idx]),
                        )
                        img_t = self.decode(z0)
                        pil_img = self._tensor_to_pil(img_t)
                        pil_img.save(os.path.join(
                            interval_dir, f"particle_{p_idx:03d}.png"))
                        pil_images.append(pil_img)
                print(f"[timing] interval {interval_count}: "
                      f"previews={_ptime.perf_counter() - _t_prev0:.2f}s "
                      f"(N={total_particles}, lookahead={la_this})", flush=True)

                if progress_callback:
                    _done = _seg_len_d + total_particles * (la_this * _LA_UNIT + _DEC_UNIT)
                    progress_callback(
                        _done / _tot_units_d, "Arranging images by similarity...",
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

                # Debug visualizations (matplotlib renders + a fresh UMAP fit)
                # — several seconds/round and NOT consumed by the web frontend
                # (which lays out from the API's own coords). Off by default so
                # the interactive server round-trip stays fast (mirror of
                # flux_fmtt_dev); the standalone CLI passes True.
                if save_debug_plots:
                    if inc_layout is not None:
                        inc_layout.plot(
                            title=f"Interval {interval_count} (step {k}, t={t_now:.3f})",
                            save_path=os.path.join(interval_dir,
                                                   "layout_incremental.png"),
                            thumb_size=80,
                        )
                    cluster_path = os.path.join(interval_dir, "cluster_dinov2.png")
                    plot_image_cluster(
                        features, pil_images,
                        title=f"DINOv2 Cluster — Interval {interval_count}",
                        save_path=cluster_path,
                        n_neighbors=min(15, total_particles - 1),
                        min_dist=0.1, thumb_size=80,
                    )
                    self._save_grid(pil_images,
                                    os.path.join(interval_dir, "grid.png"),
                                    labels)

                # REFINE hook — mirror of flux_fmtt_dev: PATH-FAITHFUL,
                # STAGE-ADAPTIVE (validated on THIS backend:
                # outputs/refine_calibration/results.json). EARLY picks
                # (t' > 0.25) continue the coarse preview path — the same
                # SHIFTED lookahead grid euler_lookahead used — down to the
                # handoff t''=0.25, then fine-solve the shifted fine-schedule
                # points below it (LPIPS preview-vs-refine ~0.009 at t'=0.9
                # vs ~0.08 naive). LATE picks (t' <= 0.25) keep the naive
                # fine solve (identical there; path-faithful inverts below
                # its own handoff). X_all untouched.
                _T_REFINE_HANDOFF = 0.25
                def _refine_particles(ridx, _k=k, _dir=interval_dir,
                                      _la=la_this,
                                      _th=_T_REFINE_HANDOFF):
                    t_pick = time_steps[_k + 1].item()
                    if t_pick <= _th + 1e-6:
                        pts = [time_steps[kk].item()
                               for kk in range(_k + 1, num_steps + 1)]
                        mode = "naive-fine"
                    else:
                        # Preview grid: uniform in NAIVE sigma, re-shifted —
                        # exactly euler_lookahead's spacing.
                        shift = float(getattr(self, "_flow_shift", 3.0))
                        inv = t_pick / max(shift - (shift - 1.0) * t_pick,
                                           1e-8)
                        naive = np.linspace(inv, 0.0, _la + 1)
                        la_pts = shift * naive / (1.0 + (shift - 1.0) * naive)
                        coarse_seg = [float(p) for p in la_pts
                                      if p > _th + 1e-6]
                        fine_below = [float(p) for p in time_steps.tolist()
                                      if p < _th - 1e-6]
                        pts = coarse_seg + [_th] + fine_below
                        mode = "path-faithful"
                    _n_ref_steps = max(len(pts) - 1, 1)
                    with torch.no_grad():
                        for _ri_pos, i in enumerate(ridx):
                            z_f = X_all[i:i+1].clone()
                            for jj in range(len(pts) - 1):
                                v_f = self.predict_velocity(
                                    z_f, pts[jj], guidance_scale,
                                    prompt_idx=_pidx([i]))
                                z_f = z_f + (pts[jj + 1] - pts[jj]) * v_f
                                # REAL per-step refine progress (routed to the
                                # job's refine_progress, not the main bar).
                                if progress_callback:
                                    progress_callback(
                                        (_ri_pos * _n_ref_steps + jj + 1)
                                        / (len(ridx) * _n_ref_steps),
                                        f"Refining image {i} — step {jj + 1}/{_n_ref_steps}",
                                        interval_count - 1, n_segments_total,
                                    )
                            img_f = self.decode(z_f)
                            self._tensor_to_pil(img_f).save(
                                os.path.join(_dir, f"refined_particle_{i}.png"))
                            print(f"  Refined particle {i} ({mode}, "
                                  f"t'={t_pick:.3f}, {len(pts) - 1} steps).")

                kept_indices = self._prompt_user_selection(
                    total_particles, interval_count, interval_dir,
                    refine_fn=_refine_particles,
                )
                prev_kept_indices = kept_indices
                selections.append(dict(step=int(k), interval=int(interval_count),
                                       kept=[int(i) for i in kept_indices]))

                kept_latents = X_all[kept_indices].clone()
                # Clones inherit their parent's prompt variation.
                kept_pidx = self.particle_prompt_idx[
                    torch.as_tensor(kept_indices, dtype=torch.long)]
                self.particle_prompt_idx = kept_pidx.repeat_interleave(C)
                num_particles = len(kept_indices)
                total_particles = num_particles * C
                X_all = kept_latents.unsqueeze(1).expand(
                    -1, C, -1, -1, -1).clone()
                X_all = X_all.reshape(
                    total_particles, self.latent_channels, latent_h, latent_w)
                is_anchor = torch.zeros(total_particles, dtype=torch.bool)
                for i in range(num_particles):
                    is_anchor[i * C] = True

        # Final decode
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

        _seg_len_f, _la_f, _tot_units_f = _seg_progress_units(
            final_seg_idx, total_particles)
        pil_images = []
        with torch.no_grad():
            for p_idx in range(total_particles):
                if progress_callback:
                    _done = _seg_len_f + p_idx * _DEC_UNIT
                    progress_callback(
                        _done / _tot_units_f,
                        f"Decoding final image {p_idx + 1}/{total_particles}...",
                        final_seg_idx, n_segments_total,
                    )
                img_t = self.decode(X_all[p_idx:p_idx+1])
                pil_img = self._tensor_to_pil(img_t)
                pil_img.save(os.path.join(final_dir,
                                          f"particle_{p_idx:03d}.png"))
                pil_images.append(pil_img)

        features = self._extract_dino_features(pil_images)
        # Skip final IncrementalLayout.update: its kept_indices semantics
        # don't match `range(total_particles)` when total_particles grew via
        # cloning since the last interval update (pre-existing limitation
        # shared with flux_fmtt_dev). The per-interval layout plots remain
        # accurate; only the after-the-final-step view is dropped.
        if save_debug_plots:
            plot_image_cluster(
                features, pil_images,
                title="DINOv2 Cluster — Final",
                save_path=os.path.join(final_dir, "cluster_dinov2.png"),
                n_neighbors=min(15, total_particles - 1),
                min_dist=0.1, thumb_size=80,
            )
            self._save_grid(pil_images,
                            os.path.join(final_dir, "grid.png"), labels)

        # ===== SAVE THE NOISE: write the reproduction recipe =====
        import json as _json
        _recipe = dict(
            master_seed=master_seed,
            model_id=self.model_id,
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            rho=float(rho),
            schedule_mode=schedule_mode,
            num_clones=int(C),
            img_shape=[int(imgH), int(imgW)],
            prompts=list(getattr(self, "prompts", None) or [self.prompt]),
            sigmas=[float(t) for t in time_steps.tolist()],
            resample_steps=[int(s) for s in resample_steps],
            selections=selections,
            lineage=lineage,
            image_seeded=bool(init_latents is not None),
            init_strength=(float(init_strength)
                           if init_latents is not None else None),
            interval_start=int(interval_start),
        )
        _lineage_path = os.path.join(output_dir, "lineage.json")
        try:
            with open(_lineage_path, "w") as fh:
                _json.dump(_recipe, fh, indent=2)
            print(f"  Lineage recipe saved: {_lineage_path}")
        except Exception as _le:
            print(f"  WARNING: lineage.json write failed: {_le}")

        self._unload_dino()

        with torch.no_grad():
            return torch.cat(
                [self.decode(X_all[i:i+1]).float()
                 for i in range(total_particles)], dim=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SANA Interactive FMTT")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_clones", type=int, default=2)
    parser.add_argument("--num_resampling_steps", type=int, default=3)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--lookahead_steps", type=int, default=4)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="interactive_sana")
    args = parser.parse_args()

    sampler = SanaInteractive(prompt=args.prompt, device="cuda")
    best = sampler.sample_interactive(
        num_steps=args.num_steps,
        num_particles=args.num_particles,
        num_clones=args.num_clones,
        num_resampling_steps=args.num_resampling_steps,
        guidance_scale=args.guidance_scale,
        rho=args.rho,
        lookahead_steps=args.lookahead_steps,
        img_shape=(args.height, args.width),
        seed=args.seed,
        output_dir=args.output_dir,
        save_debug_plots=True,   # standalone CLI keeps the grids/plots
    )
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    pil_final = SanaInteractive._tensor_to_pil(best[0:1])
    pil_final.save(args.output)
    print(f"\nFinal image saved to: {args.output}")
