"""
Krea-2-Raw Interactive FMTT Sampler (flow-matching with GLASS Flows).

Mirror of SanaInteractive / FluxDevInteractive — same public contract
(`predict_velocity`, `decode`, `euler_lookahead`, `_glass_init`,
`sample_interactive`) — but for **Krea-2-Raw**, which is the QWEN-IMAGE lineage
(NOT FLUX):

  * transformer:  Krea2Transformer2DModel  (28-layer MMDiT, in_channels=64,
                  patch_size=2 → num_channels_latents = 64/4 = 16, velocity pred)
  * vae:          AutoencoderKLQwenImage   (z_dim=16, 3 temporal-downsample
                  stages → vae_scale_factor=8; per-channel latents_mean/std
                  normalisation, NOT a single scaling_factor)
  * text encoder: Qwen3VLModel + Qwen2Tokenizer  (multi-layer hidden-state tap,
                  encoder_attention_mask required)
  * scheduler:    FlowMatchEulerDiscreteScheduler with use_dynamic_shifting=True
                  and time_shift_type="exponential" — the resolution shift `mu`
                  is computed by `calculate_shift` (base, is_distilled=False).

Time convention (same as FLUX/SANA for the GLASS bridge): sigma ≡ t with
σ=1 (pure noise) → σ=0 (clean). The transformer's `timestep` input is the
**shifted** sigma directly (the pipeline passes `t / num_train_timesteps`,
and timesteps == sigmas * num_train_timesteps for this scheduler).

Latents are kept in **packed** form `[B, num_tokens, 64]` throughout — the same
layout the transformer and scheduler operate on. GLASS math is elementwise, so
packed Gaussian noise == unpacked Gaussian noise and the bridge drops in
verbatim.
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

from diffusers import Krea2Pipeline, FlowMatchEulerDiscreteScheduler

# Reuse FLUX's incremental layout, schedule and save-the-noise helpers —
# all model-agnostic (derive_seed/randn_from_seed draw on the CPU RNG stream).
from flux_fmtt_dev import (
    IncrementalLayout,
    compute_aggressive_diversity_schedule,
    compute_linear_diversity_schedule,
    derive_seed,
    randn_from_seed,
)
from cluster_images import plot_image_cluster


DEFAULT_MODEL_ID = "krea/Krea-2-Raw"


def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=6400,
                     base_shift=0.5, max_shift=1.15):
    """Replica of Krea2Pipeline.calculate_shift (linear mu interpolation)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _time_shift_exponential(mu: float, sigma_naive):
    """Replica of FlowMatchEulerDiscreteScheduler._time_shift_exponential with
    sigma=1.0 (the value the scheduler uses). Works on float or np.ndarray."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / sigma_naive - 1.0))


class Krea2Interactive:
    """Krea-2-Raw interactive sampler with GLASS Flows."""

    def __init__(self, prompt: str, model_id: str = DEFAULT_MODEL_ID,
                 device: str = "cuda", dtype=torch.bfloat16,
                 keep_text_encoder: bool = False):
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.prompt = prompt
        # When True the Gemma text encoder + tokenizer stay resident so we can
        # re-embed prompt variations after load (set_prompts / add_prompts) for
        # prompt-augmented round-1 diversity. Costs a few GB of VRAM.
        self.keep_text_encoder = bool(keep_text_encoder)
        # Multi-prompt caches: prompts[i] -> (cond embed, cond mask). Filled in
        # _load_model with the single base prompt; extended by set_prompts /
        # add_prompts. predict_velocity selects per-particle via prompt_idx.
        self.prompts = [prompt]

        torch.backends.cuda.matmul.allow_tf32 = True
        self._load_model(prompt)

    # ------------------------------------------------------------------
    def _load_model(self, prompt: str):
        import glob
        print(f"Loading Krea-2-Raw from: {self.model_id}")
        load_target = self.model_id
        _local = os.environ.get("KREA_LOCAL_PATH")
        if _local and os.path.isfile(os.path.join(_local, "model_index.json")):
            load_target = _local
            print(f"Loading Krea2 from node-local: {_local}")
        else:
            try:
                hf_home = os.environ.get(
                    "HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                repo_dir = os.path.join(
                    hf_home, "hub",
                    "models--" + self.model_id.replace("/", "--"))
                snaps = [s for s in glob.glob(os.path.join(repo_dir, "snapshots", "*"))
                         if os.path.isfile(os.path.join(s, "model_index.json"))]
                if snaps:
                    load_target = sorted(snaps)[-1]
                    print(f"Loading Krea2 from local snapshot dir: {load_target}")
            except Exception as e:
                print(f"Could not resolve local snapshot ({type(e).__name__}); using id.")

        self.pipeline = Krea2Pipeline.from_pretrained(
            load_target, torch_dtype=self.dtype
        ).to(self.device)

        # FlowMatchEulerDiscreteScheduler already; rebuild from config to be safe
        # (keeps use_dynamic_shifting + exponential time-shift config).
        self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.patch_size = self.pipeline.patch_size                # 2
        self.vae_scale_factor = self.pipeline.vae_scale_factor    # 8
        # num_channels_latents (UNPACKED) = in_channels / patch_size**2 = 16
        self.in_channels = self.transformer.config.in_channels    # 64 (packed)
        self.num_channels_latents = self.in_channels // (self.patch_size ** 2)
        self.num_train_timesteps = float(
            self.pipeline.scheduler.config.num_train_timesteps)   # 1000

        # VAE per-channel normalisation tensors.
        self.z_dim = self.vae.config.z_dim                        # 16
        self._latents_mean = torch.tensor(
            self.vae.config.latents_mean).view(1, self.z_dim, 1, 1, 1)
        self._latents_std = torch.tensor(
            self.vae.config.latents_std).view(1, self.z_dim, 1, 1, 1)
        # Lazy latent->RGB projection for live step previews (see
        # _step_preview_rgb): None = unloaded; "gray" = no fit file found.
        self._l2rgb_w = None
        self._l2rgb_b = None
        # Async preview emission queue (lazy; see _sp_submit).
        self._sp_q = None

        # Precompute text embeddings (cond + uncond). Both padded to
        # max_sequence_length so they share seq_len → CFG-doubled batch is safe.
        print("Precomputing text embeddings...")
        with torch.no_grad():
            pe, pm = self.pipeline.encode_prompt(prompt=prompt, device=self.device)
            ne, nm = self.pipeline.encode_prompt(prompt="", device=self.device)
        # PER-PROMPT caches. encode_prompt pads to a fixed max_sequence_length,
        # so every variation shares the same text_seq_len -> embeds stack and the
        # precomputed position_ids stay valid across variations. Index 0 is the
        # artist's exact prompt (anchor).
        self._embed_list = [pe.to(self.dtype)]     # list of [1, seq, ...] embeds
        self._mask_list = [pm]                      # list of [1, seq] masks
        # Back-compat single-prompt views (variation 0).
        self.cached_prompt_embeds = self._embed_list[0]
        self.cached_prompt_mask = self._mask_list[0]
        self.cached_neg_embeds = ne.to(self.dtype)
        self.cached_neg_mask = nm
        assert pe.shape[1] == ne.shape[1], "cond/uncond seq len differ"

        if self.keep_text_encoder:
            # Keep the Gemma text encoder + tokenizer resident for warm
            # re-prompting (set_prompts / add_prompts). VAE/transformer still fit.
            print("Krea-2-Raw loaded (warm mode — text encoder resident for "
                  "prompt variations)")
        else:
            # Drop text encoder + tokenizer to free memory (single-shot mode).
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
            print("Krea-2-Raw loaded (memory-efficient mode)")

    def free_text_encoder(self):
        """Drop the Gemma text encoder + tokenizer to reclaim VRAM. Call AFTER
        all prompt variations are embedded (set_prompts) when no mid-stream
        re-prompting is needed — reclaims a few GB before the heavy solve so the
        37GB transformer + variations fit comfortably on a 44GB L40S."""
        if getattr(self.pipeline, "text_encoder", None) is None:
            return
        print("[krea2] freeing text encoder (variations already embedded)...")
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

    # ------------------------------------------------------------------
    # Prompt-variation support (mirrors FluxDevInteractive.set_prompts /
    # add_prompts / num_prompts) — enables prompt-augmented round-1 diversity.
    # ------------------------------------------------------------------
    @property
    def num_prompts(self) -> int:
        return len(getattr(self, "_embed_list", [None]))

    def _encode_one(self, prompt: str):
        """Embed a single prompt into (embed[1,seq,...], mask[1,seq])."""
        if getattr(self.pipeline, "text_encoder", None) is None:
            raise RuntimeError(
                "krea2 set_prompts/add_prompts needs the text encoder, but it "
                "was freed. Construct with keep_text_encoder=True.")
        with torch.no_grad():
            pe, pm = self.pipeline.encode_prompt(prompt=prompt, device=self.device)
        return pe.to(self.dtype), pm

    def set_prompt(self, prompt: str):
        """Re-embed a single prompt (all particles share it)."""
        self.set_prompts([prompt])

    def set_prompts(self, prompts):
        """(Re)compute cached embeddings for one OR MORE prompt variations,
        reusing the resident text encoder. Index 0 is the anchor prompt."""
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [p for p in prompts if p is not None]
        if not prompts:
            prompts = [self.prompt]
        embeds, masks = [], []
        for p in prompts:
            pe, pm = self._encode_one(p)
            embeds.append(pe)
            masks.append(pm)
        self.prompts = list(prompts)
        self.prompt = prompts[0]
        self._embed_list = embeds
        self._mask_list = masks
        self.cached_prompt_embeds = embeds[0]
        self.cached_prompt_mask = masks[0]
        print(f"[krea2] embedded {len(prompts)} prompt variation(s).")

    def add_prompts(self, prompts):
        """APPEND prompt variations to the cached embeddings (span/stream mode).
        Append-only: existing indices never move. Returns new variation count."""
        if isinstance(prompts, str):
            prompts = [prompts]
        for p in prompts:
            if p is None:
                continue
            pe, pm = self._encode_one(p)
            self._embed_list.append(pe)
            self._mask_list.append(pm)
            self.prompts.append(p)
        print(f"[krea2] prompt variations now: {len(self._embed_list)}")
        return len(self._embed_list)

    # ------------------------------------------------------------------
    # Resolution-dependent flow schedule (precomputed per image shape)
    # ------------------------------------------------------------------
    def _setup_geometry(self, imgH: int, imgW: int):
        """Compute packed-token grid, position_ids, mu, and stash on self."""
        self.latent_h = int(imgH) // self.vae_scale_factor   # e.g. 64 @ 512
        self.latent_w = int(imgW) // self.vae_scale_factor
        self.grid_h = self.latent_h // self.patch_size        # 32 @ 512
        self.grid_w = self.latent_w // self.patch_size
        self.image_seq_len = self.grid_h * self.grid_w        # 1024 @ 512
        text_seq_len = self.cached_prompt_embeds.shape[1]
        self.position_ids = self.pipeline.prepare_position_ids(
            text_seq_len, self.grid_h, self.grid_w, self.device)
        sc = self.pipeline.scheduler.config
        self._mu = _calculate_shift(
            self.image_seq_len,
            sc.get("base_image_seq_len", 256),
            sc.get("max_image_seq_len", 6400),
            sc.get("base_shift", 0.5),
            sc.get("max_shift", 1.15),
        )
        print(f"[krea2] grid={self.grid_h}x{self.grid_w} seq={self.image_seq_len} mu={self._mu:.4f}")

    def _pack(self, latents: torch.Tensor) -> torch.Tensor:
        """[B, C, lh, lw] -> [B, tokens, C*p*p]."""
        return self.pipeline._pack_latents(
            latents, latents.shape[0], self.num_channels_latents,
            self.latent_h, self.latent_w)

    # ------------------------------------------------------------------
    # Velocity prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_sigma_schedule(self, num_steps):
        """Descending time-shifted sigma schedule, length num_steps+1 (final 0).
        Same convention the inline sampler uses (mirrors flux get_sigma_schedule)."""
        import numpy as np
        sigmas_naive = np.linspace(1.0, 1.0 / num_steps, num_steps)
        sigmas_arr = _time_shift_exponential(self._mu, sigmas_naive)
        return torch.tensor(list(sigmas_arr) + [0.0],
                            device=self.device, dtype=torch.float32)

    def predict_velocity(self, z: torch.Tensor, t_cur: float,
                         guidance_scale: float = 4.5,
                         prompt_idx=None) -> torch.Tensor:
        """Single-step flow-matching velocity for PACKED latent `z`
        [B, tokens, 64]. Krea2 CFG convention: v = v_cond + g*(v_cond - v_uncond)
        (== usual CFG with scale 1+g). timestep passed to the transformer is the
        shifted sigma directly (t_cur ∈ [0,1]).

        `prompt_idx` selects, per particle, which cached prompt variation to
        condition on (prompt-augmented diversity). None -> all particles use
        variation 0 (single prompt), byte-identical to the legacy behaviour.
        A LongTensor/list of length B selects per row (row i uses variation
        prompt_idx[i])."""
        if isinstance(t_cur, torch.Tensor):
            t_cur = t_cur.item()
        batch_size = z.shape[0]
        device = z.device
        dtype = self.dtype

        # Resolve per-row prompt variation. None -> variation 0 for everyone.
        embed_list = getattr(self, "_embed_list", [self.cached_prompt_embeds])
        mask_list = getattr(self, "_mask_list", [self.cached_prompt_mask])
        n_var = len(embed_list)
        if prompt_idx is None or n_var <= 1:
            row_pidx = [0] * batch_size
        else:
            if isinstance(prompt_idx, torch.Tensor):
                row_pidx = [int(v) for v in prompt_idx.tolist()]
            else:
                row_pidx = [int(v) for v in prompt_idx]
            if len(row_pidx) < batch_size:      # pad defensively
                row_pidx += [row_pidx[-1] if row_pidx else 0] * (
                    batch_size - len(row_pidx))
            row_pidx = [min(max(p, 0), n_var - 1) for p in row_pidx[:batch_size]]

        do_cfg = guidance_scale is not None and guidance_scale > 0.0
        latent_full = z.to(dtype)

        # Krea-2-Raw is fully resident at ~26GB on a 44GB L40S. A single B=1
        # transformer forward peaks at ~27.7GB (16GB headroom), but a CFG-doubled
        # OR multi-particle BATCHED forward overflows attention. Fix: process the
        # batch in ROW CHUNKS of size KREA2_FWD_CHUNK (default 1) and run cond /
        # uncond as SEPARATE passes (mirrors the vanilla pipeline). This makes any
        # particle count fit regardless of P.
        import os as _os
        chunk = int(_os.environ.get("KREA2_FWD_CHUNK", "1"))
        chunk = max(1, chunk)

        def _fwd_chunk(latent_c, embeds_full, mask_full):
            bs = latent_c.shape[0]
            ts = torch.full((bs,), float(t_cur), device=device, dtype=dtype)
            # embeds_full/mask_full already carry `bs` rows (built per chunk so
            # each row can use its own prompt variation).
            return self.transformer(
                hidden_states=latent_c,
                encoder_hidden_states=embeds_full,
                timestep=ts,
                position_ids=self.position_ids,
                encoder_attention_mask=mask_full,
                return_dict=False,
            )[0].float()

        outs = []
        for i in range(0, batch_size, chunk):
            latent_c = latent_full[i:i + chunk]
            bs = latent_c.shape[0]
            # Build this chunk's per-row cond embeds/masks from each row's
            # selected prompt variation (cat of [1, seq, ...] slices).
            pidx_chunk = row_pidx[i:i + bs]
            cond_e = torch.cat([embed_list[p] for p in pidx_chunk], dim=0)
            cond_m = torch.cat([mask_list[p] for p in pidx_chunk], dim=0)
            v_cond = _fwd_chunk(latent_c, cond_e, cond_m)
            if do_cfg:
                v_uncond = _fwd_chunk(latent_c,
                                      self.cached_neg_embeds.expand(bs, -1, -1, -1),
                                      self.cached_neg_mask.expand(bs, -1))
                v_c = v_cond + guidance_scale * (v_cond - v_uncond)
                del v_uncond
            else:
                v_c = v_cond
            outs.append(v_c)
            del v_cond
        v = torch.cat(outs, dim=0)
        return v.to(z.dtype)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sp_submit(self, fn):
        """Run a preview save/emit OFF the solve hot path.

        The PIL resize + jpg write + callback used to run synchronously
        inside the step loop; on 2026-08-26 (node loadavg 22 on an 8-core
        allocation) solves stalled at 19-337 s/step, and synchronous CPU/NFS
        work per step amplifies any such pressure. A 2-slot queue with
        drop-on-full makes previews best-effort: under pressure frames drop,
        solve steps never wait.
        """
        import queue as _q
        import threading as _t
        if self._sp_q is None:
            self._sp_q = _q.Queue(maxsize=2)

            def _worker():
                while True:
                    f = self._sp_q.get()
                    if f is None:
                        return
                    try:
                        f()
                    except Exception:
                        pass
            _t.Thread(target=_worker, daemon=True).start()
        try:
            self._sp_q.put_nowait(fn)
        except _q.Full:
            pass   # drop the frame — the next one supersedes it anyway

    def _step_preview_rgb_diff(self, z_packed: torch.Tensor) -> torch.Tensor:
        """Differentiable core of _step_preview_rgb: unpack -> de-std ->
        linear RGB projection, all torch, grad flows. Used by the FMRG
        repulsion (reward space) — no quantile/uint8 stages. Falls back to
        the raw de-std latent if no color fit is loaded (still a valid,
        merely less perceptual, repulsion space)."""
        imgH = self.latent_h * self.vae_scale_factor
        imgW = self.latent_w * self.vae_scale_factor
        lat = self.pipeline._unpack_latents(z_packed, imgH, imgW)[:, :, 0].float()
        mean = self._latents_mean.to(lat.device, torch.float32)
        std = self._latents_std.to(lat.device, torch.float32)
        lat = lat * (std[:, :, 0] if std.dim() == 5 else std) \
            + (mean[:, :, 0] if mean.dim() == 5 else mean)
        if self._l2rgb_w is None or isinstance(self._l2rgb_w, str):
            # trigger lazy load via the non-diff path's logic
            jp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "krea2_latent2rgb.json")
            if os.path.exists(jp):
                import json as _json
                d = _json.load(open(jp))
                self._l2rgb_w = torch.tensor(d["W"], device=lat.device)
                self._l2rgb_b = torch.tensor(d["b"], device=lat.device)
        if self._l2rgb_w is None or isinstance(self._l2rgb_w, str):
            return lat
        return torch.einsum("bchw,ck->bhwk", lat,
                            self._l2rgb_w.float().to(lat.device)) \
            + self._l2rgb_b.float().to(lat.device)

    def _step_preview_rgb(self, z_packed: torch.Tensor) -> np.ndarray:
        """~Free RGB preview of PACKED latents [B, tokens, 64] -> uint8 [B,h,w,3].

        For LIVE step previews of the running denoised estimate x0_hat during
        the solve. NOT a VAE decode: Krea-2-Raw is Qwen-Image-based (5D latent,
        per-channel mean/std), so FLUX's taef1 tiny-VAE cannot decode it and
        the full Qwen VAE costs seconds per image. Instead: unpack -> de-std ->
        a fitted 16->3 linear map (krea2_latent2rgb.json, least-squares against
        real decodes; grayscale channel-mean fallback until that file exists).
        One matmul per emission; the blurry-color-ghost look is the point —
        the artist watches the model's current belief sharpen.
        """
        imgH = self.latent_h * self.vae_scale_factor
        imgW = self.latent_w * self.vae_scale_factor
        lat = self.pipeline._unpack_latents(z_packed, imgH, imgW)  # [B,z,1,lh,lw]
        lat = lat[:, :, 0].float()                                  # [B,z,lh,lw]
        mean = self._latents_mean.to(lat.device, torch.float32)
        std = self._latents_std.to(lat.device, torch.float32)
        lat = lat * std[:, :, 0] if std.dim() == 5 else lat * std
        lat = lat + (mean[:, :, 0] if mean.dim() == 5 else mean)
        if self._l2rgb_w is None:
            jp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "krea2_latent2rgb.json")
            if os.path.exists(jp):
                import json as _json
                d = _json.load(open(jp))
                self._l2rgb_w = torch.tensor(d["W"], device=lat.device)   # [z,3]
                self._l2rgb_b = torch.tensor(d["b"], device=lat.device)   # [3]
            else:
                self._l2rgb_w = "gray"
        if isinstance(self._l2rgb_w, str):   # grayscale fallback, pre-fit
            g = lat.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            rgb = g.permute(0, 2, 3, 1)
        else:
            rgb = torch.einsum("bchw,ck->bhwk", lat, self._l2rgb_w.float()) \
                  + self._l2rgb_b.float()
        # Per-image robust normalize (2..98 pct) -> uint8.
        B = rgb.shape[0]
        flat = rgb.reshape(B, -1)
        lo = torch.quantile(flat, 0.02, dim=1).view(B, 1, 1, 1)
        hi = torch.quantile(flat, 0.98, dim=1).view(B, 1, 1, 1)
        rgb = ((rgb - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
        return (rgb * 255).to(torch.uint8).cpu().numpy()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode PACKED Krea2 latent [B, tokens, 64] -> image [B,3,H,W] in [0,1].
        Qwen-Image VAE: unpack -> latents/std + mean -> 5D decode -> drop frame."""
        imgH = self.latent_h * self.vae_scale_factor
        imgW = self.latent_w * self.vae_scale_factor
        lat = self.pipeline._unpack_latents(z, imgH, imgW)  # [B, z_dim, 1, lh, lw]
        lat = lat.to(self.vae.dtype)
        mean = self._latents_mean.to(lat.device, lat.dtype)
        std = self._latents_std.to(lat.device, lat.dtype)
        # Qwen-Image VAE de-normalisation: x = latent * std + mean. (The vanilla
        # pipeline writes this as `latent / (1/std) + mean` — same thing.)
        lat = lat * std + mean
        with torch.no_grad():
            samples = self.vae.decode(lat, return_dict=False)[0][:, :, 0]
        samples = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)
        return samples

    # ------------------------------------------------------------------
    # Euler lookahead preview (exponential-shifted sigma curve)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _advance_particle_segment(self, p, k_start, k_end_incl, time_steps,
                                  guidance_scale, is_anchor_p, gp, clone_row,
                                  X_all, on_step=None):
        """Advance ONE particle through a whole segment, start to finish.

        The sequential-order primitive (Jerry 2026-08-26: each child should
        complete start-to-finish; display mirrors execution). Valid because
        within a segment every particle's ODE is independent — clones bridge
        toward the parent's state FROZEN AT SPAWN (gp['x_t']), never its live
        state. Math is the same anchor-Euler / GLASS-bridge update as the
        together path, applied to a single row; gp coefficients are
        per-segment scalars shared by all clones, clone_row indexes this
        particle's row in gp's stacked tensors.
        """
        M = gp["M_inner"] if gp is not None else max(k_end_incl + 1 - k_start, 1)
        seg_start = gp["seg_start_step"] if gp is not None else k_start
        clip = 1e-8
        with torch.no_grad():
            for k in range(k_start, k_end_incl + 1):
                t_cur = time_steps[k].item()
                t_next = time_steps[k + 1].item()
                dt = t_next - t_cur
                z_p = X_all[p:p + 1]
                if is_anchor_p:
                    v = self.predict_velocity(
                        z_p, t_cur, guidance_scale,
                        prompt_idx=self.particle_prompt_idx[p:p + 1])
                    X_all[p] = (z_p + dt * v)[0]
                    x0_hat_p = z_p - t_cur * v
                else:
                    m = k - seg_start
                    s = m / M
                    ds = 1.0 / M
                    x_parent = gp["x_t"][clone_row:clone_row + 1]
                    bar_alpha_s = s * gp["bar_alpha"]
                    bar_sigma_s = ((1.0 - s) * gp["bar_sigma_0"]
                                   + s * gp["bar_sigma"])
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
                    bproduct = max(mu1 * mu1 * inv11 + 2 * mu1 * mu2 * inv12
                                   + mu2 * mu2 * inv22, clip)
                    t_star = max(min(1.0 / (1.0 + math.sqrt(
                        max(1.0 / bproduct, 0.0))), 0.999), 0.001)
                    sigma_star = 1.0 - t_star
                    alpha_star = t_star
                    w_xt = alpha_star * (mu1 * inv11 + mu2 * inv12) / bproduct
                    w_xs = alpha_star * (mu1 * inv12 + mu2 * inv22) / bproduct
                    S_input = w_xt * x_parent + w_xs * z_p
                    v_star = self.predict_velocity(
                        S_input, sigma_star, guidance_scale,
                        prompt_idx=self.particle_prompt_idx[p:p + 1])
                    denoiser = S_input - sigma_star * v_star
                    velocity = w1 * z_p + w2 * denoiser + w3 * x_parent
                    X_all[p] = (z_p + ds * velocity)[0]
                    x0_hat_p = denoiser
                if on_step is not None and (k % 2 == 0 or k == k_end_incl):
                    try:
                        on_step(k, x0_hat_p)
                    except Exception:
                        pass

    def euler_lookahead(self, z: torch.Tensor, t_cur: float,
                        guidance_scale: float, num_steps: int = 8,
                        prompt_idx=None, on_step=None) -> torch.Tensor:
        """Deterministic Euler integration t_cur → 0 for preview, placed on the
        SAME exponential-shifted sigma curve as the main schedule (uniform in
        *naive* σ, then re-shifted), so the few preview steps land at correct
        noise levels."""
        z_cur = z.clone()
        mu = float(getattr(self, "_mu", 0.5))
        # inverse exponential shift: naive = 1 / (1 + exp(mu)*(1/s - 1))
        s = float(t_cur)
        if s <= 1e-6:
            return z_cur
        inv = 1.0 / (1.0 + math.exp(mu) * (1.0 / s - 1.0))
        naive = np.linspace(inv, 0.0, num_steps + 1)
        # re-shift (guard s=0 -> 0)
        pts = np.where(naive > 0.0, _time_shift_exponential(mu, np.clip(naive, 1e-8, 1.0)), 0.0)
        for i in range(num_steps):
            t = float(pts[i]); t_next = float(pts[i + 1])
            v = self.predict_velocity(z_cur, t, guidance_scale,
                                      prompt_idx=prompt_idx)
            # Live per-step preview of THIS particle's denoised estimate.
            # The lookahead phase dominates round wall time (P particles x
            # num_steps solves, serial) and used to emit NOTHING until each
            # tile finally landed — measured 60+s of blank solve on a real
            # round-1. x0_hat from the velocity already computed; the caller
            # renders/emits (sub-ms latent2rgb + one tiny jpg).
            if on_step is not None:
                try:
                    on_step(i, num_steps, z_cur - t * v)
                except Exception:
                    pass
            z_cur = z_cur + (t_next - t) * v
        return z_cur

    # ------------------------------------------------------------------
    # DINO features
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
                    images=img, return_tensors="pt").to(self.device)
                outputs = self.dino_model(**inputs)
                embeddings.append(outputs.pooler_output.cpu().numpy())
        features = np.concatenate(embeddings, axis=0)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norms + 1e-8)

    # ------------------------------------------------------------------
    # GLASS bridge (flow-matching time convention; identical math to SANA/FLUX)
    # ------------------------------------------------------------------
    def _glass_init(self, X_bar: torch.Tensor, sigma_start: float,
                    sigma_end: float, rho: float,
                    generator=None) -> Tuple[torch.Tensor, dict]:
        """GLASS stochastic init (packed latents; elementwise math).

        The single eps draw here is the ONLY stochasticity in a GLASS clone.
        `generator` (a CPU torch.Generator, or a list of them — one per batch
        row) makes the draw exactly reproducible (save-the-noise, mirrors
        FluxDevInteractive._glass_init); None keeps the torch.randn_like path.
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
            x_t=X_bar.clone(), alpha_start=alpha_start, sigma_start=sigma_start,
            bar_gamma=bar_gamma, bar_alpha=bar_alpha, bar_sigma=bar_sigma,
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
            # REFINE: "refine:i,j" renders full-quality versions of those
            # particles (see the refine_fn closure in sample_interactive),
            # then loops back to waiting for the real KEEP selection.
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
    # Interactive sampling — identical structure to SanaInteractive
    # ------------------------------------------------------------------
    def sample_interactive(self,
                           num_steps: int = 28,
                           num_particles: int = 4,
                           num_clones: int = 2,
                           num_resampling_steps: int = 4,
                           guidance_scale: float = 4.5,
                           rho: float = 0.4,
                           lookahead_steps: int = 8,
                           lookahead_steps_first=None,
                           schedule_mode: Optional[str] = "linear",
                           img_shape: Optional[Tuple[int, int]] = None,
                           seed: Optional[int] = None,
                           master_seed: Optional[int] = None,
                           output_dir: str = "interactive_krea2",
                           prompt_idx=None,
                           progress_callback=None,
                           on_preview=None,
                           on_step_preview=None,
                           seq_order: bool = False) -> torch.Tensor:
        imgH, imgW = img_shape if img_shape is not None else (512, 512)
        device = self.device
        os.makedirs(output_dir, exist_ok=True)

        # ── Prompt-augmented diversity: prompt_idx[i] selects which cached
        # prompt variation anchor i is conditioned on. Clones inherit their
        # parent anchor's variation. None -> every particle uses variation 0
        # (byte-identical to the legacy single-prompt behaviour).
        if prompt_idx is not None:
            _pa = torch.as_tensor(list(prompt_idx), dtype=torch.long)
            if _pa.numel() < num_particles:      # pad with last / 0
                pad_val = int(_pa[-1].item()) if _pa.numel() else 0
                _pa = torch.cat([_pa, torch.full(
                    (num_particles - _pa.numel(),), pad_val, dtype=torch.long)])
            self.particle_prompt_idx = _pa[:num_particles].clone()
        else:
            self.particle_prompt_idx = torch.zeros(num_particles, dtype=torch.long)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # ===== SAVE THE NOISE: session master seed + lineage recording =====
        # Mirrors FluxDevInteractive.sample_interactive: every stochastic draw
        # (root init noise, GLASS clone eps) is derived deterministically from
        # master_seed via derive_seed()/randn_from_seed(), and every particle
        # creation is recorded in `lineage` -> written to lineage.json.
        if master_seed is None:
            master_seed = int(seed) if seed is not None else \
                random.SystemRandom().randrange(2 ** 31)
        master_seed = int(master_seed)
        print(f"  [save-noise] master_seed={master_seed}")
        lineage: List[dict] = []
        selections: List[dict] = []
        spawn_event_counter = 0

        self._setup_geometry(imgH, imgW)
        latent_h, latent_w = self.latent_h, self.latent_w

        n_segments_total = num_resampling_steps + 1
        if progress_callback:
            progress_callback(0.02, "Loading image similarity model...",
                              0, n_segments_total)
        self._load_dino()
        if progress_callback:
            progress_callback(0.05, "Preparing...", 0, n_segments_total)

        # Packed anchors: sample unpacked Gaussian then pack (== Krea2
        # prepare_latents). Each root's noise is drawn from its own derived
        # seed (save-the-noise) so it is exactly reproducible.
        root_seeds = [derive_seed(master_seed, i) for i in range(num_particles)]
        anchors_unpacked = torch.stack([
            randn_from_seed(rs, (self.num_channels_latents, latent_h, latent_w),
                            device, self.dtype)
            for rs in root_seeds
        ])
        anchors = self._pack(anchors_unpacked)   # [P, tokens, 64]
        for _ri in range(num_particles):
            lineage.append(dict(idx=_ri, kind="root", parent=None,
                                seed=root_seeds[_ri], rho=None,
                                sigma_start=None, sigma_end=None,
                                spawn_step=None,
                                prompt_idx=int(self.particle_prompt_idx[_ri].item())))
        tok = anchors.shape[1]
        feat = anchors.shape[2]
        total_particles = num_particles
        C = num_clones

        # Exponential-shifted sigma schedule (matches scheduler.set_timesteps(mu)).
        sigmas_naive = np.linspace(1.0, 1.0 / num_steps, num_steps)
        sigmas_arr = _time_shift_exponential(self._mu, sigmas_naive)
        print(f"[krea2] flow mu={self._mu:.4f} sigma[0..2]={sigmas_arr[:3]}")
        time_steps = torch.tensor(
            list(sigmas_arr) + [0.0], device=device, dtype=torch.float32)

        if schedule_mode == "aggressive" and num_resampling_steps > 0:
            resample_steps = compute_aggressive_diversity_schedule(
                num_steps, num_resampling_steps)
        elif schedule_mode == "linear" and num_resampling_steps > 0:
            resample_steps = compute_linear_diversity_schedule(
                num_steps, num_resampling_steps)
        elif num_resampling_steps > 0:
            resample_steps = sorted(
                int(round((i + 1) * num_steps / (num_resampling_steps + 1)))
                for i in range(num_resampling_steps))
        else:
            resample_steps = []
        resample_at = set(resample_steps)
        segment_bounds = [0] + resample_steps + [num_steps]

        X_all = anchors.clone()
        is_anchor = torch.ones(total_particles, dtype=torch.bool)
        glass_params = None
        interval_count = 0
        # Live step-preview state: x0_hat buffer + emission cadence. Every
        # step costs one matmul + P tiny jpg writes (~20 ms); the env knob
        # exists so a struggling box can thin the cadence without a code edit.
        _x0_hat = None
        _sp_every = max(1, int(os.environ.get("KREA2_STEP_PREVIEW_EVERY", "2")))
        _sp_dir = os.path.join(output_dir, "steps")
        if on_step_preview is not None:
            os.makedirs(_sp_dir, exist_ok=True)

        # CADS-style condition annealing (diversity guidance tier 1, default
        # OFF). Early in the solve each particle conditions on a NOISED copy
        # of its prompt embedding (per-particle noise, seeded for the
        # save-the-noise contract), annealing to the clean embedding by
        # KREA2_CADS_T1 — trajectories fan out early, converge prompt-
        # faithful late. Per-particle expansion matters: with augment OFF all
        # particles share embed 0, and shared noise would add zero RELATIVE
        # diversity. Embeds + prompt_idx are restored every step end, so
        # boundary lookaheads/decodes always see clean conditioning.
        _cads_s = float(os.environ.get("KREA2_CADS_S", "0") or 0)
        _cads_t1 = float(os.environ.get("KREA2_CADS_T1", "0.6"))
        _cads_t2 = float(os.environ.get("KREA2_CADS_T2", "0.9"))
        _cads_saved = None   # (embed_list, mask_list, prompt_idx) originals

        # SEQUENTIAL ORDER state: presolved segment window + cached previews.
        # When seq_order, each segment is solved particle-by-particle at
        # segment start (each child completes start-to-finish and previews
        # immediately); the main loop then skips its per-step advance through
        # that window, and the boundary reuses the cached decodes.
        _seq_pils = {}
        _seq_boundary = -1
        inc_layout = IncrementalLayout(n_neighbors=15, min_dist=0.2)
        prev_kept_indices = None

        from tqdm import tqdm
        import time as _time
        pbar = tqdm(range(num_steps), total=num_steps, desc="Krea2-Interactive")

        _step_t0 = _time.time()
        for k in pbar:
            # Forensic marker for the intermittent 19-337 s/step stalls seen
            # 2026-08-26 (node loadavg 22): a timestamped line per slow step
            # lets the next occurrence be correlated against augment activity
            # and co-tenant load instead of guessed at.
            _now = _time.time()
            if k > 0 and (_now - _step_t0) > 15.0:
                print(f"[slow-step] k={k-1} took {_now - _step_t0:.1f}s "
                      f"(wall {_time.strftime('%H:%M:%S')})", flush=True)
            _step_t0 = _now

            # CADS perturb for THIS step (see setup above). Captured fresh
            # and restored+cleared at step end, so boundary spawns that
            # rewrite particle_prompt_idx are never clobbered by a stale save.
            if _cads_s > 0.0:
                _t_for_g = time_steps[k].item()
                _g = (1.0 if _t_for_g <= _cads_t1 else
                      0.0 if _t_for_g >= _cads_t2 else
                      (_cads_t2 - _t_for_g) / max(_cads_t2 - _cads_t1, 1e-6))
                if _g < 1.0:
                    _cads_saved = (self._embed_list, self._mask_list,
                                   self.particle_prompt_idx)
                    _oe, _om, _op = _cads_saved
                    _nl, _nm = [], []
                    for _pi in range(total_particles):
                        _src = int(_op[min(_pi, len(_op) - 1)].item())
                        _e = _oe[min(_src, len(_oe) - 1)]
                        _eps = randn_from_seed(
                            derive_seed(master_seed, 9100 + k, _pi),
                            tuple(_e.shape), _e.device, torch.float32)
                        _y = (math.sqrt(_g) * _e.float()
                              + _cads_s * math.sqrt(1.0 - _g) * _eps)
                        # Rescale PER TOKEN, not globally: the 08-26 A/B at
                        # s=0.15 with a global mean/std rescale produced pure
                        # glitch output (DINO sim 0.46 measured corruption,
                        # not diversity — the montage caught it). Preserving
                        # each token's norm keeps the embedding's structure.
                        _en = _e.float().norm(dim=-1, keepdim=True)
                        _yn = _y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        _y = _y * (_en / _yn)
                        _nl.append(_y.to(_e.dtype))
                        _nm.append(_om[min(_src, len(_om) - 1)])
                    self._embed_list = _nl
                    self._mask_list = _nm
                    self.particle_prompt_idx = torch.arange(total_particles)
            if progress_callback:
                seg_idx = 0
                for b in resample_steps:
                    if k >= b:
                        seg_idx += 1
                seg_start = segment_bounds[seg_idx]
                seg_end = segment_bounds[seg_idx + 1]
                seg_len = max(seg_end - seg_start, 1)
                frac = 0.05 + 0.35 * ((k - seg_start) / seg_len)
                progress_callback(
                    frac, f"Generating images — step {k + 1 - seg_start}/{seg_len}",
                    seg_idx, n_segments_total)

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
                    parent_states = torch.zeros(
                        len(clone_indices), tok, feat,
                        device=device, dtype=self.dtype)
                    for ci, idx in enumerate(clone_indices):
                        group = idx.item() // C
                        parent_states[ci] = X_all[group * C]
                    # Per-clone derived seeds -> per-row CPU generators, so each
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
                            prompt_idx=int(self.particle_prompt_idx[_ci].item())))
                    # STAGED DIVERSITY CONTRACTION (Jerry 2026-08-27, the
                    # DEFAULT not a mode): early breed stages stay close to
                    # round-1 variety, later stages tighten. rho is the
                    # parent-correlation (bar_gamma ~ rho; fresh noise
                    # ~ sqrt(1-rho^2)), so the schedule scales rho UP with
                    # the stage index: first children at 0.5x the base rho,
                    # final stage at ~1.4x (clamped). KREA2_RHO_SCHEDULE=0
                    # restores the flat rho.
                    _rho_eff = rho
                    if os.environ.get("KREA2_RHO_SCHEDULE", "1") == "1":
                        _stages = max(num_resampling_steps - 1, 1)
                        _prog = min(max(interval_count - 1, 0), _stages) / _stages
                        _s1 = float(os.environ.get("KREA2_RHO_STAGE1", "0.5"))
                        _rho_eff = min(max(rho * (_s1 + (1.4 - _s1) * _prog),
                                           0.04), 0.85)
                        print(f"[rho-schedule] stage={interval_count} "
                              f"rho {rho:.2f}->{_rho_eff:.2f}")
                    clone_inner, gp = self._glass_init(
                        parent_states, sigma_seg_start, sigma_seg_end, _rho_eff,
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

            # SEQUENTIAL presolve: at segment start, run each particle's
            # whole segment + immediate preview, particle by particle. The
            # main loop then skips its per-step advance through this window
            # (the boundary block still fires normally at _seq_boundary).
            if (seq_order and glass_params.get("seg_start_step") == k
                    and _seq_boundary < 0):
                _sb = next((b for b in segment_bounds if b > k), num_steps)
                if _sb < num_steps:   # only segments ending at a boundary
                    _seq_boundary = _sb
                    _nxt_int = interval_count + 1
                    _nxt_dir = os.path.join(output_dir, f"interval_{_nxt_int}")
                    os.makedirs(_nxt_dir, exist_ok=True)
                    _cmap = {int(ci.item()): r for r, ci in
                             enumerate(glass_params["clone_indices"])}
                    _t_prev = time_steps[_sb + 1].item()
                    import time as _wt
                    for _p in range(total_particles):
                        _p_t0 = _wt.time()
                        if progress_callback:
                            progress_callback(
                                0.10 + 0.85 * (_p / total_particles),
                                f"Painting image {_p + 1}/{total_particles}"
                                " — start to finish...",
                                interval_count, n_segments_total)
                        _onstep = None
                        if on_step_preview is not None:
                            def _onstep(_k, _x0p, _pp=_p):
                                arr1 = self._step_preview_rgb(_x0p)

                                def _sv(_a=arr1, _kk=_k):
                                    _fp = os.path.join(
                                        _sp_dir, f"p{_pp:03d}.jpg")
                                    Image.fromarray(_a[0]).resize(
                                        (256, 256), Image.BILINEAR).save(
                                        _fp, quality=70)
                                    on_step_preview(_kk, num_steps, [
                                        os.path.join(_sp_dir, f"p{_q:03d}.jpg")
                                        for _q in range(total_particles)], _pp)
                                self._sp_submit(_sv)
                        self._advance_particle_segment(
                            _p, k, _sb, time_steps, guidance_scale,
                            bool(is_anchor[_p]), glass_params,
                            _cmap.get(_p), X_all, on_step=_onstep)
                        with torch.no_grad():
                            # Round-1 sits at high sigma — honor the richer
                            # first-interval lookahead like the lockstep path
                            # does (sequential silently used the breed value,
                            # presenting coarser round-1 tiles; caught by
                            # Jerry's step-budget question 2026-08-27).
                            _la_n = (lookahead_steps_first
                                     if (_nxt_int == 1
                                         and lookahead_steps_first)
                                     else lookahead_steps)
                            _z0 = self.euler_lookahead(
                                X_all[_p:_p + 1], _t_prev, guidance_scale,
                                num_steps=_la_n,
                                prompt_idx=self.particle_prompt_idx[_p:_p + 1])
                            _img_t = self.decode(_z0)
                        _pil = self._tensor_to_pil(_img_t)
                        _pil.save(os.path.join(
                            _nxt_dir, f"particle_{_p:03d}.png"))
                        _seq_pils[_p] = _pil
                        # Forensic marker for the presolve path (the main
                        # loop's slow-step marker never sees these steps).
                        _p_el = _wt.time() - _p_t0
                        if _p_el > 25.0:
                            print(f"[slow-step] presolve particle {_p} took "
                                  f"{_p_el:.1f}s (wall "
                                  f"{_wt.strftime('%H:%M:%S')})", flush=True)
                        if on_preview is not None:
                            try:
                                on_preview(_nxt_int, _p, os.path.join(
                                    _nxt_dir, f"particle_{_p:03d}.png"))
                            except Exception:
                                pass

            m = k - glass_params["seg_start_step"]
            M = glass_params["M_inner"]
            s = m / M
            ds = 1.0 / M
            # Inside a presolved window the states are already final for
            # these k — skip the batched advance (and its emission) entirely.
            _skip_advance = (seq_order and _seq_boundary >= 0
                             and k <= _seq_boundary)

            with torch.no_grad():
                anchor_indices = is_anchor.nonzero(as_tuple=True)[0]
                if not _skip_advance and len(anchor_indices) > 0:
                    # BATCHED anchor Euler step: one forward over all anchors
                    # (was a per-particle loop). Same MATH; bf16 batched GEMM
                    # is ~1e-2 different from single-row (cuBLAS reduction order).
                    idx_a = anchor_indices
                    z_a = X_all[idx_a]
                    v_a = self.predict_velocity(
                        z_a, t_cur, guidance_scale,
                        prompt_idx=self.particle_prompt_idx[idx_a])
                    X_all[idx_a] = z_a + dt * v_a
                    # FMRG-style DIVERSITY REPULSION (flag-gated, default off;
                    # Jerry 2026-08-26 "things like fmrg should work"). Reward
                    # r = pairwise distance between the particles' one-step
                    # endpoint estimates PROJECTED through the fitted
                    # latent->RGB map — a perceptually-calibrated space
                    # (rmse 0.067 vs real decodes) reachable by ONE matmul,
                    # so grad(r) costs no model forward/backward at all
                    # (autograd through the linear projection only). The
                    # control u = lam * grad_x0 r is applied with the
                    # DPS-style identity pullback and annealed by sigma_t so
                    # late steps go untouched (path-consistency contract:
                    # spread forms early, convergence is never fought).
                    # Anchors only — clones follow their parents.
                    _fmrg_s = float(os.environ.get("KREA2_FMRG_S", "0") or 0)
                    if _fmrg_s > 0 and len(idx_a) > 1 and t_cur > 0.35:
                        try:
                            _x0a = (z_a - t_cur * v_a).detach().float()
                            _x0a.requires_grad_(True)
                            _rgb = self._step_preview_rgb_diff(_x0a)
                            _fl = _rgb.reshape(len(idx_a), -1)
                            _d = torch.cdist(_fl, _fl)
                            _r = _d.sum() / max(len(idx_a) * (len(idx_a) - 1), 1)
                            _r.backward()
                            _g = _x0a.grad
                            # FMRG-J normalization (paper Sec E, Jerry's
                            # correction): u_hat = (u/||u||)*||v||, lambda_t =
                            # eta*dt — guidance rescaled to the VELOCITY
                            # magnitude per sample, so eta is invariant to the
                            # reward's gradient scale. Paper uses eta=1.0 at
                            # FLUX scale; env KREA2_FMRG_S carries eta.
                            _gn = _g.flatten(1).norm(dim=1).clamp_min(1e-8)
                            _vn = v_a.flatten(1).norm(dim=1).clamp_min(1e-8)
                            _u = (_g / _gn.view(-1, 1, 1)) \
                                * _vn.view(-1, 1, 1) * _fmrg_s * abs(dt)
                            X_all[idx_a] = X_all[idx_a] + _u.to(X_all.dtype)
                        except Exception as _fx:
                            print(f"[fmrg] skipped at k={k}: {_fx}")
                    # Denoised estimate for live step previews: the flow's
                    # current best guess of the clean latent, from the velocity
                    # we already paid for (x0_hat = x_t - sigma_t * v).
                    if on_step_preview is not None:
                        if _x0_hat is None or _x0_hat.shape != X_all.shape:
                            _x0_hat = torch.zeros_like(X_all)
                        _x0_hat[idx_a] = z_a - t_cur * v_a

            clone_indices = glass_params["clone_indices"]
            if not _skip_advance and len(clone_indices) > 0:
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
                    min(1.0/(1.0+math.sqrt(max(1.0/bproduct, 0.0))), 0.999), 0.001)
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
                        prompt_idx=self.particle_prompt_idx[cidx])
                    denoiser = S_input - sigma_star * v_star
                    velocity = w1 * x_clone + w2 * denoiser + w3 * x_parent
                    X_all[cidx] = x_clone + ds * velocity
                    # Clones' preview estimate: the GLASS branch already
                    # computes a denoiser (clean-latent estimate at sigma_star)
                    # — reuse it, no extra forward.
                    if on_step_preview is not None:
                        if _x0_hat is None or _x0_hat.shape != X_all.shape:
                            _x0_hat = torch.zeros_like(X_all)
                        _x0_hat[cidx] = denoiser.to(_x0_hat.dtype)

            # Emit live step previews: the current denoised estimate of every
            # particle, as tiny jpgs the server can serve immediately. Emitted
            # every _sp_every steps + always on the final step of a segment.
            if (on_step_preview is not None and _x0_hat is not None
                    and not _skip_advance
                    and (k % _sp_every == 0 or (k + 1) in resample_at
                         or k == num_steps - 1)):
                try:
                    # GPU->numpy stays on the solve thread (milliseconds);
                    # PIL resize + NFS write + callback go to the worker.
                    arr = self._step_preview_rgb(_x0_hat)      # [P,h,w,3] u8

                    def _emit(_arr=arr, _k=k, _n=num_steps):
                        paths = []
                        for _pi in range(_arr.shape[0]):
                            _pp = os.path.join(_sp_dir, f"p{_pi:03d}.jpg")
                            Image.fromarray(_arr[_pi]).resize(
                                (256, 256), Image.BILINEAR).save(
                                _pp, quality=70)
                            paths.append(_pp)
                        on_step_preview(_k, _n, paths)
                    self._sp_submit(_emit)
                except Exception as _sp_exc:
                    print(f"[step-preview] emission failed at k={k}: {_sp_exc}")

            # Restore clean conditioning before any boundary work (lookahead
            # previews, decodes, spawns) and clear the save for re-capture.
            if _cads_saved is not None:
                (self._embed_list, self._mask_list,
                 self.particle_prompt_idx) = _cads_saved
                _cads_saved = None

            if k in resample_at:
                glass_params = None
                _seq_boundary = -1   # presolved window (if any) ends here
                interval_count += 1
                interval_dir = os.path.join(output_dir, f"interval_{interval_count}")
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

                pil_images = []
                with torch.no_grad():
                    for p_idx in range(total_particles):
                        # Sequential presolve already rendered + emitted this
                        # particle's preview — reuse it, don't decode twice.
                        if p_idx in _seq_pils:
                            pil_images.append(_seq_pils.pop(p_idx))
                            continue
                        if progress_callback:
                            dec_frac = 0.40 + 0.50 * (p_idx / total_particles)
                            progress_callback(
                                dec_frac,
                                f"Decoding image {p_idx + 1}/{total_particles}...",
                                seg_idx_done, n_segments_total)
                        # Per-step liveness for the lookahead phase: particle
                        # p's tile visibly sharpens while ITS solve runs (the
                        # sequential one-stroke-at-a-time rhythm), then locks
                        # in as the real tile below. Every 2nd step + final.
                        _la_emit = None
                        if on_step_preview is not None:
                            def _la_emit(_si, _sn, _x0p, _pi=p_idx):
                                if _si % 2 == 1 and _si != _sn - 1:
                                    return
                                # GPU part sync (ms); CPU+NFS part async.
                                arr1 = self._step_preview_rgb(_x0p)

                                def _save(_a=arr1, _s=_si, _n=_sn):
                                    _pp = os.path.join(_sp_dir, f"p{_pi:03d}.jpg")
                                    Image.fromarray(_a[0]).resize(
                                        (256, 256), Image.BILINEAR).save(
                                        _pp, quality=70)
                                    # 4th arg = which particle the GPU is
                                    # actively refining (UI highlights it).
                                    on_step_preview(_s, _n, [
                                        os.path.join(_sp_dir, f"p{_q:03d}.jpg")
                                        for _q in range(total_particles)], _pi)
                                self._sp_submit(_save)
                        z0 = self.euler_lookahead(
                            X_all[p_idx:p_idx+1], t_now,
                            guidance_scale, num_steps=lookahead_steps,
                            prompt_idx=self.particle_prompt_idx[p_idx:p_idx+1],
                            on_step=_la_emit)
                        img_t = self.decode(z0)
                        pil_img = self._tensor_to_pil(img_t)
                        pil_img.save(os.path.join(
                            interval_dir, f"particle_{p_idx:03d}.png"))
                        pil_images.append(pil_img)
                        # PROGRESSIVE PAINT: emit this tile the instant it's on
                        # disk so the UI shows it ~immediately instead of waiting
                        # for the whole interval to decode + cluster.
                        if on_preview is not None:
                            try:
                                on_preview(interval_count, p_idx,
                                           os.path.join(interval_dir,
                                                        f"particle_{p_idx:03d}.png"))
                            except Exception:
                                pass

                if progress_callback:
                    progress_callback(0.95, "Clustering images...",
                                      seg_idx_done, n_segments_total)

                features = self._extract_dino_features(pil_images)

                # ── DIVERSITY RESPAWN (gradient-free tier of the FMRG plan;
                # Jerry 2026-08-26 "go"). The DINO features just computed for
                # the layout double as a near-duplicate detector: when two
                # particles' previews are almost the same image (cosine > tau),
                # the later one is wasted compute — re-noise it at the CURRENT
                # sigma (SDEdit-style: x = (1-sigma)*z0_dup + sigma*eps_fresh,
                # the codebase's own renoise convention, cf. FLUX spawn_mode
                # "sdedit") so the next segment re-solves it toward somewhere
                # new. Never at the final interval (nothing would re-solve it),
                # never anchors (they carry lineage), at most half a round.
                # Selection-contract note: the respawned tile's shown preview
                # no longer matches its future — accepted because its kept
                # twin (cos>tau) shows ~the same image and carries that visual
                # signal forward if picked; respawning BOTH twins is blocked
                # by construction (j<i and j is never respawned).
                # Gate: KREA2_DIVERSITY_RESPAWN=0 disables; tau via env.
                # No "final interval" guard needed: every k in resample_at is
                # followed by another solve segment by construction (the final
                # images decode AFTER the step loop) — t_now floor suffices.
                _dr_on = os.environ.get("KREA2_DIVERSITY_RESPAWN", "1") == "1"
                if _dr_on and t_now > 0.05:
                    try:
                        _tau = float(os.environ.get(
                            "KREA2_DIVERSITY_TAU", "0.93"))
                        _F = features / (np.linalg.norm(
                            features, axis=1, keepdims=True) + 1e-8)
                        _S = _F @ _F.T
                        _respawned = []
                        _max_n = max(1, total_particles // 2)
                        for _i in range(total_particles):
                            if len(_respawned) >= _max_n:
                                break
                            for _j in range(_i):
                                if (_S[_i, _j] > _tau and _i not in _respawned
                                        and not bool(is_anchor[_i])):
                                    _rs = derive_seed(
                                        master_seed, 7000 + interval_count,
                                        _i)
                                    _eps = randn_from_seed(
                                        _rs, tuple(X_all[_i].shape),
                                        device, X_all.dtype)
                                    _z0d = self.euler_lookahead(
                                        X_all[_i:_i+1], t_now, guidance_scale,
                                        num_steps=2,
                                        prompt_idx=self.particle_prompt_idx[_i:_i+1])
                                    X_all[_i] = ((1.0 - t_now) * _z0d[0]
                                                 + t_now * _eps)
                                    _respawned.append(_i)
                                    print(f"[diversity-respawn] interval="
                                          f"{interval_count} p={_i} dup-of="
                                          f"{_j} cos={_S[_i,_j]:.3f} "
                                          f"sigma={t_now:.3f}")
                                    break
                    except Exception as _dr_exc:
                        print(f"[diversity-respawn] skipped: {_dr_exc}")
                if inc_layout is not None:
                    if interval_count == 1:
                        inc_layout.initialize(features, pil_images)
                    else:
                        inc_layout.update(
                            prev_kept_indices, features, pil_images,
                            num_clones=C, round_num=interval_count)
                    inc_layout.plot(
                        title=f"Interval {interval_count} (step {k}, t={t_now:.3f})",
                        save_path=os.path.join(interval_dir,
                                               "layout_incremental.png"),
                        thumb_size=80)

                cluster_path = os.path.join(interval_dir, "cluster_dinov2.png")
                plot_image_cluster(
                    features, pil_images,
                    title=f"DINOv2 Cluster — Interval {interval_count}",
                    save_path=cluster_path,
                    n_neighbors=min(15, total_particles - 1),
                    min_dist=0.1, thumb_size=80)

                self._save_grid(pil_images,
                                os.path.join(interval_dir, "grid.png"), labels)

                # refine_fn renders a FULL-QUALITY version of a particle on
                # request: deterministic Euler over the full remaining fine
                # schedule (time_steps[k+1:], NOT the coarse lookahead count)
                # from the stored latent, decoded with the full VAE. No GLASS
                # spawn, no fresh noise; X_all untouched. Mirrors the closure
                # in FluxDevInteractive.sample_interactive.
                def _refine_particles(ridx, _k=k, _dir=interval_dir):
                    with torch.no_grad():
                        for i in ridx:
                            z_f = X_all[i:i+1].clone()
                            _pi = self.particle_prompt_idx[i:i+1]
                            for kk in range(_k + 1, num_steps):
                                v_f = self.predict_velocity(
                                    z_f, time_steps[kk].item(), guidance_scale,
                                    prompt_idx=_pi)
                                z_f = z_f + (time_steps[kk + 1].item()
                                             - time_steps[kk].item()) * v_f
                            img_f = self.decode(z_f)
                            self._tensor_to_pil(img_f).save(
                                os.path.join(_dir, f"refined_particle_{i}.png"))
                            print(f"  Refined particle {i} "
                                  f"({num_steps - _k - 1} fine steps).")

                kept_indices = self._prompt_user_selection(
                    total_particles, interval_count, interval_dir,
                    refine_fn=_refine_particles)
                prev_kept_indices = kept_indices
                selections.append(dict(step=int(k), interval=int(interval_count),
                                       kept=[int(i) for i in kept_indices]))

                kept_latents = X_all[kept_indices].clone()
                # Each kept anchor's prompt variation propagates to its C clones
                # so the next round keeps the diversity of the selected images.
                _kept_pidx = self.particle_prompt_idx[
                    torch.as_tensor(kept_indices, dtype=torch.long)]
                num_particles = len(kept_indices)
                total_particles = num_particles * C
                X_all = kept_latents.unsqueeze(1).expand(-1, C, -1, -1).clone()
                X_all = X_all.reshape(total_particles, tok, feat)
                self.particle_prompt_idx = _kept_pidx.repeat_interleave(C)
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

        pil_images = []
        with torch.no_grad():
            for p_idx in range(total_particles):
                if progress_callback:
                    dec_frac = 0.40 + 0.50 * (p_idx / total_particles)
                    progress_callback(
                        dec_frac,
                        f"Decoding final image {p_idx + 1}/{total_particles}...",
                        final_seg_idx, n_segments_total)
                img_t = self.decode(X_all[p_idx:p_idx+1])
                pil_img = self._tensor_to_pil(img_t)
                pil_img.save(os.path.join(final_dir, f"particle_{p_idx:03d}.png"))
                pil_images.append(pil_img)

        features = self._extract_dino_features(pil_images)
        plot_image_cluster(
            features, pil_images, title="DINOv2 Cluster — Final",
            save_path=os.path.join(final_dir, "cluster_dinov2.png"),
            n_neighbors=min(15, total_particles - 1), min_dist=0.1, thumb_size=80)
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
            spawn_mode="glass",
            schedule_mode=schedule_mode,
            num_clones=int(C),
            img_shape=[int(imgH), int(imgW)],
            prompts=list(getattr(self, "prompts", [self.prompt])),
            sigmas=[float(t) for t in time_steps.tolist()],
            resample_steps=[int(s) for s in resample_steps],
            image_seeded=False,
            init_strength=None,
            selections=selections,
            lineage=lineage,
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
    parser = argparse.ArgumentParser(description="Krea-2-Raw Interactive FMTT")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_clones", type=int, default=2)
    parser.add_argument("--num_resampling_steps", type=int, default=3)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--lookahead_steps", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="interactive_krea2")
    args = parser.parse_args()

    sampler = Krea2Interactive(prompt=args.prompt, device="cuda")
    best = sampler.sample_interactive(
        num_steps=args.num_steps, num_particles=args.num_particles,
        num_clones=args.num_clones, num_resampling_steps=args.num_resampling_steps,
        guidance_scale=args.guidance_scale, rho=args.rho,
        lookahead_steps=args.lookahead_steps,
        img_shape=(args.height, args.width), seed=args.seed,
        output_dir=args.output_dir)
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    Krea2Interactive._tensor_to_pil(best[0:1]).save(args.output)
    print(f"\nFinal image saved to: {args.output}")
