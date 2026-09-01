"""
FLUX.1-Dev Interactive FMTT Sampler (standalone — no fluxfm_guidance dependency)

Interactive particle pruning with GLASS Flows for stochastic clone divergence.
Uses standard FluxPipeline from diffusers with instantaneous velocity (Euler ODE).

Based on flux_fmtt_interactive.py but removes all flow map / LoRA / VLM dependencies.

Time convention: FLUX uses t=1 (noise) → t=0 (clean).
"""

import os
import gc
import math
import random
import sys

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional, Tuple, List
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from cluster_images import plot_image_cluster


class IncrementalLayout:
    """Persistent 2D layout that accumulates all images across rounds.

    Keeps rejected images in the layout (rendered faded) so the artist
    maintains spatial memory. New clones appear near their parent.
    """

    def __init__(self, n_neighbors=15, min_dist=0.2):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.umap_model = None
        self.coords = None          # [N_total, 2] all positions (history)
        self.images = None          # all PIL images (history)
        self.active = None          # bool array: True = current round, False = rejected
        self.round_ids = None       # which round each image was last active

    def initialize(self, features, images):
        """First round: full UMAP fit."""
        import umap
        n = features.shape[0]
        if n < 4:
            # UMAP is degenerate for <4 points (n_neighbors must be >1 and <n)
            # -> deterministic horizontal-line fallback so small batches never crash.
            self.umap_model = None
            self.coords = np.array([[float(i) * 2.0, 0.0] for i in range(n)],
                                   dtype=float)
        else:
            self.umap_model = umap.UMAP(
                n_neighbors=max(2, min(self.n_neighbors, n - 1)),
                min_dist=self.min_dist,
                metric="cosine",
                random_state=42,
            )
            self.coords = self.umap_model.fit_transform(features)
        self.images = list(images)
        self.active = np.ones(n, dtype=bool)
        self.round_ids = np.zeros(n, dtype=int)
        self._gentle_repulsion()

    def update(self, kept_indices, new_features, new_images, num_clones=None,
               round_num=1):
        """Update layout: mark unselected as inactive, add new clones near parents.

        All previous images stay in the layout at their positions (faded if rejected).
        New clones appear near their parent + DINO nudge.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        n_old = len(self.images)
        n_kept = len(kept_indices)
        n_new = len(new_images)
        C = num_clones or 2

        # Mark all current active images as inactive
        # Then re-activate only the kept ones
        active_indices = np.where(self.active)[0]
        self.active[:] = False
        kept_global = active_indices[kept_indices]  # map local kept → global indices
        # Don't reactivate kept — they stay as history. New clones are the active ones.

        # Get parent positions (from the kept particles in global coords)
        parent_coords = self.coords[kept_global].copy()

        # Place new clones near their parents
        # Spread decreases at later rounds (clones are more similar, should stay closer)
        x_range = max(self.coords[:, 0].max() - self.coords[:, 0].min(), 1e-6)
        y_range = max(self.coords[:, 1].max() - self.coords[:, 1].min(), 1e-6)
        decay = max(0.3, 1.0 / (1 + 0.5 * round_num))  # round 0: 1.0, round 3: 0.4, round 5: 0.3
        spread = min(x_range, y_range) * 0.06 * decay

        new_coords = np.zeros((n_new, 2))
        is_clone = np.zeros(n_new, dtype=bool)

        for i in range(n_kept):
            parent_pos = parent_coords[i]
            for c in range(C):
                idx = i * C + c
                if idx < n_new:
                    if c == 0:
                        new_coords[idx] = parent_pos
                    else:
                        angle = 2 * np.pi * (c + np.random.uniform(-0.3, 0.3)) / max(C - 1, 1)
                        offset = spread * np.array([np.cos(angle), np.sin(angle)])
                        new_coords[idx] = parent_pos + offset
                        is_clone[idx] = True

        # No automatic repulsion — user controls spacing via the slider.

        # Append to history
        self.coords = np.vstack([self.coords, new_coords])
        self.images.extend(new_images)
        new_active = np.zeros(len(self.coords), dtype=bool)
        new_active[n_old:] = True
        self.active = new_active
        self.mobility = np.ones(len(self.coords))  # not used anymore

        new_rounds = np.zeros(len(self.coords), dtype=int)
        new_rounds[:n_old] = self.round_ids
        new_rounds[n_old:] = round_num
        self.round_ids = new_rounds

    def _gentle_repulsion(self):
        """Light repulsion to prevent overlap. Uses per-particle mobility."""
        n = self.coords.shape[0]
        if n < 2:
            return
        x_range = max(self.coords[:, 0].max() - self.coords[:, 0].min(), 1e-6)
        y_range = max(self.coords[:, 1].max() - self.coords[:, 1].min(), 1e-6)
        min_sep = max(x_range, y_range) * 0.06

        mob = self.mobility if hasattr(self, 'mobility') and self.mobility is not None \
            else np.where(self.active, 1.0, 0.03)

        for _ in range(30):
            forces = np.zeros_like(self.coords)
            for i in range(n):
                for j in range(i + 1, n):
                    dx = self.coords[j] - self.coords[i]
                    d = max(np.linalg.norm(dx), 1e-8)
                    if d < min_sep:
                        push = 0.2 * (min_sep - d) / d
                        forces[i] -= push * dx * mob[i]
                        forces[j] += push * dx * mob[j]
            self.coords += forces

    def plot(self, title, save_path, thumb_size=80):
        """Render layout. Active images full opacity, inactive faded + smaller."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(self.images)
        fig_size = max(14, 10 + n * 0.1)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        x_range = self.coords[:, 0].max() - self.coords[:, 0].min()
        y_range = self.coords[:, 1].max() - self.coords[:, 1].min()
        hw_base = max(min(x_range, y_range) * 0.045, 1e-6)

        # Draw inactive (historical) images first, faded and smaller
        for i in range(n):
            if self.active[i]:
                continue
            x, y = self.coords[i]
            hw = hw_base * 0.5
            ts = max(int(thumb_size * 0.5), 16)
            thumb = self.images[i].resize((ts, ts), Image.LANCZOS)
            arr = np.array(thumb).astype(np.float32)
            arr = arr * 0.4 + 255 * 0.6  # fade toward white
            ax.imshow(arr.astype(np.uint8),
                      extent=(x - hw, x + hw, y - hw, y + hw),
                      alpha=0.5, zorder=1)

        # Draw active images on top, full size and opacity
        for i in range(n):
            if not self.active[i]:
                continue
            x, y = self.coords[i]
            hw = hw_base
            thumb = self.images[i].resize((thumb_size, thumb_size), Image.LANCZOS)
            ax.imshow(np.array(thumb),
                      extent=(x - hw, x + hw, y - hw, y + hw),
                      zorder=2)

        pad = hw_base * 4
        ax.set_xlim(self.coords[:, 0].min() - pad, self.coords[:, 0].max() + pad)
        ax.set_ylim(self.coords[:, 1].min() - pad, self.coords[:, 1].max() + pad)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=16)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved incremental layout: {save_path}")

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"


def compute_aggressive_diversity_schedule(num_steps: int, num_resampling_steps: int) -> list:
    """Even more front-loaded schedule for maximum diversity.

    Places all resampling in the first ~25% of steps with steeper power-law (exponent 2.0).
    All branching happens at very high noise levels for maximum diversity.
    """
    R = num_resampling_steps
    if R == 0:
        return []
    max_step = num_steps * 0.25  # first 25% only (vs 40% for linear)
    steps = []
    for k in range(R):
        frac = ((k + 1) / (R + 1)) ** 2.0  # steeper curve (vs 1.5 for linear)
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


def derive_seed(master_seed: int, *indices: int) -> int:
    """Stable derived seed: fold integer indices into a master seed.

    derive_seed(master, i)    -> per-root seed for root particle i
    derive_seed(master, e, c) -> per-clone seed for spawn event e, clone
                                 (current-round) index c
    """
    s = int(master_seed) % (2 ** 63)
    for i in indices:
        s = (s * 1_000_003 + int(i)) % (2 ** 63)
    return s


def randn_from_seed(seed: int, shape, device, dtype) -> torch.Tensor:
    """Draw N(0, I) of `shape` deterministically from an integer seed.

    Uses a CPU torch.Generator — the CPU RNG stream is stable across runs,
    machines and CUDA versions (the CUDA RNG stream is not guaranteed to be) —
    then moves the draw to the target device/dtype. This is the
    "save the noise" primitive: any latent init is exactly reproducible
    from its integer seed.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed) % (2 ** 63))
    x = torch.randn(tuple(shape), generator=g, device="cpu", dtype=torch.float32)
    return x.to(device=device, dtype=dtype)


class FluxDevInteractive:
    """
    FLUX.1-Dev interactive sampler with GLASS Flows.

    Standalone class using standard FluxPipeline — no LoRA, no flow map,
    no VLM reward model. Uses instantaneous velocity for ODE steps and
    Euler lookahead for previews.
    """

    def __init__(self, prompt: str, model_id: str = DEFAULT_MODEL_ID,
                 device: str = "cuda", dtype=torch.bfloat16,
                 keep_text_encoders: bool = False):
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.prompt = prompt
        # When True, the text encoders stay resident so the SAME loaded model
        # can be re-prompted via set_prompt() without a full reload — the basis
        # for the server keeping FLUX warm across generations (saves ~17s/prompt).
        self.keep_text_encoders = keep_text_encoders

        torch.backends.cuda.matmul.allow_tf32 = True

        self._load_model(prompt)

    def _load_model(self, prompt: str):
        """Load FLUX.1-Dev pipeline and precompute the prompt embeddings. Unless
        keep_text_encoders=True, the text encoders are freed afterwards (single-
        shot memory-efficient mode)."""
        print(f"Loading FLUX.1-Dev from: {self.model_id}")
        # Resolve the cached snapshot DIRECTORY directly off the filesystem
        # (refs/main -> snapshots/<commit>) and load diffusers from that concrete
        # path. Fully deterministic — avoids huggingface_hub's Hub/offline
        # resolution, which failed intermittently inside the long-running server
        # process on some cluster nodes even with the model fully cached on the
        # shared /data filesystem. Falls back to the repo id if not found.
        load_target = self.model_id
        # Highest priority: node-local /scratch copy staged by the startup bash
        # (job_ltnt_server.sh) — avoids the /data autofs subpath that the long-
        # running offline server process can't re-trigger after the import gap.
        _flux_local = os.environ.get("FLUX_LOCAL_PATH")
        if _flux_local and os.path.isfile(os.path.join(_flux_local, "model_index.json")):
            load_target = _flux_local
            print(f"Loading FLUX from node-local: {_flux_local}")
        if load_target == self.model_id:
            try:
                hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                repo_dir = os.path.join(hf_home, "hub",
                                        "models--" + self.model_id.replace("/", "--"))
                ref_main = os.path.join(repo_dir, "refs", "main")
                if os.path.exists(ref_main):
                    with open(ref_main) as fh:
                        commit = fh.read().strip()
                    snap = os.path.join(repo_dir, "snapshots", commit)
                    if os.path.isdir(snap):
                        load_target = snap
                        print(f"Loading FLUX from local snapshot dir: {snap}")
            except Exception as e:
                print(f"Could not resolve local snapshot dir ({type(e).__name__}); "
                      f"using {self.model_id}.")
        self.pipeline = FluxPipeline.from_pretrained(
            load_target, torch_dtype=self.dtype
        ).to(self.device)

        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.vae_scale_factor = self.pipeline.vae_scale_factor  # 16 for FLUX

        # Precompute text embeddings for the initial prompt.
        self.set_prompt(prompt)

        if not self.keep_text_encoders:
            # Free text encoders to save ~10GB GPU (single-shot mode).
            print("Removing text encoders to free memory...")
            del self.pipeline.text_encoder
            del self.pipeline.text_encoder_2
            del self.pipeline.tokenizer
            del self.pipeline.tokenizer_2
            self.pipeline.text_encoder = None
            self.pipeline.text_encoder_2 = None
            self.pipeline.tokenizer = None
            self.pipeline.tokenizer_2 = None
            gc.collect()
            torch.cuda.empty_cache()
            print("FLUX.1-Dev loaded (memory-efficient mode)")
        else:
            print("FLUX.1-Dev loaded (warm mode — text encoders resident)")

    def set_prompt(self, prompt: str):
        """Re-embed a single prompt (all particles share it). Thin wrapper over
        set_prompts for the common case + backward compatibility."""
        self.set_prompts([prompt])

    def set_prompts(self, prompts):
        """(Re)compute cached text embeddings for one OR MORE prompt variations,
        reusing the already-loaded model (needs keep_text_encoders=True). Stacks
        them into cached_prompt_embeds [K, seq, dim] so different particles can be
        conditioned on different prompts (prompt-augmented diversity)."""
        if getattr(self.pipeline, "text_encoder", None) is None:
            raise RuntimeError(
                "set_prompts() needs the text encoders, but they were freed. "
                "Construct with keep_text_encoders=True for warm re-prompting.")
        if isinstance(prompts, str):
            prompts = [prompts]
        self.prompt = prompts[0]
        pe_list, ppe_list, text_ids = [], [], None
        with torch.no_grad():
            for p in prompts:
                pe, ppe, tid = self.pipeline.encode_prompt(prompt=p, prompt_2=p)
                pe_list.append(pe)
                ppe_list.append(ppe)
                text_ids = tid
        self.prompts = list(prompts)  # recorded verbatim in lineage.json
        self.cached_prompt_embeds = torch.cat(pe_list, dim=0)          # [K, seq, dim]
        self.cached_pooled_prompt_embeds = torch.cat(ppe_list, dim=0)  # [K, dim]
        self.cached_text_ids = text_ids
        print(f"Embedded {len(prompts)} prompt variation(s).")

    @property
    def num_prompts(self) -> int:
        pe = getattr(self, "cached_prompt_embeds", None)
        return int(pe.shape[0]) if pe is not None else 1

    def add_prompts(self, prompts):
        """APPEND prompt variations to the cached embeddings (span mode).

        Append-only on purpose: a background thread can add variations while a
        generation thread reads indices < the old length — existing indices
        never move (the tensor swap is a single attribute assignment).
        Needs keep_text_encoders=True. Returns the new variation count.
        """
        if getattr(self.pipeline, "text_encoder", None) is None:
            raise RuntimeError(
                "add_prompts() needs the text encoders, but they were freed. "
                "Construct with keep_text_encoders=True for warm re-prompting.")
        if isinstance(prompts, str):
            prompts = [prompts]
        if not hasattr(self, "prompts") or not self.prompts:
            self.prompts = [self.prompt]
        with torch.no_grad():
            for p in prompts:
                pe, ppe, _ = self.pipeline.encode_prompt(prompt=p, prompt_2=p)
                self.cached_prompt_embeds = torch.cat(
                    [self.cached_prompt_embeds, pe], dim=0)
                self.cached_pooled_prompt_embeds = torch.cat(
                    [self.cached_pooled_prompt_embeds, ppe], dim=0)
                self.prompts.append(p)
        print(f"[flux] prompt variations now: {len(self.prompts)}")
        return len(self.prompts)

    # =========================================================================
    # SPAN MODE (round 1): stream INDEPENDENT full images in small batches
    # =========================================================================

    def sample_span_round(self, *,
                          master_seed: int,
                          output_dir: str,
                          batch_size: int = 8,
                          max_images: int = 64,
                          num_steps: int = 14,
                          guidance_scale: float = 3.5,
                          img_shape: Optional[Tuple[int, int]] = (512, 512),
                          prompt_plan=None,
                          on_batch=None,
                          progress_callback=None) -> dict:
        """SPAN ROUND 1 for FLUX — mirror of SanaInteractive.sample_span_round.

        Independent FULL fast solves (num_steps Euler steps on the mu-shifted
        schedule), streamed batch-by-batch via on_batch. Image i's init noise
        comes from derive_seed(master_seed, i) (root convention) so every span
        image is breedable / exactly reproducible (recipe: lineage_span.json).
        See the SANA implementation for the callback contracts.
        """
        import json as _json
        import time as _time

        imgH, imgW = img_shape if img_shape is not None else (512, 512)
        master_seed = int(master_seed)
        interval_dir = os.path.join(output_dir, "interval_1")
        os.makedirs(interval_dir, exist_ok=True)

        if getattr(self, "dino_model", None) is None:
            self._load_dino()

        latent_h = 2 * (int(imgH) // (self.vae_scale_factor * 2))
        latent_w = 2 * (int(imgW) // (self.vae_scale_factor * 2))

        # mu-shifted schedule — identical construction to sample_interactive.
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
            self.pipeline.scheduler, num_steps, self.device,
            sigmas=sigmas_arr, mu=mu)
        pts = [float(t) / 1000.0 for t in timesteps.tolist()] + [0.0]
        print(f"[span] FLUX num_steps={num_steps} batch={batch_size} "
              f"cap={max_images}")

        lineage = []
        all_latents, all_paths, all_features = [], [], []
        all_prompt_idx, all_seeds = [], []
        stop_reason = "cap"
        batch_idx = 0
        n_done = 0

        while n_done < max_images:
            bsz = min(batch_size, max_images - n_done)
            g0 = n_done
            t0 = _time.perf_counter()

            if prompt_plan is not None:
                try:
                    p_idx = list(prompt_plan(batch_idx, bsz))[:bsz]
                except Exception as _pe:
                    print(f"[span] prompt_plan failed ({_pe}); base prompt.")
                    p_idx = [0] * bsz
            else:
                p_idx = [0] * bsz
            p_idx = [min(max(int(i), 0), self.num_prompts - 1) for i in p_idx]
            p_idx_t = torch.as_tensor(p_idx, dtype=torch.long)

            seeds = [derive_seed(master_seed, g0 + j) for j in range(bsz)]
            z = torch.stack([
                randn_from_seed(s, (16, latent_h, latent_w),
                                self.device, self.dtype)
                for s in seeds
            ])

            with torch.no_grad():
                for k in range(num_steps):
                    if progress_callback:
                        progress_callback(batch_idx, n_done, k, num_steps)
                    t_cur, t_next = float(pts[k]), float(pts[k + 1])
                    v = self.predict_velocity(z, t_cur, guidance_scale,
                                              prompt_idx=p_idx_t)
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

        _recipe = dict(
            master_seed=master_seed, model_id=self.model_id, mode="span_round1",
            num_steps=int(num_steps), guidance_scale=float(guidance_scale),
            img_shape=[int(imgH), int(imgW)],
            prompts=list(getattr(self, "prompts", None) or [self.prompt]),
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

    # =========================================================================
    # Velocity prediction (instantaneous, single timestep)
    # =========================================================================

    def predict_velocity(self, z: torch.Tensor, t_cur: float,
                         guidance_scale: float = 3.5,
                         prompt_idx=None) -> torch.Tensor:
        """
        Predict instantaneous velocity using FLUX transformer.

        Args:
            z: Latent tensor [B, C, H_lat, W_lat]
            t_cur: Current time in [0, 1] (1=noise, 0=clean)
            guidance_scale: Embedded guidance scale
            prompt_idx: optional LongTensor [B] selecting, per particle, which
                cached prompt variation to condition on (for prompt-augmented
                diversity). None → all particles use variation 0 (single prompt).

        Returns:
            Velocity tensor [B, C, H_lat, W_lat]
        """
        batch_size = z.shape[0]
        device = z.device

        if isinstance(t_cur, torch.Tensor):
            t_cur = t_cur.item()

        H_latent, W_latent = z.shape[2], z.shape[3]
        H_image = H_latent * self.vae_scale_factor
        W_image = W_latent * self.vae_scale_factor

        # Timestep in [0, 1000] scale
        timestep = torch.tensor([t_cur * 1000], device=device, dtype=z.dtype).expand(batch_size)

        # Guidance embedding
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)
        else:
            guidance = None

        # Position embeddings
        latent_image_ids = self.pipeline._prepare_latent_image_ids(
            batch_size, H_latent // 2, W_latent // 2, device, z.dtype
        )

        # Pack latents: [B, C, H, W] -> [B, seq_len, C*4]
        z_packed = self.pipeline._pack_latents(z, batch_size, z.shape[1], H_latent, W_latent)

        # Text IDs
        text_ids = torch.zeros(self.cached_prompt_embeds.shape[1], 3, device=device, dtype=z.dtype)

        # Prompt embeddings, per particle. cached_prompt_embeds is [K, seq, dim]
        # (K prompt variations); pick variation 0 for everyone, or gather per
        # particle when prompt_idx is given.
        if prompt_idx is None:
            prompt_embeds = self.cached_prompt_embeds[:1].expand(batch_size, -1, -1)
            pooled_prompt_embeds = self.cached_pooled_prompt_embeds[:1].expand(batch_size, -1)
        else:
            idx = prompt_idx.to(self.cached_prompt_embeds.device)
            prompt_embeds = self.cached_prompt_embeds.index_select(0, idx)
            pooled_prompt_embeds = self.cached_pooled_prompt_embeds.index_select(0, idx)

        # Forward pass
        noise_pred = self.transformer(
            hidden_states=z_packed,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        # Unpack: [B, seq_len, C*4] -> [B, C, H, W]
        velocity = self.pipeline._unpack_latents(noise_pred, H_image, W_image, self.vae_scale_factor)
        return velocity

    # =========================================================================
    # Decode / helpers
    # =========================================================================

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image [B, 3, H, W] in [0, 1] (full VAE — finals)."""
        z_scaled = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.no_grad():
            samples = self.vae.decode(z_scaled, return_dict=False)[0]
        samples = torch.clamp((samples + 1) / 2, 0, 1)
        return samples

    def encode_image_to_latent(self, pil_image, img_shape: Optional[Tuple[int, int]] = None):
        """Inverse of `decode`: PIL image -> clean GLASS-convention latent z0
        ([1, 16, H_lat, W_lat]), ready to be re-noised for image-seeded
        (SDEdit) exploration.

        decode() does:  z_scaled = z / scaling_factor + shift_factor ; img = vae.decode(z_scaled)
        so encode is the exact inverse:
            z_scaled = vae.encode(img_in_[-1,1]).sample()      # raw VAE latent
            z0       = (z_scaled - shift_factor) * scaling_factor
        The VAE works in [-1, 1] (decode clamps (x+1)/2 -> [0,1]); we map the
        PIL image [0,1] -> [-1,1] before encoding. The image is resized so the
        latent matches the session's (latent_h, latent_w).
        """
        from PIL import Image as _PILImage
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        if img_shape is not None:
            imgH, imgW = int(img_shape[0]), int(img_shape[1])
            # Match the exact latent grid sample_interactive uses (see below).
            latent_h = 2 * (imgH // (self.vae_scale_factor * 2))
            latent_w = 2 * (imgW // (self.vae_scale_factor * 2))
            target_w = latent_w * self.vae_scale_factor
            target_h = latent_h * self.vae_scale_factor
            if pil_image.size != (target_w, target_h):
                pil_image = pil_image.resize((target_w, target_h), _PILImage.LANCZOS)
        arr = np.asarray(pil_image).astype(np.float32) / 255.0   # [H, W, 3] in [0,1]
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        x = (x * 2.0 - 1.0).to(self.device, dtype=self.dtype)    # -> [-1, 1]
        with torch.no_grad():
            z_scaled = self.vae.encode(x).latent_dist.sample()
        z0 = (z_scaled - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z0.to(dtype=self.dtype)

    def preview_decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents for the interactive PREVIEW thumbnails. Separate hook
        from `decode` so previews can use a fast tiny VAE while the final pick
        keeps the full VAE. Step 1: full VAE, batched. Step 2: TAEF1."""
        return self.decode(z)

    def euler_lookahead(self, z: torch.Tensor, t_cur: float,
                        guidance_scale: float, num_steps: int = 8,
                        prompt_idx=None, step_cb=None) -> torch.Tensor:
        """Deterministic Euler integration from t_cur to t=0 for preview.

        step_cb(step_done, num_steps): optional per-step hook so callers can
        report REAL progress during the preview integration (each step is a
        full batched forward — the dominant between-round cost).
        """
        z_cur = z.clone()
        t = t_cur
        dt = -t_cur / num_steps
        for _i in range(num_steps):
            v = self.predict_velocity(z_cur, t, guidance_scale, prompt_idx=prompt_idx)
            z_cur = z_cur + dt * v
            t = t + dt
            if step_cb is not None:
                step_cb(_i + 1, num_steps)
            if t < 1e-6:
                break
        return z_cur

    # =========================================================================
    # DINOv3-B feature extraction
    # =========================================================================

    def _load_dino(self):
        from transformers import AutoImageProcessor, AutoModel
        dino_id = "facebook/dinov2-base"
        print(f"Loading DINOv2-base ({dino_id})...")
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_id)
        self.dino_model = AutoModel.from_pretrained(dino_id, low_cpu_mem_usage=False).to(self.device).eval()
        print("DINOv2-base loaded and kept on GPU")

    def _unload_dino(self):
        if hasattr(self, "dino_model") and self.dino_model is not None:
            del self.dino_model
            del self.dino_processor
            self.dino_model = None
            self.dino_processor = None
            gc.collect()
            torch.cuda.empty_cache()

    def _extract_dino_features(self, pil_images: List[Image.Image]) -> np.ndarray:
        # Batched: process all previews in a single forward pass instead of a
        # per-image loop (was total_particles× the GPU launches).
        with torch.no_grad():
            inputs = self.dino_processor(images=list(pil_images),
                                         return_tensors="pt").to(self.device)
            outputs = self.dino_model(**inputs)
            features = outputs.pooler_output.cpu().numpy()
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norms + 1e-8)

    # =========================================================================
    # Interactive helpers
    # =========================================================================

    @staticmethod
    def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
        arr = (img_tensor[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _save_grid(pil_images: List[Image.Image], save_path: str, labels: List[str] = None):
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
        print(f"  Particle previews:  {interval_dir}/particle_*.png")
        print(f"  Labeled grid:       {interval_dir}/grid.png")
        print(f"  Total particles:    {total}")
        print("=" * 60)
        while True:
            raw = input(f"  Enter indices to KEEP (0-{total-1}), comma-separated: ").strip()
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
                indices = [int(x.strip()) for x in raw.split(",")]
                if not indices or any(i < 0 or i >= total for i in indices):
                    print(f"  Invalid. Must be in 0-{total-1}.")
                    continue
                indices = sorted(set(indices))
                print(f"  Keeping particles: {indices}")
                return indices
            except ValueError:
                print("  Invalid input.")

    # =========================================================================
    # GLASS Flows (same math as SD3.5 version, FLUX time convention)
    # =========================================================================

    def _glass_init(self, X_bar: torch.Tensor, sigma_start: float,
                    sigma_end: float, rho: float,
                    generator=None) -> Tuple[torch.Tensor, dict]:
        """GLASS stochastic init. sigma = t_flux (1=noise, 0=clean).

        The single eps draw here is the ONLY stochasticity in a GLASS clone —
        the inner bridge ODE is deterministic after it. `generator` (a CPU
        torch.Generator, or a list of them — one per batch row) makes the draw
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

    # =========================================================================
    # SDEdit-style spawn (alternative to GLASS in-trajectory branching)
    # =========================================================================

    @staticmethod
    def _rho_to_t_RN(rho: float, t_seg_start: float) -> float:
        """Map divergence knob rho in [0,1] to renoise depth in FLUX time
        (t=1 noise, t=0 clean).

        rho = 1 -> t_RN = t_seg_start  (no renoise, clone == parent).
        rho = 0 -> t_RN = 1            (full noise, clone ~ independent).

        Small rho = renoise higher = more divergence; large rho = less.
        Matches GLASS rho semantics so the API knob is consistent.
        """
        rho = max(0.0, min(1.0, rho))
        return t_seg_start + (1.0 - rho) * (1.0 - t_seg_start)

    def _sdedit_init(self, X_bar: torch.Tensor, sigma_seg_start: float,
                     rho: float, generator=None) -> Tuple[torch.Tensor, float]:
        """SDEdit-style spawn: renoise the parent latent UP to a higher noise
        level t_RN > sigma_seg_start via the forward kernel (GLASS Eq. 21 in
        FLUX time, alpha_t = 1-t, sigma_t = t):

            x_{t_RN} = (alpha_RN/alpha_t)*x_t
                       + sqrt(sigma_RN^2 - (alpha_RN/alpha_t)^2 * sigma_t^2)*eps

        Returns the renoised latent and t_RN. The clone then runs a standard
        deterministic Euler ODE from t_RN onward (NOT the GLASS inner bridge),
        so it diverges more than the path-consistent GLASS step.
        """
        clip = 1e-12
        t = sigma_seg_start
        t_RN = self._rho_to_t_RN(rho, t)
        if t_RN <= t + 1e-9:
            return X_bar.clone(), t
        a_t, a_RN = 1.0 - t, 1.0 - t_RN
        s_t, s_RN = t, t_RN
        mean_coef = a_RN / max(a_t, clip)
        var = s_RN * s_RN - (a_RN * a_RN / max(a_t * a_t, clip)) * s_t * s_t
        noise_coef = math.sqrt(max(var, 0.0))
        # Single eps draw = the ONLY stochasticity of an SDEdit spawn. A CPU
        # generator makes it exactly reproducible; None = original behavior.
        if generator is None:
            eps = torch.randn_like(X_bar)
        else:
            eps = torch.randn(tuple(X_bar.shape), generator=generator,
                              device="cpu", dtype=torch.float32
                              ).to(device=X_bar.device, dtype=X_bar.dtype)
        return mean_coef * X_bar + noise_coef * eps, t_RN

    # =========================================================================
    # Interactive sampling with GLASS
    # =========================================================================

    def sample_interactive(self,
                           num_steps: int = 28,
                           num_particles: int = 4,
                           num_clones: int = 2,
                           num_resampling_steps: int = 4,
                           guidance_scale: float = 3.5,
                           rho: float = 0.4,
                           spawn_mode: str = "glass",
                           lookahead_steps: int = 8,
                           # Round-1 previews live at σ≈0.9 where the image is
                           # largely uncommitted — the old fast setting (2 Euler
                           # steps) gave blurry near-identical blobs. Spend MORE
                           # lookahead steps on the FIRST interval only (later
                           # intervals are more committed and stay cheap).
                           # None -> fall back to lookahead_steps.
                           lookahead_steps_first: Optional[int] = None,
                           schedule_mode: Optional[str] = "linear",
                           img_shape: Optional[Tuple[int, int]] = None,
                           seed: Optional[int] = None,
                           master_seed: Optional[int] = None,
                           output_dir: str = "interactive_fmtt",
                           progress_callback=None,
                           save_debug_plots: bool = False,
                           prompt_idx=None,
                           init_latent=None,
                           init_strength: float = 0.55,
                           # ── SPAN MODE breeding: MULTIPLE clean parent
                           # latents [P, 16, h, w] (the selected span-round
                           # images). Anchors are re-noised round-robin over
                           # the parents (each with its own per-root eps).
                           # init_latent (singular, classic explore-from-pin)
                           # is unchanged; init_latents takes precedence.
                           init_latents=None,
                           # Global ids of the parents (lineage only).
                           init_parent_ids=None,
                           # Offset for interval_N dir names: a span round
                           # already wrote interval_1, so breeding rounds
                           # start at interval_{1+interval_start}.
                           interval_start: int = 0) -> torch.Tensor:
        """
        Interactive FMTT with GLASS Flows for FLUX.1-Dev.

        Anchor-clone design:
        - N anchor particles run standard deterministic Euler ODE
        - Each anchor has (C-1) clones that diverge via GLASS stochastic init
        - At resampling: user selects, selected become anchors, clones get GLASS

        Time convention: t=1 (noise) -> t=0 (clean).
        """
        imgH, imgW = img_shape if img_shape is not None else (512, 512)
        device = self.device
        os.makedirs(output_dir, exist_ok=True)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # ===== SAVE THE NOISE: session master seed + lineage recording =====
        # Every stochastic draw in this sampler (root init noise, GLASS clone
        # eps, SDEdit spawn eps) is derived deterministically from master_seed
        # via derive_seed()/randn_from_seed(), and every particle creation is
        # recorded in `lineage` -> written to lineage.json at the end. A
        # KB-sized recipe that makes every node exactly reproducible.
        if master_seed is None:
            master_seed = int(seed) if seed is not None else \
                random.SystemRandom().randrange(2 ** 31)
        master_seed = int(master_seed)
        print(f"  [save-noise] master_seed={master_seed}")
        lineage: List[dict] = []
        selections: List[dict] = []
        spawn_event_counter = 0

        # Segment bookkeeping for progress reporting. Each segment is the span
        # between two adjacent resample points (or the start / final step); a
        # "session" in the UI corresponds to exactly one segment.
        n_segments_total = num_resampling_steps + 1
        if progress_callback:
            # First segment's 0-10% reserved by the server for model loading.
            progress_callback(0.02, "Loading image similarity model...", 0, n_segments_total)
        self._load_dino()
        if progress_callback:
            progress_callback(0.05, "Preparing...", 0, n_segments_total)

        latent_h = 2 * (int(imgH) // (self.vae_scale_factor * 2))
        latent_w = 2 * (int(imgW) // (self.vae_scale_factor * 2))

        # ===== Initialize N anchor particles =====
        # Round 1 has N particles (no clones yet). Cloning starts at the first
        # resample: each selected particle spawns C new variations via GLASS.
        # DEFAULT (init_latent is None): pure-noise anchors at t=1 — identical to
        # the original behavior. When init_latent (a clean z0 from an existing
        # image) is given, the anchors are instead built by re-noising z0 to a
        # partial level t' below (image-seeded SDEdit exploration). That block
        # runs AFTER the time schedule is built so t' snaps to a real step.
        root_seeds = [derive_seed(master_seed, i) for i in range(num_particles)]
        anchors = torch.stack([
            randn_from_seed(rs, (16, latent_h, latent_w), device, self.dtype)
            for rs in root_seeds
        ])
        total_particles = num_particles
        C = num_clones

        # Time schedule (resolution-dependent shift)
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

        # ===== Image-seeded (SDEdit) start — only when init_latent given =====
        # Re-noise the supplied clean latent z0 to a partial level t' and start
        # the trajectory there. Default path (init_latent is None) leaves the
        # pure-noise `anchors` and k_start=0 untouched -> behavior IDENTICAL.
        #   FLUX forward kernel: x_t = alpha(t)*z0 + sigma(t)*eps,
        #   with alpha(t) = 1 - t and sigma(t) = t  (t=1 noise, t=0 clean) —
        #   the SAME alpha/sigma convention _glass_init uses.
        # Higher init_strength = higher t' = re-noise further = explore more /
        # keep less of the original. We clamp t' to [0.30, 0.95] for sanity,
        # then SNAP it down to the first schedule step <= t' so integration
        # begins on a real grid point.
        k_start = 0
        _seeded_root_meta = None  # (t_renoise, k_start) when image-seeded
        _anchor_parent = None     # per-anchor parent row (span breeding)
        if init_latents is not None:
            # SPAN breeding: multi-parent tensor rides the same code path.
            init_latent = init_latents
        if init_latent is not None:
            try:
                t_prime = float(init_strength)
            except (TypeError, ValueError):
                t_prime = 0.55
            t_prime = max(0.30, min(0.95, t_prime))
            z0 = init_latent.to(device=device, dtype=self.dtype)
            if z0.dim() == 3:
                z0 = z0.unsqueeze(0)
            # Crop/pad-safe: z0 must match the latent grid. If it doesn't (e.g.
            # a differently-sized source), fall back to pure noise rather than
            # error — encode_image_to_latent already resizes to (latent_h,
            # latent_w), so this is just a guard.
            if tuple(z0.shape[1:]) == (16, latent_h, latent_w):
                # Find first schedule step whose t <= t' (snap down).
                k_start = 0
                for _ki in range(num_steps):
                    if time_steps[_ki].item() <= t_prime:
                        k_start = _ki
                        break
                else:
                    k_start = num_steps - 1
                t_renoise = time_steps[k_start].item()
                a_rn = 1.0 - t_renoise
                s_rn = t_renoise
                # Per-anchor INDEPENDENT noise so the N seeded anchors diverge.
                # Deterministic: reuses the per-root seeds (the pure-noise
                # anchors drawn above are discarded on this path).
                # MULTI-PARENT (span breeding): anchors are assigned to the P
                # parent latents round-robin; P=1 reproduces the classic
                # explore-from-pin behavior exactly.
                P = z0.shape[0]
                _anchor_parent = [i % P for i in range(num_particles)]
                z0b = torch.stack([z0[p] for p in _anchor_parent])
                eps = torch.stack([
                    randn_from_seed(rs, (16, latent_h, latent_w),
                                    device, self.dtype)
                    for rs in root_seeds
                ])
                anchors = a_rn * z0b + s_rn * eps
                _seeded_root_meta = (float(t_renoise), int(k_start))
                print(f"  [image-seed] {P} parent latent(s), init_strength="
                      f"{init_strength} -> t'={t_prime:.3f} "
                      f"snapped to step {k_start} (t={t_renoise:.4f}); "
                      f"re-noised z0 with per-anchor eps.")
            else:
                print(f"  [image-seed] WARNING: init_latent shape {tuple(z0.shape)} "
                      f"!= expected (*,16,{latent_h},{latent_w}); falling back to pure noise.")

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

        # ===== Seeded-session resample rebuild (image-seed / board pin) =====
        # When image-seeded we start integration at k_start (the snapped t'
        # step), so the FULL-range schedule above can place every resample
        # checkpoint BEFORE k_start -> they never fire -> the artist gets ZERO
        # interaction rounds. Recompute the checkpoints to fall within the
        # REMAINING trajectory (k_start, num_steps), evenly spaced, giving up
        # to num_resampling_steps rounds. The non-seeded path (init_latent is
        # None / k_start == 0) is left UNCHANGED.
        if init_latent is not None and k_start > 0 and num_resampling_steps > 0:
            _rebuilt = sorted(set(
                k_start + int(round((i + 1) * (num_steps - k_start)
                                    / (num_resampling_steps + 1)))
                for i in range(num_resampling_steps)
            ))
            # Keep only checkpoints strictly inside (k_start, num_steps); unique.
            resample_steps = [s for s in _rebuilt if k_start < s < num_steps]
            resample_at = set(resample_steps)
            segment_bounds = [k_start] + resample_steps + [num_steps]
            # Progress bookkeeping must reflect the seeded segment count so the
            # bar/final-segment index are not over-counted.
            n_segments_total = len(resample_steps) + 1
            print(f"  [image-seed] seeded resample schedule (k_start={k_start}): "
                  f"steps {resample_steps} -> {n_segments_total} segment(s)")

        # Build initial particle tensor — round 1 is N anchors, no clones.
        X_all = anchors.clone()
        is_anchor = torch.ones(total_particles, dtype=torch.bool)

        # Per-particle prompt-variation index (prompt-augmented diversity). When
        # prompt_idx is given (one entry per initial anchor), each particle is
        # conditioned on its own prompt variation; clones inherit their parent's.
        if prompt_idx is not None:
            self.particle_prompt_idx = torch.as_tensor(
                list(prompt_idx), dtype=torch.long)[:total_particles]
        else:
            self.particle_prompt_idx = torch.zeros(total_particles, dtype=torch.long)

        # Lineage entries for the roots. Image-seeded roots are SDEdit
        # re-noises of z0 (kind "sdedit"); pure-noise roots are kind "root".
        for _ri in range(num_particles):
            _e = dict(idx=_ri, kind="root", parent=None, seed=root_seeds[_ri],
                      rho=None, sigma_start=None, sigma_end=None,
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

        glass_params = None
        interval_count = 0
        inc_layout = IncrementalLayout(n_neighbors=15, min_dist=0.2)
        prev_kept_indices = None

        # When image-seeded, begin integration at k_start (the first step <= t');
        # default path has k_start=0 so the full schedule runs as before.
        pbar = tqdm(range(k_start, num_steps), total=num_steps - k_start,
                    desc="FLUX-Dev-Interactive")

        # ── Honest progress accounting ──────────────────────────────────────
        # The reported fraction is (work done / total work) for the segment,
        # in "main ODE step" units, INCLUDING the preview lookahead + decode
        # tail that used to be hidden behind a 40→90% jump. Per segment:
        #   * each main step         = 1 unit (batched anchor+clone forwards)
        #   * preview lookahead step = ~0.7 units (one batched forward over
        #                              all particles)
        #   * preview VAE decode     ≈ 1 unit; DINO + layout tail ≈ 2 units
        # Monotone by construction: phases are cumulative within a segment and
        # the server clamps with max(prev, new).
        _la_first = lookahead_steps_first if lookahead_steps_first else lookahead_steps
        _LA_UNIT = 0.7
        _DEC_UNITS = 1.0
        _TAIL_UNITS = 2.0

        def _seg_progress_units(seg_idx):
            """Return (seg_len, la_for_segment, total_units) for one segment."""
            seg_len = max(segment_bounds[seg_idx + 1] - segment_bounds[seg_idx], 1)
            la = _la_first if seg_idx == 0 else lookahead_steps
            if seg_idx == n_segments_total - 1:
                la = 0  # final segment decodes latents directly (no lookahead)
            return seg_len, la, seg_len + la * _LA_UNIT + _DEC_UNITS + _TAIL_UNITS

        for k in pbar:
            if progress_callback:
                # Find the segment this step belongs to: [seg_start, seg_end).
                seg_idx = 0
                for b in resample_steps:
                    if k >= b:
                        seg_idx += 1
                seg_start = segment_bounds[seg_idx]
                seg_len, _la_seg, _total_units = _seg_progress_units(seg_idx)
                frac = (k - seg_start) / _total_units
                progress_callback(
                    frac,
                    f"Generating images — step {k + 1 - seg_start}/{seg_len}",
                    seg_idx, n_segments_total,
                )
            t_cur = time_steps[k].item()
            t_next = time_steps[k + 1].item()
            dt = t_next - t_cur  # negative (going from noise to clean)

            # ===== At segment start: GLASS init for clones =====
            if glass_params is None:
                # BRIDGE-BOUNDARY FIX (evidence: outputs/glass_sigma_sweep/
                # {cascade,exact_d4}_results.json). The resample event at fine
                # step b fires AFTER stepping b, i.e. with states at
                # time_steps[b+1] — but the bridge used to target
                # time_steps[b] (first bound > k). Consequences: (a) segments
                # longer than 1 step took an extra inner step at k=b and
                # OVERSHOT s past 1; (b) a segment starting ON a resample step
                # (adjacent events) was INTERRUPTED mid-bridge (off-marginal
                # states expanded). Fix: target the ACTUAL expansion boundary
                # time_steps[b+1] with M_inner = b+1-k inner steps, so s runs
                # 0..(M-1)/M and the bridge lands EXACTLY on-marginal (s=1)
                # at the moment selection/expansion happens. Events 1 step
                # apart give a single EXACT step (exactness beats step count);
                # wider spacing automatically gives >= 2 inner steps.
                seg_end_step = None
                for b in resample_steps:
                    if b >= k:
                        seg_end_step = b + 1
                        break
                if seg_end_step is None:
                    # FINAL-ROUND BRIDGE CAP (validated: outputs/glass_sigma_
                    # sweep/followup_results.json). With no next resample
                    # boundary the bridge used to run all the way to sigma=0,
                    # which (a) reduces per-child novelty — parent-pull over
                    # the whole descent (parent-child DINO cosdist 0.444 vs
                    # 0.518 for Euler completion) — and (b) makes
                    # euler_lookahead previews of mid-descent clones
                    # systematically wrong (their true future is bridge
                    # dynamics, not the ODE the preview integrates). Cap the
                    # final bridge at a short handoff matching a typical
                    # non-final segment (M <= 3, landing on-marginal at s=1),
                    # then plain Euler to 0 (handoff switch in the step block
                    # below). Lineage sigma_end records the handoff sigma.
                    seg_end_step = min(k + 3, num_steps)

                sigma_seg_start = time_steps[k].item()
                sigma_seg_end = time_steps[seg_end_step].item()
                M_inner = seg_end_step - k

                print(f"\n  Segment: steps {k}-{seg_end_step}, "
                      f"t={sigma_seg_start:.4f}->{sigma_seg_end:.4f}, M={M_inner}")

                clone_indices = (~is_anchor).nonzero(as_tuple=True)[0]
                if len(clone_indices) > 0 and spawn_mode == "sdedit":
                    # SDEdit-style spawn: renoise each clone's parent UP to a
                    # higher noise level, then run it as a plain Euler particle
                    # (no GLASS inner bridge -> more divergence, less path
                    # consistency). clone_indices stays empty so the GLASS step
                    # block below is skipped; these clones step as Euler.
                    for idx in clone_indices:
                        _ci = int(idx.item())
                        group = _ci // C
                        _cs = derive_seed(master_seed, spawn_event_counter, _ci)
                        _g = torch.Generator(device="cpu").manual_seed(_cs)
                        renoised, t_RN = self._sdedit_init(
                            X_all[group * C:group * C + 1], sigma_seg_start, rho,
                            generator=_g)
                        X_all[idx:idx + 1] = renoised
                        lineage.append(dict(
                            idx=_ci, kind="sdedit", parent=group * C, seed=_cs,
                            rho=float(rho),
                            sigma_start=float(sigma_seg_start),
                            sigma_end=float(t_RN), spawn_step=int(k),
                            prompt_idx=int(
                                self.particle_prompt_idx[_ci].item())))
                    spawn_event_counter += 1
                    print(f"  SDEdit spawn: renoised {len(clone_indices)} clones "
                          f"rho={rho} -> t_RN={t_RN:.4f} (Euler from here)")
                    glass_params = {"seg_start_step": k, "M_inner": M_inner,
                                    "clone_indices": torch.tensor([], dtype=torch.long),
                                    "sdedit_clone_indices": clone_indices}
                elif len(clone_indices) > 0:
                    parent_states = torch.zeros(len(clone_indices), 16, latent_h, latent_w,
                                                device=device, dtype=self.dtype)
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

            # ===== Step all particles =====
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
                # Anchors: standard deterministic Euler ODE. In sdedit spawn
                # mode the renoised clones also step as plain Euler particles,
                # as do final-segment clones after their bridge handoff.
                euler_indices = is_anchor.nonzero(as_tuple=True)[0]
                sdedit_clone_indices = glass_params.get("sdedit_clone_indices")
                if sdedit_clone_indices is not None and len(sdedit_clone_indices) > 0:
                    euler_indices = torch.cat([euler_indices, sdedit_clone_indices])
                handoff_clone_indices = glass_params.get("handoff_clone_indices")
                if handoff_clone_indices is not None and len(handoff_clone_indices) > 0:
                    euler_indices = torch.cat([euler_indices, handoff_clone_indices])
                if len(euler_indices) > 0:
                    # BATCHED anchor Euler step: one forward over all anchors
                    # (was a per-particle loop). predict_velocity is fully
                    # batched and supports a per-row prompt_idx. Same MATH as
                    # the sequential loop (bf16 batched-GEMM is ~1e-2 different
                    # from single-row, an inherent cuBLAS reduction-order effect).
                    idx_e = euler_indices
                    z_e = X_all[idx_e]
                    v_e = self.predict_velocity(
                        z_e, t_cur, guidance_scale,
                        prompt_idx=self.particle_prompt_idx[idx_e])
                    X_all[idx_e] = z_e + dt * v_e

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
                    # BATCHED GLASS clone step: w_xt/w_xs/sigma_star/w1/w2/w3/ds
                    # are per-segment SCALARS (identical for every clone in the
                    # segment), so stack all clones into one forward. x_t is
                    # indexed by enumerate-position ci (parents gathered for
                    # clone_indices); X_all is indexed by the global clone idx.
                    cidx = clone_indices
                    x_parent = x_t                       # [N_clone, ...] parents, ci-aligned
                    x_clone = X_all[cidx]                # [N_clone, ...] clones
                    S_input = w_xt * x_parent + w_xs * x_clone
                    v_star = self.predict_velocity(
                        S_input, sigma_star, guidance_scale,
                        prompt_idx=self.particle_prompt_idx[cidx])
                    denoiser = S_input - sigma_star * v_star
                    velocity = w1 * x_clone + w2 * denoiser + w3 * x_parent
                    X_all[cidx] = x_clone + ds * velocity

            # ===== Interactive pruning at resampling intervals =====
            if k in resample_at:
                glass_params = None

                interval_count += 1
                # interval_start offsets dir names when a SPAN round already
                # owns interval_1 (breeding rounds then start at interval_2).
                interval_dir = os.path.join(
                    output_dir, f"interval_{interval_count + interval_start}")
                os.makedirs(interval_dir, exist_ok=True)

                t_now = t_next
                print(f"\n--- Interval {interval_count} (step {k}/{num_steps}, "
                      f"t={t_now:.4f}) ---")
                print(f"Computing Euler look-ahead previews for {total_particles} particles...")

                # Segment that just ended — interval_count is 1-indexed.
                seg_idx_done = interval_count - 1

                # Label by actual anchor/clone status (handles round 1 where
                # every particle is an anchor and there are no clone groups).
                labels = []
                anchor_group = -1
                for i in range(total_particles):
                    if is_anchor[i]:
                        anchor_group += 1
                        labels.append(f"{i} (anchor g{anchor_group})")
                    else:
                        labels.append(f"{i} (clone g{anchor_group})")

                # a) Euler look-ahead + decode — BATCHED across all particles.
                #    Was a per-particle loop (total_particles × the GPU
                #    launches for both the lookahead integration and the VAE
                #    decode); now one batched pass each. Per-phase timing is
                #    printed so we can see exactly where the round goes.
                import time as _time
                _t0 = _time.perf_counter()
                # Round-1 preview boost: the first interval's states sit at
                # σ≈0.9 (largely uncommitted) — the old fast setting (2 Euler
                # steps) produced blurry near-identical blobs that read as
                # "the model is broken". Spend the richer first-round
                # lookahead here; later intervals are more committed and keep
                # the cheap setting. interval_count is 1-indexed (==1 is R1).
                la_this = _la_first if interval_count == 1 else lookahead_steps
                _seg_len_d, _la_d, _tot_units_d = _seg_progress_units(seg_idx_done)
                if progress_callback:
                    progress_callback(
                        _seg_len_d / _tot_units_d,
                        f"Sketching previews of {total_particles} images...",
                        seg_idx_done, n_segments_total,
                    )
                # Per-lookahead-step ticks: with the round-1 boost the preview
                # integration is the longest single phase — without ticks the
                # bar froze for its whole duration (zero-dead-air violation).
                _sketch_cb = None
                if progress_callback:
                    def _sketch_cb(_done, _n, _sl=_seg_len_d, _tu=_tot_units_d,
                                   _si=seg_idx_done, _np=total_particles):
                        progress_callback(
                            (_sl + _done * _LA_UNIT) / _tu,
                            f"Sketching {_np} previews — step {_done}/{_n}",
                            _si, n_segments_total,
                        )
                with torch.no_grad():
                    z0_all = self.euler_lookahead(
                        X_all, t_now, guidance_scale, num_steps=la_this,
                        prompt_idx=self.particle_prompt_idx,
                        step_cb=_sketch_cb,
                    )
                    _t_look = _time.perf_counter()
                    if progress_callback:
                        progress_callback(
                            (_seg_len_d + la_this * _LA_UNIT) / _tot_units_d,
                            f"Decoding {total_particles} previews...",
                            seg_idx_done, n_segments_total,
                        )
                    img_all = self.preview_decode(z0_all)
                    _t_dec = _time.perf_counter()

                pil_images = []
                for p_idx in range(total_particles):
                    pil_img = self._tensor_to_pil(img_all[p_idx:p_idx+1])
                    pil_img.save(os.path.join(interval_dir, f"particle_{p_idx:03d}.png"))
                    pil_images.append(pil_img)

                if progress_callback:
                    progress_callback(
                        (_seg_len_d + la_this * _LA_UNIT + _DEC_UNITS) / _tot_units_d,
                        "Arranging images by similarity...",
                        seg_idx_done, n_segments_total,
                    )

                # b) DINOv3-B features + layout
                print("Extracting DINOv3-B features...")
                features = self._extract_dino_features(pil_images)
                _t_dino = _time.perf_counter()
                print(f"[timing] interval {interval_count}: "
                      f"lookahead={_t_look - _t0:.2f}s "
                      f"decode={_t_dec - _t_look:.2f}s "
                      f"dino={_t_dino - _t_dec:.2f}s "
                      f"(N={total_particles}, steps={la_this})")

                # Incremental layout: initialize on first round, update on subsequent
                if inc_layout is not None:
                    if interval_count == 1:
                        inc_layout.initialize(features, pil_images)
                    else:
                        inc_layout.update(prev_kept_indices, features, pil_images,
                                          num_clones=C, round_num=interval_count)

                # Debug visualizations (matplotlib renders + a fresh UMAP fit in
                # plot_image_cluster) — several seconds/round and NOT consumed by
                # the web frontend (which lays out from the API's own coords).
                # Off by default so the interactive server stays fast; the
                # standalone CLI can pass save_debug_plots=True.
                if save_debug_plots:
                    if inc_layout is not None:
                        inc_layout.plot(
                            title=f"Interval {interval_count} (step {k}, t={t_now:.3f})",
                            save_path=os.path.join(interval_dir, "layout_incremental.png"),
                            thumb_size=80,
                        )
                    cluster_path = os.path.join(interval_dir, "cluster_dinov3.png")
                    plot_image_cluster(
                        features, pil_images,
                        title=f"DINOv3-B Cluster — Interval {interval_count} "
                              f"(step {k}, t={t_now:.3f})",
                        save_path=cluster_path,
                        n_neighbors=min(15, total_particles - 1),
                        min_dist=0.1, thumb_size=80,
                    )
                    self._save_grid(pil_images, os.path.join(interval_dir, "grid.png"), labels)

                # d) Prompt user. refine_fn renders a FULL-QUALITY version of
                #    a particle on request — PATH-FAITHFUL, STAGE-ADAPTIVE
                #    (validated: outputs/refine_calibration/results.json). For
                #    EARLY picks (t' above the handoff) the naive fine solve
                #    diverges from the coarse preview the artist saw (LPIPS up
                #    to ~0.08-0.16 at t'~0.9); instead we continue the
                #    particle's COARSE preview path (the SAME euler_lookahead
                #    grid the preview used) down to t''=0.25, then fine-solve
                #    the remaining fine-schedule points below t'' (~10x closer
                #    to the preview: LPIPS ~0.009 at t'=0.9). For LATE picks
                #    (t' <= t'') the naive fine solve is kept — identical
                #    there, and path-faithful inverts below its own handoff.
                #    No GLASS spawn, no fresh noise; X_all untouched.
                _T_REFINE_HANDOFF = 0.25
                def _refine_particles(ridx, _k=k, _dir=interval_dir,
                                      _la=la_this, _th=_T_REFINE_HANDOFF):
                    t_pick = time_steps[_k + 1].item()
                    if t_pick <= _th + 1e-6:
                        # LATE pick: naive fine solve over the remaining
                        # fine schedule (existing behavior, unchanged).
                        pts = [time_steps[kk].item()
                               for kk in range(_k + 1, num_steps + 1)]
                        mode = "naive-fine"
                    else:
                        # EARLY pick: preview grid (uniform t' -> 0, same as
                        # euler_lookahead) above the handoff, one step onto
                        # the handoff, then fine-schedule points below it.
                        la_pts = np.linspace(t_pick, 0.0, _la + 1)
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
                                    prompt_idx=self.particle_prompt_idx[i:i+1])
                                z_f = z_f + (pts[jj + 1] - pts[jj]) * v_f
                                # REAL per-step refine progress (routed to the
                                # job's refine_progress, not the main bar).
                                if progress_callback:
                                    progress_callback(
                                        (_ri_pos * _n_ref_steps + jj + 1)
                                        / (len(ridx) * _n_ref_steps),
                                        f"Refining image {i} — step {jj + 1}/{_n_ref_steps}",
                                        seg_idx_done, n_segments_total,
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
                prev_kept_indices = kept_indices  # save for next incremental update
                selections.append(dict(step=int(k), interval=int(interval_count),
                                       kept=[int(i) for i in kept_indices]))

                # e) Selected -> new anchors, shrink N
                kept_latents = X_all[kept_indices].clone()
                num_particles = len(kept_indices)
                total_particles = num_particles * C

                X_all = kept_latents.unsqueeze(1).expand(-1, C, -1, -1, -1).clone()
                X_all = X_all.reshape(total_particles, 16, latent_h, latent_w)

                # Each kept particle becomes a group of C (anchor + clones); the
                # whole group inherits the kept particle's prompt variation.
                self.particle_prompt_idx = (
                    self.particle_prompt_idx[kept_indices].repeat_interleave(C))

                is_anchor = torch.zeros(total_particles, dtype=torch.bool)
                for i in range(num_particles):
                    is_anchor[i * C] = True

                print(f"  Selected {num_particles} -> {num_particles} anchors x {C} = "
                      f"{total_particles} total (GLASS init at next segment)")

        # ===== Final: decode all =====
        final_dir = os.path.join(output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        print(f"\n--- Final output (step {num_steps}/{num_steps}) ---")
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
        _decoded_final = []          # cache batched tensors for the return (avoid a 2nd decode)
        _DEC_CHUNK = 8               # VRAM-safe chunk for the batched VAE decode
        _seg_len_f, _la_f, _tot_units_f = _seg_progress_units(final_seg_idx)
        with torch.no_grad():
            for start in range(0, total_particles, _DEC_CHUNK):
                end = min(start + _DEC_CHUNK, total_particles)
                if progress_callback:
                    _done = _seg_len_f + _DEC_UNITS * (start / max(1, total_particles))
                    progress_callback(
                        _done / _tot_units_f,
                        f"Decoding {total_particles} final images...",
                        final_seg_idx, n_segments_total,
                    )
                imgs_t = self.decode(X_all[start:end])      # BATCHED [chunk, C, H, W]
                _decoded_final.append(imgs_t.float().cpu())
                for j in range(imgs_t.shape[0]):
                    pil_img = self._tensor_to_pil(imgs_t[j:j+1])
                    pil_img.save(os.path.join(final_dir, f"particle_{start + j:03d}.png"))
                    pil_images.append(pil_img)

        features = self._extract_dino_features(pil_images)
        plot_image_cluster(
            features, pil_images,
            title="DINOv3-B Cluster — Final",
            save_path=os.path.join(final_dir, "cluster_dinov3.png"),
            n_neighbors=min(15, total_particles - 1),
            min_dist=0.1, thumb_size=80,
        )

        # Final incremental layout
        if inc_layout is not None and inc_layout.coords is not None:
            inc_layout.update(list(range(total_particles)), features, pil_images)
            inc_layout.plot(
                title="Final — Incremental Layout",
                save_path=os.path.join(final_dir, "layout_incremental.png"),
                thumb_size=80,
            )

        self._save_grid(pil_images, os.path.join(final_dir, "grid.png"), labels)

        print(f"\n  Final images saved to: {final_dir}/")
        print(f"  Incremental layout: {final_dir}/layout_incremental.png")

        # ===== SAVE THE NOISE: write the reproduction recipe =====
        import json as _json
        _recipe = dict(
            master_seed=master_seed,
            model_id=self.model_id,
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            rho=float(rho),
            spawn_mode=spawn_mode,
            schedule_mode=schedule_mode,
            num_clones=int(C),
            img_shape=[int(imgH), int(imgW)],
            prompts=list(getattr(self, "prompts", None) or [self.prompt]),
            sigmas=[float(t) for t in time_steps.tolist()],
            resample_steps=[int(s) for s in resample_steps],
            image_seeded=bool(init_latent is not None),
            init_strength=(float(init_strength)
                           if init_latent is not None else None),
            interval_start=int(interval_start),
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

        return torch.cat(_decoded_final, dim=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FLUX.1-Dev Interactive FMTT")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--num_particles", type=int, default=6)
    parser.add_argument("--num_clones", type=int, default=3)
    parser.add_argument("--num_resampling_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--lookahead_steps", type=int, default=8)
    parser.add_argument("--schedule_mode", type=str, default="linear", choices=["linear", "aggressive", "none"])
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="interactive_fmtt")
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args()

    init_kwargs = dict(prompt=args.prompt, device="cuda")
    if args.model_id:
        init_kwargs["model_id"] = args.model_id

    sampler = FluxDevInteractive(**init_kwargs)

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

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    pil_final = FluxDevInteractive._tensor_to_pil(best_image[0:1])
    pil_final.save(args.output)
    print(f"\nFinal image saved to: {args.output}")
