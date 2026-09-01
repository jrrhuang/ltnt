# LTNT — Model-agnostic sampler interface (adding a new backend)

LTNT's exploration engine (GLASS interactive sampling) is **model-agnostic**: it runs over any
flow-matching / diffusion image model that exposes a *true, self-consistent* velocity field. FLUX.1-dev,
SANA, and Krea-2-Raw ship today; this doc is the contract for adding another.

> ⚠ **Hard requirement: the base flow must be SELF-CONSISTENT (un-distilled).** GLASS samples stochastic
> transitions from a deterministic flow and does lookahead along it; a model **distilled into a few-step,
> non-path-consistent flow** (FLUX-schnell/klein, SDXL-Turbo, SANA-Sprint, Krea-Turbo, …) breaks the
> trajectory consistency → clones go off-manifold. Use the **base/dev** flow, not the turbo/distilled one.

## Required interface

A backend is a class `MyModelInteractive` (see `flux_fmtt_dev.py`, `sana_fmtt.py`, `krea2_fmtt.py`).
It MUST implement:

| Method | Signature | Contract |
|---|---|---|
| `__init__` | `(prompt, model_id=…, device, dtype=…)` | Load the model resident on `device`; cache the prompt embedding. Resolve weights from a local snapshot path (env override, e.g. `*_LOCAL_PATH`) to survive cluster autofs/offline-mode. |
| `predict_velocity` | `(z, t_cur, guidance_scale, prompt_idx=None) -> v` | **The only truly model-specific piece.** Return the flow velocity `v(z, t)` at latent `z`, time/σ `t_cur`, with CFG. `prompt_idx` (optional) selects a per-particle prompt embedding for augmentation diversity. |
| `decode` | `(z) -> image_tensor` | Latent → pixel tensor in `[0,1]`, shape `[B,3,H,W]`. (VAE decode + the model's latent normalization — note Krea-2 uses `latent*std+mean`.) |
| `euler_lookahead` | `(z, t_cur, guidance_scale, num_steps, prompt_idx=None) -> z_hat` | Cheap deterministic Euler integration `t_cur → 0` for the per-checkpoint PREVIEW (the "foreshadow the final" image). Few steps = fast/rough; this is where the progressive-refinement spec lives. |
| `sample_interactive` | `(num_steps, num_particles, num_clones, num_resampling_steps, guidance_scale, rho, spawn_mode, lookahead_steps, …) -> images` | The GLASS interactive loop: integrate the flow, at each resample checkpoint emit previews + wait for the user's pick, then GLASS-clone the kept particles. Reuse the shared pattern from `sana_fmtt`/`flux_fmtt`; only `predict_velocity`/`decode`/`encode` differ per model. |
| `_extract_dino_features` | `(pil_images) -> np.ndarray` | DINOv2 features (L2-normalized) for the spatial UMAP/MDS layout. Identical across backends — copy it. |

## Optional extensions (gated by signature inspection in `server.py`)

The server feature-detects these via `inspect.signature(...)` and degrades gracefully if absent, so a minimal
backend works without them:

- `set_prompts(prompts)` — embed K prompt variations into `cached_prompt_embeds [K,…]` for **prompt-augmented
  semantic diversity** (requires keeping text encoders resident: `keep_text_encoders=True`). Without it, all
  particles share one prompt. FLUX-only today. (`set_prompt(p)` = the single-prompt convenience wrapper.)
- `encode_image_to_latent(pil, img_shape)` — VAE-encode a seed image to a clean latent for **explore-from-pin**
  (image-seeded SDEdit; server re-noises to `init_strength`). FLUX-only today.

## How the server dispatches (`server.py`)

`_do_generate` branches on `req["model"]` (`flux`/`sana`/`krea2`) to construct the right class, then calls
`sample_interactive(**kwargs)`; it strips kwargs the backend doesn't accept via signature inspection, so adding
a backend that lacks `lookahead_steps`/`prompt_idx`/`init_latent` is safe. FLUX is kept warm
(`_warm_flux`) across generations; heavy single-resident models (Krea-2 ~37GB @512) free the previous sampler
first. To register a new backend: add an `elif model == "mymodel":` branch + a frontend dropdown entry.

## Memory / resolution notes
- One big model resident at a time on an L40S (48GB). Krea-2 is 512-only (1024 OOMs).
- Resolve weights from a node-local snapshot dir (stage to `/scratch`, set `*_LOCAL_PATH`) — HF offline
  resolution is flaky on cluster nodes (the "autofs gremlin").
