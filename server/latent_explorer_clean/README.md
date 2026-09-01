# LTNT

**LTNT** ("latent") is a tool for **navigating the latent space of a diffusion / flow-matching
image model** — not one-shot prompt→image, but an interactive loop where you spread out across the
space, prune what you don't want, and breed what you do.

> **Status: research MVP, pre-1.0.** This is a working proof-of-concept, not a polished product.
> Interfaces change, rough edges exist, and it's built to self-host on your own GPU. Issues and PRs welcome.

## What it does

1. **Diverse spread** — a prompt yields a *spatial cluster of diverse candidates*. An on-device LLM
   expands your prompt into varied interpretations (medium / style / mood) so the first round genuinely
   spans the model's space instead of collapsing to one look.
2. **Prune** — keep the candidates you like; the rest fade.
3. **Breed ("make more like these")** — children spawn near your picks and stay *coherent with them*
   (GLASS stochastic-transition sampling from a self-consistent flow), while the space re-organizes by
   visual similarity (DINOv2 features + UMAP).
4. **Create-map atlas** — a fly-through **create-map** view lays out a whole neighborhood of the latent
   space as a navigable atlas you can span and generate from interactively.

## Models

LTNT runs over **self-consistent (un-distilled) flow-matching models**. One model is resident at a time:

| Model | Repo | Notes |
|-------|------|-------|
| **Krea-2-Raw** (default) | `krea/Krea-2-Raw` | ~37 GB, 512px-only; heaviest but the validated default |
| **FLUX.1-dev** | `black-forest-labs/FLUX.1-dev` | gated; **non-commercial license** (see below) |
| **SD3.5** | `stabilityai/stable-diffusion-3.5-medium` | gated |

> ⚠ Distilled/turbo few-step models (schnell/klein, SDXL-Turbo, SANA-Sprint, …) break the GLASS path
> consistency — use the **base/dev** flow. To add a backend, implement the small interface in
> [`SAMPLER_INTERFACE.md`](SAMPLER_INTERFACE.md).

## Quickstart (Docker + GPU)

Requires an NVIDIA GPU with the container toolkit. FLUX/SD3.5 fit on a 24–48 GB GPU; Krea-2-Raw needs ~37 GB.

```bash
cp .env.example .env         # optional: add HF_TOKEN for gated FLUX / SD3.5
docker build -t ltnt .
docker run --gpus all -p 8001:8001 \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface:/workspace/huggingface \
  ltnt
# open http://localhost:8001
```

### Local (no Docker)

```bash
pip install -r requirements.txt              # into a CUDA PyTorch env
cd ltnt_frontend && npm ci && npm run build && cd ..
python server.py                             # serves UI + API on :8001
```

## Tests

```bash
python tests/test_logic.py                   # GPU-free unit tests
```

## Method / background

GLASS interactive sampling (stochastic transitions from a deterministic, self-consistent flow +
Euler-lookahead previews) plus flow-map test-time alignment. See [`SAMPLER_INTERFACE.md`](SAMPLER_INTERFACE.md)
for the backend contract and the inline docstrings in `flux_fmtt_dev.py` / `server.py`.

## License

**TODO** — a license has not yet been chosen for this repository. Until one is added, no open-source
license is granted.

**Model licenses differ and are separate from this repo's license.** In particular **FLUX.1-dev is
released under a non-commercial license** — do not use the FLUX backend commercially. Review each model's
license before use.
