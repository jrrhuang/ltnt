# LTNT — Latent Navigator

**An interactive tool for artists to explore the latent space of diffusion
models.** Generate a spread of images, pick the ones that speak to you, and
breed them — LTNT spawns children of your picks, from wide global
re-imaginings down to local detail variations, with an LLM proposing
alternative readings of your prompt along the way. You are the reward signal.

Built on flow-matching models (FLUX.1-dev + a distilled flow map for
~seconds-fast iteration; Krea-2 for maximum quality) and a sequential
Monte Carlo loop where selection happens in the browser.

## Two ways to run

### 1. Your own GPU (any CUDA machine, ≥40 GB VRAM recommended)

```bash
git clone <this-repo> && cd ltnt
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
# Accept the FLUX.1-dev license on HuggingFace, then:
export HF_TOKEN=<your token>
bash run.sh          # downloads models on first boot (~35 GB), serves :8001
```

Open http://localhost:8001 — type a prompt, GENERATE, click images you
like, BREED. That's the whole loop.

### 2. RunPod (no GPUs of your own — you pay per GPU-hour)

1. Create a RunPod account and add credits.
2. Deploy a pod from the Docker image built by this repo's CI
   (`ghcr.io/<owner>/ltnt:latest`), GPU type A40/L40S/A100 (≥40 GB),
   expose port **8001**, and set the env var `HF_TOKEN` to your
   HuggingFace token (with FLUX.1-dev access accepted).
3. First boot downloads models (~35 GB, one-time per volume — attach a
   RunPod network volume at `/models` to persist them).
4. Open the pod's port-8001 URL. Same loop, in your browser.

## Models
| backend | role | notes |
|---|---|---|
| FLUX-FM | fast path — breeds in seconds | FLUX.1-dev + 512px flow-map LoRA (auto-downloaded) |
| Krea-2 | quality path | gated on HF; optional |

## How it works (short version)
Round 1 casts a spread from noise. Selecting images and breeding spawns
children by re-noising each pick to a scheduled depth and transporting back
with the flow map — deep re-noise early (global structural variety, with the
LLM's wild prompt readings), shallow late (local refinement, readings that
restyle rather than recompose). Diversity decreases approximately linearly
across rounds by design. Every session is seeded and replayable, and each
round records labeled metadata (`interval_*/meta.json`).

## License
MIT (see LICENSE).
