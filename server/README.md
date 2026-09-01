# LTNT.APP

**LTNT** ("latent") is a creative AI image tool whose headline is **intuitive latent-space
exploration** of flow-matching / diffusion image models — not one-shot prompt→image, but a fast loop where you
*navigate, branch, and cluster* the latent manifold to discover images you love.

> Status: **pre-1.0, under active development.** The architecture is evolving (a unified infinite-canvas
> redesign is proposed). Interfaces may change. Issues/PRs welcome.

## The loop
1. **Generate** — a prompt yields a spatial cluster of *diverse* candidates (an on-device LLM expands your
   prompt into varied interpretations — different medium / style / mood — so round 1 spans the space).
2. **Pick + "Make more like these"** — select favorites; children spawn near them and stay *coherent with your
   pick* (GLASS stochastic-transition sampling from a self-consistent flow), while unpicked images fade.
3. **Explore** — the canvas auto-organizes by visual similarity (DINO features + UMAP).
4. **Board** — pin favorites to a persistent collection; **explore-from-pin** to seed a new session from one.
5. **Progressive refinement** — early rounds are fast, lightly-approximated previews (enough to pick a
   direction); deeper rounds refine cleaner. Optimize early for speed, later for fidelity.

## Model-agnostic
LTNT runs over any **self-consistent (un-distilled) flow-matching model**. FLUX.1-dev, SANA, and Krea-2-Raw
ship today. To add a backend, implement the small interface in **[SAMPLER_INTERFACE.md](SAMPLER_INTERFACE.md)**.
> ⚠ Distilled/turbo few-step models (schnell/klein, SDXL-Turbo, SANA-Sprint, …) break the GLASS path
> consistency — use the **base/dev** flow.

## Run it (local, zero API keys)
The core app — generate, explore, cluster, make-more, board — runs **with no API keys**, on your own GPU.
(Only the optional external *edit* providers need keys; see [`.env.example`](.env.example).)

```bash
# 1. Backend deps: a CUDA PyTorch env with diffusers/transformers/fastapi/uvicorn/umap-learn/scikit-learn.
#    (A pinned requirements.txt is a TODO; today the maintainer uses a conda env.)
# 2. Cache a supported model (e.g. FLUX.1-dev) in your HF cache; set HF_HOME if needed.
# 3. Build the frontend (served by the backend):
cd ltnt_frontend && npm install && npm run build && cd ..
# 4. Launch the server (serves the built UI + API on :8001):
python server.py            # then open http://localhost:8001
```
VRAM: FLUX/SANA fit comfortably on a 24–48GB GPU; Krea-2-Raw is heavy (~37GB, 512px-only). One model resident
at a time.

## Tests
```bash
python tests/test_logic.py   # GPU-free unit tests (diversity schedules, layout guards)
```

## Method / background
GLASS interactive sampling (stochastic transitions from a deterministic, self-consistent flow + Euler-lookahead
previews) plus flow-map test-time alignment. See `SAMPLER_INTERFACE.md` for the backend contract and the inline
docstrings in `flux_fmtt_dev.py` / `server.py`.

## License
TBD (see project decisions). NB **model licenses differ**: FLUX-dev is non-commercial; SANA and the Qwen
prompt-augment model are Apache-2.0 — pick your default backend accordingly.
