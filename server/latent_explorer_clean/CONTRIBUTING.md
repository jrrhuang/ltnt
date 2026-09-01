# Contributing to LTNT

Thanks for your interest! LTNT is pre-1.0 and moving fast — please open an issue to discuss substantial
changes before a large PR.

## Dev setup
- Backend: a CUDA PyTorch env with diffusers / transformers / fastapi / uvicorn / umap-learn / scikit-learn.
- Frontend: `cd ltnt_frontend && npm install`. Edit `src/`, then `npm run build` (the FastAPI server serves
  `ltnt_frontend/dist`; static assets reload without a server restart — backend changes need a restart).
- Run: `python server.py` → http://localhost:8001. Tests: `python tests/test_logic.py`.

## Ground rules
- **No secrets in the repo.** All keys load via `os.getenv` (see `.env.example`); never commit a real `.env`.
- **Keep it model-agnostic.** New model backends implement the contract in `SAMPLER_INTERFACE.md`; the base
  flow MUST be self-consistent (un-distilled) or GLASS breaks.
- **Add a test** for new pure logic (`tests/test_logic.py`) — especially edge cases (small-N, input bounds).
- **Don't hold the model lock across a user wait** without a timeout — a stuck/abandoned interactive session
  must self-release so it can't wedge generation for everyone.
