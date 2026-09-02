# LTNT

LTNT is a creative AI image tool for intuitively exploring the latent space of
flow-matching and diffusion image models. It gives you a fast loop where you
navigate, branch, and cluster the latent manifold to find images you love.

Enter a prompt and LTNT generates a whole population of images, lays them out
so visually similar ones sit near each other, and lets you pick the ones worth
developing. Each pick generates new variations that stay close to it while
resolving differently. A few rounds in, the spread narrows as your taste
sharpens.

## The loop

1. **Generate.** A prompt yields a spatial cluster of diverse candidates.
2. **Make more like these.** Select favorites, and variations appear near them,
   staying coherent with the pick.
3. **Explore.** The canvas auto-organizes by visual similarity, using DINOv2
   features.
4. **Board.** Pin favorites to a persistent collection. Explore from a pin to
   seed a new session.
5. **Refine.** Early rounds are fast previews, enough to judge direction.
   Deeper rounds resolve cleanly.

## Model-agnostic

LTNT runs on any self-consistent flow-matching model. FLUX.1-dev with a
distilled flow-map LoRA is the default at about 13 seconds per round on an
L40S. Plain FLUX.1-dev takes 25 seconds, and Krea-2-Raw takes 83 seconds at
higher quality. To add a backend, implement the interface in
[SAMPLER_INTERFACE.md](server/SAMPLER_INTERFACE.md).

Distilled and turbo few-step models break the GLASS path consistency. Use a
base or dev flow.

## Installation

Needs an NVIDIA GPU with at least 40 GB of memory, a CUDA 12.x driver, Python
3.10 or newer, and 40 GB of free disk.

```bash
git clone https://github.com/jrrhuang/ltnt.git
cd ltnt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
bash download_models.sh
```

FLUX.1-dev is gated. Accept the license at
https://huggingface.co/black-forest-labs/FLUX.1-dev, then log in with
`huggingface-cli login`. `download_models.sh` skips anything already present
and can be re-run. Krea-2-Raw is gated separately and downloads only when
`LTNT_WITH_KREA=1`.

## Usage

```bash
bash run.sh
```

Open http://localhost:8001, enter a prompt, and press GENERATE. The first
prompt loads the model and takes a few minutes. Subsequent rounds take seconds.

The core loop of generate, explore, cluster, make-more, and board runs with no
API keys. Only the optional external edit providers need them, listed in
[.env.example](server/.env.example).

Set `PORT` to serve elsewhere, `LTNT_HOST=0.0.0.0` to bind all interfaces, and
`LTNT_MODELS` to keep weights outside the repository. When the server runs on a
remote machine, forward the port and open the same address locally.

```bash
ssh -N -L 8001:localhost:8001 user@host
```

## Method

Variations come from GLASS interactive sampling, which takes stochastic
transitions from a deterministic, self-consistent flow, with Euler-lookahead
previews at each checkpoint. Branching happens partway through generation,
while the image is still unresolved. See
[SAMPLER_INTERFACE.md](server/SAMPLER_INTERFACE.md) for the backend contract and
`server/spawn/` for the variation methods.

## Deployment

TBD.

## License

MIT. Model licenses differ, and FLUX.1-dev is non-commercial.
