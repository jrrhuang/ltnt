# LTNT

LTNT is a creative AI image tool for intuitively exploring the latent space of
flow-matching and diffusion image models. It gives you a fast loop where you
navigate, branch, and cluster the latent manifold to find images you love.

Enter a prompt and LTNT generates a whole population of images, lays them out
so visually similar ones sit near each other, and lets you pick the ones worth
developing. Each pick generates new variations that stay close to it while
resolving differently. A few rounds in, the spread narrows as your taste
sharpens.

## Workflow

1. **Generate.** A prompt gives you a canvas of candidates covering different
   readings of it.
2. **Make more like these.** Select your favorites. New variations appear near
   them and stay close to what you picked. Unpicked images fade back but stay
   on the canvas.
3. **Explore.** The canvas organizes itself by visual similarity, so related
   images sit together and you can see the range the model has for your prompt.
4. **Refine.** Each round generates variations closer to your picks, so the
   spread narrows as your taste sharpens.

## Requirements

- NVIDIA GPU with at least 40 GB of memory
- CUDA 12.x driver
- Python 3.10 or newer
- 40 GB of free disk space
- A Hugging Face account with access to FLUX.1-dev

## Installation

```bash
git clone https://github.com/jrrhuang/ltnt.git
cd ltnt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
bash download_models.sh
```

FLUX.1-dev is gated. Accept the license at
https://huggingface.co/black-forest-labs/FLUX.1-dev before downloading.
`download_models.sh` fetches the flow-map LoRA, FLUX.1-dev, and DINOv2 into
`models/`. It skips anything already present, so an interrupted download
resumes by running it again.

## Usage

```bash
bash run.sh
```

Open http://localhost:8001, enter a prompt, and press GENERATE. The first
prompt loads the model and takes a few minutes. Subsequent rounds take seconds.

When the server runs on a remote machine, forward the port and open the same
address locally.

```bash
ssh -N -L 8001:localhost:8001 user@host
```

## How it works

A prompt starts many particles through a flow-matching or diffusion trajectory.
The trajectory pauses partway, before the images have resolved, and each
particle is previewed. Previews are embedded with DINOv2 and projected into two
dimensions, which is what puts visually similar images near each other.

Selecting an image generates variations from it. A variation renoises that
image's intermediate state and integrates back down, so it resolves differently
while staying conditioned on what the trajectory had already committed to.
Branching partway through, while the image is still unresolved, is what gives
the variations room to differ.

`server/spawn/` holds the variation methods, picked by name from a registry.
`distance` sets how far a variation travels from the image it came from, from 0
at that image to 1 at an independent sample, and a narrowing schedule lowers it
across rounds.

## Models

| Model | Key | Notes |
|---|---|---|
| FLUX.1-dev with the flow-map LoRA | `fluxfm` | The default. 13 s per round. |
| FLUX.1-dev | `flux` | The same backbone without the flow map. 25 s per round. |
| Krea-2 | `krea2` | Higher quality, 83 s per round. Gated, and fetched only when `LTNT_WITH_KREA=1`. |

Times are for three variations on one L40S. LTNT runs on self-consistent
flow-matching models. Distilled few-step models break the assumption the
variation step relies on.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `PORT` | `8001` | Port the server listens on. |
| `LTNT_HOST` | `127.0.0.1` | Interface to bind. Set `0.0.0.0` in a container. |
| `LTNT_MODELS` | `./models` | Where weights are stored. |
| `LTNT_WITH_KREA` | `0` | Fetch Krea-2 during download. |
| `FLUXFM_CLONE_STEPS` | `4` | Flow-map jumps per variation. Raise for more detail. |

## Deployment

TBD.

## License

MIT.
