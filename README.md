# LTNT

LTNT ("latent") is a creative AI image tool. Type a prompt and you get a whole
canvas of images. Pick the ones you like, ask for more like those, and LTNT
grows new variations from your picks. Repeat until you land on the image you
want.

## The loop

1. **Generate.** A prompt gives you a canvas of candidates covering different
   readings of it.
2. **Make more like these.** Select your favorites. Children spawn near them
   and stay coherent with what you picked. Unpicked images fade back but stay
   on the canvas.
3. **Explore.** The canvas organizes itself by visual similarity, so related
   images sit together and you can see the range the model has for your prompt.
4. **Refine.** Each round spawns closer to your picks, so the spread narrows as
   your taste sharpens.

## Setup

An NVIDIA GPU with at least 40 GB of memory, a CUDA 12.x driver, Python 3.10 or
newer, and about 40 GB of disk.

```bash
git clone https://github.com/jrrhuang/ltnt.git
cd ltnt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
bash download_models.sh
```

FLUX.1-dev is gated, so accept the license at
https://huggingface.co/black-forest-labs/FLUX.1-dev before downloading.
`download_models.sh` fetches the flow-map LoRA, FLUX.1-dev, and DINOv2 into
`models/`. It skips anything already present, so an interrupted download
resumes by running it again.

## Run

```bash
bash run.sh
```

Open http://localhost:8001, type a prompt, and press GENERATE. The first prompt
loads the model and takes a few minutes. After that a round takes seconds.

Running on a remote GPU, forward the port and open the same address locally:

```bash
ssh -N -L 8001:localhost:8001 user@host
```

## How it works

A prompt starts many particles through a flow-matching or diffusion trajectory.
The trajectory pauses partway, before the images have resolved, and each
particle is previewed. Previews are embedded with DINOv2 and projected into two
dimensions, which is what puts visually similar images near each other.

Selecting a particle clones it. A clone renoises the parent's intermediate
state and integrates back down, so the child resolves differently while staying
conditioned on what the parent had already committed to. Branching from an
intermediate state is what makes children alternatives to their parent rather
than edits of a finished image.

`server/spawn/` holds the spawn methods, picked by name from a registry.
`distance` sets how far a child travels from its parent, from 0 at the parent
to 1 at an independent sample, and a narrowing schedule lowers it across
rounds.

## Models

| Model | Key | Notes |
|---|---|---|
| FLUX.1-dev with the flow-map LoRA | `fluxfm` | The default. 13 s per round. |
| FLUX.1-dev | `flux` | The same backbone without the flow map. 25 s per round. |
| Krea-2 | `krea2` | Higher quality, 83 s per round. Gated, and fetched only when `LTNT_WITH_KREA=1`. |

Times are for a brood of three on one L40S. LTNT runs on self-consistent
flow-matching models. Distilled few-step models break the assumption the
cloning step relies on.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `PORT` | `8001` | Port the server listens on. |
| `LTNT_HOST` | `127.0.0.1` | Interface to bind. Set `0.0.0.0` in a container. |
| `LTNT_MODELS` | `./models` | Where weights are stored. |
| `LTNT_WITH_KREA` | `0` | Fetch Krea-2 during download. |
| `FLUXFM_CLONE_STEPS` | `4` | Flow-map jumps in a child's descent. Raise for more detail per child. |

## Rented GPU

TBD.

## License

MIT.
