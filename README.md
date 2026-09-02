# LTNT

LTNT ("latent") is a creative AI image tool for intuitive latent-space
exploration of flow-matching and diffusion image models. It gives you a fast
loop where you navigate, branch, and cluster the latent manifold to find images
you love.

Enter a prompt and LTNT generates a whole population of images, lays them out
so visually similar ones sit near each other, and lets you pick the ones worth
developing. Each pick spawns children that stay connected to their parent while
resolving differently. A few rounds grow a tree, and the spread narrows as your
taste sharpens.

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

## Models

| Model | Key | Notes |
|---|---|---|
| FLUX.1-dev with the flow-map LoRA | `fluxfm` | The default. 13 s per round. |
| FLUX.1-dev | `flux` | The same backbone without the flow map. 25 s per round. |
| Krea-2 | `krea2` | Higher quality, 83 s per round. Gated, and fetched only when `LTNT_WITH_KREA=1`. |

Times are for a brood of three on one L40S.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `PORT` | `8001` | Port the server listens on. |
| `LTNT_HOST` | `127.0.0.1` | Interface to bind. Set `0.0.0.0` in a container. |
| `LTNT_MODELS` | `./models` | Where weights are stored. |
| `LTNT_WITH_KREA` | `0` | Fetch Krea-2 during download. |
| `FLUXFM_CLONE_STEPS` | `4` | Flow-map jumps in a child's descent. Raise for more detail per child. |

How children come from a parent lives in `server/spawn/`. A spawn method is
picked by name from a registry, and `distance` controls how far a child travels
from its parent, from 0 at the parent to 1 at an independent sample. A
narrowing schedule lowers it across rounds.

## Rented GPU

TBD.

## License

MIT.
