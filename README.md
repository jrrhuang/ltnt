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

1. **Generate.** A prompt fills the canvas with candidates.
2. **Pick + "make more like these."** LTNT generates variations from your
   selections.
3. **Explore.** The canvas organizes itself by visual similarity.
4. **Refine.** Repeat until the spread settles on what you want.

Variations branch partway through generation, while the image is still
unresolved. DINOv2 embeddings decide where images land on the canvas.

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
and can be re-run.

## Usage

```bash
bash run.sh
```

Open http://localhost:8001, enter a prompt, and press GENERATE. The first
prompt loads the model and takes a few minutes. Subsequent rounds take seconds.

Set `PORT` to serve elsewhere, `LTNT_HOST=0.0.0.0` to bind all interfaces, and
`LTNT_MODELS` to keep weights outside the repository. When the server runs on a
remote machine, forward the port and open the same address locally.

```bash
ssh -N -L 8001:localhost:8001 user@host
```

## Models

FLUX.1-dev with a distilled flow-map LoRA is the default and takes about 13
seconds per round on an L40S. Plain FLUX.1-dev takes 25 seconds, and Krea-2
takes 83 seconds for higher quality. Krea-2 is gated separately and downloads
only when `LTNT_WITH_KREA=1`.

LTNT runs on self-consistent flow-matching models. Distilled few-step models
are unsupported.

## License

MIT.
