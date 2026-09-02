# LTNT

LTNT is a creative AI image tool for intuitively exploring the latent space of
flow-matching and diffusion image models. You generate variations, select the
ones you like, and refine toward them over successive rounds.

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
4. **Refine.** Repeat.

Variations branch partway through generation, while the image is still
unresolved. DINOv2 embeddings decide where images land on the canvas.

## Installation

Needs an NVIDIA GPU with at least 40 GB of memory, a CUDA 12.x driver, Python
3.10 or newer, and 40 GB of free disk.

Accept the FLUX.1-dev license at
https://huggingface.co/black-forest-labs/FLUX.1-dev.

```bash
git clone https://github.com/jrrhuang/ltnt.git
cd ltnt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
bash download_models.sh
```

## Usage

```bash
bash run.sh
```

Open http://localhost:8001, enter a prompt, and press GENERATE. The first
prompt loads the model and takes a few minutes. Subsequent rounds take seconds.

For a remote GPU, forward the port and open the same address locally.

```bash
ssh -N -L 8001:localhost:8001 user@host
```

Set `LTNT_MODELS` to store weights outside the repository.

## Models

LTNT supports FLUX.1-dev with the distilled flow-map LoRA, plain FLUX.1-dev,
and Krea-2. The flow-map model is the default and the fastest.

Krea-2 is gated separately. Set `LTNT_WITH_KREA=1` when running
`download_models.sh` to download it.

## License

MIT.
