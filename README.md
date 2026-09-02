# LTNT

LTNT is a creative AI image tool for interactively exploring a model's latent
space by generating variations, selecting what you like, and iteratively
refining the results according to your preferences, guiding the search toward
images you love.

Enter a prompt and LTNT generates a population of images, then arranges them on
a 2D map so visually similar images appear nearby. Select the images you like
to generate related variations and continue exploring over multiple rounds.
Variations branch from intermediate diffusion states and stay connected to the
image you selected while resolving differently. The spread narrows across
rounds as your preferences become more specific.

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

LTNT supports FLUX.1-dev with Gabe Guo's distilled
[flow-map LoRA](https://huggingface.co/gabeguofanclub/flux-1-dev-flowmap-lsd)
for fast interaction, plain FLUX.1-dev, and Krea-2. The flow-map model is the
default.

Krea-2 is gated separately. Set `LTNT_WITH_KREA=1` when running
`download_models.sh` to download it.

## License

MIT.
