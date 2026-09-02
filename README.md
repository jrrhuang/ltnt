# Latent Navigator (LTNT)

LTNT is a tool for navigating the range of images a generative model can
produce for one prompt. It generates a population instead of a single sample,
arranges the population so visually similar images sit near each other, and
grows new variations from the images you select.

A generative model encodes many possible interpretations of a prompt. They
differ in composition, style, lighting, and semantic emphasis. A prompt box
returns one of them at a time and gives no view of the others. LTNT shows the
population and lets your selections decide which regions are developed further.
Repeated selection concentrates the population on the directions you keep
choosing.

## The loop

1. **Generate.** Enter a prompt. The system runs many particles in parallel
   through the diffusion trajectory and previews each one at the first
   checkpoint.
2. **Arrange.** Previews are embedded with DINOv2 and projected into two
   dimensions, so visually similar previews appear nearby.
3. **Select.** Click the images worth developing. Unselected images stay on the
   canvas as desaturated thumbnails.
4. **Breed.** Each selection spawns children that remain connected to it and
   differ from each other. Children are placed near their parent.
5. **Repeat.** Two or three rounds grow a tree of variations rooted in the
   first population.

Branching happens at intermediate states. Part of the image is still
undetermined at that point, so children can resolve differently while staying
conditioned on the same parent.

## Requirements

- NVIDIA GPU with at least 40 GB of memory
- CUDA 12.x driver
- Python 3.10 or newer
- ~40 GB of disk for model weights
- A Hugging Face account with access to `black-forest-labs/FLUX.1-dev`

## Run on your own GPU

Accept the FLUX.1-dev license at
https://huggingface.co/black-forest-labs/FLUX.1-dev, then create a read token
at https://huggingface.co/settings/tokens.

```bash
git clone https://github.com/jrrhuang/ltnt.git
cd ltnt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login          # or: export HF_TOKEN=<your token>
bash download_models.sh        # ~35 GB into models/, re-runnable
bash run.sh
```

Open http://localhost:8001, type a prompt, and press GENERATE.

`download_models.sh` fetches the flow-map LoRA, FLUX.1-dev, and DINOv2.
It skips anything already present, so an interrupted download resumes by
running it again. Set `LTNT_MODELS` to place weights elsewhere.

To serve on a different port or interface:

```bash
PORT=9000 LTNT_HOST=0.0.0.0 bash run.sh
```

On a remote machine, keep the default loopback binding and forward the port:

```bash
ssh -N -L 8001:localhost:8001 user@host
```

## Run on a rented GPU

TBD.

## Models

| Model | Key | Notes |
|---|---|---|
| FLUX.1-dev with the flow-map LoRA | `fluxfm` | The default. A round of children takes seconds. |
| FLUX.1-dev | `flux` | The same backbone without the flow map. Slower per round. |
| Krea-2 | `krea2` | Higher quality and slower. Separately gated, and fetched only when `LTNT_WITH_KREA=1`. |

Measured on one L40S, generating a population of four and then a brood of
three: 13 s per brood with the flow map, 25 s with FLUX, 83 s with Krea-2.

Set `LTNT_MODELS` to place weights outside the repository.

## Spawning

`server/spawn/` holds the strategies that turn a parent into children. Each is
a callable mapping a parent latent to a child latent, selected by name:

| Strategy | Mechanism | Parameters |
|---|---|---|
| `renoise` | Renoise the parent latent to `t + distance(1-t)`, then integrate down. | `distance`, `steps` |
| `lookahead` | Renoise the parent's clean-image estimate, then integrate down. | `distance`, `steps` |
| `glass` | GLASS bridge to a noise floor, then integrate down. | `distance`, `inner_steps`, `sigma_floor` |

`distance` sets how far a child travels from its parent. A value near 0 keeps
the parent's structure and a value near 1 approaches an independent sample.
Prompt variation composes with any strategy through `plan_brood`, which assigns
per-child conditioning without touching the spawn mechanism. See
`server/spawn/README.md`.

## Tests

```bash
pip install pytest
python -m pytest server/spawn/tests -q
```

The suite runs on CPU and needs no model weights.

## License

MIT.
