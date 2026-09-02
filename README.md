# LTNT

A latent-space navigator for image models. Instead of one image per prompt,
LTNT generates a population, shows you how its members relate, and lets you
breed the ones you like into further generations.

A text-to-image model does not hold a single picture for a prompt. It holds a
space of interpretations — compositions, palettes, framings, readings of the
same words — and a prompt box samples that space blindly. LTNT makes the space
navigable: generate a spread, select what interests you, and spawn children of
your selections. Repeat, and the tree that grows is the record of your choices.

## The loop

1. **Generate.** Enter a prompt. The system runs a spread of generations in
   parallel and previews each one partway through the diffusion trajectory.
2. **Arrange.** Previews are embedded with DINOv2 and placed on a canvas where
   proximity means visual kinship. Distinct readings of the prompt form
   distinct clusters.
3. **Select.** Click the images worth developing. Unselected images stay on the
   canvas, desaturated.
4. **Breed.** Each selection spawns children that share its direction and
   differ from each other. Children are placed around their parent.
5. **Repeat.** Two or three rounds produce a tree rooted in the first spread.

Branching happens at intermediate states, not finished images. A half-resolved
image is still undecided, so its children are genuine alternatives rather than
edits of a settled picture.

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

| Model | Role |
|---|---|
| FLUX.1-dev + flow-map LoRA | Fast path. Seconds per generation. Downloaded automatically. |
| Krea-2 | Quality path. Requires separate access to the Krea weights. |

Set `LTNT_MODELS` to place weights outside the repository.

## Spawning

`server/spawn/` holds the strategies that turn a parent into children. Each is
a callable mapping a parent latent to a child latent, selected by name:

| Strategy | Mechanism | Parameters |
|---|---|---|
| `renoise` | Renoise the parent latent to `t + rho(1-t)`, then integrate down. | `rho`, `steps` |
| `lookahead` | Renoise the parent's clean-image estimate, then integrate down. | `tau`, `steps` |
| `glass` | GLASS bridge to a noise floor, then integrate down. | `rho`, `inner_steps`, `sigma_floor` |

`rho` sets how far a child travels from its parent: near 0 keeps the parent's
structure, near 1 approaches an independent sample. Prompt variation composes
with any strategy through `plan_brood`, which assigns per-child conditioning
without touching the spawn mechanism. See `server/spawn/README.md`.

## Tests

```bash
pip install pytest
python -m pytest server/spawn/tests -q
```

The suite runs on CPU and needs no model weights.

## License

MIT.
