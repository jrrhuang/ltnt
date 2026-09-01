# spawn

How a child latent is produced from a parent latent, and how prompt
variation is layered on top.

## Interfaces

| Object | Role |
|---|---|
| `SpawnStrategy` | `(parent, ctx) -> child`. One child, one call. |
| `SpawnContext` | Model handle, conditioning, `t_parent`, `t_end`, guidance. |
| `Conditioning` | Prompt text and its encodings for one child. |
| `ReadingPool` | Alternate prompts, encoded on demand. |
| `ChildPlan` | A strategy and a conditioning for one child. |

Times are FLUX times: `1.0` is noise, `0.0` is clean.

## Strategies

| Name | Mechanism | Key parameters |
|---|---|---|
| `renoise` | Renoise the parent latent to `t + rho(1-t)`, integrate down. | `rho`, `steps` |
| `lookahead` | Renoise the parent's clean-image estimate to `tau`, integrate down. | `tau`, `steps` |
| `glass` | GLASS bridge to `sigma_floor`, then integrate down. | `rho`, `inner_steps`, `sigma_floor`, `steps` |

`steps` is the number of flow-map jumps in the descent. One jump is a
single-NFE generation and is visibly softer than the parent's full solve.

## Use

```python
from spawn import make, plan_brood, SpawnContext
from spawn.adapter import SamplerModel

strategy = make("renoise", rho=0.95, steps=4)
model = SamplerModel(sampler)
plans = plan_brood(7, strategy, base_cond, pool=readings, augmented=3)

for p in plans:
    ctx = SpawnContext(model, p.cond, t_parent=0.98, t_end=0.0,
                       guidance=3.5)
    child = p.strategy(parent, ctx)
```

Swap the mechanism by changing the `make(...)` name. Add prompt variation
by passing a `ReadingPool` and the number of children that should use it;
strategies are unaffected.

## Adding a strategy

Implement `__call__(parent, ctx) -> Tensor` and register it in
`STRATEGIES`. `base.renoise`, `base.descend`, and `base.endpoint` cover
the common pieces.
