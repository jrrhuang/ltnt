"""The brood seam: one call carries a whole generation.

A spawn method takes the selected parent states at one artist decision
boundary and returns the candidate population at the next. That is the
unit the artist works in, and it is the unit at which methods differ: a
bridge carries auxiliary state across its inner steps, an amortized jump
does not, and a repulsion method needs to see the siblings it is pushing
away from. A per-child interface can express none of those without
smuggling state through the object, and it forces one forward pass per
child.

`PerChild` adapts an independent strategy to this interface, so a method
that genuinely treats children separately stays a few lines.
"""
from dataclasses import dataclass, field, replace
from typing import List, Optional, Protocol, Sequence

import torch

from .base import Conditioning, SpawnContext


@dataclass(frozen=True)
class ChildSpec:
    """What one child of a brood should be."""
    index: int
    distance: float
    cond: Conditioning
    guidance: float = 3.5
    seed: Optional[int] = None


@dataclass(frozen=True)
class Segment:
    """The span a brood crosses, in FLUX time: 1 is noise, 0 is clean."""
    t_parent: float
    t_end: float


@dataclass
class BroodResult:
    """The children of one brood, and what produced them."""
    children: torch.Tensor              # [n_children, ...]
    specs: Sequence[ChildSpec]
    method: str = ""
    forward_passes: int = 0


class SpawnMethod(Protocol):
    """Selected parents at one boundary to a population at the next."""

    def spawn_brood(self, parent: torch.Tensor, specs: Sequence[ChildSpec],
                    segment: Segment, model) -> BroodResult:
        ...


def _generator(spec: ChildSpec, fallback: Optional[torch.Generator]):
    if spec.seed is None:
        return fallback
    return torch.Generator().manual_seed(spec.seed)


@dataclass
class PerChild:
    """Adapts an independent strategy to the brood interface.

    Each child is produced by its own call, so this is the reference
    implementation a batched method must agree with.
    """
    strategy: object
    name: str = ""

    def spawn_brood(self, parent: torch.Tensor, specs: Sequence[ChildSpec],
                    segment: Segment, model) -> BroodResult:
        out, passes = [], []
        for spec in specs:
            calls = []
            ctx = SpawnContext(model=_Counting(model, calls), cond=spec.cond,
                               t_parent=segment.t_parent,
                               t_end=segment.t_end, guidance=spec.guidance,
                               generator=_generator(spec, None))
            strategy = replace(self.strategy, distance=spec.distance)
            out.append(strategy(parent, ctx))
            passes.append(len(calls))
        return BroodResult(children=torch.cat(out, 0), specs=list(specs),
                           method=self.name or type(self.strategy).__name__,
                           forward_passes=sum(passes))


@dataclass
class _Counting:
    """Wraps a model to count forward passes."""
    inner: object
    calls: list

    def velocity(self, x, t, cond, t_next, guidance):
        self.calls.append(t)
        return self.inner.velocity(x, t, cond, t_next, guidance)


@dataclass
class BatchedRenoise:
    """Renoise, with the whole brood carried as one batch.

    Children differ only in their spawn level and conditioning, so the
    descent runs over a stacked tensor: `steps` forward passes for the
    brood instead of `steps` per child. Each child draws its noise from
    its own generator, so the result matches the per-child reference
    exactly when the seeds match.
    """
    steps: int = 4

    def spawn_brood(self, parent: torch.Tensor, specs: Sequence[ChildSpec],
                    segment: Segment, model) -> BroodResult:
        from .base import renoise
        t_p, t_end = segment.t_parent, segment.t_end
        levels = [t_p + s.distance * (1.0 - t_p) for s in specs]
        states = [renoise(parent, t_p, lvl,
                          torch.randn(parent.shape,
                                      generator=_generator(spec, None),
                                      device=parent.device,
                                      dtype=parent.dtype))
                  for lvl, spec in zip(levels, specs)]
        x = torch.cat(states, 0)

        # Children start at different levels, so each row carries its own
        # time. The model sees one batch per step; the per-row schedule
        # keeps each child on the trajectory it would have had alone.
        cur = list(levels)
        passes = 0
        for i in range(self.steps):
            nxt = [lvl + (t_end - lvl) * ((i + 1) / self.steps)
                   for lvl in levels]
            v = model.velocity_batch(x, cur, [s.cond for s in specs], nxt,
                                     [s.guidance for s in specs])
            passes += 1
            dt = torch.tensor([n - c for n, c in zip(nxt, cur)],
                              device=x.device, dtype=x.dtype)
            x = x + dt.view(-1, *([1] * (x.dim() - 1))) * v
            cur = nxt
        return BroodResult(children=x, specs=list(specs),
                           method="batched_renoise", forward_passes=passes)


METHODS = {"per_child": PerChild, "batched_renoise": BatchedRenoise}
