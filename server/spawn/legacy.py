"""The spawn mathematics production runs, moved behind the brood seam.

This is a transcription, not a redesign. It reproduces the sampler's
clone path exactly — the same renoise, the same chained descent, the same
per-child noise draw — so the two can be compared on real output before
the inline copy is removed. Improvements belong in the methods in
`strategies.py`, not here.
"""
import os
from dataclasses import dataclass
from typing import Sequence

import torch

from .base import renoise
from .method import BroodResult, ChildSpec, Segment


def clone_steps() -> int:
    """How many chained jumps the descent uses."""
    return max(1, int(os.environ.get("FLUXFM_CLONE_STEPS", "4")))


def spawn_level(t_seg_start: float, distance: float) -> float:
    """The renoise level for a child, in the sampler's live convention.

    Larger distance renoises further toward pure noise. The sampler also
    contains an inverted helper (`_rho_to_t_RN`, where 1 means no
    renoise) used by a different branch; this is the one the per-child
    ladder runs.
    """
    return t_seg_start + distance * (1.0 - t_seg_start)


@dataclass
class LegacyWDM:
    """Renoise the parent to the child's level, then chain jumps down.

    One child per model call, matching the sampler. `BatchedRenoise` in
    `method.py` computes the same thing in one call per step.
    """
    steps: int = 0          # 0 means read FLUXFM_CLONE_STEPS

    def spawn_brood(self, parent: torch.Tensor, specs: Sequence[ChildSpec],
                    segment: Segment, model) -> BroodResult:
        n_steps = self.steps or clone_steps()
        t_start, t_end = segment.t_parent, segment.t_end
        out, passes = [], 0
        for spec in specs:
            t_rn = spawn_level(t_start, spec.distance)
            gen = (torch.Generator(device="cpu").manual_seed(spec.seed)
                   if spec.seed is not None else None)
            eps = torch.randn(parent.shape, generator=gen,
                              dtype=parent.dtype).to(parent.device) \
                if gen is not None else torch.randn_like(parent)
            x = renoise(parent, t_start, t_rn, eps)
            t = t_rn
            for i in range(n_steps):
                t_next = t_rn + (t_end - t_rn) * ((i + 1) / n_steps)
                v = model.velocity(x, t, spec.cond, t_next, spec.guidance)
                passes += 1
                x = x + (t_next - t) * v
                t = t_next
            out.append(x)
        return BroodResult(children=torch.cat(out, 0), specs=list(specs),
                           method="legacy_wdm", forward_passes=passes)


def descend_chain(x, t_from: float, t_to: float, steps: int, velocity):
    """Chain `steps` flow-map jumps from t_from down to t_to.

    `velocity(x, t, t_next)` returns the map's output for one jump. This
    is the sampler's descent, extracted so both the running app and the
    brood methods compute it in one place.
    """
    t = t_from
    for i in range(steps):
        t_next = t_from + (t_to - t_from) * ((i + 1) / steps)
        x = x + (t_next - t) * velocity(x, t, t_next)
        t = t_next
    return x


def child_seed(master: int, stage: int, index: int) -> int:
    """A stable seed for one child.

    Derived from the master seed and the child's position rather than
    drawn from ambient global state, so a brood replays identically
    regardless of what else the process did first, and regardless of the
    order the children happen to be computed in.
    """
    h = (int(master) & 0xFFFFFFFF) * 0x9E3779B1
    h ^= (int(stage) + 1) * 0x85EBCA77
    h ^= (int(index) + 1) * 0xC2B2AE3D
    return h & 0x7FFFFFFF


def child_noise(like: torch.Tensor, master, stage: int, index: int):
    """Noise for one child: seeded when a master seed exists, else free."""
    if master is None:
        return torch.randn_like(like)
    gen = torch.Generator(device="cpu").manual_seed(
        child_seed(master, stage, index))
    return torch.randn(like.shape, generator=gen,
                       dtype=torch.float32).to(like.device, like.dtype)
