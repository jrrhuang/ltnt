"""Spawn interfaces: how one child is produced from one parent."""
from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass
class Conditioning:
    """Text conditioning for a single child."""
    prompt: str
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor


@dataclass
class SpawnContext:
    """Everything a strategy needs besides the parent latent.

    t_parent and t_end are FLUX times: 1.0 is noise, 0.0 is clean.
    """
    model: "FlowModel"
    cond: Conditioning
    t_parent: float
    t_end: float
    guidance: float
    generator: Optional[torch.Generator] = None


class SpawnStrategy(Protocol):
    """Maps a parent latent at t_parent to a child latent at t_end."""

    def __call__(self, parent: torch.Tensor,
                 ctx: SpawnContext) -> torch.Tensor:
        ...


class FlowModel(Protocol):
    """The sampler capability a strategy depends on."""

    def velocity(self, x: torch.Tensor, t: float, cond: Conditioning,
                 t_next: float, guidance: float) -> torch.Tensor:
        ...


def renoise(x: torch.Tensor, t_from: float, t_to: float,
            noise: torch.Tensor) -> torch.Tensor:
    """Exact forward kernel between two noise levels, t_to >= t_from."""
    a_from, a_to = 1.0 - t_from, 1.0 - t_to
    ratio = a_to / max(a_from, 1e-8)
    var = max(t_to ** 2 - (ratio ** 2) * t_from ** 2, 0.0)
    return ratio * x + (var ** 0.5) * noise


def descend(x: torch.Tensor, t_from: float, t_to: float, steps: int,
            ctx: SpawnContext) -> torch.Tensor:
    """Integrate from t_from down to t_to in `steps` flow-map jumps."""
    t = t_from
    for i in range(steps):
        t_next = t_from + (t_to - t_from) * ((i + 1) / steps)
        v = ctx.model.velocity(x, t, ctx.cond, t_next, ctx.guidance)
        x = x + (t_next - t) * v
        t = t_next
    return x


def endpoint(x: torch.Tensor, t: float, ctx: SpawnContext) -> torch.Tensor:
    """Clean-image estimate from a latent at time t."""
    v = ctx.model.velocity(x, t, ctx.cond, 0.0, ctx.guidance)
    return x - t * v
