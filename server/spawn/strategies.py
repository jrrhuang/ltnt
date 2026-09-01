"""Spawn strategies. Each maps a parent latent to one child latent."""
import math
from dataclasses import dataclass

import torch

from .base import SpawnContext, descend, endpoint, renoise


@dataclass
class Renoise:
    """Renoise the parent state upward, then integrate down.

    rho is the fraction of the remaining distance to pure noise:
    t_spawn = t_parent + rho * (1 - t_parent). rho near 0 keeps the
    parent; rho near 1 approaches an independent sample.
    """
    rho: float = 0.95
    steps: int = 4

    def __call__(self, parent: torch.Tensor,
                 ctx: SpawnContext) -> torch.Tensor:
        t_spawn = ctx.t_parent + self.rho * (1.0 - ctx.t_parent)
        noise = torch.randn(parent.shape, generator=ctx.generator,
                            device=parent.device, dtype=parent.dtype)
        x = renoise(parent, ctx.t_parent, t_spawn, noise)
        return descend(x, t_spawn, ctx.t_end, self.steps, ctx)


@dataclass
class Lookahead:
    """Renoise the parent's clean-image estimate, then integrate down.

    Preserves composition more strongly than Renoise at equal noise
    level, since the estimate is re-noised rather than the latent.
    """
    tau: float = 0.75
    steps: int = 4

    def __call__(self, parent: torch.Tensor,
                 ctx: SpawnContext) -> torch.Tensor:
        x0 = endpoint(parent, ctx.t_parent, ctx)
        noise = torch.randn(parent.shape, generator=ctx.generator,
                            device=parent.device, dtype=parent.dtype)
        x = (1.0 - self.tau) * x0 + self.tau * noise
        return descend(x, self.tau, ctx.t_end, self.steps, ctx)


@dataclass
class GlassBridge:
    """GLASS transition kernel integrated over an auxiliary variable.

    The bridge target must carry noise: at t_end = 0 both the parent
    coupling and the bridge width vanish, so t_end is raised to
    sigma_floor and the remainder is integrated by `descend`.
    """
    rho: float = 0.4
    inner_steps: int = 8
    sigma_floor: float = 0.35
    steps: int = 4

    def __call__(self, parent: torch.Tensor,
                 ctx: SpawnContext) -> torch.Tensor:
        t_bridge = max(ctx.t_end, self.sigma_floor)
        gp = self._params(ctx.t_parent, t_bridge)
        noise = torch.randn(parent.shape, generator=ctx.generator,
                            device=parent.device, dtype=parent.dtype)
        x = gp["gamma"] * parent + gp["sigma_0"] * noise
        for i in range(self.inner_steps):
            x = self._step(x, parent, i / self.inner_steps,
                           1.0 / self.inner_steps, gp, ctx)
        return descend(x, t_bridge, ctx.t_end, self.steps, ctx)

    def _params(self, t_start: float, t_end: float) -> dict:
        gamma = self.rho * t_end / max(t_start, 1e-8)
        return dict(
            alpha_start=1.0 - t_start,
            sigma_start=t_start,
            gamma=gamma,
            alpha=(1.0 - t_end) - gamma * (1.0 - t_start),
            sigma=math.sqrt(max(t_end ** 2 * (1.0 - self.rho ** 2), 0.0)),
            sigma_0=1.0,
        )

    def _step(self, x, parent, s, ds, gp, ctx):
        clip = 1e-8
        alpha_s = s * gp["alpha"]
        sigma_s = (1.0 - s) * gp["sigma_0"] + s * gp["sigma"]
        w1 = (gp["sigma"] - gp["sigma_0"]) / max(sigma_s, clip)
        w2 = gp["alpha"] - alpha_s * w1
        w3 = -gp["gamma"] * w1

        mu1 = gp["alpha_start"]
        mu2 = alpha_s + gp["gamma"] * gp["alpha_start"]
        s11 = gp["sigma_start"] ** 2
        s12 = gp["gamma"] * s11
        s22 = sigma_s ** 2 + gp["gamma"] ** 2 * s11
        det = max(s11 * s22 - s12 * s12, clip)
        i11, i12, i22 = s22 / det, -s12 / det, s11 / det
        b = max(mu1 * mu1 * i11 + 2 * mu1 * mu2 * i12 + mu2 * mu2 * i22,
                clip)
        t_star = min(max(1.0 / (1.0 + math.sqrt(max(1.0 / b, 0.0))),
                         0.001), 0.999)
        sigma_star = 1.0 - t_star
        w_parent = t_star * (mu1 * i11 + mu2 * i12) / b
        w_x = t_star * (mu1 * i12 + mu2 * i22) / b

        probe = w_parent * parent + w_x * x
        # t_next must differ from t: a two-timestep flow map returns a
        # null jump when they are equal.
        v = ctx.model.velocity(probe, sigma_star, ctx.cond,
                               max(sigma_star - 1e-2, 0.0), ctx.guidance)
        clean = probe - sigma_star * v
        return x + ds * (w1 * x + w2 * clean + w3 * parent)


STRATEGIES = {
    "renoise": Renoise,
    "lookahead": Lookahead,
    "glass": GlassBridge,
}


def make(name: str, **params):
    """Build a strategy by name. Unknown keys raise TypeError."""
    if name not in STRATEGIES:
        raise KeyError(f"unknown spawn strategy {name!r}; "
                       f"have {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)
