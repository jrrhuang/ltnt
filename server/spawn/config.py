"""Spawn configuration: one record, serialisable, per session."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .strategies import make

# Measured diversity of a brood, 1 - mean pairwise DINO similarity, as a
# function of spawn noise level. Used to place boundaries so diversity
# falls linearly across stages.
DIVERSITY_CURVE = [(0.116, 0.5), (0.153, 0.7), (0.173, 0.9), (0.261, 1.0)]


@dataclass
class SpawnConfig:
    """Everything that decides how children are made.

    strategy names a registry entry; params are its constructor
    arguments. augmented is how many children of each brood draw from
    the reading pool.
    """
    strategy: str = "renoise"
    params: Dict[str, Any] = field(default_factory=lambda: {"rho": 0.95,
                                                            "steps": 4})
    augmented: int = 0
    diversity_high: float = 0.24
    diversity_low: float = 0.12

    def build(self):
        return make(self.strategy, **self.params)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_request(cls, req: Dict[str, Any]) -> "SpawnConfig":
        spawn = dict(req.get("spawn") or {})
        cfg = cls()
        if "strategy" in spawn:
            cfg.strategy = spawn["strategy"]
            cfg.params = dict(spawn.get("params") or {})
        elif req.get("clone_mode") == "glass":
            cfg.strategy = "glass"
            cfg.params = {"rho": float(req.get("rho") or 0.4)}
        else:
            cfg.strategy = "renoise"
            cfg.params = {"rho": float(req.get("rho") or 0.95), "steps": 4}
        cfg.augmented = int(spawn.get("augmented", 0))
        return cfg


def invert_diversity(target: float) -> float:
    """Noise level whose measured brood diversity is `target`."""
    pts = DIVERSITY_CURVE
    if target <= pts[0][0]:
        return pts[0][1]
    if target >= pts[-1][0]:
        return pts[-1][1]
    for (d0, t0), (d1, t1) in zip(pts, pts[1:]):
        if d0 <= target <= d1:
            return t0 + (t1 - t0) * (target - d0) / max(d1 - d0, 1e-6)
    return pts[-1][1]


def boundary_steps(sigmas: List[float], n_stages: int,
                   high: float = 0.24, low: float = 0.12) -> List[int]:
    """Schedule steps at which broods spawn.

    Stage i targets a linearly decreasing diversity, inverted through
    DIVERSITY_CURVE and snapped to the nearest schedule step. The curve
    is convex near noise, so the resulting steps are front-loaded.
    """
    if n_stages <= 0:
        return []
    out = []
    for i in range(n_stages):
        frac = i / max(n_stages - 1, 1) if n_stages > 1 else 0.0
        target = high - (high - low) * frac
        tau = invert_diversity(target)
        step = min(range(1, len(sigmas)),
                   key=lambda k: abs(sigmas[k] - tau))
        out.append(step)
    return sorted(set(out))
