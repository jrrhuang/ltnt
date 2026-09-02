"""Child-spawning: strategies, brood planning, and conditioning.

    from spawn import make, plan_brood, SpawnContext

    strategy = make("renoise", rho=0.95, steps=4)
    plans = plan_brood(7, strategy, base_cond, pool=readings, augmented=3)
    for p in plans:
        ctx = SpawnContext(model, p.cond, t_parent, t_end, guidance)
        child = p.strategy(parent, ctx)
"""
from .base import (Conditioning, FlowModel, SpawnContext, SpawnStrategy,
                   descend, endpoint, renoise)
from .brood import ChildPlan, ReadingPool, mixed_brood, plan_brood
from .schedule import Narrowing, with_distance
from .strategies import STRATEGIES, GlassBridge, Lookahead, Renoise, make

__all__ = [
    "Conditioning", "FlowModel", "SpawnContext", "SpawnStrategy",
    "renoise", "descend", "endpoint",
    "Renoise", "Lookahead", "GlassBridge", "STRATEGIES", "make",
    "ChildPlan", "ReadingPool", "plan_brood", "mixed_brood",
    "Narrowing", "with_distance",
]
