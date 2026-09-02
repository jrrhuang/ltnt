"""Brood planning: which strategy and which conditioning each child gets.

Conditioning is orthogonal to the spawn strategy, so prompt variation
composes with any strategy without touching it.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .base import Conditioning
from .schedule import with_distance


@dataclass
class ReadingPool:
    """Alternate prompts and their encodings, assigned to children.

    encode maps a prompt to (prompt_embeds, pooled_prompt_embeds); it is
    called once per prompt and cached.
    """
    readings: Sequence[str]
    encode: Callable[[str], tuple]
    _cache: dict = field(default_factory=dict)

    def conditioning(self, index: int) -> Conditioning:
        prompt = self.readings[index % len(self.readings)]
        if prompt not in self._cache:
            self._cache[prompt] = self.encode(prompt)
        embeds, pooled = self._cache[prompt]
        return Conditioning(prompt, embeds, pooled)


@dataclass
class ChildPlan:
    index: int
    strategy: object
    cond: Optional[Conditioning]


def plan_brood(n_children: int, strategy, base_cond: Conditioning,
               pool: Optional[ReadingPool] = None,
               augmented: int = 0,
               distances: Optional[Sequence[float]] = None
               ) -> List[ChildPlan]:
    """Assign strategy, distance and conditioning to each child.

    The first `augmented` children draw from the pool; the rest keep the
    base conditioning. `distances` gives each child its own spawn
    distance, so one brood spans a range rather than repeating a point.
    """
    plans = []
    for i in range(n_children):
        use_pool = pool is not None and i < augmented
        cond = pool.conditioning(i) if use_pool else base_cond
        child = (with_distance(strategy, distances[i])
                 if distances else strategy)
        plans.append(ChildPlan(index=i, strategy=child, cond=cond))
    return plans


def mixed_brood(n_children: int, strategies: Sequence,
                base_cond: Conditioning,
                pool: Optional[ReadingPool] = None,
                augmented: int = 0) -> List[ChildPlan]:
    """Like plan_brood, cycling through several strategies."""
    plans = []
    for i in range(n_children):
        use_pool = pool is not None and i < augmented
        cond = pool.conditioning(i) if use_pool else base_cond
        plans.append(ChildPlan(index=i,
                               strategy=strategies[i % len(strategies)],
                               cond=cond))
    return plans
