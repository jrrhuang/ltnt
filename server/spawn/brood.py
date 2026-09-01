"""Brood planning: which strategy and which conditioning each child gets.

Conditioning is orthogonal to the spawn strategy, so prompt variation
composes with any strategy without touching it.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .base import Conditioning


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
               augmented: int = 0) -> List[ChildPlan]:
    """Assign strategy and conditioning to each child.

    The first `augmented` children draw from the pool; the rest keep the
    base conditioning. augmented=0 gives a pure-renoise brood,
    augmented=n_children gives an all-variant brood.
    """
    plans = []
    for i in range(n_children):
        use_pool = pool is not None and i < augmented
        cond = pool.conditioning(i) if use_pool else base_cond
        plans.append(ChildPlan(index=i, strategy=strategy, cond=cond))
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
