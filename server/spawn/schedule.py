"""How far children travel from their parent, as a function of stage.

The first brood fills the space around the artist's selection. Each later
selection is a sharper statement of preference, so each later brood stays
closer to what was chosen. `Narrowing` is that curve; `ladder` spreads one
brood around it so a single round still offers a range.

Pure Python: no model, no tensors.
"""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Narrowing:
    """Spawn distance per stage, in canonical units.

    0 keeps the parent, 1 is an independent sample. Every strategy
    accepts this same quantity and translates it itself. `start` is stage 1,
    `end` is the last stage of `stages`; intermediate stages interpolate
    with `curve` as the exponent — above 1 holds width for longer and
    narrows late, below 1 narrows immediately.
    """
    start: float = 0.95
    end: float = 0.45
    stages: int = 4
    curve: float = 1.0
    spread: float = 0.15

    def distance(self, stage: int) -> float:
        """Centre distance for a stage, counted from 1."""
        if self.stages <= 1:
            return self.start
        step = min(max(stage, 1), self.stages) - 1
        frac = (step / (self.stages - 1)) ** self.curve
        return self.start + (self.end - self.start) * frac

    def ladder(self, stage: int, n_children: int) -> List[float]:
        """Per-child distances for one brood, spread around `rho(stage)`.

        A brood at a single distance is mutually redundant; laddering gives
        the artist a near variant and a far one in the same round. The
        spread shrinks with the centre, so a late brood stays tight.
        """
        if n_children <= 0:
            return []
        centre = self.distance(stage)
        half = self.spread * centre
        if n_children == 1:
            return [centre]
        # Shift the window into range rather than clipping it: a clipped
        # ladder at a centre near 1 is narrower than the stage after it,
        # which inverts the narrowing the schedule exists to express.
        lo, hi = centre - half, centre + half
        if hi > 1.0:
            lo, hi = lo - (hi - 1.0), 1.0
        if lo < 0.0:
            lo, hi = 0.0, hi - lo
        lo, hi = max(lo, 0.0), min(hi, 1.0)
        return [lo + (hi - lo) * i / (n_children - 1)
                for i in range(n_children)]


def with_distance(strategy, distance: float):
    """A copy of `strategy` at a given distance.

    Every strategy declares `distance`, so this sets one field and never
    inspects which native parameter a method happens to use.
    """
    from dataclasses import replace
    return replace(strategy, distance=distance)
