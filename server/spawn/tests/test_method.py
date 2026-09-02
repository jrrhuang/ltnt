"""Brood-seam tests. CPU only, no model weights."""
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spawn import Conditioning, make  # noqa: E402
from spawn.method import (BatchedRenoise, BroodResult, ChildSpec,  # noqa: E402
                          PerChild, Segment)


@dataclass
class ToyModel:
    """A velocity field that depends on the state, in batch or alone.

    `velocity_batch` must agree row-for-row with `velocity`, which is what
    lets the batched and per-child paths be compared exactly.
    """
    scale: float = 0.3
    batch_calls: int = 0
    single_calls: int = 0

    def velocity(self, x, t, cond, t_next, guidance):
        self.single_calls += 1
        return self.scale * torch.tanh(x) * float(t)

    def velocity_batch(self, x, ts, conds, t_nexts, guidances):
        self.batch_calls += 1
        t = torch.tensor(ts, device=x.device, dtype=x.dtype)
        return self.scale * torch.tanh(x) * t.view(-1, *([1] * (x.dim() - 1)))


def specs(n, distances=None, seeds=None):
    cond = Conditioning("p", torch.zeros(1, 2), torch.zeros(1, 2))
    distances = distances or [0.4 + 0.1 * i for i in range(n)]
    seeds = seeds if seeds is not None else list(range(n))
    return [ChildSpec(index=i, distance=distances[i], cond=cond,
                      guidance=3.5, seed=seeds[i]) for i in range(n)]


def test_brood_returns_one_row_per_spec():
    model = ToyModel()
    out = PerChild(make("renoise", steps=3)).spawn_brood(
        torch.ones(1, 16), specs(5), Segment(0.7, 0.0), model)
    assert isinstance(out, BroodResult)
    assert out.children.shape[0] == 5


def test_batched_matches_per_child_exactly():
    """The batched path is an optimisation, not a different method."""
    parent, seg, sp = torch.ones(1, 16) * 0.5, Segment(0.7, 0.0), specs(6)
    ref = PerChild(make("renoise", steps=4)).spawn_brood(
        parent, sp, seg, ToyModel())
    fast = BatchedRenoise(steps=4).spawn_brood(parent, sp, seg, ToyModel())
    assert torch.allclose(ref.children, fast.children, atol=1e-5), \
        (ref.children - fast.children).abs().max().item()


def test_batching_collapses_the_forward_passes():
    parent, seg, sp = torch.ones(1, 16), Segment(0.7, 0.0), specs(7)
    ref = PerChild(make("renoise", steps=4)).spawn_brood(
        parent, sp, seg, ToyModel())
    fast = BatchedRenoise(steps=4).spawn_brood(parent, sp, seg, ToyModel())
    assert ref.forward_passes == 28          # 7 children x 4 steps
    assert fast.forward_passes == 4          # one batch per step
    assert fast.forward_passes < ref.forward_passes / 5


def test_seeds_make_a_brood_reproducible():
    parent, seg = torch.ones(1, 16), Segment(0.7, 0.0)
    a = BatchedRenoise().spawn_brood(parent, specs(4), seg, ToyModel())
    b = BatchedRenoise().spawn_brood(parent, specs(4), seg, ToyModel())
    assert torch.allclose(a.children, b.children)


def test_different_seeds_give_different_children():
    parent, seg = torch.ones(1, 16), Segment(0.7, 0.0)
    a = BatchedRenoise().spawn_brood(
        parent, specs(3, seeds=[1, 2, 3]), seg, ToyModel())
    b = BatchedRenoise().spawn_brood(
        parent, specs(3, seeds=[4, 5, 6]), seg, ToyModel())
    assert not torch.allclose(a.children, b.children)


def test_per_child_distance_still_governs_parent_retention():
    parent, seg, model = torch.ones(1, 64) * 2.0, Segment(0.7, 0.0), ToyModel()
    out = BatchedRenoise(steps=3).spawn_brood(
        parent, specs(2, distances=[0.05, 0.95], seeds=[7, 7]), seg, model)
    near = float(out.children[0:1] @ parent.T / (parent @ parent.T))
    far = float(out.children[1:2] @ parent.T / (parent @ parent.T))
    assert near > far


def test_a_joint_method_is_expressible_at_this_seam():
    """Sibling repulsion needs the other children; the seam allows it."""

    @dataclass
    class Repelling:
        steps: int = 2
        strength: float = 0.5

        def spawn_brood(self, parent, sp, segment, model):
            base = BatchedRenoise(steps=self.steps).spawn_brood(
                parent, sp, segment, model)
            x = base.children
            centre = x.mean(dim=0, keepdim=True)
            return BroodResult(children=x + self.strength * (x - centre),
                               specs=base.specs, method="repelling",
                               forward_passes=base.forward_passes)

    parent, seg, sp = torch.ones(1, 32), Segment(0.7, 0.0), specs(5)
    plain = BatchedRenoise(steps=2).spawn_brood(parent, sp, seg, ToyModel())
    repelled = Repelling().spawn_brood(parent, sp, seg, ToyModel())
    spread = lambda t: float(t.std(dim=0).mean())  # noqa: E731
    assert spread(repelled.children) > spread(plain.children)
    assert repelled.forward_passes == plain.forward_passes


def test_batched_result_records_what_made_it():
    out = BatchedRenoise().spawn_brood(
        torch.ones(1, 8), specs(3), Segment(0.6, 0.0), ToyModel())
    assert out.method == "batched_renoise"
    assert [s.index for s in out.specs] == [0, 1, 2]


@pytest.mark.parametrize("n", [1, 2, 8])
def test_brood_sizes(n):
    out = BatchedRenoise(steps=2).spawn_brood(
        torch.ones(1, 8), specs(n), Segment(0.7, 0.0), ToyModel())
    assert out.children.shape[0] == n


# ---- the transcription must match the batched method exactly ----

def test_legacy_wdm_matches_batched_renoise():
    """LegacyWDM is production's arithmetic; BatchedRenoise is the fast
    path. Same seeds must give the same children, or the batched path is
    not an optimisation but a change of method."""
    from spawn.legacy import LegacyWDM
    parent, seg, sp = torch.ones(1, 16) * 0.5, Segment(0.7, 0.0), specs(5)
    slow = LegacyWDM(steps=4).spawn_brood(parent, sp, seg, ToyModel())
    fast = BatchedRenoise(steps=4).spawn_brood(parent, sp, seg, ToyModel())
    assert torch.allclose(slow.children, fast.children, atol=1e-5), \
        (slow.children - fast.children).abs().max().item()
    assert slow.forward_passes == 20 and fast.forward_passes == 4


def test_legacy_spawn_level_follows_the_live_convention():
    from spawn.legacy import spawn_level
    assert spawn_level(0.9, 0.0) == pytest.approx(0.9)
    assert spawn_level(0.9, 1.0) == pytest.approx(1.0)
    assert spawn_level(0.9, 0.5) == pytest.approx(0.95)


def test_legacy_clone_steps_reads_the_environment(monkeypatch):
    from spawn import legacy
    monkeypatch.setenv("FLUXFM_CLONE_STEPS", "7")
    assert legacy.clone_steps() == 7
    monkeypatch.delenv("FLUXFM_CLONE_STEPS")
    assert legacy.clone_steps() == 4


def test_descend_chain_matches_a_hand_rolled_loop():
    from spawn.legacy import descend_chain
    model, x0 = ToyModel(), torch.ones(1, 12) * 0.4
    got = descend_chain(x0, 0.9, 0.0, 4,
                        lambda x, t, tn: model.velocity(x, t, None, tn, 3.5))
    x, t = x0, 0.9
    for i in range(4):
        tn = 0.9 + (0.0 - 0.9) * ((i + 1) / 4)
        x = x + (tn - t) * model.velocity(x, t, None, tn, 3.5)
        t = tn
    assert torch.allclose(got, x, atol=1e-7)


def test_child_seed_is_stable_and_position_dependent():
    from spawn.legacy import child_seed
    assert child_seed(31337, 1, 0) == child_seed(31337, 1, 0)
    assert child_seed(31337, 1, 0) != child_seed(31337, 1, 1)
    assert child_seed(31337, 1, 0) != child_seed(31337, 2, 0)
    assert child_seed(31337, 1, 0) != child_seed(31338, 1, 0)
    assert 0 <= child_seed(31337, 3, 5) < 2 ** 31


def test_child_noise_replays_from_the_master_seed():
    from spawn.legacy import child_noise
    like = torch.zeros(1, 32)
    a = child_noise(like, 31337, 1, 2)
    b = child_noise(like, 31337, 1, 2)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, child_noise(like, 31337, 1, 3))


def test_child_noise_without_a_master_seed_is_free():
    from spawn.legacy import child_noise
    like = torch.zeros(1, 64)
    assert not torch.allclose(child_noise(like, None, 1, 0),
                              child_noise(like, None, 1, 0))
