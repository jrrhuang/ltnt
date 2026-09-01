"""Spawn tests. CPU only, no model weights."""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spawn import (Conditioning, SpawnContext, descend, endpoint,  # noqa: E402
                   make, mixed_brood, plan_brood, renoise)
from spawn.brood import ReadingPool  # noqa: E402
from spawn.config import (SpawnConfig, boundary_steps,  # noqa: E402
                          invert_diversity)


@dataclass
class LinearModel:
    """Velocity field of the exact interpolant toward a fixed target.

    For x_t = (1-t) target + t noise, the flow-matching velocity is
    noise - target = (x_t - target) / t.
    """
    target: torch.Tensor
    calls: list

    def velocity(self, x, t, cond, t_next, guidance):
        self.calls.append((round(float(t), 6), round(float(t_next), 6)))
        return (x - self.target) / max(t, 1e-6)


def make_ctx(target=None, t_parent=0.9, t_end=0.0):
    target = torch.zeros(1, 4) if target is None else target
    model = LinearModel(target=target, calls=[])
    cond = Conditioning("p", torch.zeros(1, 2), torch.zeros(1, 2))
    return SpawnContext(model=model, cond=cond, t_parent=t_parent,
                        t_end=t_end, guidance=3.5), model


def test_renoise_matches_forward_kernel_moments():
    torch.manual_seed(0)
    x0 = torch.randn(4096, 8)
    t_from, t_to = 0.4, 0.9
    x_from = (1 - t_from) * x0 + t_from * torch.randn_like(x0)
    out = renoise(x_from, t_from, t_to, torch.randn_like(x_from))
    direct = (1 - t_to) * x0 + t_to * torch.randn_like(x0)
    assert out.std().item() == pytest.approx(direct.std().item(), rel=0.05)


def test_renoise_is_identity_at_equal_levels():
    x = torch.randn(2, 4)
    out = renoise(x, 0.7, 0.7, torch.randn_like(x))
    assert torch.allclose(out, x, atol=1e-5)


def test_descend_reaches_target_time_in_n_steps():
    ctx, model = make_ctx()
    x = torch.randn(1, 4)
    descend(x, 0.9, 0.0, 4, ctx)
    assert len(model.calls) == 4
    assert model.calls[0][0] == pytest.approx(0.9)
    assert model.calls[-1][1] == pytest.approx(0.0)


def test_descend_never_calls_with_null_jump():
    ctx, model = make_ctx()
    descend(torch.randn(1, 4), 0.98, 0.0, 6, ctx)
    assert all(t != t_next for t, t_next in model.calls)


def test_descend_converges_to_target():
    target = torch.ones(1, 4) * 3.0
    ctx, _ = make_ctx(target=target)
    x = (1 - 0.9) * target + 0.9 * torch.randn(1, 4)
    out = descend(x, 0.9, 0.0, 8, ctx)
    assert torch.allclose(out, target, atol=1e-3)


def test_endpoint_recovers_clean_image():
    target = torch.ones(1, 4) * -2.0
    ctx, _ = make_ctx(target=target)
    t = 0.6
    x = (1 - t) * target + t * torch.randn(1, 4)
    assert torch.allclose(endpoint(x, t, ctx), target, atol=1e-4)


@pytest.mark.parametrize("rho,expected", [(0.0, 0.9), (0.5, 0.95),
                                          (1.0, 1.0)])
def test_renoise_strategy_spawn_level_follows_rho(rho, expected):
    ctx, model = make_ctx(t_parent=0.9)
    make("renoise", rho=rho, steps=2)(torch.randn(1, 4), ctx)
    assert model.calls[0][0] == pytest.approx(expected, abs=1e-6)


def test_lookahead_spawns_at_tau_regardless_of_parent_time():
    ctx, model = make_ctx(t_parent=0.98)
    make("lookahead", tau=0.7, steps=2)(torch.randn(1, 4), ctx)
    assert model.calls[0] == (0.98, 0.0)      # endpoint estimate first
    assert model.calls[1][0] == pytest.approx(0.7)


def test_glass_bridge_never_targets_zero_noise():
    ctx, model = make_ctx(t_parent=0.98, t_end=0.0)
    strategy = make("glass", rho=0.4, inner_steps=3, sigma_floor=0.35,
                    steps=2)
    strategy(torch.randn(1, 4), ctx)
    params = strategy._params(0.98, max(0.0, strategy.sigma_floor))
    assert params["gamma"] > 0.0
    assert params["sigma"] > 0.0


def test_glass_step_counts():
    ctx, model = make_ctx()
    make("glass", inner_steps=5, steps=3)(torch.randn(1, 4), ctx)
    assert len(model.calls) == 5 + 3


def test_registry_rejects_unknown_strategy():
    with pytest.raises(KeyError):
        make("nope")


def test_plan_brood_assigns_pool_then_base():
    base = Conditioning("base", torch.zeros(1, 2), torch.zeros(1, 2))
    pool = ReadingPool(readings=["a", "b"],
                       encode=lambda p: (torch.zeros(1, 2),
                                         torch.zeros(1, 2)))
    plans = plan_brood(5, make("renoise"), base, pool=pool, augmented=2)
    assert [p.cond.prompt for p in plans] == ["a", "b", "base", "base",
                                              "base"]


def test_plan_brood_without_pool_is_all_base():
    base = Conditioning("base", torch.zeros(1, 2), torch.zeros(1, 2))
    plans = plan_brood(3, make("renoise"), base)
    assert all(p.cond is base for p in plans)


def test_mixed_brood_cycles_strategies():
    base = Conditioning("base", torch.zeros(1, 2), torch.zeros(1, 2))
    a, b = make("renoise"), make("lookahead")
    plans = mixed_brood(4, [a, b], base)
    assert [p.strategy for p in plans] == [a, b, a, b]


def test_reading_pool_encodes_each_prompt_once():
    seen = []

    def encode(p):
        seen.append(p)
        return torch.zeros(1, 2), torch.zeros(1, 2)

    pool = ReadingPool(readings=["x", "y"], encode=encode)
    for i in range(6):
        pool.conditioning(i)
    assert seen == ["x", "y"]


def test_config_from_legacy_request():
    cfg = SpawnConfig.from_request({"clone_mode": "glass", "rho": 0.4})
    assert (cfg.strategy, cfg.params["rho"]) == ("glass", 0.4)
    cfg = SpawnConfig.from_request({"clone_mode": "flow_map", "rho": 0.9})
    assert (cfg.strategy, cfg.params["rho"]) == ("renoise", 0.9)


def test_config_roundtrip_and_build():
    cfg = SpawnConfig(strategy="lookahead", params={"tau": 0.6, "steps": 3})
    assert cfg.to_dict()["params"]["tau"] == 0.6
    assert cfg.build().tau == 0.6


def test_invert_diversity_is_monotone_and_clamped():
    assert invert_diversity(0.05) == 0.5
    assert invert_diversity(0.9) == 1.0
    assert invert_diversity(0.15) < invert_diversity(0.20)


def test_boundaries_are_front_loaded():
    sigmas = [1.0 - i / 28 for i in range(28)]
    steps = boundary_steps(sigmas, n_stages=3)
    assert steps == sorted(steps)
    assert steps[0] < 6
    assert steps[-1] > steps[0]
    assert max(steps) < len(sigmas)
