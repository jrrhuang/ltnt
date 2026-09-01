"""GPU-free unit tests for LTNT pure backend logic.
Run: python tests/test_logic.py  (no GPU; imports torch on CPU). Extend over time."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flux_fmtt_dev import (
    compute_aggressive_diversity_schedule as agg,
    compute_linear_diversity_schedule as lin,
)

def _check_schedule(fn, num_steps, R, frac_cap):
    s = fn(num_steps, R)
    assert isinstance(s, list)
    assert len(s) == R, f"{fn.__name__}({num_steps},{R})->{s}: want {R} steps"
    assert s == sorted(s), f"{fn.__name__} not sorted: {s}"
    assert len(set(s)) == len(s), f"{fn.__name__} dup steps: {s}"
    assert all(1 <= x < num_steps for x in s), f"{fn.__name__} step OOB: {s}"
    assert max(s) <= num_steps * frac_cap + 1, f"{fn.__name__} not front-loaded: {s}"
    return s

def test_schedules_shape_and_bounds():
    for ns in (16, 20, 28):
        for R in (1, 2, 3, 4):
            _check_schedule(agg, ns, R, 0.25)
            _check_schedule(lin, ns, R, 0.40)

def test_zero_resampling_is_empty():
    assert agg(28, 0) == []
    assert lin(28, 0) == []

def test_aggressive_more_frontloaded_than_linear():
    for ns in (20, 28):
        for R in (2, 3, 4):
            assert max(agg(ns, R)) <= max(lin(ns, R)), f"agg>lin ns={ns} R={R}"

def test_exactly_R_even_when_crowded():
    s = agg(8, 4)
    assert len(s) == 4 and len(set(s)) == 4, f"crowded: {s}"

def test_incremental_layout_small_n_no_crash():
    # Regression for the n<4 UMAP crash guard (n_neighbors must be > 1).
    import numpy as np
    from flux_fmtt_dev import IncrementalLayout
    for n in (1, 2, 3):
        lay = IncrementalLayout()
        feats = np.random.RandomState(0).randn(n, 8).astype("float32")
        lay.initialize(feats, [None] * n)
        assert lay.coords.shape == (n, 2), f"n={n}: {lay.coords.shape}"
        assert np.isfinite(lay.coords).all(), f"n={n} non-finite"

def test_explore_tree_depth():
    # Regression for the create_map tree-structure indexing (get_particle_depth).
    from flux_explore_precompute import get_particle_depth as gpd
    for C in (2, 3):
        for R in (2, 3, 4):
            assert gpd(0, C, R) == 0, f"particle 0 must be a root (C={C},R={R})"
            assert gpd(1, C, R) == R, f"particle 1 must be a leaf (C={C},R={R})"
            for i in range(C ** R):
                d = gpd(i, C, R)
                assert 0 <= d <= R, f"depth {d} out of [0,{R}] at i={i} C={C}"

def test_explore_schedule():
    from flux_explore_precompute import compute_linear_diversity_schedule as lin2
    assert lin2(28, 0) == []
    for ns in (20, 28):
        for R in (2, 3, 4):
            sch = lin2(ns, R)
            assert len(sch) == R, f"explore schedule len {len(sch)} != {R}"
            assert sch == sorted(sch) and len(set(sch)) == len(sch), f"explore sched bad: {sch}"
            assert all(1 <= x < ns for x in sch), f"explore sched OOB: {sch}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
