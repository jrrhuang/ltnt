"""Radial, similarity-preserving hierarchical layout.

build_radial_layout(features, parents, depths, *, r0=0.12, decay=0.5, seed=42)
  features : (N, D) DINO feature matrix
  parents  : list[int|None] of length N; parents[i] is index of i's parent (None for roots)
  depths   : list[int]    of length N; depths[i] is i's depth (0 for roots)

Returns (N, 2) float array of (x, y) in [0, 1]^2.

Algorithm
---------
1. Root positions: UMAP on root features, rescaled so min pairwise root
   distance >= 3 * r0. Recentred into [0,1]^2.
2. For each subtree: children of parent p are placed around p at radius
   r_d = r0 * decay^d. The angular arrangement comes from the top-2
   PCA components of the siblings' feature residuals (feature minus
   sibling centroid). Siblings whose features are alike land near each
   other; outliers sit opposite. Scaled so max sibling offset == r_d.
3. If C < 3 or sibling features collapse, fall back to equiangular
   placement on the ring.

Why decay=0.5: for C>=3, min angular gap between siblings is 2pi/C,
so sibling-to-sibling distance is 2*r*sin(pi/C) >= r (for C=3, ~1.73r).
With child radius r_d = r_{d-1}/2, a child's subtree fits inside half
the spacing between its parent and the parent's siblings => no overlap.
"""
from __future__ import annotations

import numpy as np


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Project rows of X onto its top-2 PCA components. X is (n, D)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD of centered data; columns of Vt are principal directions
    try:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((X.shape[0], 2), dtype=np.float32)
    k = min(2, Vt.shape[0])
    proj = Xc @ Vt[:k].T  # (n, k)
    if k < 2:
        pad = np.zeros((proj.shape[0], 2 - k), dtype=proj.dtype)
        proj = np.concatenate([proj, pad], axis=1)
    return proj.astype(np.float32)


def _root_layout(root_feats: np.ndarray, r0: float, seed: int) -> np.ndarray:
    """UMAP roots to 2D, rescale so min pairwise distance >= 3*r0,
    centre in [0,1]^2. Returns (n_roots, 2)."""
    n = root_feats.shape[0]
    if n == 1:
        return np.array([[0.5, 0.5]], dtype=np.float32)

    if n <= 3:
        coords = _pca_2d(root_feats)
    else:
        try:
            import umap
            reducer = umap.UMAP(
                n_neighbors=min(15, n - 1),
                min_dist=0.3,
                metric="cosine",
                random_state=seed,
            )
            coords = reducer.fit_transform(root_feats).astype(np.float32)
        except Exception:
            coords = _pca_2d(root_feats)

    # Compute current min pairwise distance; scale so min == 3*r0
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_d = float(dists.min())
    if min_d < 1e-6:
        # Degenerate: spread roots on a circle
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        coords = np.stack([np.cos(angles), np.sin(angles)], axis=1) * (3 * r0)
    else:
        coords = coords * (3 * r0 / min_d)

    # Centre into [0,1]^2 with margin wide enough to fit the full subtree
    # reach around each root: r0 * (1 + decay + decay^2 + ...) = r0/(1-decay).
    # With r0=0.12, decay=0.5 that's 0.24; pad a little.
    c_min = coords.min(axis=0)
    c_max = coords.max(axis=0)
    span = np.maximum(c_max - c_min, 1e-6)
    margin = 0.26
    usable = 1.0 - 2 * margin
    # Preserve aspect ratio: scale by max span dim
    scale = usable / max(span)
    coords = (coords - c_min) * scale
    # Centre leftover space
    used = span * scale
    coords += margin + (usable - used) / 2.0
    return coords.astype(np.float32)


def _sibling_offsets(sib_feats: np.ndarray, radius: float,
                     radius_floor_frac: float = 0.55) -> np.ndarray:
    """Place siblings around (0,0) near `radius`.

    Directions come from the top-2 PCA of sibling-feature residuals, so
    similar siblings share angular neighborhood and outliers sit opposite.
    Each sibling is placed at a radius in `[radius_floor_frac * radius,
    radius]`, linearly interpolated by its PCA norm. This avoids the
    "close siblings collapse to origin" visual failure mode while still
    encoding magnitude of dissimilarity in radial distance.

    Degenerate cases (C=1, C=2 with collinear features, all siblings at
    origin after PCA) fall back to an evenly-spaced ring.
    """
    C = sib_feats.shape[0]
    if C == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if C == 1:
        return np.array([[radius, 0.0]], dtype=np.float32)

    proj = _pca_2d(sib_feats)
    norms = np.linalg.norm(proj, axis=1)

    if norms.max() < 1e-6:
        angles = np.linspace(0, 2 * np.pi, C, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32) * radius

    # Unit directions from PCA
    unit = proj / np.maximum(norms[:, None], 1e-12)

    # For C=2, PCA collapses to a line: sibling 1 along +u, sibling 2 along -u.
    # That's still a valid 2D direction (better than forcing horizontal).
    # But if the two happen to project to the same direction (rare), spread them.
    if C == 2 and np.dot(unit[0], unit[1]) > 0.99:
        unit[1] = -unit[0]

    # Radial distances: compress range into [floor, radius]
    min_n, max_n = norms.min(), norms.max()
    floor_r = radius_floor_frac * radius
    if max_n - min_n < 1e-6:
        scaled_r = np.full_like(norms, radius)
    else:
        scaled_r = floor_r + (norms - min_n) * (radius - floor_r) / (max_n - min_n)
    out = unit * scaled_r[:, None]

    # Nudge any two siblings that still coincide (e.g. identical PCA direction)
    for i in range(C):
        for j in range(i + 1, C):
            if np.linalg.norm(out[i] - out[j]) < 0.05 * radius:
                theta = 0.20 * (j - i)
                rot = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
                out[j] = out[j] @ rot

    return out.astype(np.float32)


def build_radial_layout(
    features: np.ndarray,
    parents: list,
    depths: list,
    *,
    r0: float = 0.12,
    decay: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    N = features.shape[0]
    coords = np.zeros((N, 2), dtype=np.float32)

    # Roots
    root_ids = [i for i in range(N) if parents[i] is None]
    root_feats = features[root_ids] if root_ids else features[:1]
    root_xy = _root_layout(root_feats, r0=r0, seed=seed)
    for k, i in enumerate(root_ids):
        coords[i] = root_xy[k]

    # Group children by parent
    children_of: dict[int, list[int]] = {}
    for i in range(N):
        p = parents[i]
        if p is None:
            continue
        children_of.setdefault(p, []).append(i)

    # BFS by depth so parents are placed before children
    max_depth = max(depths) if depths else 0
    for d in range(1, max_depth + 1):
        radius = r0 * (decay ** (d - 1))  # radius around depth-(d-1) parent
        # Find every parent at depth d-1 that has children
        for p_idx, kids in children_of.items():
            if depths[p_idx] != d - 1:
                continue
            sib_feats = features[kids]
            offs = _sibling_offsets(sib_feats, radius=radius)
            for k, c in enumerate(kids):
                coords[c] = coords[p_idx] + offs[k]

    # Gentle clip — in practice margins above keep everything well inside
    coords = np.clip(coords, 0.005, 0.995)
    return coords
