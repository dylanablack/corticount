#!/usr/bin/env python3
"""
solve_phi_psi.py
===================
Solve Laplace's equation on a 2D cortical ribbon to obtain:
  φ   — cortical depth (0 at ventricular surface, 1 at pial surface)
  sv/psi  — ventricular arclength (painted along columns = streamlines of ∇φ)

Boundary conditions:
  ventricular  (code 1):  Dirichlet  φ = 0
  pial         (code 2):  Dirichlet  φ = 1
  left cut     (code 3):  Neumann    ∂φ/∂n = 0
  right cut    (code 4):  Neumann    ∂φ/∂n = 0

Inputs:
  --mask              cortex_mask.tif (binary, 255 = cortex)
  --boundary_labels   boundary_labels.tif (uint8: 0 bg, 1-4 on mask edge)

Outputs:
  {prefix}_phi.tif    cortical depth  (uint16, 0-65535 mapped to 0-1)
  {prefix}_sv.tif     ventricular arclength (uint16, 0-65535 mapped to 0-1)
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import linalg as splinalg
from skimage import io
from scipy.ndimage import binary_erosion

# -------- Constants --------

CODES = {1: "ventricular", 2: "pial", 3: "left", 4: "right"}
DIRICHLET_CODES = {1, 2}     # ventricular=0, pial=1
NEUMANN_CODES = {3, 4}       # zero-flux

NBRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# -------- Data --------

@dataclass(frozen=True)
class BoundarySets:
    ventricular: NDArray[np.bool_]   # Dirichlet φ=0
    pial: NDArray[np.bool_]          # Dirichlet φ=1
    left: NDArray[np.bool_]          # Neumann
    right: NDArray[np.bool_]         # Neumann
    cortex_mask: NDArray[np.bool_]   # full interior


# -------- I/O helpers --------

def save_u16(path: str, arr: np.ndarray, mask: np.ndarray | None = None) -> None:
    """Save a [0,1] float array as uint16 TIFF."""
    a = np.asarray(arr, dtype=np.float32)
    if mask is not None:
        a = np.where(mask, a, 0.0)
    a = np.where(np.isfinite(a), a, 0.0)
    a = np.clip(a, 0.0, 1.0)
    io.imsave(path, (a * 65535).astype(np.uint16), check_contrast=False)


def load_boundary_sets(mask_path: str, labels_path: str) -> BoundarySets:
    """
    Load mask and boundary labels directly.

    The boundary_labels image (from the annotator) already has every mask-edge
    pixel labelled 1–4.  We just read it and split into boolean arrays.
    """
    mask = io.imread(mask_path) > 0
    labels = io.imread(labels_path).astype(np.uint8)

    vent = (labels == 1) & mask
    pial = (labels == 2) & mask
    left = (labels == 3) & mask
    right = (labels == 4) & mask

    boundary_union = vent | pial | left | right

    # Check every mask-edge pixel has a label
    interior = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    mask_edge = mask & ~interior
    unlabelled = mask_edge & ~boundary_union
    n_unlabelled = int(np.sum(unlabelled))
    if n_unlabelled > 0:
        print(f"[WARN] {n_unlabelled} mask-edge pixels have no boundary label. "
              f"They will be treated as Neumann.")

    # Report
    for code, name in CODES.items():
        n = int(np.sum(labels == code))
        print(f"  {name:12s}: {n:6d} boundary pixels")

    return BoundarySets(vent, pial, left, right, mask)


# -------- Laplacian solve --------

def solve_phi(bs: BoundarySets,
              rtol: float = 1e-6,
              maxiter: int = 50_000) -> NDArray[np.float32]:
    """
    Solve ∇²φ = 0 inside the cortex mask.

    Dirichlet:  φ = 0 on ventricular,  φ = 1 on pial.
    Neumann:    ∂φ/∂n = 0 on left, right, and any unlabelled mask-edge pixels.

    The Neumann condition is implemented by simply excluding exterior neighbours
    from the stencil: each interior/Neumann pixel's value equals the average of
    its within-mask neighbours only.
    """

    H, W = bs.cortex_mask.shape
    mask = bs.cortex_mask

    # Dirichlet pixels: fixed values, not unknowns
    dirichlet = bs.ventricular | bs.pial
    dirichlet_val = np.zeros((H, W), dtype=np.float32)
    dirichlet_val[bs.pial] = 1.0

    # Unknown pixels: everything in mask that isn't Dirichlet
    unknown = mask & ~dirichlet
    N = int(np.sum(unknown))
    print(f"  Solving Laplacian: {N} unknowns, "
          f"{int(np.sum(dirichlet))} Dirichlet pixels")

    # Map unknown pixels to linear indices
    idx = np.full((H, W), -1, dtype=np.int64)
    idx[unknown] = np.arange(N)

    # Assemble sparse system
    rows, cols, vals = [], [], []
    rhs = np.zeros(N, dtype=np.float64)

    ys, xs = np.nonzero(unknown)
    for k, (y, x) in enumerate(zip(ys, xs)):
        n_interior = 0
        for dy, dx in NBRS4:
            ny, nx = y + dy, x + dx

            # neighbour outside image or outside mask: skip entirely.
            # This is the zero-flux (Neumann) condition: the face doesn't
            # participate in the averaging, so no phantom value is introduced.
            if ny < 0 or ny >= H or nx < 0 or nx >= W or not mask[ny, nx]:
                continue

            if dirichlet[ny, nx]:
                # Dirichlet neighbour: known value goes to RHS
                rhs[k] += float(dirichlet_val[ny, nx])
            else:
                # Unknown neighbour: coefficient in matrix
                rows.append(k)
                cols.append(int(idx[ny, nx]))
                vals.append(-1.0)

            n_interior += 1

        # Diagonal = number of within-mask neighbours
        rows.append(k)
        cols.append(k)
        vals.append(float(n_interior))

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

    # Solve with conjugate gradients
    try:
        u, info = splinalg.cg(A, rhs, rtol=rtol, atol=0.0, maxiter=maxiter)
    except TypeError:
        # Older scipy without rtol/atol keyword
        u, info = splinalg.cg(A, rhs, tol=rtol, maxiter=maxiter)

    if info != 0:
        print(f"  [WARN] CG did not converge (info={info})")

    phi = np.zeros((H, W), dtype=np.float32)
    phi[dirichlet] = dirichlet_val[dirichlet]
    phi[unknown] = u.astype(np.float32)
    phi[~mask] = 0.0
    return np.clip(phi, 0.0, 1.0)


# -------- Ventricular arclength --------

def _order_boundary_pixels(bw: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """
    Order the pixels in a thin binary curve by greedy 8-connected walk.
    Starts from an endpoint (degree-1 pixel) if one exists.
    """
    ys, xs = np.nonzero(bw)
    if len(ys) == 0:
        return []

    remaining = set(zip(ys.tolist(), xs.tolist()))
    nbr8 = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

    # Find degree-1 pixels (endpoints)
    def degree(p):
        return sum((p[0] + dy, p[1] + dx) in remaining for dy, dx in nbr8)

    endpoints = [p for p in remaining if degree(p) == 1]
    start = endpoints[0] if endpoints else next(iter(remaining))

    path = [start]
    remaining.remove(start)

    while remaining:
        cur = path[-1]
        # Try 8-connected neighbours
        found = False
        for dy, dx in nbr8:
            nb = (cur[0] + dy, cur[1] + dx)
            if nb in remaining:
                path.append(nb)
                remaining.remove(nb)
                found = True
                break
        if not found:
            # Jump to nearest remaining pixel
            rem_arr = np.array(list(remaining))
            dists = (rem_arr[:, 0] - cur[0]) ** 2 + (rem_arr[:, 1] - cur[1]) ** 2
            j = int(np.argmin(dists))
            nb = (int(rem_arr[j, 0]), int(rem_arr[j, 1]))
            path.append(nb)
            remaining.remove(nb)

    return path


def ventricular_arclength_seeds(
    bs: BoundarySets,
    seed_stride: int = 4
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Trace the ventricular boundary as an ordered curve and compute normalised
    arc-length at evenly spaced seed points.
    """
    path = _order_boundary_pixels(bs.ventricular)
    if len(path) < 2:
        raise RuntimeError("Ventricular boundary has fewer than 2 pixels.")

    coords = np.array(path, dtype=np.float64)   # (N, 2) with columns (y, x)
    seg_lengths = np.hypot(np.diff(coords[:, 1]), np.diff(coords[:, 0]))
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    arc /= arc[-1] + 1e-12   # normalise to [0, 1]

    stride = max(1, seed_stride)
    indices = list(range(0, len(path), stride))
    seeds = [(int(coords[i, 0]), int(coords[i, 1])) for i in indices]
    svals = [float(arc[i]) for i in indices]

    print(f"  Ventricular arclength: {len(path)} pixels, {len(seeds)} seeds")
    return seeds, svals


# -------- Paint sv along columns --------

def paint_sv_along_columns(
    phi: NDArray[np.float32],
    mask: NDArray[np.bool_],
    seeds: list[tuple[int, int]],
    svals: list[float],
    step: float = 0.75,
    max_steps: int = 20_000,
) -> NDArray[np.float32]:
    """
    Trace streamlines of ∇φ from ventricular seeds toward the pial surface,
    painting each visited pixel with the seed's arclength value.

    Unpainted pixels are filled by BFS within the mask, then by nearest-neighbour
    as a final fallback.
    """
    H, W = phi.shape
    gy, gx = np.gradient(phi.astype(np.float64), edge_order=2)
    sv = np.full((H, W), np.nan, dtype=np.float32)

    for (y0, x0), s in zip(seeds, svals):
        # Nudge half a pixel along the gradient to start inside the ribbon
        g_x, g_y = float(gx[y0, x0]), float(gy[y0, x0])
        norm = np.hypot(g_x, g_y) + 1e-12
        y, x = float(y0) + 0.5 * g_y / norm, float(x0) + 0.5 * g_x / norm

        # Fall back to seed position if nudge lands outside
        iy, ix = int(np.clip(round(y), 0, H - 1)), int(np.clip(round(x), 0, W - 1))
        if not mask[iy, ix]:
            y, x = float(y0), float(x0)

        for _ in range(max_steps):
            iy = int(np.clip(round(y), 0, H - 1))
            ix = int(np.clip(round(x), 0, W - 1))

            if np.isnan(sv[iy, ix]):
                sv[iy, ix] = s

            # Bilinear gradient sample
            xi = np.clip(x, 0, W - 1)
            yi = np.clip(y, 0, H - 1)
            x0i, y0i = int(np.floor(xi)), int(np.floor(yi))
            x1i, y1i = min(x0i + 1, W - 1), min(y0i + 1, H - 1)
            wx, wy = xi - x0i, yi - y0i

            gxi = ((1 - wx) * (1 - wy) * gx[y0i, x0i] +
                   wx * (1 - wy) * gx[y0i, x1i] +
                   (1 - wx) * wy * gx[y1i, x0i] +
                   wx * wy * gx[y1i, x1i])
            gyi = ((1 - wx) * (1 - wy) * gy[y0i, x0i] +
                   wx * (1 - wy) * gy[y0i, x1i] +
                   (1 - wx) * wy * gy[y1i, x0i] +
                   wx * wy * gy[y1i, x1i])

            n = np.hypot(gxi, gyi)
            if n < 1e-8:
                break

            x += (gxi / n) * step
            y += (gyi / n) * step

            iy_next = int(np.clip(round(y), 0, H - 1))
            ix_next = int(np.clip(round(x), 0, W - 1))
            if not mask[iy_next, ix_next]:
                break
            if phi[iy_next, ix_next] >= 0.999:
                # Paint the pial pixel too
                if np.isnan(sv[iy_next, ix_next]):
                    sv[iy_next, ix_next] = s
                break

    # Fill unpainted mask pixels by BFS (preserves value from nearest painted pixel)
    painted = np.isfinite(sv) & mask
    n_painted = int(np.sum(painted))
    n_mask = int(np.sum(mask))
    print(f"  Streamlines painted {n_painted}/{n_mask} mask pixels "
          f"({100 * n_painted / max(1, n_mask):.1f}%)")

    out = sv.copy()
    q = deque(zip(*np.nonzero(painted)))
    while q:
        y, x = q.popleft()
        v = out[y, x]
        for dy, dx in NBRS4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not np.isfinite(out[ny, nx]):
                out[ny, nx] = v
                q.append((ny, nx))

    # Nearest-neighbour for any remaining NaNs
    missing = mask & ~np.isfinite(out)
    n_missing = int(np.sum(missing))
    if n_missing > 0:
        print(f"  [WARN] {n_missing} pixels filled by nearest-neighbour fallback")
        known = mask & np.isfinite(out)
        if not np.any(known):
            raise RuntimeError("No painted pixels — check seeds and mask.")
        _, (iy, ix) = ndi.distance_transform_edt(~known, return_indices=True)
        out[missing] = out[iy[missing], ix[missing]]

    out[~mask] = 0.0
    return np.clip(out, 0.0, 1.0)


# -------- Main --------

def main():
    ap = argparse.ArgumentParser(
        description="Solve Laplacian φ and paint ventricular arclength sv.")
    ap.add_argument("--mask", required=True,
                    help="Cortex mask image (255 = cortex)")
    ap.add_argument("--boundary_labels", required=True,
                    help="Boundary labels image (uint8: 1=vent, 2=pial, 3=left, 4=right)")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--prefix", default="section")
    ap.add_argument("--seed_stride", type=int, default=4,
                    help="Seed every k-th ventricular pixel for sv painting")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--maxiter", type=int, default=50_000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load boundaries
    print("Loading boundaries...")
    bs = load_boundary_sets(args.mask, args.boundary_labels)

    # Solve φ
    print("Solving Laplacian for φ...")
    phi = solve_phi(bs, rtol=args.rtol, maxiter=args.maxiter)
    phi_path = os.path.join(args.outdir, f"{args.prefix}_phi.tif")
    save_u16(phi_path, phi, mask=bs.cortex_mask)
    print(f"  Saved {phi_path}")

    # Ventricular arclength
    print("Computing ventricular arclength...")
    seeds, svals = ventricular_arclength_seeds(bs, seed_stride=args.seed_stride)

    print("Painting sv along columns...")
    sv = paint_sv_along_columns(phi, bs.cortex_mask, seeds, svals)
    sv_path = os.path.join(args.outdir, f"{args.prefix}_sv.tif")
    save_u16(sv_path, sv, mask=bs.cortex_mask)
    print(f"  Saved {sv_path}")

    print("Done.")


if __name__ == "__main__":
    main()