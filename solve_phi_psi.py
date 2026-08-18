#!/usr/bin/env python3
"""
solve_phi_psi.py
===================
Solve two Laplace equations on a 2D cortical ribbon to obtain a normalized
orthogonal coordinate system:
  φ (phi) — cortical depth (0 at ventricular surface, 1 at pial surface)
  ψ (psi) — normalized arclength (0 at left anchor, 1 at right boundary)

Boundary conditions for φ:
  Dirichlet: Ventricular (0), Pial (1)
  Neumann: Left cut, Right cut (zero-flux)

Boundary conditions for ψ:
  Dirichlet: Left cut (0), Right cut (1)
  Neumann: Ventricular, Pial (zero-flux)

Inputs:
  --mask             cortex_mask.tif (binary, 255 = cortex)
  --boundary_labels  boundary_labels.tif (uint8: 1=vent, 2=pial, 3=left, 4=right)

Outputs:
  {prefix}_phi.tif   cortical depth (uint16, 0-65535 mapped to 0-1)
  {prefix}_psi.tif   arclength (uint16, 0-65535 mapped to 0-1)
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as splinalg
from skimage import io
from scipy.ndimage import binary_erosion

# -------- Constants --------
CODES = {1: "ventricular", 2: "pial", 3: "left", 4: "right"}
NBRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# -------- Data --------
@dataclass(frozen=True)
class BoundarySets:
    ventricular: NDArray[np.bool_]
    pial: NDArray[np.bool_]
    left: NDArray[np.bool_]
    right: NDArray[np.bool_]
    cortex_mask: NDArray[np.bool_]

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
    """Load mask and boundary labels directly."""
    mask = io.imread(mask_path) > 0
    labels = io.imread(labels_path).astype(np.uint8)

    vent = (labels == 1) & mask
    pial = (labels == 2) & mask
    left = (labels == 3) & mask
    right = (labels == 4) & mask

    boundary_union = vent | pial | left | right
    interior = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    mask_edge = mask & ~interior
    unlabelled = mask_edge & ~boundary_union
    
    n_unlabelled = int(np.sum(unlabelled))
    if n_unlabelled > 0:
        print(f"[WARN] {n_unlabelled} mask-edge pixels have no boundary label. "
              f"They will be treated as Neumann (zero-flux).")

    for code, name in CODES.items():
        n = int(np.sum(labels == code))
        print(f"  {name:12s}: {n:6d} boundary pixels")

    return BoundarySets(vent, pial, left, right, mask)

# -------- Generic Laplacian Solver --------
def solve_laplacian(
    mask: NDArray[np.bool_],
    dirichlet_0: NDArray[np.bool_],
    dirichlet_1: NDArray[np.bool_],
    field_name: str,
    rtol: float = 1e-6,
    maxiter: int = 50_000
) -> NDArray[np.float32]:
    """
    Solve ∇²u = 0 inside the cortex mask.
    Pixels in dirichlet_0 are fixed to 0.0.
    Pixels in dirichlet_1 are fixed to 1.0.
    Any boundary edge of the mask NOT in dirichlet_0 or dirichlet_1 
    is implicitly treated as a Neumann (zero-flux) boundary.
    """
    H, W = mask.shape
    dirichlet = dirichlet_0 | dirichlet_1

    unknown = mask & ~dirichlet
    N = int(np.sum(unknown))
    print(f"  Solving {field_name}: {N} unknowns, {int(np.sum(dirichlet))} Dirichlet pixels")

    idx = np.full((H, W), -1, dtype=np.int64)
    idx[unknown] = np.arange(N)

    rows, cols, vals = [], [], []
    rhs = np.zeros(N, dtype=np.float64)

    ys, xs = np.nonzero(unknown)
    for k, (y, x) in enumerate(zip(ys, xs)):
        n_interior = 0
        for dy, dx in NBRS4:
            ny, nx = y + dy, x + dx

            # Out of bounds or outside mask = Neumann (zero-flux)
            if ny < 0 or ny >= H or nx < 0 or nx >= W or not mask[ny, nx]:
                continue

            if dirichlet[ny, nx]:
                if dirichlet_1[ny, nx]:
                    rhs[k] += 1.0 
                # If dirichlet_0, it adds 0.0 to rhs, so we do nothing.
            else:
                rows.append(k)
                cols.append(int(idx[ny, nx]))
                vals.append(-1.0)

            n_interior += 1

        rows.append(k)
        cols.append(k)
        vals.append(float(n_interior))

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

    try:
        u, info = splinalg.cg(A, rhs, rtol=rtol, atol=0.0, maxiter=maxiter)
    except TypeError:
        u, info = splinalg.cg(A, rhs, tol=rtol, maxiter=maxiter)

    if info != 0:
        print(f"  [WARN] CG did not converge (info={info})")

    out_field = np.zeros((H, W), dtype=np.float32)
    out_field[dirichlet_1] = 1.0
    out_field[unknown] = u.astype(np.float32)
    out_field[~mask] = 0.0
    
    return np.clip(out_field, 0.0, 1.0)


# -------- Main --------
def main():
    ap = argparse.ArgumentParser(
        description="Solve Dual Laplacian for φ (depth) and ψ (arclength).")
    ap.add_argument("--mask", required=True,
                    help="Cortex mask image (255 = cortex)")
    ap.add_argument("--boundary_labels", required=True,
                    help="Boundary labels image (uint8: 1=vent, 2=pial, 3=left, 4=right)")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--prefix", default="section")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--maxiter", type=int, default=50_000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading boundaries...")
    bs = load_boundary_sets(args.mask, args.boundary_labels)

    # 1. Solve φ (Depth Field)
    # 0 at ventricle, 1 at pia. (Left/Right act as Neumann)
    print("\n--- Computing Depth Field (φ) ---")
    phi = solve_laplacian(
        mask=bs.cortex_mask, 
        dirichlet_0=bs.ventricular, 
        dirichlet_1=bs.pial, 
        field_name="φ",
        rtol=args.rtol, maxiter=args.maxiter
    )
    phi_path = os.path.join(args.outdir, f"{args.prefix}_phi.tif")
    save_u16(phi_path, phi, mask=bs.cortex_mask)
    print(f"  Saved {phi_path}")

    # 2. Solve ψ (Arclength Field)
    # 0 at left cut, 1 at right cut. (Ventricle/Pia act as Neumann)
    print("\n--- Computing Arclength Field (ψ) ---")
    psi = solve_laplacian(
        mask=bs.cortex_mask, 
        dirichlet_0=bs.left, 
        dirichlet_1=bs.right, 
        field_name="ψ",
        rtol=args.rtol, maxiter=args.maxiter
    )
    psi_path = os.path.join(args.outdir, f"{args.prefix}_psi.tif")
    save_u16(psi_path, psi, mask=bs.cortex_mask)
    print(f"  Saved {psi_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()

