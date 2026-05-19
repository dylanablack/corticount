#!/usr/bin/env python3
"""
generate_streamlines.py
=======================
Trace cortical columns (streamlines of ∇φ) from ventricular seeds to the pial
surface.  Seeds are placed at even arc-length intervals along the ventricular
boundary, which is read directly from the boundary labels image.

Inputs:
  --phi               phi TIFF from the Laplacian solver (uint16 or float, 0–1)
  --mask              cortex mask TIFF
  --boundary_labels   boundary labels TIFF (1=vent, 2=pial, 3=left, 4=right)
  --perimeter_json    perimeter JSON from the annotator (polygon vertices + edge types)

Outputs:
  {prefix}_streamlines.json       line coordinates + metadata
  {prefix}_streamlines.csv        per-streamline summary
  {prefix}_streamlines_overlay.png  visualisation on phi or background image
  {prefix}_vent_rim_qc.png        ventricular boundary QC image
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from skimage import io, draw, exposure

from scipy.ndimage import binary_erosion

# -------- I/O helpers --------

def _squeeze_gray(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2:
        return a
    a = np.squeeze(a)
    if a.ndim == 2:
        return a
    if a.ndim >= 3:
        return a[..., 0]
    return a


def load_phi(path: str) -> NDArray[np.float32]:
    raw = _squeeze_gray(io.imread(path))
    if raw.dtype.kind in "ui":
        vmax = 65535.0 if (raw.dtype == np.uint16 or int(np.max(raw)) > 255) else 255.0
        phi = raw.astype(np.float32) / vmax
    else:
        phi = raw.astype(np.float32)
    return np.clip(phi, 0.0, 1.0)


def load_mask(path: str) -> NDArray[np.bool_]:
    return _squeeze_gray(io.imread(path)) > 0


def load_gray01(path: Optional[str], fallback: NDArray[np.float32]) -> NDArray[np.float32]:
    """Load a grayscale image as float [0,1], or use fallback."""
    if not path:
        return exposure.rescale_intensity(
            fallback, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)
    g = _squeeze_gray(io.imread(path)).astype(np.float32)
    return exposure.rescale_intensity(
        g, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)


# -------- Ventricular arc from polygon vertices --------

def extract_ventricular_polyline(json_path: str) -> NDArray[np.float64]:
    """
    Read the perimeter JSON and return the ventricular arc as an ordered
    polyline in (x, y) pixel coordinates.

    The annotator stores vertices and per-edge type codes.  Edge i goes from
    verts[i] to verts[(i+1) % n].  We collect all consecutive ventricular
    edges (code 1) into a single polyline.
    """
    with open(json_path) as f:
        J = json.load(f)

    verts = J["vertices"]              # [[x, y], ...]
    codes = J["edge_types"]            # int code per edge
    n = len(verts)

    # Find runs of consecutive ventricular edges
    is_vent = [int(c) == 1 for c in codes]

    # Collect vertex indices belonging to ventricular edges.
    # Edge i connects verts[i] → verts[(i+1)%n], so it contributes both endpoints.
    vent_indices = []
    for i in range(n):
        if is_vent[i]:
            vent_indices.append(i)
            vent_indices.append((i + 1) % n)

    if not vent_indices:
        raise RuntimeError("No ventricular edges (code 1) found in perimeter JSON.")

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for idx in vent_indices:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)

    polyline = np.array([verts[i] for i in ordered], dtype=np.float64)  # (M, 2) = (x, y)
    return polyline


def make_seeds_from_polyline(
    polyline: NDArray[np.float64],
    mask: NDArray[np.bool_],
    n_streams: int | None = None,
    seed_stride: int = 30,
    um_per_px: float | None = None,
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Place seeds at even arc-length intervals along the ventricular polyline,
    then snap each to the nearest mask-edge pixel.

    Parameters
    ----------
    polyline : (M, 2) array of (x, y) coordinates from the polygon vertices.
    mask : cortex mask (used to snap seeds to boundary pixels).
    n_streams : if given, place this many seeds; otherwise use seed_stride.
    seed_stride : approximate spacing in pixels (used when n_streams is None).
    """
    # Arc-length along the polyline
    dx = np.diff(polyline[:, 0])
    dy = np.diff(polyline[:, 1])
    seg_lengths = np.hypot(dx, dy)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    L = arc[-1]

    if L < 1e-6:
        raise RuntimeError("Ventricular polyline has zero length.")

    # Determine number of seeds
    if n_streams is not None and n_streams > 0:
        ns = n_streams
    else:
        ns = max(1, int(round(L / max(1, seed_stride))))

    # Target arc-length positions (centered in each interval)
    targets = (np.arange(ns) + 0.5) * (L / ns)

    # Interpolate (x, y) at each target arc-length
    seed_xy = np.column_stack([
        np.interp(targets, arc, polyline[:, 0]),
        np.interp(targets, arc, polyline[:, 1]),
    ])
    svals = (targets / L).tolist()

    # Snap to nearest mask boundary pixel
    interior = binary_erosion(mask, structure=np.ones((3, 3), bool))
    edge = mask & ~interior
    ey, ex = np.nonzero(edge)

    if len(ey) == 0:
        raise RuntimeError("No mask-edge pixels found.")

    seeds = []
    for x, y in seed_xy:
        d2 = (ex - x) ** 2 + (ey - y) ** 2
        j = int(np.argmin(d2))
        seeds.append((int(ey[j]), int(ex[j])))

    spacing = L / ns
    spacing_str = f"{spacing:.1f} px"
    if um_per_px:
        spacing_str += f" ({spacing * um_per_px:.1f} µm)"
    print(f"  Ventricular arc: L={L:.1f} px, {ns} seeds, spacing ≈ {spacing_str}")

    return seeds, svals


# ── Gradient field ────────────────────────────────────────────────────────────

def compute_gradient(
    phi: NDArray[np.float32],
    sigma: float = 0.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute ∇φ once. Optionally smooth phi before differentiating."""
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        phi_s = gaussian_filter(phi.astype(np.float64), sigma=sigma)
    else:
        phi_s = phi.astype(np.float64)
    gy, gx = np.gradient(phi_s, edge_order=2)
    return gx.astype(np.float32), gy.astype(np.float32)


# ── Streamline tracer ─────────────────────────────────────────────────────────

def _bilinear(field: NDArray[np.float32], x: float, y: float) -> float:
    """Bilinear interpolation of a 2D field at continuous (x, y)."""
    H, W = field.shape
    xi, yi = np.clip(x, 0, W - 1), np.clip(y, 0, H - 1)
    x0 = int(np.floor(xi));  x1 = min(x0 + 1, W - 1)
    y0 = int(np.floor(yi));  y1 = min(y0 + 1, H - 1)
    wx, wy = xi - x0, yi - y0
    return float(
        (1 - wx) * (1 - wy) * field[y0, x0] +
        wx * (1 - wy) * field[y0, x1] +
        (1 - wx) * wy * field[y1, x0] +
        wx * wy * field[y1, x1]
    )


def trace_streamline(
    gx: NDArray[np.float32],
    gy: NDArray[np.float32],
    phi: NDArray[np.float32],
    mask: NDArray[np.bool_],
    seed: tuple[int, int],
    step: float = 0.9,
    max_steps: int = 20_000,
    stop_phi: float = 0.995,
    start_nudge: float = 1.0,
) -> tuple[list[tuple[float, float]], str]:
    """
    Trace a single streamline from seed along +∇φ.

    Returns the line as [(x, y), ...] and a reason string for termination.
    """
    H, W = phi.shape
    y0, x0 = seed

    # Nudge seed along gradient to start inside the ribbon
    g0x, g0y = float(gx[y0, x0]), float(gy[y0, x0])
    n0 = np.hypot(g0x, g0y) + 1e-12
    x = float(x0) + start_nudge * g0x / n0
    y = float(y0) + start_nudge * g0y / n0

    iy, ix = int(np.clip(round(y), 0, H - 1)), int(np.clip(round(x), 0, W - 1))
    if not mask[iy, ix]:
        x, y = float(x0), float(y0)

    line = [(x, y)]
    reason = "max_steps"

    for _ in range(max_steps):
        gxi = _bilinear(gx, x, y)
        gyi = _bilinear(gy, x, y)
        n = np.hypot(gxi, gyi)
        if n < 1e-8:
            reason = "flat_grad"
            break

        # Step with adaptive step-halving near mask boundary
        dx, dy = (gxi / n) * step, (gyi / n) * step
        moved = False
        for _ in range(8):
            xn, yn = x + dx, y + dy
            iy = int(np.clip(round(yn), 0, H - 1))
            ix = int(np.clip(round(xn), 0, W - 1))
            if mask[iy, ix]:
                x, y = xn, yn
                moved = True
                break
            dx *= 0.5
            dy *= 0.5
        if not moved:
            reason = "left_mask"
            break

        line.append((x, y))

        iy = int(np.clip(round(y), 0, H - 1))
        ix = int(np.clip(round(x), 0, W - 1))
        if phi[iy, ix] >= stop_phi:
            reason = "hit_pial"
            break

    return line, reason


# ── Visualisation ─────────────────────────────────────────────────────────────

def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    """HSL to RGB (0–255)."""
    if s == 0:
        v = int(255 * l)
        return v, v, v

    def hue2rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    r = hue2rgb(p, q, h + 1/3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1/3)
    return int(255 * r), int(255 * g), int(255 * b)


def sv_color(t: float) -> tuple[int, int, int]:
    """Map arclength fraction [0,1] to a colour (purple → yellow)."""
    t = float(np.clip(t, 0.0, 1.0))
    h = 0.75 * (1 - t) + 0.1666 * t
    return _hsl_to_rgb(h, 1.0, 0.55)


def paint_overlay(
    bg: NDArray[np.float32],
    mask: NDArray[np.bool_],
    seeds: list[tuple[int, int]],
    lines: list[list[tuple[float, float]]],
    svals: list[float],
    thickness: int = 2,
) -> NDArray[np.uint8]:
    H, W = bg.shape
    base = np.dstack([bg] * 3)
    base[mask] = np.clip(base[mask] + 0.15, 0.0, 1.0)
    rgb = (base * 255).astype(np.uint8)

    r = max(1, thickness)

    # Seeds as red dots
    for y, x in seeds:
        rr, cc = draw.disk((y, x), radius=r, shape=(H, W))
        rgb[rr, cc] = (255, 0, 0)

    # Streamlines coloured by arclength
    for s, ln in zip(svals, lines):
        if len(ln) < 2:
            continue
        color = sv_color(s)
        for i in range(len(ln) - 1):
            x0, y0 = ln[i]
            x1, y1 = ln[i + 1]
            rr, cc, _ = draw.line_aa(
                int(round(y0)), int(round(x0)),
                int(round(y1)), int(round(x1)))
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    if dr * dr + dc * dc > r * r:
                        continue
                    rrs = np.clip(rr + dr, 0, H - 1)
                    ccs = np.clip(cc + dc, 0, W - 1)
                    rgb[rrs, ccs] = color

    return rgb


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(
    phi_path: str,
    mask_path: str,
    labels_path: str,
    perimeter_json: str,
    outdir: str,
    prefix: str,
    n_streams: Optional[int] = None,
    seed_stride: int = 30,
    step: float = 0.9,
    max_steps: int = 20_000,
    stop_phi: float = 0.995,
    grad_sigma: float = 0.8,
    start_nudge: float = 1.0,
    um_per_px: Optional[float] = None,
    thickness: int = 2,
    background: Optional[str] = None,
    overlay_image: Optional[str] = None,
):
    os.makedirs(outdir, exist_ok=True)

    # Load data
    phi = load_phi(phi_path)
    mask = load_mask(mask_path)
    labels = _squeeze_gray(io.imread(labels_path)).astype(np.uint8)
    H, W = phi.shape

    if phi.shape != mask.shape or labels.shape != (H, W):
        raise RuntimeError(
            f"Shape mismatch: phi {phi.shape}, mask {mask.shape}, labels {labels.shape}")

    in_std = float(np.std(phi[mask])) if mask.any() else 0.0
    print(f"  phi: min={phi.min():.4f}  max={phi.max():.4f}  std(mask)={in_std:.4f}")
    if in_std < 1e-5:
        raise RuntimeError("phi is nearly constant inside mask — check solver output.")

    # ── Ventricular seeds from polygon geometry ───────────────────────────
    polyline = extract_ventricular_polyline(perimeter_json)
    seeds, svals = make_seeds_from_polyline(
        polyline, mask,
        n_streams=n_streams, seed_stride=seed_stride, um_per_px=um_per_px)

    seed_phi = np.array([phi[y, x] for y, x in seeds])
    if seed_phi.size:
        print(f"  Seed φ: min={seed_phi.min():.3f}  median={np.median(seed_phi):.3f}  "
              f"max={seed_phi.max():.3f}")

    # ── Compute gradient once ─────────────────────────────────────────────
    gx, gy = compute_gradient(phi, sigma=grad_sigma)

    # ── Trace streamlines ─────────────────────────────────────────────────
    lines = []
    reasons = Counter()
    for sd in seeds:
        ln, reason = trace_streamline(
            gx, gy, phi, mask, sd,
            step=step, max_steps=max_steps, stop_phi=stop_phi,
            start_nudge=start_nudge)
        lines.append(ln)
        reasons[reason] += 1

    n_good = sum(1 for ln in lines if len(ln) >= 2)
    print(f"  Traced: {n_good}/{len(lines)} streamlines  |  "
          f"stop reasons: {dict(reasons)}")

    # ── Lengths ───────────────────────────────────────────────────────────
    lengths_px = []
    for ln in lines:
        if len(ln) < 2:
            lengths_px.append(0.0)
            continue
        arr = np.array(ln)
        lengths_px.append(float(np.sum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1])))))
    lengths_px = np.array(lengths_px)

    nz = lengths_px[lengths_px > 0]
    if nz.size:
        p5, p50, p95 = np.percentile(nz, [5, 50, 95])
        print(f"  Length (px): p5={p5:.1f}  median={p50:.1f}  p95={p95:.1f}")
        if um_per_px:
            print(f"  Length (µm): p5={p5*um_per_px:.1f}  median={p50*um_per_px:.1f}  "
                  f"p95={p95*um_per_px:.1f}")

    # ── QC: ventricular boundary overlay ──────────────────────────────────
    vent_pixels = (labels == 1) & mask
    qc = np.zeros((H, W, 3), dtype=np.float32)
    qc[..., 0] = vent_pixels.astype(np.float32)                    # red = ventricular
    from scipy.ndimage import binary_erosion
    full_edge = mask & ~binary_erosion(mask, structure=np.ones((3, 3), bool))
    qc[..., 1] = full_edge.astype(np.float32) * 0.4               # green = full mask edge
    # Mark seed positions in blue
    for y, x in seeds:
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    qc[ny, nx] = [0, 0, 1]
    qc_path = os.path.join(outdir, f"{prefix}_vent_rim_qc.png")
    io.imsave(qc_path, (np.clip(qc, 0, 1) * 255).astype(np.uint8),
              check_contrast=False)

    # ── Overlay on background ─────────────────────────────────────────────
    bg = load_gray01(background, phi)
    rgb = paint_overlay(bg, mask, seeds, lines, svals, thickness=thickness)
    overlay_path = os.path.join(outdir, f"{prefix}_streamlines_overlay.png")
    io.imsave(overlay_path, rgb, check_contrast=False)
    print(f"  Saved {overlay_path}")

    # Optional extra overlay
    if overlay_image:
        bg2 = load_gray01(overlay_image, phi)
        rgb2 = paint_overlay(bg2, mask, seeds, lines, svals, thickness=thickness)
        stem = os.path.splitext(os.path.basename(overlay_image))[0]
        extra_path = os.path.join(outdir, f"{prefix}_overlay_on_{stem}.png")
        io.imsave(extra_path, rgb2, check_contrast=False)
        print(f"  Saved {extra_path}")

    # ── CSV + JSON ────────────────────────────────────────────────────────
    lengths_um = lengths_px * um_per_px if um_per_px else np.zeros_like(lengths_px)

    csv_path = os.path.join(outdir, f"{prefix}_streamlines.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "sv", "length_px", "length_um", "n_points"])
        for i, (s, lp, lu, ln) in enumerate(
                zip(svals, lengths_px, lengths_um, [len(l) for l in lines])):
            w.writerow([i, f"{s:.6f}", f"{lp:.3f}",
                        f"{lu:.3f}" if um_per_px else "", ln])

    json_path = os.path.join(outdir, f"{prefix}_streamlines.json")
    with open(json_path, "w") as f:
        json.dump({
            "image_shape_hw": [H, W],
            "um_per_px": um_per_px,
            "n_streams": len(seeds),
            "stop_phi": stop_phi,
            "grad_sigma": grad_sigma,
            "lines": [[(float(x), float(y)) for x, y in ln] for ln in lines],
            "svals": [float(s) for s in svals],
            "length_px": lengths_px.tolist(),
            "seed_phi_stats": {
                "min": float(seed_phi.min()) if seed_phi.size else None,
                "median": float(np.median(seed_phi)) if seed_phi.size else None,
                "max": float(seed_phi.max()) if seed_phi.size else None,
            },
        }, f)

    print(f"  Saved {csv_path}")
    print(f"  Saved {json_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Trace cortical streamlines from ventricular seeds along ∇φ.")
    ap.add_argument("--phi", required=True, help="φ TIFF (uint16 or float)")
    ap.add_argument("--mask", required=True, help="Cortex mask TIFF")
    ap.add_argument("--boundary_labels", required=True, help="Boundary labels TIFF")
    ap.add_argument("--perimeter_json", required=True, help="Perimeter JSON from annotator")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--prefix", default="section")
    ap.add_argument("--n_streams", type=int, default=None,
                    help="Number of evenly spaced seeds (overrides spacing options)")
    ap.add_argument("--seed_stride", type=int, default=30,
                    help="Seed spacing in pixels (default: 30)")
    ap.add_argument("--seed_spacing_um", type=float, default=None,
                    help="Seed spacing in µm (requires --um_per_px; overrides --seed_stride)")
    ap.add_argument("--grad_sigma", type=float, default=0.8,
                    help="Gaussian smoothing σ for φ before gradient computation")
    ap.add_argument("--step", type=float, default=0.9)
    ap.add_argument("--start_nudge", type=float, default=1.0)
    ap.add_argument("--stop_phi", type=float, default=0.995)
    ap.add_argument("--max_steps", type=int, default=20_000)
    ap.add_argument("--um_per_px", type=float, default=None)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--background", default=None,
                    help="Background image for overlay (default: φ)")
    ap.add_argument("--overlay_image", default=None,
                    help="Extra image to overlay streamlines onto")
    args = ap.parse_args()

    # Convert µm spacing to pixel stride
    seed_stride = args.seed_stride
    if args.seed_spacing_um is not None:
        if args.um_per_px is None:
            ap.error("--seed_spacing_um requires --um_per_px")
        seed_stride = max(1, int(round(args.seed_spacing_um / args.um_per_px)))
        print(f"  Seed spacing: {args.seed_spacing_um} µm → {seed_stride} px "
              f"(at {args.um_per_px} µm/px)")

    run(phi_path=args.phi, mask_path=args.mask, labels_path=args.boundary_labels,
        perimeter_json=args.perimeter_json,
        outdir=args.outdir, prefix=args.prefix,
        n_streams=args.n_streams, seed_stride=seed_stride,
        step=args.step, max_steps=args.max_steps, stop_phi=args.stop_phi,
        grad_sigma=args.grad_sigma, start_nudge=args.start_nudge,
        um_per_px=args.um_per_px, thickness=args.thickness,
        background=args.background, overlay_image=args.overlay_image)


if __name__ == "__main__":
    main()