"""
Interactive boundary annotator

Draw one continuous closed polygon around the cortical ribbon on a 2D section
Each edge is assigned a boundary-condition type as you draw:

1 = ventricular (Dirichlet,  phi = 0)
2 = pial (Dirichlet, phi = 1)
3 = left cut (Neumann d phi/dn = 0)
4 = right cut (Neumann d phi/dn = 0)

Press number keys to change the edge type before clicking the next point.
Press 'f' to close the polygon and then 's' to save.

Outputs (in full-res pixel coordinates):
    {stem}_perimeter.json   - vertices + per-edge type codes
    {stem}_cortex_mask.tif  - binary mask (polygon interior)
    {stem}_boundary_labels.tif  - boundary condition label per mask-edge pixel

Boundary labels are derived from the polygon geometry.
Every mask pixel that borders the exterior is assigned the type of its nearest polygon edge such that:
1. There is no unlabelled boundary pixel.
2. There is no dilation artefacts at the Dirichlet/Neumann corners.
3. The boundary labels are always on the actual mask edge.

Zooming and panning use Matplotlib's standard toolbar; left-clicks while a
toolbar tool (zoom/pan) is active do not add vertices. The toolbar tool
auto-exits after a single zoom rectangle or pan drag so the next click adds
a vertex again.

Keys:
    1/2/3/4     set the current edge type
    Left click  add vertex
    Right/Bksp  undo last vertex
    f           close polygon
    c           change closing edge type after closing (cycles 1->2->3->4)
    s           save outputs
    r           reset
    h           print help
    q/Esc       quit

Usage:
    python create_boundaries.py --image DAPI.tif --outstem my_section
"""

from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from skimage import io, draw, measure, transform
from scipy.ndimage import binary_erosion
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# -------- Boundary condition codes --------

CODES = {
    'ventricular': 1, # Dirichlet, phi=0
    'pial':        2, # Dirichlet, phi=1
    'left':        3, # Neumann
    'right':       4, # Neumann
}

COLOURS = {
    'ventricular': 'tab:blue',
    'pial':        'tab:red',
    'left':        'tab:green',
    'right':       'tab:orange',
}

NAME_FOR_CODE = {v: k for k, v in CODES.items()}
NAME_CYCLE = ['ventricular', 'pial', 'left', 'right']

HELP_TEXT = """
Keys:
    1/2/3/4     set the current edge type (vent / pial / left / right)
    Left click  add vertex (ignored while zoom/pan tool is active)
    Right click undo last vertex
    Backspace   undo last vertex
    f           close polygon
    c           cycle the closing edge's type (only after closing)
    s           save outputs
    r           reset
    h           print this help
    q / Esc     quit

Zoom / pan:
    Click the magnifier or hand on the Matplotlib toolbar, then drag on the
    image. The tool auto-exits after one zoom rectangle or one pan drag, so
    your next left-click adds a vertex again. While zoom/pan is selected the
    title shows a [ZOOM RECT - clicks disabled] / [PAN - ...] banner.
"""


def _stretch(a: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.5) -> np.ndarray:
    """Percentile contrast stretch to float32 in [0, 1]."""
    a = a.astype(np.float32)
    if a.ndim == 2:
        lo, hi = np.percentile(a, (lo_p, hi_p))
    else:  # per-channel for RGB
        lo = np.percentile(a, lo_p, axis=(0, 1))
        hi = np.percentile(a, hi_p, axis=(0, 1))
    return np.clip((a - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)


def _to_display_array(img: np.ndarray) -> np.ndarray:
    """
    Normalise an arbitrary image array to one of:
        (H, W)         - grayscale
        (H, W, 3)      - RGB
    Handles channels-first, Z-stacks, RGBA, multi-channel fluorescence, and
    bit depths beyond uint8. Anything ambiguous is collapsed to a max-projection.
    """
    img = np.squeeze(img)

    if img.ndim == 2:
        return img

    if img.ndim == 3:
        s = img.shape
        # Channels-last RGB or RGBA.
        if s[2] == 3:
            return img
        if s[2] == 4:
            # Drop alpha; treat as RGB.
            return img[..., :3]
        # Channels-first (C, H, W) where C is small. Heuristic: smallest axis
        # is channel-like, and the other two are spatial. We only treat axis 0
        # as channels if it's much smaller than the other two AND <= 4.
        if s[0] <= 4 and s[0] < s[1] and s[0] < s[2]:
            if s[0] == 3:
                return np.moveaxis(img, 0, -1)
            if s[0] == 4:
                return np.moveaxis(img, 0, -1)[..., :3]
            # 1 or 2 channels: max-project.
            return img.max(axis=0)
        # Anything else (Z-stack, multi-channel fluorescence, etc.):
        # collapse the longest non-spatial axis. We assume the two largest
        # axes are spatial.
        axes_by_size = np.argsort(s)
        non_spatial = int(axes_by_size[0])  # smallest axis = "stack" axis
        return img.max(axis=non_spatial)

    if img.ndim >= 4:
        # Reduce extra axes by max-projection until 2D or RGB-shaped.
        out = img
        while out.ndim > 3:
            out = out.max(axis=0)
        return _to_display_array(out)

    raise ValueError(f"Unsupported image shape: {img.shape}")


# ------- Geometry helpers -------- #

def assign_boundary_labels(mask, verts, edge_codes):
    """
    For every mask-edge pixel, find the nearest polygon edge and assign its code.

    Parameters:
    ----------
    mask : (H, W) bool array
        True inside the cortical ribbon.
    verts : (N, 2) array, columns are (x, y) in pixel coordinates.
    edge_codes : length-N list of ints
        Boundary code for edge i (from verts[i] -> verts[(i+1) % N]).

    Returns:
    -------
    labels : (H, W) uint8
        0 for non-boundary; 1-4 on mask-edge pixels.
    """

    H, W = mask.shape
    interior = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    boundary_mask = mask & ~interior # mask pixels with at least one non-mask neighbour always on mask edge

    by, bx = np.nonzero(boundary_mask) # return the boundary coordinates
    n_boundary = len(by) # get the number of boundary coords
    if n_boundary == 0:
        return np.zeros((H, W), dtype=np.uint8) # return a zero array

    n_edges = len(verts) # number of edges from outline
    bx_f = bx.astype(np.float64) # bx as float
    by_f = by.astype(np.float64) # by as float

    # For each boundary pixel, find the nearest polygon edge
    best_code = np.zeros(n_boundary, dtype=np.uint8)
    best_dist = np.full(n_boundary, np.inf)

    for i in range(n_edges):
        j = (i + 1) % n_edges
        ax, ay = float(verts[i][0]), float(verts[i][1])
        ex, ey = float(verts[j][0]), float(verts[j][1])

        # Vectorised point-to-segment distance, add comments for clarity.
        dx, dy = ex - ax, ey - ay
        len2 = dx * dx + dy * dy
        if len2 < 1e-12:
            d2 = (bx_f - ax) ** 2 + (by_f - ay) ** 2
        else:
            t = np.clip(((bx_f - ax) * dx + (by_f - ay) * dy) / len2, 0.0, 1.0)
            cx = ax + t * dx
            cy = ay + t * dy
            d2 = (bx_f - cx) ** 2 + (by_f - cy) ** 2

        closer = d2 < best_dist
        best_dist[closer] = d2[closer]
        best_code[closer] = edge_codes[i]

    labels = np.zeros((H, W), dtype=np.uint8)
    labels[by, bx] = best_code
    return labels


def _polygon_is_simple(verts) -> bool:
    """
    Check that a closed polygon has no self-intersecting non-adjacent edges.
    O(N^2) but N is small (user-drawn).
    """
    n = len(verts)
    if n < 4:
        return True
    V = np.asarray(verts, dtype=np.float64)

    def _seg_intersect(p1, p2, p3, p4):
        # Proper segment intersection test (excludes shared endpoints).
        def ccw(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        # Skip self and the two adjacent edges (they share endpoints).
        for k in range(i + 2, n):
            if i == 0 and k == n - 1:
                continue
            c, d = V[k], V[(k + 1) % n]
            if _seg_intersect(a, b, c, d):
                return False
    return True


# -------- Annotator UI -------- #

class PerimeterAnnotator:
    def __init__(self, img: np.ndarray, outstem: str, display_max_side: int = 1600):
        self.img_full = img
        self.H, self.W = img.shape[:2]
        self.outstem = outstem

        # No default type: force the user to choose before the first vertex
        # makes it harder to silently mislabel the first edge.
        self.current_type: str | None = None
        self.verts: list[list[float]] = [] # [[x, y], ...]
        self.edge_type_names: list[str] = [] # name for edge from verts[i] -> verts[i+1]
        self.is_closed = False

        # Display
        disp_img = img
        if display_max_side and max(self.H, self.W) > display_max_side:
            scale = display_max_side / float(max(self.H, self.W))
            newH, newW = max(1, int(round(self.H * scale))), max(1, int(round(self.W * scale)))
            disp_img = transform.resize(img, (newH, newW),
                                        preserve_range=True, anti_aliasing=True).astype(img.dtype)

        # Bound the figure aspect ratio so very tall/wide sections stay usable.
        aspect = self.H / max(self.W, 1)
        aspect = float(np.clip(aspect, 0.4, 2.5))
        self.fig, self.ax = plt.subplots(figsize=(8, 8 * aspect))

        kw = dict(extent=(0, self.W, self.H, 0), interpolation='nearest')
        if disp_img.ndim == 2:
            self.ax.imshow(disp_img, cmap='gray', **kw)
        else:
            self.ax.imshow(disp_img, **kw)
        self.ax.set_xlim(0, self.W); self.ax.set_ylim(self.H, 0)
        self.ax.set_axis_off()

        # Legend
        handles = [Line2D([0], [0], color=COLOURS[k], lw=3,
                          label=f"{k} ({CODES[k]})")
                   for k in ('ventricular', 'pial', 'left', 'right')]
        self.ax.legend(handles=handles, loc='lower right', fontsize=8)

        # Drawing state
        self._segments: list = []
        self._seg_colours: list = []
        self.lc = LineCollection([], linewidth=2, antialiased=False)
        self.ax.add_collection(self.lc)

        # Toolbar state: auto-exit zoom/pan after one operation so users
        # don't get stuck with clicks "doing nothing" after a zoom.
        self._last_tb_mode = ''
        self._tb_exit_timer = None
        self._closed = False
        self._cids: list = []

        self._update_title()
        self._cids.append(self.fig.canvas.mpl_connect('button_press_event', self._on_click))
        self._cids.append(self.fig.canvas.mpl_connect('button_release_event', self._on_release))
        self._cids.append(self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion))
        self._cids.append(self.fig.canvas.mpl_connect('key_press_event', self._on_key))
        self._cids.append(self.fig.canvas.mpl_connect('close_event', self._on_close))

    # -------- Display -------- #
    def _toolbar_mode_str(self) -> str:
        tb = getattr(self.fig.canvas, 'toolbar', None)
        if tb is None:
            return ''
        m = (getattr(tb, 'mode', '') or '').strip()
        return m

    def _update_title(self):
        tb_mode = self._toolbar_mode_str()
        banner = f"  [{tb_mode.upper()} - clicks disabled]" if tb_mode else ''
        if self.is_closed:
            closing = self.edge_type_names[-1] if self.edge_type_names else '?'
            msg = (f"Polygon closed. closing-edge = {closing} | "
                   f"c = cycle closing type | s = save | r = reset | q = quit"
                   f"{banner}")
        else:
            n = len(self.verts)
            t = self.current_type if self.current_type else "NONE - press 1/2/3/4"
            msg = (f"Points: {n} | Edge type: {t} "
                   f"(1=vent 2=pial 3=left 4=right){banner}\n"
                   f"L-click: add | R-click/Bksp: undo | f: close | h: help | "
                   f"toolbar: zoom/pan (auto-exits)")
        self.ax.set_title(msg, fontsize=9)
        self.fig.canvas.draw_idle()

    def _refresh_lines(self):
        segs = list(self._segments)
        cols = list(self._seg_colours)
        # Preview closing edge
        if not self.is_closed and len(self.verts) >= 2 and self.current_type is not None:
            segs.append((tuple(self.verts[-1]), tuple(self.verts[0])))
            cols.append(COLOURS[self.current_type])
        self.lc.set_segments(segs)
        self.lc.set_colors(cols if cols else [(0, 0, 0, 0)])
        self.fig.canvas.draw_idle()

    def _toolbar_active(self) -> bool:
        """True if zoom or pan is selected (so clicks shouldn't add vertices)."""
        return self._toolbar_mode_str() != ''

    # -------- Events -------- #

    def _on_click(self, event):
        if self._closed:
            return
        if event.inaxes != self.ax:
            return
        if self._toolbar_active():
            return  # let the toolbar handle zoom/pan
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            self._add_point(float(event.xdata), float(event.ydata))
        elif event.button == 3:
            self._undo()

    def _on_release(self, event):
        """After a zoom-rect or pan drag completes, auto-exit toolbar mode."""
        if self._closed or event.button != 1:
            return
        mode = self._toolbar_mode_str().lower()
        if 'zoom' in mode or 'pan' in mode:
            # Defer the toggle: calling tb.zoom()/tb.pan() inside the release
            # callback can race with matplotlib's own release_zoom/release_pan
            # handler. A short timer lets that finish first.
            self._schedule_toolbar_exit()

    def _on_motion(self, event):
        """Refresh title when toolbar mode changes (e.g. user just clicked zoom)."""
        if self._closed:
            return
        cur = self._toolbar_mode_str()
        if cur != self._last_tb_mode:
            self._last_tb_mode = cur
            self._update_title()

    def _on_close(self, event):
        self._cleanup()

    def _schedule_toolbar_exit(self):
        if self._closed:
            return
        tb = getattr(self.fig.canvas, 'toolbar', None)
        if tb is None:
            return
        # Cancel any pending timer first.
        if self._tb_exit_timer is not None:
            try:
                self._tb_exit_timer.stop()
            except Exception:
                pass
            self._tb_exit_timer = None
        try:
            timer = self.fig.canvas.new_timer(interval=50)
            timer.single_shot = True
            timer.add_callback(self._exit_toolbar_mode)
            timer.start()
            self._tb_exit_timer = timer
        except Exception:
            # Backend doesn't support timers - fall back to immediate toggle.
            self._exit_toolbar_mode()

    def _exit_toolbar_mode(self):
        self._tb_exit_timer = None
        if self._closed:
            return
        try:
            if not plt.fignum_exists(self.fig.number):
                return
        except Exception:
            return
        tb = getattr(self.fig.canvas, 'toolbar', None)
        if tb is None:
            return
        mode = self._toolbar_mode_str().lower()
        try:
            if 'zoom' in mode:
                tb.zoom()  # toggle off
            elif 'pan' in mode:
                tb.pan()   # toggle off
        except Exception:
            pass
        self._last_tb_mode = self._toolbar_mode_str()
        self._update_title()

    def _cleanup(self):
        """Stop pending timers and detach all callbacks. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._tb_exit_timer is not None:
            try:
                self._tb_exit_timer.stop()
            except Exception:
                pass
            self._tb_exit_timer = None
        for cid in self._cids:
            try:
                self.fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._cids = []

    def _on_key(self, event):
        if self._closed:
            return
        key = event.key
        changed = False
        if key in '1234':
            names = {'1': 'ventricular', '2': 'pial', '3': 'left', '4': 'right'}
            self.current_type = names[key]
            changed = True
        elif key in ('backspace', 'delete'):
            self._undo()
            changed = True
        elif key == 'f':
            self._finish()
            changed = True
        elif key == 'c':
            self._cycle_closing_type()
            changed = True
        elif key == 'r':
            self._reset()
            changed = True
        elif key == 's':
            self._save()
            changed = True
        elif key == 'h':
            print(HELP_TEXT)
        elif key in ('q', 'escape'):
            self._cleanup()
            plt.close(self.fig)
            return
        if changed:
            self._update_title()
            self._refresh_lines()

    # -------- Drawing -------- #

    def _add_point(self, x, y):
        if self.is_closed:
            print("Polygon is closed. Press 'r' to reset.")
            return
        if self.current_type is None:
            print("Choose an edge type first (press 1, 2, 3 or 4).")
            return
        # Clip to image bounds so vertices can't sit off-canvas.
        x = float(np.clip(x, 0.0, self.W))
        y = float(np.clip(y, 0.0, self.H))
        if self.verts:
            self._segments.append((tuple(self.verts[-1]), (x, y)))
            self._seg_colours.append(COLOURS[self.current_type])
            self.edge_type_names.append(self.current_type)
        self.verts.append([x, y])
        self._update_title()
        self._refresh_lines()

    def _undo(self):
        if self.is_closed or not self.verts:
            return
        self.verts.pop()
        if self._segments:
            self._segments.pop()
            self._seg_colours.pop()
            self.edge_type_names.pop()
        self._update_title()
        self._refresh_lines()

    def _finish(self):
        if len(self.verts) < 3:
            print("Need greater than or equal to 3 points.")
            return
        if self.current_type is None:
            print("Choose an edge type for the closing edge first (1/2/3/4).")
            return
        # Closing edge uses the current type
        self._segments.append((tuple(self.verts[-1]), tuple(self.verts[0])))
        self._seg_colours.append(COLOURS[self.current_type])
        self.edge_type_names.append(self.current_type)
        self.is_closed = True
        self._update_title()
        self._refresh_lines()
        print(f"Polygon is closed. Closing edge = '{self.current_type}'. "
              f"Press 'c' to cycle its type, or 's' to save.")

    def _cycle_closing_type(self):
        """After closing, cycle the closing edge's type without resetting."""
        if not self.is_closed or not self.edge_type_names:
            return
        cur = self.edge_type_names[-1]
        try:
            idx = NAME_CYCLE.index(cur)
        except ValueError:
            idx = -1
        new = NAME_CYCLE[(idx + 1) % len(NAME_CYCLE)]
        self.edge_type_names[-1] = new
        self._seg_colours[-1] = COLOURS[new]
        self.current_type = new
        print(f"Closing edge -> {new}")

    def _reset(self):
        self.verts.clear()
        self.edge_type_names.clear()
        self._segments.clear()
        self._seg_colours.clear()
        self.is_closed = False
        self.current_type = None
        self._update_title()
        self._refresh_lines()
        print("Reset.")

    # -------- Saving -------- #

    def _unique_outstem(self, base: str) -> str:
        """Return a stem that doesn't collide with existing outputs."""
        candidates = [f"{base}_perimeter.json",
                      f"{base}_cortex_mask.tif",
                      f"{base}_boundary_labels.tif"]
        if not any(os.path.exists(p) for p in candidates):
            return base
        k = 1
        while True:
            new_base = f"{base}_v{k}"
            cands = [f"{new_base}_perimeter.json",
                     f"{new_base}_cortex_mask.tif",
                     f"{new_base}_boundary_labels.tif"]
            if not any(os.path.exists(p) for p in cands):
                print(f"Outputs exist for stem '{base}'; using '{new_base}' instead.")
                return new_base
            k += 1

    def _save(self):
        if not self.is_closed:
            print("Close polygon first (press 'f').")
            return
        n = len(self.verts)
        edge_codes = [CODES[name] for name in self.edge_type_names]

        if len(edge_codes) != n:
            print(f"Bug: {n} verts but {len(edge_codes)} edges. Reset and redraw.")
            return

        # Validate polygon is simple (no self-intersections).
        if not _polygon_is_simple(self.verts):
            print("Polygon self-intersects. Please reset ('r') and redraw "
                  "without crossings - saving was aborted to avoid silently "
                  "losing part of the annotation.")
            return

        os.makedirs(os.path.dirname(self.outstem) or '.', exist_ok=True)
        stem = self._unique_outstem(self.outstem)

        # JSON with vertices and edge types
        perim = {
            'vertices': self.verts,
            'edge_types': edge_codes,
            'edge_type_names': self.edge_type_names,
            'codes_legend': CODES,
            'image_shape': [int(self.H), int(self.W)],
            'note': 'edge i goes from verts[i] to verts[(i+1) % n]',
        }

        json_path = f"{stem}_perimeter.json"
        with open(json_path, 'w') as f:
            json.dump(perim, f, indent=2)
        print(f"    Saved {json_path}")

        # Cortex mask polygon fill
        xs = [v[0] for v in self.verts]
        ys = [v[1] for v in self.verts]
        rr, cc = draw.polygon(ys, xs, shape=(self.H, self.W))
        mask = np.zeros((self.H, self.W), dtype=bool)
        mask[rr, cc] = True

        # Keep on the largest connected component
        labels = measure.label(mask, connectivity=2)
        if labels.max() > 1:
            areas = [np.sum(labels == k) for k in range(1, labels.max() + 1)]
            mask = labels == (1 + int(np.argmax(areas)))

        mask_u8 = (mask.astype(np.uint8)) * 255
        mask_path = f"{stem}_cortex_mask.tif"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(mask_path, mask_u8, check_contrast=False)
        print(f"    Saved {mask_path}")

        # Boundary labels from mask edge and polygon geometry
        verts_arr = np.array(self.verts)
        boundary_labels = assign_boundary_labels(mask, verts_arr, edge_codes)

        lab_path = f"{stem}_boundary_labels.tif"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(lab_path, boundary_labels, check_contrast=False)

        # Report counts and invariant check.
        for code, name in NAME_FOR_CODE.items():
            count = np.sum(boundary_labels == code)
            print(f"    {name:12s} (code {code}): {count:6d} pixels")

        # Invariant: every mask-edge pixel must carry a label.
        interior = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        edge_pixels = int(np.sum(mask & ~interior))
        labelled = int(np.sum(boundary_labels > 0))
        unlabelled = edge_pixels - labelled
        if unlabelled != 0:
            print(f"    WARNING: {unlabelled} mask-edge pixels are unlabelled "
                  f"(edge={edge_pixels}, labelled={labelled}).")
        else:
            print(f"    Invariant OK: all {edge_pixels} mask-edge pixels labelled.")
        print(f"    Saved {lab_path}")
        print(f"Done. Press 'q' to quit or 'r' to redo.")


# -------- Usage -------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', required=True, help='Background image (e.g., DAPI)')
    ap.add_argument('--outstem', default=None,
                    help='Output file stem (default: image path without extension)')
    ap.add_argument('--display_max_side', type=int, default=1600,
                    help='Downsample display to this max dimension (0 = disable)')
    ap.add_argument('--lo_pct', type=float, default=1.0,
                    help='Lower percentile for contrast stretch (default 1.0)')
    ap.add_argument('--hi_pct', type=float, default=99.5,
                    help='Upper percentile for contrast stretch (default 99.5)')
    args = ap.parse_args()

    # Prefer tifffile for TIFFs so PIL's decompression-bomb limit doesn't bite.
    img = None
    ext = os.path.splitext(args.image)[1].lower()
    if ext in ('.tif', '.tiff'):
        try:
            import tifffile
            img = tifffile.imread(args.image)
        except Exception as e:
            print(f"tifffile read failed ({e}); falling back to skimage.io.imread")
    if img is None:
        img = io.imread(args.image)

    # Normalise dims/channels for display regardless of source layout.
    img = _to_display_array(img)

    # Percentile stretch so dim fluorescence isn't crushed by hot pixels.
    img = _stretch(img, lo_p=args.lo_pct, hi_p=args.hi_pct)

    outstem = args.outstem or os.path.splitext(args.image)[0]
    _ann = PerimeterAnnotator(img, outstem, display_max_side=args.display_max_side)
    print("Draw a perimeter. Keys 1-4 set edge type. 'f' to close, 's' to save.")
    print("Use the toolbar magnifier/hand to zoom and pan. Press 'h' for help.")
    plt.show()


if __name__ == '__main__':
    main()
