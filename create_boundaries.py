"""
Interactive boundary annotator

Draw one continuous closed polygon around the cortical ribbon on a 2D section
Each edge is assigned a boundary-condition type as you draw:

1 = ventricular (Dirichlet,  φ = 0)
2 = pial (Dirichlet, φ = 1)
3 = left cut (Neumann ∂φ/∂n = 0)
4 = right cut (Neumann ∂φ/∂n = 0)

Press number kleys to change the edge type before clicking the next point. 
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

Keys:
    1/2/3/4     set the current edge type
    Left click  add vertex
    Right/Bksp  undo last vertex
    f           close polygon
    s           save outputs
    r           reset
    h           print help
    q/Esc       quit

Usage:
    python create_boundaries.py --image DAPI.tif --outstem my_section
"""

from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from skimage import io, draw, measure, transform
from scipy.ndimage import binary_erosion

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

# -------- Annotator UI -------- #

class PerimeterAnnotator:
    def __init__(self, img: np.ndarray, outstem: str, display_max_side: int = 1600):
        self.img_full = img
        self.H, self.W = img.shape[:2]
        self.outstem = outstem

        self.current_type = 'left'
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
        self.fig, self.ax = plt.subplots(figsize=(8, 8 * self.H / self.W))
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

        self._update_title()
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    # -------- Display -------- #
    def _update_title(self):
        if self.is_closed:
            msg = "Polygon closed. s = save | r = reset | q = quit"
        else:
            n = len(self.verts)
            msg = (f"Points: {n} | Edge type: {self.current_type} "
                   f"(1=vent 2=pial 3=left 4=right)\n"
                   f"L-click: add | R-click/Bksp: undo | f: close | h: help")
        self.ax.set_title(msg, fontsize=9)
        self.fig.canvas.draw_idle()
    
    def _refresh_lines(self):
        segs = list(self._segments)
        cols = list(self._seg_colours)
        # Preview closing edge
        if not self.is_closed and len(self.verts) >= 2:
            segs.append((tuple(self.verts[-1]), tuple(self.verts[0])))
            cols.append(COLOURS[self.current_type])
        self.lc.set_segments(segs)
        self.lc.set_colors(cols if cols else [(0, 0, 0, 0)])
        self.fig.canvas.draw_idle()

    # -------- Events -------- #

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self._add_point(float(event.xdata), float(event.ydata))
        elif event.button == 3:
            self._undo()
    
    def _on_key(self, event):
        key = event.key
        if key in '1234':
            names = {'1': 'ventricular', '2': 'pial', '3': 'left', '4': 'right'}
            self.current_type = names[key]
        elif key in ('backspace', 'delete'):
            self._undo()
        elif key == 'f':
            self._finish()
        elif key == 'r':
            self._reset()
        elif key == 's':
            self._save()
        elif key in ('q', 'escape'):
            plt.close(self.fig)
            return
        self._update_title()
        self._refresh_lines()

    # -------- Drawing -------- #

    def _add_point(self, x, y):
        if self.is_closed:
            print("Polygon is closed. Press 'r' to reset.")
            return
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
        # Closing edge uses the current type
        self._segments.append((tuple(self.verts[-1]), tuple(self.verts[0])))
        self._seg_colours.append(COLOURS[self.current_type])
        self.edge_type_names.append(self.current_type)
        self.is_closed = True
        self._update_title()
        self._refresh_lines()
        print("Polygon is closed. Set desired type and press 's' to save.")

    def _reset(self):
        self.verts.clear()
        self.edge_type_names.clear()
        self._segments.clear()
        self._seg_colours.clear()
        self.is_closed = False
        self._update_title()
        self._refresh_lines()
        print("Reset.")

    # -------- Saving -------- #

    def _save(self):
        if not self.is_closed:
            print("Close polygon first (press 'f').")
            return
        n = len(self.verts)
        edge_codes = [CODES[name] for name in self.edge_type_names]

        if len(edge_codes) != n:
            print(f"Bug: {n} verts but {len(edge_codes)} edges. Reset and redraw.")
            return
        
        os.makedirs(os.path.dirname(self.outstem) or '.', exist_ok=True)

        # JSON with vertices and edge types
        perim = {
            'vertices': self.verts,
            'edge_types': edge_codes,
            'edge_type_names': self.edge_type_names,
            'codes_legend': CODES,
            'note': 'edge i goes from verts[i] to verts[(i+1) % n]',
        }

        json_path = f"{self.outstem}_perimeter.json"
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
        mask_path = f"{self.outstem}_cortex_mask.tif"
        io.imsave(mask_path, mask_u8)
        print(f"    Saved {mask_path}")

        # Boundary labels from mask edge and polygon geometry
        verts_arr = np.array(self.verts)
        boundary_labels = assign_boundary_labels(mask, verts_arr, edge_codes)

        lab_path = f"{self.outstem}_boundary_labels.tif"
        io.imsave(lab_path, boundary_labels)

        # Report counts
        for code, name in NAME_FOR_CODE.items():
            count = np.sum(boundary_labels == code)
            print(f"    {name:12s} (code {code}): {count:6d} pixels")
        # unlabelled_edge = np.sum(mask & ~np.pad(mask, 1, constant_values=False)[1:-1, 1:-1]) - np.sum(boundary_labels > 0)
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
    args = ap.parse_args()

    img = io.imread(args.image)
    img = np.squeeze(img)
    if img.ndim == 3 and img.shape[2] > 3:
        lo, hi = float(img.min()), float(img.max())
        img = ((img.astype(np.float32) - lo) / max(1e-6, hi - lo))
    
    outstem = args.outstem or os.path.splitext(args.image)[0]
    _ann = PerimeterAnnotator(img, outstem, display_max_side=args.display_max_side)
    print("Draw a perimeter. Keys 1-4 set edge type. 'f' to close, 's' to save.")
    plt.show()

if __name__ == '__main__':
    main()
