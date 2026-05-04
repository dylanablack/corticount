"""
Interactive boundary annotator

Draw one continuous closed polygon around the cortical ribbon on a 2D section
Each edge is assigned a boundary-condition type as you draw:

1 = ventricular (Dirichlet,  φ = 0)
2 = pial (Dirichlet, φ = 1)
3 = left cut (Neummann ∂φ/∂n = 0)

"""

from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from skimage import io, draw, measure, transform

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

def assign_boundary_labels(mask):
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
    from scipy.ndimage import binary_erosion

    H, W = mask.shape
    interior = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    boundary_mask = mask & ~interior # mask pixels with at least one non-mask neighbour

    return boundary_mask
    




