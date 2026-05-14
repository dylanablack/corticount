# corticount :sparkles:

Corticount is a small package for histological analysis of mammalian cortical tissue.
It can be used to define a 2D coordinate system that respects the curvature of the cortex, and to divide the cortex into segmented bins that align with radial neurite geometry.

Corticount is intended to be used alongside segmentation tools such as Cellpose, and optionally outputs cortical divisions as ImageJ ROIs. 

## :wrench: How it works

Corticount requires only user-defined 2D outlines of the cortex determined from microscopy images.
These outlines define boundary conditions over which the Laplacian is computed.


## :zap: Quick start

```
# User defined boundaries, identifying pia, ventricle, and left/right boundaries.
python create_boundaries.py \
--image path/to/image.tiff # microscopy image to tracer over.
--outstem stem_name

# Running the solver.
python solve_phi_psi.py \

# Generate radial streamlines, which define the lateral boundaries of regions and cortical thickness measurements.
python generate_streamlines.py \

# Export regions as segmentations and/or ImageJ ROIs for downstream analysis.
python export_corticounts.py \

```

## Design choices

The interactive boundary annotator is currently implemented with matplotlib, as it was easiest to associate line segments with boundary identities. However, future updates may include an annotator using Napari, which might be better suited for larger background images.
