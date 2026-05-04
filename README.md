# corticount :sparkles:

Cortical neurobiological research regularly requires quantifying cellular features in 2D histological sections. 
The latest segmentation methods allow for large scale quantifications of these features.
However, a concise and principled allocation of these features to specific cortical regions is still needed. 
This is especially true for the cortex, which is a curved ribbon of tissue complicating generation of the sensible coordinate system.

This project addresses this by defining a coordinate system based on the curved structure of the cortical ribbon.

## :wrench: How it works

Corticount works in the following way: 
- **User-defined boundaries of the cortex in 2D**. Pial and ventricular surfaces are defined as Dirichlet boundaries, and the lateral borders (e.g., rhinal sulcus) are Neumann with 0 flux. 
- **Laplace equation solved under these boundary conditions**, producing a smooth, continuous gradient from ventricle to pia. 
