# Life X - 01_strocchi_LV

Life X recently published their own implementation of the LDRB algorithm and with that they also publushed a lot of example meshes. This demo aims to try out this implementation of the LDRB algorithm on these example meshes. The LifeX example meshes can be found at https://zenodo.org/record/5810269#.YeEjWi8w1B0, which also contains a DOI: https://doi.org/10.5281/zenodo.5810269.

This demo assumes that you have downloaded the folder with the meshes in the same format as they are uploaded on zenodo, so that the gmsh files are located in a folder called `lifex_fiber_generation_examples/mesh`.

First we import the necessary packages. Note that we also import `meshio` which is used for converted from `.msh` (gmsh) to `.xdmf` (FEnICS).


```python
import dolfin
```

```python
import ldrb
```

Load the mesh and markers. This is a large mesh that you probably want to run in parallel, but you need to first convert the mesh to a fenics mesh and that has to be done in serial. You can make sure that the mesh is saved by setting `unlink = False`

```python
# Last argument here is the markers, but these are not used
mesh, ffun, _ = ldrb.gmsh2dolfin(
    "lifex_fiber_generation_examples/mesh/01_strocchi_LV.msh",
    unlink=False,
)
# Run this first in serial and exit here
# exit()
```

To run the in paralell you can do
```
mpirun -n 4 python demo_lifex_strocchi_LV.py
```


These are the actualy markers (but we only supprt one base at the moment)

```python
original_markers = {"epi": 10, "endo": 20, "aortic_valve": 50, "mitral_valve": 60}
```

So we just use these markers instead

```python
markers = {"epi": 10, "lv": 20, "base": 40}
```

And update the markers accordingly

```python
ffun.array()[ffun.array() == original_markers["aortic_valve"]] = markers["base"]
ffun.array()[ffun.array() == original_markers["mitral_valve"]] = markers["base"]
```

Select linear langange elements

```python
fiber_space = "P_1"
```


Compute the fiber-sheet system

```python
fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
    mesh=mesh,
    fiber_space=fiber_space,
    ffun=ffun,
    markers=markers,
    alpha_endo_lv=60,  # Fiber angle on the endocardium
    alpha_epi_lv=-60,  # Fiber angle on the epicardium
    beta_endo_lv=0,  # Sheet angle on the endocardium
    beta_epi_lv=0,  # Sheet angle on the epicardium
)
```

And save the results

```python
with dolfin.XDMFFile(mesh.mpi_comm(), "01_strocchi_LV_fiber.xdmf") as xdmf:
    xdmf.write(fiber)
```


![_](_static/figures/01_strocchi_LV_fiber_1.png)
![_](_static/figures/01_strocchi_LV_fiber_2.png)
