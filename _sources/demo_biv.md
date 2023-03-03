# Creating fibers on a simple BiV ellipsoid

In this demo we will create fibers from the simple idealized BiV ellipsoid. We will use a library called [`cardiac-geometries`](https://computationalphysiology.github.io/cardiac_geometries/) to generate the mesh.
This library can be installed with `pip`, i.e `pip install cardiac-geometries`, but note that it also requires `gmsh` with OpenCASCADE to be installed.



```python
import dolfin as df
```

```python
import ldrb
import cardiac_geometries
```



Here we just create a lv mesh using `cardiac-geometries``


```python
geometry = cardiac_geometries.create_biv_ellipsoid()
```


The mesh is stored as an attribute mesh


```python
mesh = geometry.mesh
```


The facet function (function with marking for the boundaries) is stored as an attribute `ffun`


```python
ffun = geometry.ffun
```


A dictionary with keys and values for the markers as stored in the attribute `markers`. These markers are loaded from the underlying gmsh files, and has to be slightly modified to work with `ldrb` as `ldrb` expected a dictionary with the keys being `"base"`, `"epi"`, `"lv"` and `"rv"`.


```python
markers = geometry.markers
markers = {
    "base": geometry.markers["BASE"][0],
    "epi": geometry.markers["EPI"][0],
    "lv": geometry.markers["ENDO_LV"][0],
    "rv": geometry.markers["ENDO_RV"][0],
}
```


Also if you want to to this demo in parallel you should create the mesh
in serial and save it to e.g xdmf



```python
with df.XDMFFile(mesh.mpi_comm(), "mesh.xdmf") as xdmf:
    xdmf.write(mesh)
```


And when you run the code in parallel you should load the mesh from the file.


```python
mesh = df.Mesh()
with df.XDMFFile("mesh.xdmf") as xdmf:
    xdmf.read(mesh)
```

You should also save the facet function

```python
with df.XDMFFile(mesh.mpi_comm(), "ffun.xdmf") as xdmf:
    xdmf.write(ffun)
```

and read it agin

```python
ffun = df.MeshFunction("size_t", mesh, 2)
with df.XDMFFile("ffun.xdmf") as xdmf:
    xdmf.read(ffun)
```


Choose space for the fiber fields
This is a string on the form {family}_{degree}

```python
fiber_space = "Lagrange_2"
```

Compute the microstructure

```python
fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
    mesh=mesh,
    fiber_space=fiber_space,
    ffun=ffun,
    markers=markers,
    alpha_endo_lv=80,  # Fiber angle on the LV endocardium
    alpha_epi_lv=-30,  # Fiber angle on the LV epicardium
    beta_endo_lv=0,  # Sheet angle on the LV endocardium
    beta_epi_lv=0,  # Sheet angle on the LV epicardium
    alpha_endo_sept=80,  # Fiber angle on the Septum endocardium
    alpha_epi_sept=-60,  # Fiber angle on the Septum epicardium
    beta_endo_sept=0,  # Sheet angle on the Septum endocardium
    beta_epi_sept=0,  # Sheet angle on the Septum epicardium
    alpha_endo_rv=60,  # Fiber angle on the RV endocardium
    alpha_epi_rv=-80,  # Fiber angle on the RV epicardium
    beta_endo_rv=0,  # Sheet angle on the RV endocardium
    beta_epi_rv=0,
)
```

Store the results

```python
with df.HDF5File(mesh.mpi_comm(), "biv.h5", "w") as h5file:
    h5file.write(fiber, "/fiber")
    h5file.write(sheet, "/sheet")
    h5file.write(sheet_normal, "/sheet_normal")
```

If you run in parallel you should skip the visualization step and do that in
serial in stead. In that case you can read the the functions from the xml
Using the following code


```python
V = ldrb.space_from_string(fiber_space, mesh, dim=3)
```

```python
fiber = df.Function(V)
sheet = df.Function(V)
sheet_normal = df.Function(V)
```

```python
with df.HDF5File(mesh.mpi_comm(), "biv.h5", "r") as h5file:
    h5file.read(fiber, "/fiber")
    h5file.read(sheet, "/sheet")
    h5file.read(sheet_normal, "/sheet_normal")
```


You can also store files in XDMF which will also compute the fiber angle as scalars on the glyph to be visualised in Paraview. Note that these functions don't work (yet) using mpirun

```python
# (These function are not tested in parallel)
ldrb.fiber_to_xdmf(fiber, "biv_fiber")
# ldrb.fiber_to_xdmf(sheet, "biv_sheet")
# ldrb.fiber_to_xdmf(sheet_normal, "biv_sheet_normal")
```

![_](_static/figures/biv_fiber.png)
[Link to source code](https://github.com/finsberg/ldrb/blob/main/demos/demo_biv.py)
