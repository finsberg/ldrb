# Generating fibers for patient specific geometries

In this demo we will show how to generate fiber orientations from a patient specific geometry. We will use a mesh of an LV that is constructed using gmsh (https://gmsh.info), see https://github.com/finsberg/ldrb/blob/master/demos/mesh.msh

It is important that the mesh contains physical surfaces of the endocardium (lv and rv if present), the base and the epicardium. You can find an example of how to generate such a geometry using the python API for gmsh here: https://github.com/finsberg/pulse/blob/0d7b5995f62f41df4eec9f5df761fa03da725f69/pulse/geometries.py#L160

First we import the necessary packages. Note that we also import `meshio` which is used for converted from `.msh` (gmsh) to `.xdmf` (FEnICS).

```python
from pathlib import Path
```

```python
import dolfin
import meshio
```

```python
import ldrb
```


Next we define some helper functions for reading the mesh from gmsh. The mesh consists of different cell-types. In this demo we only work with tetrahedral (volume) and triangle (surface) cell-types, but there are also lower dimensional cell types, see e.g https://github.com/finsberg/pulse/blob/0d7b5995f62f41df4eec9f5df761fa03da725f69/pulse/geometries.py#L61


```python
def create_mesh(mesh, cell_type):
    # From http://jsdokken.com/converted_files/tutorial_pygmsh.html
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]},
    )
    return out_mesh
```

```python
def read_meshfunction(fname, obj):
    with dolfin.XDMFFile(Path(fname).as_posix()) as f:
        f.read(obj, "name_to_read")
```


Next we define a function for converting the gmsh file into an xdmf file. This function returns the mesh, as well as the facet function (which contains the markers for all the facets) as well as a dictionary containing the available markers.


```python
def gmsh2dolfin(msh_file):

    msh = meshio.gmsh.read(msh_file)

    triangle_mesh = create_mesh(msh, "triangle")
    tetra_mesh = create_mesh(msh, "tetra")

    triangle_mesh_name = Path("triangle_mesh.xdmf")
    meshio.write(triangle_mesh_name, triangle_mesh)

    tetra_mesh_name = Path("mesh.xdmf")
    meshio.write(
        tetra_mesh_name,
        tetra_mesh,
    )

    mesh = dolfin.Mesh()

    with dolfin.XDMFFile(tetra_mesh_name.as_posix()) as infile:
        infile.read(mesh)

    cfun = dolfin.MeshFunction("size_t", mesh, 3)
    read_meshfunction(tetra_mesh_name, cfun)
    tetra_mesh_name.unlink()
    tetra_mesh_name.with_suffix(".h5").unlink()

    ffun_val = dolfin.MeshValueCollection("size_t", mesh, 2)
    read_meshfunction(triangle_mesh_name, ffun_val)
    ffun = dolfin.MeshFunction("size_t", mesh, ffun_val)
    for value in ffun_val.values():
        mesh.domains().set_marker(value, 2)
    ffun.array()[ffun.array() == max(ffun.array())] = 0
    triangle_mesh_name.unlink()
    triangle_mesh_name.with_suffix(".h5").unlink()

    markers = msh.field_data

    return mesh, ffun, markers
```


In the final function we simply generate the fibers by first converting the mesh and then generating the fibers


```python
def main():
    mesh, ffun, markers = gmsh2dolfin("mesh.msh")

    ldrb_markers = {
        "base": markers["BASE"][0],
        "lv": markers["ENDO"][0],
        "epi": markers["EPI"][0],
    }

    fiber_space = "P_1"

    angles = dict(
        alpha_endo_lv=60,  # Fiber angle on the endocardium
        alpha_epi_lv=-60,  # Fiber angle on the epicardium
        beta_endo_lv=0,  # Sheet angle on the endocardium
        beta_epi_lv=0,
    )

    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh=mesh, fiber_space=fiber_space, ffun=ffun, markers=ldrb_markers, **angles
    )

    # Save to xdmf
    # with dolfin.XDMFFile(mesh.mpi_comm(), "patient_fiber.xdmf") as xdmf:
    #     xdmf.write(fiber)

    ldrb.fiber_to_xdmf(fiber, "patient_fiber")
```

```python
if __name__ == "__main__":
    main()
```
