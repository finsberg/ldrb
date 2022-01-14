# # Life X examples
#
# Life X recently published their own implementation of the LDRB algorithm and with that they also publushed a lot of example meshes. This demo aims to try out this implementation of the LDRB algorithm on these example meshes. The LifeX example meshes can be found at https://zenodo.org/record/5810269#.YeEjWi8w1B0, which also contains a DOI: https://doi.org/10.5281/zenodo.5810269.
#
# This demo assumes that you have downloaded the folder with the meshes in the same format as they are uploaded on zenodo, so that the gmsh files are located in a folder called `lifex_fiber_generation_examples/mesh``.
#
# First we import the necessary packages. Note that we also import `meshio` which is used for converted from `.msh` (gmsh) to `.xdmf` (FEnICS).

from pathlib import Path

import dolfin
import meshio

import ldrb
from ldrb.utils import mpi_comm_world


# Next we define some helper functions for reading the mesh from gmsh. The mesh consists of different cell-types. In this demo we only work with tetrahedral (volume) and triangle (surface) cell-types, but there are also lower dimensional cell types, see e.g https://github.com/finsberg/pulse/blob/0d7b5995f62f41df4eec9f5df761fa03da725f69/pulse/geometries.py#L61


# In the final function we simply generate the fibers by first converting the mesh and then generating the fibers


def strocchi_LV():

    # You should probably run this with mpirun
    mesh, ffun = gmsh2dolfin("lifex_fiber_generation_examples/mesh/01_strocchi_LV.msh")

    # These are the actualy markers (but we only supprt one base at the moment)
    original_markers = {"epi": 10, "endo": 20, "aortic_valve": 50, "mitral_valve": 60}
    # So we just use these markers
    markers = {"epi": 10, "lv": 20, "base": 40}

    # And update the markers accordingly
    ffun.array()[ffun.array() == original_markers["aortic_valve"]] = markers["base"]
    ffun.array()[ffun.array() == original_markers["mitral_valve"]] = markers["base"]

    fiber_space = "P_1"

    angles = dict(
        alpha_endo_lv=60,  # Fiber angle on the endocardium
        alpha_epi_lv=-60,  # Fiber angle on the epicardium
        beta_endo_lv=0,  # Sheet angle on the endocardium
        beta_epi_lv=0,
    )

    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh=mesh, fiber_space=fiber_space, ffun=ffun, markers=markers, **angles
    )

    # Save to xdmf
    with dolfin.XDMFFile(mesh.mpi_comm(), "patient_fiber.xdmf") as xdmf:
        xdmf.write(fiber)


def fastl_LA():
    mesh, ffun = gmsh2dolfin("lifex_fiber_generation_examples/mesh/03_fastl_LA.msh")


def hollow_sphere_LA():
    mesh, ffun = gmsh2dolfin(
        "lifex_fiber_generation_examples/mesh/hollow_sphere_LA.msh",
    )


def idealized_LV():
    mesh, ffun = gmsh2dolfin("lifex_fiber_generation_examples/mesh/idealized_LV.msh")


def main():
    strocchi_LV()


if __name__ == "__main__":
    main()
