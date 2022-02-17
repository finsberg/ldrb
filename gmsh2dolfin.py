#!/usr/bin/env python3
# Laplace-Dirichlet Rule-Based (LDRB) myocardial fiber orientations
#
# Copyright (C) 2022 Simula Research Laboratory
#
# Authors:
#   - James D. Trotter <james@simula.no>
#   - Henrik Finsberg <henriknf@simula.no>
#
# Last modified: 2022-02-17
#
# This script uses meshio (https://github.com/nschloe/meshio) to
# convert meshes in Gmsh format to DOLFIN format. This is needed to
# use the meshes with the FEniCS PDE solver framework, and to use the
# ldrb tool to generate myocardial fibre orientations.
#
# To run the script, you will need both meshio and h5py.
#
# For more information on using meshio to convert from Gmsh to DOLFIN
# meshes, see http://jsdokken.com/converted_files/tutorial_pygmsh.html
#
# Basic usage is as follows:
#
#  $ ./gmsh2dolfin.py <msh-file> <surface-xdmf-file> <volume-xdmf-file>
#
import os
import sys
import time

program_name = "gmsh2dolfin.py"
program_version = "1.0.0"
program_copyright = "Copyright (C) 2021 James D. Trotter"
program_license = (
    "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law."
)
program_invocation_name = None
program_invocation_short_name = None


def create_mesh(meshio, mesh, cell_type):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(
        points=mesh.points, cells={cell_type: cells}, cell_data={"markers": [cell_data]},
    )
    return out_mesh


def help():
    print("Usage: {} [OPTION..] MSHFILE XDMFSURFFILE XDMFVOLFILE".format(program_name))
    print("")
    print(" Convert Gmsh's MSH4.1 ASCII files to DOLFIN XDMF format.")
    print("")
    print(" The following arguments are mandatory:")
    print("")
    print("  MSHFILE\tpath to a Gmsh MSH file")
    print("  XDMFSURFFILE\tpath to output a DOLFIN XDMF file for the surface mesh")
    print("  XDMFVOLFILE\tpath to output a DOLFIN XDMF file for the volume mesh")
    print("")
    print(" Other options are:")
    print("  -v, --verbose		be more verbose")
    print("")
    print("  -h, --help		display this help and exit")
    print("  --version		display version information and exit")
    print("")
    print("Report bugs to: <james@simula.no>")


def main(args):
    if "-h" in args or "--help" in args:
        help()
        sys.exit(0)

    if "--version" in args:
        print("{} {}".format(program_name, program_version))
        print(program_copyright)
        print(program_license)
        sys.exit(0)

    verbose = False
    if "--verbose" in args:
        verbose = True
        args.remove("--verbose")

    if len(args) < 1:
        print(
            "{}: please specify a Gmsh MSH file".format(program_invocation_short_name),
            file=sys.stderr,
        )
        sys.exit(1)
    mshpath = args[0]

    if len(args) < 2:
        print(
            "{}: please specify a DOLFIN XDMF output file for writing the mesh surface".format(
                program_invocation_short_name,
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    xdmfsurfpath = args[1]
    h5surfpath = os.path.splitext(xdmfsurfpath)[0] + ".h5"

    if len(args) < 3:
        print(
            "{}: please specify a DOLFIN XDMF output file for writing the mesh volume".format(
                program_invocation_short_name,
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    xdmfvolpath = args[2]
    h5volpath = os.path.splitext(xdmfvolpath)[0] + ".h5"

    if len(args) > 3:
        print(
            "{}: Invalid argument {}".format(program_invocation_short_name, args[3]),
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import meshio
    except:
        print(
            "{}: meshio not found".format(program_invocation_short_name),
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import h5py
    except:
        print(
            "{}: h5py not found".format(program_invocation_short_name), file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print("Reading mesh: ", end="", flush=True, file=sys.stderr)
    t0 = time.time()
    msh = meshio.gmsh.read(mshpath)
    if verbose:
        t1 = time.time()
        size = None
        try:
            size = os.path.getsize(mshpath)
        except:
            pass
        if size is not None:
            print(
                "{:.6f} seconds ({:.1f} MB/s)".format(t1 - t0, 1e-6 * size / (t1 - t0)),
                file=sys.stderr,
            )
        else:
            print("{:.6f} seconds".format(t1 - t0), file=sys.stderr)

    if verbose:
        print("Creating surface mesh: ", end="", flush=True, file=sys.stderr)
    t0 = time.time()
    surface = create_mesh(meshio, msh, "triangle")
    if verbose:
        t1 = time.time()
        print("{:.6f} seconds".format(t1 - t0), file=sys.stderr)

    if verbose:
        print("Writing surface mesh: ", end="", flush=True, file=sys.stderr)
    t0 = time.time()
    meshio.write(xdmfsurfpath, surface)
    if verbose:
        t1 = time.time()
        size = None
        try:
            size = os.path.getsize(h5surfpath)
        except:
            pass
        if size is not None:
            print(
                "{:.6f} seconds ({:.1f} MB/s)".format(t1 - t0, 1e-6 * size / (t1 - t0)),
                file=sys.stderr,
            )
        else:
            print("{:.6f} seconds".format(t1 - t0), file=sys.stderr)

    if verbose:
        print("Creating volume mesh: ", end="", flush=True, file=sys.stderr)
    t0 = time.time()
    volume = create_mesh(meshio, msh, "tetra")
    if verbose:
        t1 = time.time()
        print("{:.6f} seconds".format(t1 - t0), file=sys.stderr)

    if verbose:
        print("Writing volume mesh: ", end="", flush=True, file=sys.stderr)
    t0 = time.time()
    meshio.write(xdmfvolpath, volume)
    if verbose:
        t1 = time.time()
        size = None
        try:
            size = os.path.getsize(h5volpath)
        except:
            pass
        if size is not None:
            print(
                "{:.6f} seconds ({:.1f} MB/s)".format(t1 - t0, 1e-6 * size / (t1 - t0)),
                file=sys.stderr,
            )
        else:
            print("{:.6f} seconds".format(t1 - t0), file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    program_invocation_name = sys.argv[0]
    program_invocation_short_name = program_invocation_name.split("/")[-1]
    main(sys.argv[1:])
