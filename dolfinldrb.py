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
# This script uses ldrb (https://github.com/finsberg/ldrb) to generate
# myocardial fibre orientations for meshes in DOLFIN XDMF format.
# This is needed to represent realistic, anisotropic conductivity in
# cardiac electrophysiology simulations.
#
# To run the script, you will need ldrb 2021.0.0, FEniCS 2019.1.0,
# h5py 2.10.0 and mpi4py.
#
# Basic usage is as follows:
#
#  $ ./dolfinldrb.py <surf-xdmf-file> <vol-xdmf-file> <fibre-xdmf-file> <sheet-xdmf-file> <normal-xdmf-file>
#

import logging
import os
import sys
import time

program_name = "dolfinldrb.py"
program_version = "1.0.0"
program_copyright = "Copyright (C) 2022 Simula Research Laboratory"
program_license = (
    "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.")
program_invocation_name = None
program_invocation_short_name = None

def help():
    print("Usage: {} [OPTION..] XDMFSURFFILE XDMFVOLFILE FIBREFILE SHEETFILE NORMALFILE"
          .format(program_name))
    print("")
    print(" Rule-based generation of myocardial fibre orientations for cardiac meshes.")
    print("")
    print(" The following arguments are mandatory:")
    print("")
    print("  XDMFSURFFILE\tpath to DOLFIN XDMF file for the surface mesh")
    print("  XDMFVOLFILE\tpath to DOLFIN XDMF file for the volume mesh")
    print("  FIBREFILE\tpath to DOLFIN XDMF output file for fibre directions")
    print("  SHEETFILE\tpath to DOLFIN XDMF output file for sheet directions")
    print("  NORMALFILE\tpath to DOLFIN XDMF output file for sheet-normal directions")
    print("")
    print(" Options for subdomain markers:")
    print("  --base-id=N\t\tmarker for biventricular base (default: 1)")
    print("  --epi-id=N\t\tmarker for epicardium (default: 2)")
    print("  --left-endo-id=N\tmarker for left ventricle endocardium (default: 3)")
    print("  --right-endo-id=N\tmarker for right ventricle endocardium (default: 4)")
    print("")
    print(" Fibre and sheet angles (in degrees):")
    print("  --left-endo-alpha=DEG\t\tleft ventricle endocardium fibre angle (default: 30)")
    print("  --left-epi-alpha=DEG\t\tleft ventricle epicardium fibre angle (default: -30)")
    print("  --left-endo-beta=DEG\t\tleft ventricle endocardium sheet angle (default: 0)")
    print("  --left-epi-beta=DEG\t\tleft ventricle epicardium sheet angle (default: 0)")
    print("  --right-endo-alpha=DEG\tright ventricle endocardium fibre angle (default: 80)")
    print("  --right-epi-alpha=DEG\t\tright ventricle epicardium fibre angle (default: -80)")
    print("  --right-endo-beta=DEG\t\tright ventricle endocardium sheet angle (default: 0)")
    print("  --right-epi-beta=DEG\t\tright ventricle epicardium sheet angle (default: 0)")
    print("  --sept-endo-alpha=DEG\t\tseptum endocardium fibre angle (default: 60)")
    print("  --sept-epi-alpha=DEG\t\tseptum epicardium fibre angle (default: -60)")
    print("  --sept-endo-beta=DEG\t\tseptum endocardium sheet angle (default: 0)")
    print("  --sept-epi-beta=DEG\t\tseptum epicardium sheet angle (default: 0)")
    print("")
    print(" Solver options:")
    print("  --element-type=ELEMENT\tfinite element type for representing fibre ")
    print("\t\t\t\torientations. Choose one of: CG_1 or DG_0.")
    print("\t\t\t\tThe default is CG_1.")
    print("  --krylov-solver\t\tUse a Krylov solver to solve the linear equation systems.")
    print("\t\t\t\tThe conjugate gradient method is used together with")
    print("\t\t\t\tan algebraic multigrid (BoomerAMG) preconditioner from Hypre.")
    print("  --krylov-solver-atol=TOL\tconvergence tolerance for the residual (default: 1e-15)")
    print("  --krylov-solver-rtol=TOL\tconvergence tolerance for the relative residual (default: 1e-10)")
    print("  --krylov-solver-max-its=N\tmaximum number of iterations (default: 10000)")
    print("")
    print(" Other options are:")
    print("  -v, --verbose\t\tbe more verbose")
    print("")
    print("  -h, --help\t\tdisplay this help and exit")
    print("  --version\t\tdisplay version information and exit")
    print("")
    print("Report bugs to: <james@simula.no>")

def main(args):
    if '-h' in args or '--help' in args:
        help()
        sys.exit(0)

    if '--version' in args:
        print("{} {}".format(program_name, program_version))
        print(program_copyright)
        print(program_license)
        sys.exit(0)

    # Default program options
    xdmfsurfpath = None
    xdmfvolpath = None
    fibrepath = None
    sheetpath = None
    normalpath = None
    base_id = 1
    epi_id = 2
    left_endo_id  =  3
    right_endo_id  =  4
    left_endo_alpha = 30
    left_epi_alpha = -30
    left_endo_beta = 0
    left_epi_beta = 0
    right_endo_alpha = 80
    right_epi_alpha = -80
    right_endo_beta = 0
    right_epi_beta = 0
    sept_endo_alpha = 60
    sept_epi_alpha = -60
    sept_endo_beta = 0
    sept_epi_beta = 0
    element_type = 'CG_1'
    krylov_solver = False
    krylov_solver_atol = 1e-15
    krylov_solver_rtol = 1e-10
    krylov_solver_max_its = 10000
    verbose = 0

    element_types = ['CG_1', 'DG_0']

    # Parse program options
    i = 0
    while i < len(args):
        if args[i] == '--verbose':
            verbose = verbose + 1
            i = i+1
            continue

        # Parse mesh markers
        try:
            if args[i].startswith('--base-id='):
                base_id = int(args[i].split('--base-id=')[1])
                i = i+1
                continue
            elif args[i] == '--base-id':
                base_id = int(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--epi-id='):
                epi_id = int(args[i].split('--epi-id=')[1])
                i = i+1
                continue
            elif args[i] == '--epi-id':
                epi_id = int(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--left-endo-id='):
                left_endo_id = int(args[i].split('--left-endo-id=')[1])
                i = i+1
                continue
            elif args[i] == '--left-endo-id':
                left_endo_id = int(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--right-endo-id='):
                right_endo_id = int(args[i].split('--right-endo-id=')[1])
                i = i+1
                continue
            elif args[i] == '--right-endo-id':
                right_endo_id = int(args[i+1])
                i = i+2
                continue
        except:
            print("{}: Invalid argument {}"
                  .format(program_invocation_short_name, args[i]),
                  file=sys.stderr)
            sys.exit(1)

        # Parse angles
        try:
            if args[i].startswith('--left-endo-alpha='):
                left_endo_alpha = float(args[i].split('--left-endo-alpha=')[1])
                i = i+1
                continue
            elif args[i] == '--left-endo-alpha':
                left_endo_alpha = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--left-epi-alpha='):
                left_epi_alpha = float(args[i].split('--left-epi-alpha=')[1])
                i = i+1
                continue
            elif args[i] == '--left-epi-alpha':
                left_epi_alpha = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--left-endo-beta='):
                left_endo_beta = float(args[i].split('--left-endo-beta=')[1])
                i = i+1
                continue
            elif args[i] == '--left-endo-beta':
                left_endo_beta = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--left-epi-beta='):
                left_epi_beta = float(args[i].split('--left-epi-beta=')[1])
                i = i+1
                continue
            elif args[i] == '--left-epi-beta':
                left_epi_beta = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--right-endo-alpha='):
                right_endo_alpha = float(args[i].split('--right-endo-alpha=')[1])
                i = i+1
                continue
            elif args[i] == '--right-endo-alpha':
                right_endo_alpha = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--right-epi-alpha='):
                right_epi_alpha = float(args[i].split('--right-epi-alpha=')[1])
                i = i+1
                continue
            elif args[i] == '--right-epi-alpha':
                right_epi_alpha = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--right-endo-beta='):
                right_endo_beta = float(args[i].split('--right-endo-beta=')[1])
                i = i+1
                continue
            elif args[i] == '--right-endo-beta':
                right_endo_beta = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--right-epi-beta='):
                right_epi_beta = float(args[i].split('--right-epi-beta=')[1])
                i = i+1
                continue
            elif args[i] == '--right-epi-beta':
                right_epi_beta = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--sept-endo-alpha='):
                sept_endo_alpha = float(args[i].split('--sept-endo-alpha=')[1])
                i = i+1
                continue
            elif args[i] == '--sept-endo-alpha':
                sept_endo_alpha = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--sept-epi-alpha='):
                sept_epi_alpha = float(args[i].split('--sept-epi-alpha=')[1])
                i = i+1
                continue
            elif args[i] == '--sept-epi-alpha':
                sept_epi_alpha = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--sept-endo-beta='):
                sept_endo_beta = float(args[i].split('--sept-endo-beta=')[1])
                i = i+1
                continue
            elif args[i] == '--sept-endo-beta':
                sept_endo_beta = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--sept-epi-beta='):
                sept_epi_beta = float(args[i].split('--sept-epi-beta=')[1])
                i = i+1
                continue
            elif args[i] == '--sept-epi-beta':
                sept_epi_beta = float(args[i+1])
                i = i+2
                continue
        except:
            print("{}: Invalid argument {}"
                  .format(program_invocation_short_name, args[i]),
                  file=sys.stderr)
            sys.exit(1)

        # Parse element type
        try:
            if args[i].startswith('--element-type='):
                element_type = args[i].split('--element-type=')[1]
                if element_type not in element_types:
                    raise ValueError
                i = i+1
                continue
            elif args[i] == '--element-type':
                element_type = args[i+1]
                if element_type not in element_types:
                    raise ValueError
                i = i+2
                continue
        except:
            print("{}: Invalid argument {}"
                  .format(program_invocation_short_name, args[i]),
                  file=sys.stderr)
            sys.exit(1)

        # Parse solver options
        try:
            if args[i] == '--krylov-solver':
                krylov_solver = True
                i = i+1
                continue
            if args[i].startswith('--krylov-solver-atol='):
                krylov_solver_atol = float(args[i].split('--krylov-solver-atol=')[1])
                i = i+1
                continue
            elif args[i] == '--krylov-solver-atol':
                krylov_solver_atol = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--krylov-solver-rtol='):
                krylov_solver_rtol = float(args[i].split('--krylov-solver-rtol=')[1])
                i = i+1
                continue
            elif args[i] == '--krylov-solver-rtol':
                krylov_solver_rtol = float(args[i+1])
                i = i+2
                continue
            if args[i].startswith('--krylov-solver-max-its='):
                krylov_solver_max_its = float(args[i].split('--krylov-solver-max-its=')[1])
                i = i+1
                continue
            elif args[i] == '--krylov-solver-max-its':
                krylov_solver_max_its = float(args[i+1])
                i = i+2
                continue
        except:
            print("{}: Invalid argument {}"
                  .format(program_invocation_short_name, args[i]),
                  file=sys.stderr)
            sys.exit(1)

        # Parse mandatory arguments
        if xdmfsurfpath is None:
            xdmfsurfpath = args[i]
            i = i+1
            continue
        elif xdmfvolpath is None:
            xdmfvolpath = args[i]
            i = i+1
            continue
        elif fibrepath is None:
            fibrepath = args[i]
            i = i+1
            continue
        elif sheetpath is None:
            sheetpath = args[i]
            i = i+1
            continue
        elif normalpath is None:
            normalpath = args[i]
            i = i+1
            continue
        else:
            print("{}: Invalid argument {}"
                  .format(program_invocation_short_name, args[i]),
                  file=sys.stderr)
            sys.exit(1)

        print("{}: Invalid argument {}"
              .format(program_invocation_short_name, args[i]),
              file=sys.stderr)
        sys.exit(1)

    # Check the mandatory arguments
    if xdmfsurfpath is None:
        print("{}: please specify a DOLFIN XDMF for "
              "the surface mesh".format(
                  program_invocation_short_name),
                  file=sys.stderr)
        sys.exit(1)
    elif xdmfvolpath is None:
        print("{}: please specify a DOLFIN XDMF for "
              "the volume mesh".format(
                  program_invocation_short_name),
                  file=sys.stderr)
        sys.exit(1)
    elif fibrepath is None:
        print("{}: please specify a DOLFIN XDMF output file "
              "for writing fibre directions".format(
                  program_invocation_short_name),
                  file=sys.stderr)
        sys.exit(1)
    elif sheetpath is None:
        print("{}: please specify a DOLFIN XDMF output file "
              "for writing sheet directions".format(
                  program_invocation_short_name),
                  file=sys.stderr)
        sys.exit(1)
    elif normalpath is None:
        print("{}: please specify a DOLFIN XDMF output file "
              "for writing sheet normal directions".format(
                  program_invocation_short_name),
                  file=sys.stderr)
        sys.exit(1)

    h5surfpath = os.path.splitext(xdmfsurfpath)[0] + '.h5'
    h5volpath = os.path.splitext(xdmfvolpath)[0] + '.h5'
    h5fibrepath = os.path.splitext(fibrepath)[0] + '.h5'
    h5sheetpath = os.path.splitext(sheetpath)[0] + '.h5'
    h5normalpath = os.path.splitext(normalpath)[0] + '.h5'

    marker_ids = {
        'base': base_id,
        'epi': epi_id,
        'lv': left_endo_id,
        'rv': right_endo_id}
    angles = {
        'alpha_endo_lv': left_endo_alpha,
        'alpha_epi_lv': left_epi_alpha,
        'beta_endo_lv': left_endo_beta,
        'beta_epi_lv': left_epi_beta,
        'alpha_endo_rv': right_endo_alpha,
        'alpha_epi_rv': right_epi_alpha,
        'beta_endo_rv': right_endo_beta,
        'beta_epi_rv': right_epi_beta,
        'alpha_endo_sept': sept_endo_alpha,
        'alpha_epi_sept': sept_epi_alpha,
        'beta_endo_sept': sept_endo_beta,
        'beta_epi_sept': sept_epi_beta}

    try:
        import ldrb
    except:
        print("{}: ldrb not found"
              .format(program_invocation_short_name),
              file=sys.stderr)
        sys.exit(1)

    try:
        import dolfin
    except:
        print("{}: dolfin not found"
              .format(program_invocation_short_name),
              file=sys.stderr)
        sys.exit(1)

    try:
        import h5py
    except:
        print("{}: h5py not found"
              .format(program_invocation_short_name),
              file=sys.stderr)
        sys.exit(1)

    rank = 0
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except:
        print("{}: warning: mpi4py not found"
              .format(program_invocation_short_name),
              file=sys.stderr)

    if verbose > 1 and rank == 0:
        dolfin.set_log_level(logging.DEBUG)

    if verbose and rank == 0:
        print("Reading volume mesh: ", end='', flush=True, file=sys.stderr)
    t0 = time.time()
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(xdmfvolpath) as f:
        f.read(mesh)
    if verbose and rank == 0:
        t1 = time.time()
        size = None
        try: size = os.path.getsize(h5volpath)
        except: pass
        if size is not None:
            print("{:.6f} seconds ({:.1f} MB/s)"
                  .format(t1-t0, 1e-6*size / (t1-t0)), file=sys.stderr)
        else: print("{:.6f} seconds".format(t1-t0), file=sys.stderr)

    if verbose and rank == 0:
        print("Reading surface mesh: ", end='', flush=True, file=sys.stderr)
    t0 = time.time()
    surface_marker_collection = dolfin.MeshValueCollection("size_t", mesh, 2)
    with dolfin.XDMFFile(xdmfsurfpath) as f:
        f.read(surface_marker_collection, "markers")
    surface_markers = dolfin.MeshFunction("size_t", mesh, surface_marker_collection)
    if verbose and rank == 0:
        t1 = time.time()
        size = None
        try: size = os.path.getsize(h5surfpath)
        except: pass
        if size is not None:
            print("{:.6f} seconds ({:.1f} MB/s)"
                  .format(t1-t0, 1e-6*size / (t1-t0)), file=sys.stderr)
        else: print("{:.6f} seconds".format(t1-t0), file=sys.stderr)

    if verbose and rank == 0:
        print("Computing fibre orientations: ", end='', flush=True, file=sys.stderr)
    t0 = time.time()
    fibre, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh=mesh,
        fiber_space=element_type,
        ffun=surface_markers,
        markers=marker_ids,
        use_krylov_solver=krylov_solver,
        krylov_solver_atol=krylov_solver_atol,
        krylov_solver_rtol=krylov_solver_rtol,
        krylov_solver_max_its=krylov_solver_max_its,
        log_level=logging.INFO if verbose > 2 and rank == 0 else logging.ERROR,
        **angles)
    if verbose and rank == 0:
        t1 = time.time()
        print("{:.6f} seconds".format(t1-t0), file=sys.stderr)

    if verbose and rank == 0:
        print("Writing fibre directions: ", end='', flush=True, file=sys.stderr)
    t0 = time.time()
    with dolfin.XDMFFile(fibrepath) as f:
        f.write(fibre)
    if verbose and rank == 0:
        t1 = time.time()
        size = None
        try: size = os.path.getsize(h5fibrepath)
        except: pass
        if size is not None:
            print("{:.6f} seconds ({:.1f} MB/s)"
                  .format(t1-t0, 1e-6*size / (t1-t0)),
                  file=sys.stderr)
        else:
            print("{:.6f} seconds".format(t1-t0), file=sys.stderr)

    if verbose and rank == 0:
        print("Writing sheet directions: ", end='', flush=True, file=sys.stderr)
    t0 = time.time()
    with dolfin.XDMFFile(sheetpath) as f:
        f.write(sheet)
    if verbose and rank == 0:
        t1 = time.time()
        size = None
        try: size = os.path.getsize(h5sheetpath)
        except: pass
        if size is not None:
            print("{:.6f} seconds ({:.1f} MB/s)"
                  .format(t1-t0, 1e-6*size / (t1-t0)),
                  file=sys.stderr)
        else:
            print("{:.6f} seconds".format(t1-t0), file=sys.stderr)

    if verbose and rank == 0:
        print("Writing sheet normal directions: ", end='', flush=True, file=sys.stderr)
    t0 = time.time()
    with dolfin.XDMFFile(normalpath) as f:
        f.write(sheet_normal)
    if verbose and rank == 0:
        t1 = time.time()
        size = None
        try: size = os.path.getsize(h5normalpath)
        except: pass
        if size is not None:
            print("{:.6f} seconds ({:.1f} MB/s)"
                  .format(t1-t0, 1e-6*size / (t1-t0)),
                  file=sys.stderr)
        else:
            print("{:.6f} seconds".format(t1-t0), file=sys.stderr)

    sys.exit(0)


if __name__ == '__main__':
    program_invocation_name = sys.argv[0]
    program_invocation_short_name = program_invocation_name.split('/')[-1]
    main(sys.argv[1:])
