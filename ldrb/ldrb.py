import logging
from collections import namedtuple
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import dolfin as df
import numpy as np
from dolfin.mesh.meshfunction import MeshFunction

from . import utils

FiberSheetSystem = namedtuple("FiberSheetSystem", "fiber, sheet, sheet_normal")


def laplace(
    mesh: df.Mesh,
    markers: Optional[Dict[str, int]],
    fiber_space: str = "CG_1",
    ffun: Optional[df.MeshFunction] = None,
    use_krylov_solver: bool = False,
    krylov_solver_atol: Optional[float] = None,
    krylov_solver_rtol: Optional[float] = None,
    krylov_solver_max_its: Optional[int] = None,
    verbose: bool = False,
    strict: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Solve the laplace equation and project the gradients
    of the solutions.
    """

    # Create scalar laplacian solutions
    df.info("Calculating scalar fields")
    scalar_solutions = scalar_laplacians(
        mesh=mesh,
        markers=markers,
        ffun=ffun,
        use_krylov_solver=use_krylov_solver,
        krylov_solver_atol=krylov_solver_atol,
        krylov_solver_rtol=krylov_solver_rtol,
        krylov_solver_max_its=krylov_solver_max_its,
        verbose=verbose,
        strict=strict,
    )

    # Create gradients
    df.info("\nCalculating gradients")
    data = project_gradients(
        mesh=mesh,
        fiber_space=fiber_space,
        scalar_solutions=scalar_solutions,
    )

    return data


def standard_dofs(n: int) -> np.ndarray:
    """
    Get the standard list of dofs for a given length
    """

    x_dofs = np.arange(0, 3 * n, 3)
    y_dofs = np.arange(1, 3 * n, 3)
    z_dofs = np.arange(2, 3 * n, 3)
    scalar_dofs = np.arange(0, n)
    return np.stack([x_dofs, y_dofs, z_dofs, scalar_dofs], -1)


def compute_fiber_sheet_system(
    lv_scalar: np.ndarray,
    lv_gradient: np.ndarray,
    epi_scalar: np.ndarray,
    epi_gradient: np.ndarray,
    apex_gradient: np.ndarray,
    dofs: Optional[np.ndarray] = None,
    rv_scalar: Optional[np.ndarray] = None,
    rv_gradient: Optional[np.ndarray] = None,
    lv_rv_scalar: Optional[np.ndarray] = None,
    marker_scalar: Optional[np.ndarray] = None,
    alpha_endo_lv: float = 40,
    alpha_epi_lv: float = -50,
    alpha_endo_rv: Optional[float] = None,
    alpha_epi_rv: Optional[float] = None,
    alpha_endo_sept: Optional[float] = None,
    alpha_epi_sept: Optional[float] = None,
    beta_endo_lv: float = -65,
    beta_epi_lv: float = 25,
    beta_endo_rv: Optional[float] = None,
    beta_epi_rv: Optional[float] = None,
    beta_endo_sept: Optional[float] = None,
    beta_epi_sept: Optional[float] = None,
) -> FiberSheetSystem:
    """
    Compute the fiber-sheets system on all degrees of freedom.
    """
    if dofs is None:
        dofs = standard_dofs(len(lv_scalar))
    if rv_scalar is None:
        rv_scalar = np.zeros_like(lv_scalar)
    if lv_rv_scalar is None:
        lv_rv_scalar = np.zeros_like(lv_scalar)
    if rv_gradient is None:
        rv_gradient = np.zeros_like(lv_gradient)

    alpha_endo_rv = alpha_endo_rv or alpha_endo_lv
    alpha_epi_rv = alpha_epi_rv or alpha_epi_lv
    alpha_endo_sept = alpha_endo_sept or alpha_endo_lv
    alpha_epi_sept = alpha_epi_sept or alpha_epi_lv

    beta_endo_rv = beta_endo_rv or beta_endo_lv
    beta_epi_rv = beta_epi_rv or beta_epi_lv
    beta_endo_sept = beta_endo_sept or beta_endo_lv
    beta_epi_sept = beta_epi_sept or beta_epi_lv

    df.info("Compute fiber-sheet system")
    df.info("Angles: ")
    df.info(
        (
            "alpha: "
            "\n endo_lv: {endo_lv}"
            "\n epi_lv: {epi_lv}"
            "\n endo_septum: {endo_sept}"
            "\n epi_septum: {epi_sept}"
            "\n endo_rv: {endo_rv}"
            "\n epi_rv: {epi_rv}"
            ""
        ).format(
            endo_lv=alpha_endo_lv,
            epi_lv=alpha_epi_lv,
            endo_sept=alpha_endo_sept,
            epi_sept=alpha_epi_sept,
            endo_rv=alpha_endo_rv,
            epi_rv=alpha_epi_rv,
        ),
    )
    df.info(
        (
            "beta: "
            "\n endo_lv: {endo_lv}"
            "\n epi_lv: {epi_lv}"
            "\n endo_septum: {endo_sept}"
            "\n epi_septum: {epi_sept}"
            "\n endo_rv: {endo_rv}"
            "\n epi_rv: {epi_rv}"
            ""
        ).format(
            endo_lv=beta_endo_lv,
            epi_lv=beta_epi_lv,
            endo_sept=beta_endo_sept,
            epi_sept=beta_epi_sept,
            endo_rv=beta_endo_rv,
            epi_rv=beta_epi_rv,
        ),
    )

    f0 = np.zeros_like(lv_gradient)
    s0 = np.zeros_like(lv_gradient)
    n0 = np.zeros_like(lv_gradient)
    if marker_scalar is None:
        marker_scalar = np.zeros_like(lv_scalar)

    tol = 0.1

    from .calculus import _compute_fiber_sheet_system

    _compute_fiber_sheet_system(
        f0,
        s0,
        n0,
        dofs[:, 0],
        dofs[:, 1],
        dofs[:, 2],
        dofs[:, 3],
        lv_scalar,
        rv_scalar,
        epi_scalar,
        lv_rv_scalar,
        lv_gradient,
        rv_gradient,
        epi_gradient,
        apex_gradient,
        marker_scalar,
        alpha_endo_lv,
        alpha_epi_lv,
        alpha_endo_rv,
        alpha_epi_rv,
        alpha_endo_sept,
        alpha_epi_sept,
        beta_endo_lv,
        beta_epi_lv,
        beta_endo_rv,
        beta_epi_rv,
        beta_endo_sept,
        beta_epi_sept,
        tol,
    )

    return FiberSheetSystem(fiber=f0, sheet=s0, sheet_normal=n0)


def dofs_from_function_space(mesh: df.Mesh, fiber_space: str) -> np.ndarray:
    """
    Get the dofs from a function spaces define in the
    fiber_space string.
    """
    Vv = utils.space_from_string(fiber_space, mesh, dim=3)
    V = utils.space_from_string(fiber_space, mesh, dim=1)
    # Get dofs
    dim = Vv.mesh().geometry().dim()
    start, end = Vv.sub(0).dofmap().ownership_range()
    x_dofs = np.arange(0, end - start, dim)
    y_dofs = np.arange(1, end - start, dim)
    z_dofs = np.arange(2, end - start, dim)

    start, end = V.dofmap().ownership_range()
    scalar_dofs = [
        dof
        for dof in range(end - start)
        if V.dofmap().local_to_global_index(dof)
        not in V.dofmap().local_to_global_unowned()
    ]

    return np.stack([x_dofs, y_dofs, z_dofs, scalar_dofs], -1)


def dolfin_ldrb(
    mesh: df.Mesh,
    fiber_space: str = "CG_1",
    ffun: Optional[df.MeshFunction] = None,
    markers: Optional[Dict[str, int]] = None,
    log_level: int = logging.INFO,
    use_krylov_solver: bool = False,
    krylov_solver_atol: Optional[float] = None,
    krylov_solver_rtol: Optional[float] = None,
    krylov_solver_max_its: Optional[int] = None,
    strict: bool = False,
    save_markers: bool = False,
    alpha_endo_lv: float = 40,
    alpha_epi_lv: float = -50,
    alpha_endo_rv: Optional[float] = None,
    alpha_epi_rv: Optional[float] = None,
    alpha_endo_sept: Optional[float] = None,
    alpha_epi_sept: Optional[float] = None,
    beta_endo_lv: float = -65,
    beta_epi_lv: float = 25,
    beta_endo_rv: Optional[float] = None,
    beta_epi_rv: Optional[float] = None,
    beta_endo_sept: Optional[float] = None,
    beta_epi_sept: Optional[float] = None,
):
    r"""
    Create fiber, cross fibers and sheet directions

    Arguments
    ---------
    mesh : dolfin.Mesh
        The mesh
    fiber_space : str
        A string on the form {familiy}_{degree} which
        determines for what space the fibers should be calculated for.
        If not provdied, then a first order Lagrange space will be used,
        i.e Lagrange_1.
    ffun : dolfin.MeshFunctionSizet (optional)
        A facet function containing markers for the boundaries.
        If not provided, the markers stored within the mesh will
        be used.
    markers : dict (optional)
        A dictionary with the markers for the
        different bondaries defined in the facet function
        or within the mesh itself.
        The follwing markers must be provided:
        'base', 'lv', 'epi, 'rv' (optional).
        If the markers are not provided the following default
        vales will be used: base = 10, rv = 20, lv = 30, epi = 40
    log_level : int
        How much to print. DEBUG=10, INFO=20, WARNING=30.
        Default: INFO
    use_krylov_solver: bool
        If True use Krylov solver, by default False
    krylov_solver_atol: float (optional)
        If a Krylov solver is used, this option specifies a
        convergence criterion in terms of the absolute
        residual. Default: 1e-15.
    krylov_solver_rtol: float (optional)
        If a Krylov solver is used, this option specifies a
        convergence criterion in terms of the relative
        residual. Default: 1e-10.
    krylov_solver_max_its: int (optional)
        If a Krylov solver is used, this option specifies the
        maximum number of iterations to perform. Default: 10000.
    strict: bool
        If true raise RuntimeError if solutions does not sum to 1.0
    save_markers: bool
        If true save markings of the geometry. This is nice if you
        want to see that the LV, RV and Septum are marked correctly.
    angles : kwargs
        Keyword arguments with the fiber and sheet angles.
        It is possible to set different angles on the LV,
        RV and septum, however it either the RV or septum
        angles are not provided, then the angles on the LV
        will be used. The default values are taken from the
        original paper, namely

        .. math::

            \alpha_{\text{endo}} &= 40 \\
            \alpha_{\text{epi}} &= -50 \\
            \beta_{\text{endo}} &= -65 \\
            \beta_{\text{epi}} &= 25

        The following keyword arguments are possible:

        alpha_endo_lv : scalar
            Fiber angle at the LV endocardium.
        alpha_epi_lv : scalar
            Fiber angle at the LV epicardium.
        beta_endo_lv : scalar
            Sheet angle at the LV endocardium.
        beta_epi_lv : scalar
            Sheet angle at the LV epicardium.
        alpha_endo_rv : scalar
            Fiber angle at the RV endocardium.
        alpha_epi_rv : scalar
            Fiber angle at the RV epicardium.
        beta_endo_rv : scalar
            Sheet angle at the RV endocardium.
        beta_epi_rv : scalar
            Sheet angle at the RV epicardium.
        alpha_endo_sept : scalar
            Fiber angle at the septum endocardium.
        alpha_epi_sept : scalar
            Fiber angle at the septum epicardium.
        beta_endo_sept : scalar
            Sheet angle at the septum endocardium.
        beta_epi_sept : scalar
            Sheet angle at the septum epicardium.


    """
    df.set_log_level(log_level)

    if not isinstance(mesh, df.Mesh):
        raise TypeError("Expected a dolfin.Mesh as the mesh argument.")

    if ffun is None:
        ffun = df.MeshFunction("size_t", mesh, 2, mesh.domains())
    # Solve the Laplace-Dirichlet problem
    verbose = log_level < logging.INFO
    data = laplace(
        mesh=mesh,
        fiber_space=fiber_space,
        markers=markers,
        ffun=ffun,
        use_krylov_solver=use_krylov_solver,
        krylov_solver_atol=krylov_solver_atol,
        krylov_solver_rtol=krylov_solver_rtol,
        krylov_solver_max_its=krylov_solver_max_its,
        verbose=verbose,
        strict=strict,
    )

    dofs = dofs_from_function_space(mesh, fiber_space)
    marker_scalar = np.zeros_like(data["lv_scalar"])
    system = compute_fiber_sheet_system(
        dofs=dofs,
        marker_scalar=marker_scalar,
        alpha_endo_lv=alpha_endo_lv,
        alpha_epi_lv=alpha_epi_lv,
        alpha_endo_rv=alpha_endo_rv,
        alpha_epi_rv=alpha_epi_rv,
        alpha_endo_sept=alpha_endo_sept,
        alpha_epi_sept=alpha_epi_sept,
        beta_endo_lv=beta_endo_lv,
        beta_epi_lv=beta_epi_lv,
        beta_endo_rv=beta_endo_rv,
        beta_epi_rv=beta_epi_rv,
        beta_endo_sept=beta_endo_sept,
        beta_epi_sept=beta_epi_sept,
        **data,
    )  # type:ignore

    if save_markers:
        Vv = utils.space_from_string(fiber_space, mesh, dim=1)
        markers_fun = df.Function(Vv)
        markers_fun.vector().set_local(marker_scalar)
        markers_fun.vector().apply("insert")
        df.File("markers.pvd") << markers_fun

    df.set_log_level(log_level)
    return fiber_system_to_dolfin(system, mesh, fiber_space)


def fiber_system_to_dolfin(
    system: FiberSheetSystem,
    mesh: df.Mesh,
    fiber_space: str,
) -> FiberSheetSystem:
    """
    Convert fiber-sheet system of numpy arrays to dolfin
    functions.
    """
    Vv = utils.space_from_string(fiber_space, mesh, dim=3)

    f0 = df.Function(Vv)
    f0.vector().set_local(system.fiber)
    f0.vector().apply("insert")
    f0.rename("fiber", "fibers")

    s0 = df.Function(Vv)
    s0.vector().set_local(system.sheet)
    s0.vector().apply("insert")
    s0.rename("sheet", "fibers")

    n0 = df.Function(Vv)
    n0.vector().set_local(system.sheet_normal)
    n0.vector().apply("insert")
    n0.rename("sheet_normal", "fibers")

    return FiberSheetSystem(fiber=f0, sheet=s0, sheet_normal=n0)


def apex_to_base(
    mesh: df.Mesh,
    base_marker: int,
    ffun: df.MeshFunction,
    use_krylov_solver: bool = False,
    krylov_solver_atol: Optional[float] = None,
    krylov_solver_rtol: Optional[float] = None,
    krylov_solver_max_its: Optional[int] = None,
    verbose: bool = False,
):
    """
    Find the apex coordinate and compute the laplace
    equation to find the apex to base solution

    Arguments
    ---------
    mesh : dolfin.Mesh
        The mesh
    base_marker : int
        The marker value for the basal facets
    ffun : dolfin.MeshFunctionSizet (optional)
        A facet function containing markers for the boundaries.
        If not provided, the markers stored within the mesh will
        be used.
    use_krylov_solver: bool
        If True use Krylov solver, by default False
    krylov_solver_atol: float (optional)
        If a Krylov solver is used, this option specifies a
        convergence criterion in terms of the absolute
        residual. Default: 1e-15.
    krylov_solver_rtol: float (optional)
        If a Krylov solver is used, this option specifies a
        convergence criterion in terms of the relative
        residual. Default: 1e-10.
    krylov_solver_max_its: int (optional)
        If a Krylov solver is used, this option specifies the
        maximum number of iterations to perform. Default: 10000.
    verbose: bool
        If true, print more info, by default False
    """
    # Find apex by solving a laplacian with base solution = 0
    # Create Base variational problem
    V = df.FunctionSpace(mesh, "CG", 1)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    a = df.dot(df.grad(u), df.grad(v)) * df.dx
    L = v * df.Constant(1) * df.dx

    apex = df.Function(V)

    base_bc = df.DirichletBC(V, 1, ffun, base_marker, "topological")

    # Solver options
    solver = solve_system(
        a,
        L,
        base_bc,
        apex,
        solver_parameters={"linear_solver": "cg", "preconditioner": "amg"},
        use_krylov_solver=use_krylov_solver,
        krylov_solver_atol=krylov_solver_atol,
        krylov_solver_rtol=krylov_solver_rtol,
        krylov_solver_max_its=krylov_solver_max_its,
        verbose=verbose,
    )

    if utils.DOLFIN_VERSION_MAJOR < 2018:
        dof_x = utils.gather_broadcast(V.tabulate_dof_coordinates()).reshape((-1, 3))
        apex_values = utils.gather_broadcast(apex.vector().get_local())
        ind = apex_values.argmax()
        apex_coord = dof_x[ind]
    else:
        dof_x = V.tabulate_dof_coordinates()
        apex_values = apex.vector().get_local()
        local_max_val = apex_values.max()

        local_apex_coord = dof_x[apex_values.argmax()]
        comm = utils.mpi_comm_world()

        from mpi4py import MPI

        global_max, apex_coord = comm.allreduce(
            sendobj=(local_max_val, local_apex_coord),
            op=MPI.MAXLOC,
        )

    df.info("  Apex coord: ({0:.2f}, {1:.2f}, {2:.2f})".format(*apex_coord))

    # Update rhs
    L = v * df.Constant(0) * df.dx
    apex_domain = df.CompiledSubDomain(
        "near(x[0], {0}) && near(x[1], {1}) && near(x[2], {2})".format(*apex_coord),
    )
    apex_bc = df.DirichletBC(V, 0, apex_domain, "pointwise")

    # Solve the poisson equation
    bcs = [base_bc, apex_bc]
    if solver is not None:
        # Reuse existing solver
        A, b = df.assemble_system(a, L, bcs)
        solver.set_operator(A)
        solver.solve(apex.vector(), b)
    else:
        solve_system(
            a,
            L,
            bcs,
            apex,
            use_krylov_solver=use_krylov_solver,
            krylov_solver_atol=krylov_solver_atol,
            krylov_solver_rtol=krylov_solver_rtol,
            krylov_solver_max_its=krylov_solver_max_its,
            solver_parameters={"linear_solver": "gmres"},
            verbose=verbose,
        )

    return apex


def project_gradients(
    mesh: df.Mesh,
    scalar_solutions: Dict[str, df.Function],
    fiber_space: str = "CG_1",
) -> Dict[str, np.ndarray]:
    """
    Calculate the gradients using projections

    Arguments
    ---------
    mesh : dolfin.Mesh
        The mesh
    fiber_space : str
        A string on the form {familiy}_{degree} which
        determines for what space the fibers should be calculated for.
    scalar_solutions: dict
        A dictionary with the scalar solutions that you
        want to compute the gradients of.
    """
    Vv = utils.space_from_string(fiber_space, mesh, dim=3)
    V = utils.space_from_string(fiber_space, mesh, dim=1)

    data = {}
    V_cg = df.FunctionSpace(mesh, df.VectorElement("Lagrange", mesh.ufl_cell(), 1))
    for case, scalar_solution in scalar_solutions.items():

        scalar_solution_int = df.interpolate(scalar_solution, V)

        if case != "lv_rv":
            gradient_cg = df.project(df.grad(scalar_solution), V_cg, solver_type="cg")
            gradient = df.interpolate(gradient_cg, Vv)

            # Add gradient data
            data[case + "_gradient"] = gradient.vector().get_local()

        # Add scalar data
        if case != "apex":
            data[case + "_scalar"] = scalar_solution_int.vector().get_local()

    # Return data
    return data


def scalar_laplacians(
    mesh: df.Mesh,
    markers: Optional[Dict[str, int]] = None,
    ffun: Optional[MeshFunction] = None,
    use_krylov_solver: bool = False,
    krylov_solver_atol: Optional[float] = None,
    krylov_solver_rtol: Optional[float] = None,
    krylov_solver_max_its: Optional[int] = None,
    verbose: bool = False,
    strict: bool = False,
) -> Dict[str, df.Function]:
    """
    Calculate the laplacians

    Arguments
    ---------
    mesh : dolfin.Mesh
       A dolfin mesh
    markers : dict (optional)
        A dictionary with the markers for the
        different bondaries defined in the facet function
        or within the mesh itself.
        The follwing markers must be provided:
        'base', 'lv', 'epi, 'rv' (optional).
        If the markers are not provided the following default
        vales will be used: base = 10, rv = 20, lv = 30, epi = 40.
    fiber_space : str
        A string on the form {familiy}_{degree} which
        determines for what space the fibers should be calculated for.
    use_krylov_solver: bool
        If True use Krylov solver, by default False
    krylov_solver_atol: float (optional)
        If a Krylov solver is used, this option specifies a
        convergence criterion in terms of the absolute
        residual. Default: 1e-15.
    krylov_solver_rtol: float (optional)
        If a Krylov solver is used, this option specifies a
        convergence criterion in terms of the relative
        residual. Default: 1e-10.
    krylov_solver_max_its: int (optional)
        If a Krylov solver is used, this option specifies the
        maximum number of iterations to perform. Default: 10000.
    verbose: bool
        If true, print more info, by default False
    strict: bool
        If true raise RuntimeError if solutions does not sum to 1.0
    """

    if not isinstance(mesh, df.Mesh):
        raise TypeError("Expected a dolfin.Mesh as the mesh argument.")

    # Init connectivities
    mesh.init(2)
    if ffun is None:
        ffun = df.MeshFunction("size_t", mesh, 2, mesh.domains())

    # Boundary markers, solutions and cases
    cases, boundaries, markers = find_cases_and_boundaries(ffun, markers)
    markers_str = "\n".join(["{}: {}".format(k, v) for k, v in markers.items()])
    df.info(
        ("Compute scalar laplacian solutions with the markers: \n" "{}").format(
            markers_str,
        ),
    )

    check_boundaries_are_marked(
        mesh=mesh,
        ffun=ffun,
        markers=markers,
        boundaries=boundaries,
    )

    # Compte the apex to base solutons
    num_vertices = mesh.num_vertices()
    num_cells = mesh.num_cells()
    if mesh.mpi_comm().size > 1:
        num_vertices = mesh.mpi_comm().allreduce(num_vertices)
        num_cells = mesh.mpi_comm().allreduce(num_cells)
    df.info("  Num vertices: {0}".format(num_vertices))
    df.info("  Num cells: {0}".format(num_cells))

    if "mv" in cases and "av" in cases:
        # Use Doste approach
        pass

    # Else use the Bayer approach
    return bayer(
        cases=cases,
        mesh=mesh,
        markers=markers,
        ffun=ffun,
        verbose=verbose,
        use_krylov_solver=use_krylov_solver,
        strict=strict,
        krylov_solver_atol=krylov_solver_atol,
        krylov_solver_rtol=krylov_solver_rtol,
        krylov_solver_max_its=krylov_solver_max_its,
    )


def find_cases_and_boundaries(
    ffun: df.MeshFunction,
    markers: Optional[Dict[str, int]],
) -> Tuple[List[str], List[str], Dict[str, int]]:

    if markers is None:
        markers = utils.default_markers()

    potential_cases = {"rv", "lv", "epi"}
    potential_boundaries = potential_cases | {"base", "mv", "av"}

    cases = []
    boundaries = []

    for marker in markers:
        msg = f"Unknown marker {marker}. Expected one of {potential_boundaries}"
        if marker not in potential_boundaries:
            logging.warning(msg)
        if marker in potential_boundaries:
            boundaries.append(marker)
        if marker in potential_cases:
            cases.append(marker)

    return cases, boundaries, markers


def check_boundaries_are_marked(
    mesh: df.Mesh,
    ffun: df.MeshFunction,
    markers: Dict[str, int],
    boundaries: List[str],
) -> None:
    # Check that all boundary faces are marked
    num_boundary_facets = df.BoundaryMesh(mesh, "exterior").num_cells()
    if num_boundary_facets != sum(
        [np.sum(ffun.array() == markers[boundary]) for boundary in boundaries],
    ):

        raise RuntimeError(
            (
                "Not all boundary faces are marked correctly. Make sure all "
                "boundary facets are marked as: {}"
                ""
            ).format(", ".join(["{} = {}".format(k, v) for k, v in markers.items()])),
        )


def bayer(
    cases,
    mesh,
    markers,
    ffun,
    verbose: bool,
    use_krylov_solver: bool,
    strict: bool,
    krylov_solver_atol: Optional[float] = None,
    krylov_solver_rtol: Optional[float] = None,
    krylov_solver_max_its: Optional[int] = None,
) -> Dict[str, df.Function]:

    apex = apex_to_base(
        mesh,
        markers["base"],
        ffun,
        use_krylov_solver=use_krylov_solver,
        krylov_solver_atol=krylov_solver_atol,
        krylov_solver_rtol=krylov_solver_rtol,
        krylov_solver_max_its=krylov_solver_max_its,
        verbose=verbose,
    )

    # Find the rest of the laplace soltions
    V = apex.function_space()
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    a = df.dot(df.grad(u), df.grad(v)) * df.dx
    L = v * df.Constant(0) * df.dx

    solutions = dict((what, df.Function(V)) for what in cases)
    solutions["apex"] = apex
    sol = solutions["apex"].vector().copy()
    sol[:] = 0.0

    # Iterate over the three different cases
    df.info("Solving Laplace equation")
    solver_parameters = {"linear_solver": "mumps"}
    if "superlu_dist" in df.linear_solver_methods():
        solver_parameters = {"linear_solver": "superlu_dist"}

    for case in cases:
        df.info(
            " {0} = 1, {1} = 0".format(
                case,
                ", ".join([c for c in cases if c != case]),
            ),
        )
        # Solve linear system
        bcs = [
            df.DirichletBC(
                V,
                1 if what == case else 0,
                ffun,
                markers[what],
                "topological",
            )
            for what in cases
        ]

        solve_system(
            a,
            L,
            bcs,
            solutions[case],
            solver_parameters=solver_parameters,
            use_krylov_solver=use_krylov_solver,
            krylov_solver_atol=krylov_solver_atol,
            krylov_solver_rtol=krylov_solver_rtol,
            krylov_solver_max_its=krylov_solver_max_its,
            verbose=verbose,
        )

        sol += solutions[case].vector()

    if "rv" in cases:
        # Solve one extra equation that is 1 on LV and
        # 0 on the RV
        solutions["lv_rv"] = df.Function(V)
        bcs = [
            df.DirichletBC(
                V,
                1,
                ffun,
                markers["lv"],
                "topological",
            ),
            df.DirichletBC(
                V,
                0,
                ffun,
                markers["rv"],
                "topological",
            ),
        ]

        solve_system(
            a,
            L,
            bcs,
            solutions["lv_rv"],
            solver_parameters=solver_parameters,
            use_krylov_solver=use_krylov_solver,
            krylov_solver_atol=krylov_solver_atol,
            krylov_solver_rtol=krylov_solver_rtol,
            krylov_solver_max_its=krylov_solver_max_its,
            verbose=verbose,
        )

    if not np.all(sol[:] > 0.999):
        msg = "Solution does not always sum to one."
        if strict:
            raise RuntimeError(msg)
        logging.warning(msg)

    # Return the solutions
    return solutions


def solve_krylov(
    a,
    L,
    bcs,
    u: df.Function,
    verbose: bool = False,
    ksp_type="cg",
    ksp_norm_type="unpreconditioned",
    ksp_atol=1e-15,
    ksp_rtol=1e-10,
    ksp_max_it=10000,
    ksp_error_if_not_converged=False,
    pc_type="hypre",
) -> df.PETScKrylovSolver:

    pc_hypre_type = "boomeramg"
    ksp_monitor = verbose
    ksp_view = verbose

    pc_view = verbose
    solver = df.PETScKrylovSolver()
    df.PETScOptions.set("ksp_type", ksp_type)
    df.PETScOptions.set("ksp_norm_type", ksp_norm_type)
    df.PETScOptions.set("ksp_atol", ksp_atol)
    df.PETScOptions.set("ksp_rtol", ksp_rtol)
    df.PETScOptions.set("ksp_max_it", ksp_max_it)
    df.PETScOptions.set("ksp_error_if_not_converged", ksp_error_if_not_converged)
    if ksp_monitor:
        df.PETScOptions.set("ksp_monitor")
    if ksp_view:
        df.PETScOptions.set("ksp_view")
    df.PETScOptions.set("pc_type", pc_type)
    df.PETScOptions.set("pc_hypre_type", pc_hypre_type)
    if pc_view:
        df.PETScOptions.set("pc_view")
    solver.set_from_options()

    A, b = df.assemble_system(a, L, bcs)
    solver.set_operator(A)
    solver.solve(u.vector(), b)
    df.info("Sucessfully solved using Krylov solver")
    return solver


def solve_regular(a, L, bcs, u, solver_parameters):
    if solver_parameters is None:
        solver_parameters = {"linear_solver": "gmres"}
    df.solve(a == L, u, bcs, solver_parameters=solver_parameters)


def solve_system(
    a,
    L,
    bcs,
    u: df.Function,
    solver_parameters: Optional[Dict[str, str]] = None,
    use_krylov_solver: bool = False,
    krylov_solver_atol: Optional[float] = None,
    krylov_solver_rtol: Optional[float] = None,
    krylov_solver_max_its: Optional[int] = None,
    verbose: bool = False,
) -> Optional[df.PETScKrylovSolver]:
    if use_krylov_solver:
        try:
            return solve_krylov(
                a,
                L,
                bcs,
                u,
                ksp_atol=krylov_solver_atol,
                ksp_rtol=krylov_solver_rtol,
                ksp_max_it=krylov_solver_max_its,
                verbose=verbose,
            )
        except Exception:
            df.info("Failed to solve using Krylov solver. Try a regular solve...")
            solve_regular(a, L, bcs, u, solver_parameters)
            return None
    solve_regular(a, L, bcs, u, solver_parameters)
    return None
