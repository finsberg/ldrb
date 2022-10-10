from collections import namedtuple
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import dolfin as df
import numpy as np
import ufl


def has_mshr():
    try:
        import mshr  # noqa: F401
    except ImportError:
        return False
    return True


def has_meshio() -> bool:
    try:
        import meshio  # noqa: F401
    except ImportError:
        return False
    return True


Geometry = namedtuple("Geometry", "mesh, ffun, markers")

if df.__version__.startswith("20"):
    # Year based versioning
    DOLFIN_VERSION_MAJOR = float(df.__version__.split(".")[0])
else:
    try:
        DOLFIN_VERSION_MAJOR = float(".".join(df.__version__.split(".")[:2]))
    except Exception:
        DOLFIN_VERSION_MAJOR = 1.6


def create_mesh(mesh, cell_type):
    if not has_meshio():
        raise ImportError("Please install 'meshio' - python3 -m pip install meshio")
    import meshio

    # From http://jsdokken.com/converted_files/tutorial_pygmsh.html
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]},
    )
    return out_mesh


def read_meshfunction(fname, obj):
    with df.XDMFFile(Path(fname).as_posix()) as f:
        f.read(obj, "name_to_read")


# Next we define a function for converting the gmsh file into an xdmf file. This function returns the mesh, as well as the facet function (which contains the markers for all the facets) as well as a dictionary containing the available markers.


def convert_msh_to_xdmf(msh_file, triangle_mesh_name, tetra_mesh_name):
    if mpi_comm_world().size > 1:
        msg = (
            "Cannot convert from gmsh to xdmf in parallel. "
            "Please run the conversion in serial and then run the "
            "ldrb algorithm in parallell afterwards. "
        )
        raise RuntimeError(msg)
    if not has_meshio():
        raise ImportError("Please install 'meshio' - python3 -m pip install meshio")
    import meshio

    # Convert to dolfin
    msh = meshio.gmsh.read(msh_file)
    triangle_mesh = create_mesh(msh, "triangle")
    tetra_mesh = create_mesh(msh, "tetra")
    meshio.write(triangle_mesh_name, triangle_mesh)
    meshio.write(
        tetra_mesh_name,
        tetra_mesh,
    )
    markers = msh.field_data
    return markers


def gmsh2dolfin(msh_file, unlink: bool = True):

    name = Path(msh_file).stem

    triangle_mesh_name = Path(f"triangle_mesh_{name}.xdmf")
    tetra_mesh_name = Path(f"mesh_{name}.xdmf")

    markers = {}
    if not (triangle_mesh_name.is_file() and tetra_mesh_name.is_file()):
        markers = convert_msh_to_xdmf(msh_file, triangle_mesh_name, tetra_mesh_name)

    mesh = df.Mesh()

    with df.XDMFFile(tetra_mesh_name.as_posix()) as infile:
        infile.read(mesh)

    cfun = df.MeshFunction("size_t", mesh, 3)
    read_meshfunction(tetra_mesh_name, cfun)

    ffun_val = df.MeshValueCollection("size_t", mesh, 2)
    read_meshfunction(triangle_mesh_name, ffun_val)
    ffun = df.MeshFunction("size_t", mesh, ffun_val)

    if unlink:
        triangle_mesh_name.unlink()
        triangle_mesh_name.with_suffix(".h5").unlink()
        tetra_mesh_name.unlink()
        tetra_mesh_name.with_suffix(".h5").unlink()

    return mesh, ffun, markers


def mpi_comm_world():
    if DOLFIN_VERSION_MAJOR >= 2018:
        return df.MPI.comm_world
    return df.mpi_comm_world()


def value_size(obj: ufl.Coefficient) -> Union[List[int], int]:
    if DOLFIN_VERSION_MAJOR >= 2018:
        value_shape = obj.value_shape()
        if len(value_shape) == 0:
            return 1
        return [0]
    return obj.value_size()


def mark_biv_mesh(
    mesh: df.Mesh,
    ffun: Optional[df.MeshFunction] = None,
    markers: Optional[Dict[str, int]] = None,
    tol: float = 0.01,
    values: Dict[str, int] = {"lv": 0, "septum": 1, "rv": 2},
) -> df.MeshFunction:

    from .ldrb import scalar_laplacians

    scalars = scalar_laplacians(mesh=mesh, ffun=ffun, markers=markers)

    for cell in df.cells(mesh):

        lv = scalars["lv"](cell.midpoint())
        rv = scalars["rv"](cell.midpoint())
        epi = scalars["epi"](cell.midpoint())

        print(cell.index(), "lv = {}, rv = {}".format(lv, rv))

        if (lv > tol or epi > 1 - tol) and rv < tol:
            print("LV")
            value = values["lv"]
            if lv < tol and rv > lv:
                value = values["rv"]
        elif (rv > tol or epi > 1 - tol) and lv < tol:
            print("RV")
            value = values["rv"]
        else:
            print("SEPTUM")
            value = values["septum"]

        mesh.domains().set_marker((cell.index(), value), 3)

    sfun = df.MeshFunction("size_t", mesh, 3, mesh.domains())
    return sfun


def mark_facets(mesh: df.Mesh, ffun: df.MeshFunction):
    """
    Mark mesh according to facet function
    """
    for facet in df.facets(mesh):

        if ffun[facet] == 2**64 - 1:
            ffun[facet] = 0

        mesh.domains().set_marker((facet.index(), ffun[facet]), 2)


def default_markers() -> Dict[str, int]:
    """
    Default markers for the mesh boundaries
    """
    return dict(base=10, rv=20, lv=30, epi=40)


def create_lv_mesh(
    N: int = 13,
    a_endo: float = 1.5,
    b_endo: float = 0.5,
    c_endo: float = 0.5,
    a_epi: float = 2.0,
    b_epi: float = 1.0,
    c_epi: float = 1.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    base_x: float = 0.0,
    markers: Optional[Dict[str, int]] = None,
) -> Geometry:
    r"""
    Create an lv-ellipsoidal mesh.

    An ellipsoid is given by the equation

    .. math::

        \frac{x^2}{a} + \frac{y^2}{b} + \frac{z^2}{c} = 1

    We create two ellipsoids, one for the endocardium and one
    for the epicardium and subtract them and then cut the base.
    For simplicity we assume that the longitudinal axis is in
    in :math:`x`-direction and as default the base is located
    at the :math:`x=0` plane.
    """
    if not has_mshr():
        raise RuntimeError("Cannot create mesh with mshr. It is not installed")

    import mshr

    df.info("Creating LV mesh. This could take some time...")
    # LV
    # The center of the LV ellipsoid
    center = df.Point(*center)

    # Markers
    if markers is None:
        markers = default_markers()
        markers.pop("rv", None)

    class Endo(df.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] - center.x()) ** 2 / a_endo**2 + (
                x[1] - center.y()
            ) ** 2 / b_endo**2 + (
                x[2] - center.z()
            ) ** 2 / c_endo**2 - 1.1 < df.DOLFIN_EPS and on_boundary

    class Base(df.SubDomain):
        def inside(self, x, on_boundary):
            return x[0] - base_x < df.DOLFIN_EPS and on_boundary

    class Epi(df.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] - center.x()) ** 2 / a_epi**2 + (
                x[1] - center.y()
            ) ** 2 / b_epi**2 + (
                x[2] - center.z()
            ) ** 2 / c_epi**2 - 0.9 > df.DOLFIN_EPS and on_boundary

    # The plane cutting the base
    diam = -2 * a_epi
    box = mshr.Box(df.Point(base_x, -diam, -diam), df.Point(diam, diam, diam))

    # LV epicardium
    el_lv = mshr.Ellipsoid(center, a_epi, b_epi, c_epi)
    # LV endocardium
    el_lv_endo = mshr.Ellipsoid(center, a_endo, b_endo, c_endo)

    # LV geometry (subtract the smallest ellipsoid)
    lv = el_lv - el_lv_endo

    # LV geometry
    m = lv - box

    # Create mesh
    print("Generate mesh. This can take some time...")
    mesh = mshr.generate_mesh(m, N)

    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    endo = Endo()
    endo.mark(ffun, markers["lv"])

    base = Base()
    base.mark(ffun, markers["base"])

    epi = Epi()
    epi.mark(ffun, markers["epi"])

    return Geometry(mesh=mesh, ffun=ffun, markers=markers)


def create_biv_mesh(
    N: int = 13,
    a_endo_lv: float = 1.5,
    b_endo_lv: float = 0.5,
    c_endo_lv: float = 0.5,
    a_epi_lv: float = 2.0,
    b_epi_lv: float = 1.0,
    c_epi_lv: float = 1.0,
    center_lv: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    a_endo_rv: float = 1.45,
    b_endo_rv: float = 1.25,
    c_endo_rv: float = 0.75,
    a_epi_rv: float = 1.75,
    b_epi_rv: float = 1.5,
    c_epi_rv: float = 1.0,
    center_rv: Tuple[float, float, float] = (0.0, 0.5, 0.0),
    base_x: float = 0.0,
    markers: Optional[Dict[str, int]] = None,
) -> Geometry:
    r"""
    Create an biv-ellipsoidal mesh.

    An ellipsoid is given by the equation

    .. math::

        \frac{x^2}{a} + \frac{y^2}{b} + \frac{z^2}{c} = 1

    We create three ellipsoids, one for the LV and RV endocardium
    and one for the epicardium and subtract them and then cut the base.
    For simplicity we assume that the longitudinal axis is in
    in :math:`x`-direction and as default the base is located
    at the :math:`x=0` plane.

    """
    if not has_mshr():
        raise RuntimeError("Cannot create mesh with mshr. It is not installed")

    import mshr

    df.info("Creating BiV mesh. This could take some time...")

    # The center of the LV ellipsoid
    center_lv = df.Point(*center_lv)
    # The center of the RV ellipsoid (slightly translated)
    center_rv = df.Point(*center_rv)

    # Markers
    if markers is None:
        markers = default_markers()

    class EndoLV(df.SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] - center_lv.x()) ** 2 / a_endo_lv**2 + (
                x[1] - center_lv.y()
            ) ** 2 / b_endo_lv**2 + (
                x[2] - center_lv.z()
            ) ** 2 / c_endo_lv**2 - 1 < df.DOLFIN_EPS and on_boundary

    class Base(df.SubDomain):
        def inside(self, x, on_boundary):
            return x[0] - base_x < df.DOLFIN_EPS and on_boundary

    class EndoRV(df.SubDomain):
        def inside(self, x, on_boundary):
            return (
                (x[0] - center_rv.x()) ** 2 / a_endo_rv**2
                + (x[1] - center_rv.y()) ** 2 / b_endo_rv**2
                + (x[2] - center_rv.z()) ** 2 / c_endo_rv**2
                - 1
                < df.DOLFIN_EPS
                and (x[0] - center_lv.x()) ** 2 / a_epi_lv**2
                + (x[1] - center_lv.y()) ** 2 / b_epi_lv**2
                + (x[2] - center_lv.z()) ** 2 / c_epi_lv**2
                - 0.9
                > df.DOLFIN_EPS
            ) and on_boundary

    class Epi(df.SubDomain):
        def inside(self, x, on_boundary):
            return (
                (x[0] - center_rv.x()) ** 2 / a_epi_rv**2
                + (x[1] - center_rv.y()) ** 2 / b_epi_rv**2
                + (x[2] - center_rv.z()) ** 2 / c_epi_rv**2
                - 0.9
                > df.DOLFIN_EPS
                and (x[0] - center_lv.x()) ** 2 / a_epi_lv**2
                + (x[1] - center_lv.y()) ** 2 / b_epi_lv**2
                + (x[2] - center_lv.z()) ** 2 / c_epi_lv**2
                - 0.9
                > df.DOLFIN_EPS
                and on_boundary
            )

    # The plane cutting the base
    a_epi = max(a_epi_lv, a_epi_rv)
    diam = -5 * a_epi
    box = mshr.Box(df.Point(base_x, a_epi, a_epi), df.Point(diam, diam, diam))
    # Generate mesh

    # LV epicardium
    el_lv = mshr.Ellipsoid(center_lv, a_epi_lv, b_epi_lv, c_epi_lv)
    # LV endocardium
    el_lv_endo = mshr.Ellipsoid(center_lv, a_endo_lv, b_endo_lv, c_endo_lv)

    # LV geometry (subtract the smallest ellipsoid)
    lv = el_lv - el_lv_endo

    # LV epicardium
    el_rv = mshr.Ellipsoid(center_rv, a_epi_rv, b_epi_rv, c_epi_rv)
    # LV endocardium
    el_rv_endo = mshr.Ellipsoid(center_rv, a_endo_rv, b_endo_rv, c_endo_rv)

    # RV geometry (subtract the smallest ellipsoid)
    rv = el_rv - el_rv_endo - el_lv

    # BiV geometry
    m = lv + rv - box

    # Create mesh
    mesh = mshr.generate_mesh(m, N)

    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    endolv = EndoLV()
    endolv.mark(ffun, markers["lv"])
    base = Base()
    base.mark(ffun, markers["base"])
    endorv = EndoRV()
    endorv.mark(ffun, markers["rv"])
    epi = Epi()
    epi.mark(ffun, markers["epi"])

    return Geometry(mesh=mesh, ffun=ffun, markers=markers)


# These functions are copied from cbcpost https://bitbucket.org/simula_cbc/cbcpost
def broadcast(array, from_process):
    "Broadcast array to all processes"
    if not hasattr(broadcast, "cpp_module"):
        cpp_code = """

        namespace dolfin {
            std::vector<double> broadcast(const MPI_Comm mpi_comm, const Array<double>& inarray, int from_process)
            {
                int this_process = dolfin::MPI::rank(mpi_comm);
                std::vector<double> outvector(inarray.size());

                if(this_process == from_process) {
                    for(int i=0; i<inarray.size(); i++)
                    {
                        outvector[i] = inarray[i];
                    }
                }
                dolfin::MPI::barrier(mpi_comm);
                dolfin::MPI::broadcast(mpi_comm, outvector, from_process);

                return outvector;
            }
        }
        """
        cpp_module = df.compile_extension_module(
            cpp_code,
            additional_system_headers=["dolfin/common/MPI.h"],
        )

        broadcast.cpp_module = cpp_module

    cpp_module = broadcast.cpp_module

    if df.MPI.rank(df.mpi_comm_world()) == from_process:
        array = np.array(array, dtype=np.float)
        shape = array.shape
        shape = np.array(shape, dtype=np.float_)
    else:
        array = np.array([], dtype=np.float)
        shape = np.array([], dtype=np.float_)

    shape = cpp_module.broadcast(df.mpi_comm_world(), shape, from_process)
    array = array.flatten()

    out_array = cpp_module.broadcast(df.mpi_comm_world(), array, from_process)

    if len(shape) > 1:
        out_array = out_array.reshape(*shape)

    return out_array


def gather(array, on_process=0, flatten=False):
    "Gather array from all processes on a single process"

    if not hasattr(gather, "cpp_module"):
        cpp_code = """
        namespace dolfin {
            std::vector<double> gather(const MPI_Comm mpi_comm, const Array<double>& inarray, int on_process)
            {
                int this_process = dolfin::MPI::rank(mpi_comm);

                std::vector<double> outvector(dolfin::MPI::size(mpi_comm)*dolfin::MPI::sum(mpi_comm, inarray.size()));
                std::vector<double> invector(inarray.size());

                for(int i=0; i<inarray.size(); i++)
                {
                    invector[i] = inarray[i];
                }

                dolfin::MPI::gather(mpi_comm, invector, outvector, on_process);
                return outvector;
            }
        }
        """
        gather.cpp_module = df.compile_extension_module(
            cpp_code,
            additional_system_headers=["dolfin/common/MPI.h"],
        )

    cpp_module = gather.cpp_module
    array = np.array(array, dtype=np.float)
    out_array = cpp_module.gather(df.mpi_comm_world(), array, on_process)

    if flatten:
        return out_array

    dist = distribution(len(array))
    cumsum = [0] + [sum(dist[: i + 1]) for i in range(len(dist))]
    out_array = [[out_array[cumsum[i] : cumsum[i + 1]]] for i in range(len(cumsum) - 1)]

    return out_array


def distribution(number):
    "Get distribution of number on all processes"
    if not hasattr(distribution, "cpp_module"):
        cpp_code = """
        namespace dolfin {
            std::vector<unsigned int> distribution(const MPI_Comm mpi_comm, int number)
            {
                // Variables to help in synchronization
                int num_processes = dolfin::MPI::size(mpi_comm);
                int this_process = dolfin::MPI::rank(mpi_comm);

                std::vector<uint> distribution(num_processes);

                for(uint i=0; i<num_processes; i++) {
                    if(i==this_process) {
                        distribution[i] = number;
                    }
                    dolfin::MPI::barrier(mpi_comm);
                    dolfin::MPI::broadcast(mpi_comm, distribution, i);
                }
                return distribution;
          }
        }
        """
        distribution.cpp_module = df.compile_extension_module(
            cpp_code,
            additional_system_headers=["dolfin/common/MPI.h"],
        )

    cpp_module = distribution.cpp_module
    return cpp_module.distribution(df.mpi_comm_world(), number)


def gather_broadcast(arr):
    arr = gather(arr, flatten=True)
    arr = broadcast(arr, 0)
    return arr


def assign_to_vector(v, a):
    """
    Assign the value of the array a to the dolfin vector v
    """
    lr = v.local_range()
    v[:] = a[lr[0] : lr[1]]


def space_from_string(space_string: str, mesh: df.Mesh, dim: int) -> df.FunctionSpace:
    """
    Constructed a finite elements space from a string
    representation of the space

    Arguments
    ---------
    space_string : str
        A string on the form {familiy}_{degree} which
        determines the space. Example 'Lagrange_1'.
    mesh : dolfin.Mesh
        The mesh
    dim : int
        1 for scalar space, 3 for vector space.
    """
    family, degree = space_string.split("_")

    if dim == 3:
        V = df.FunctionSpace(
            mesh,
            df.VectorElement(
                family=family,
                cell=mesh.ufl_cell(),
                degree=int(degree),
                quad_scheme="default",
            ),
        )
    elif dim == 1:
        V = df.FunctionSpace(
            mesh,
            df.FiniteElement(
                family=family,
                cell=mesh.ufl_cell(),
                degree=int(degree),
                quad_scheme="default",
            ),
        )
    else:
        raise df.error("Cannot create function space of dimension {dim}")

    return V
