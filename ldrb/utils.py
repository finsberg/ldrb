from typing import Dict
from typing import List
from typing import Union

import dolfin as df
import numpy as np
import ufl


def default_markers() -> Dict[str, int]:
    """
    Default markers for the mesh boundaries
    """
    return dict(base=10, rv=20, lv=30, epi=40)


if df.__version__.startswith("20"):
    # Year based versioning
    DOLFIN_VERSION_MAJOR = float(df.__version__.split(".")[0])
else:
    try:
        DOLFIN_VERSION_MAJOR = float(".".join(df.__version__.split(".")[:2]))
    except Exception:
        DOLFIN_VERSION_MAJOR = 1.6


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
        A string on the form {family}_{degree} which
        determines the space. Example 'Lagrange_1'.
    mesh : df.Mesh
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


class Projector:
    def __init__(
        self,
        V: df.FunctionSpace,
        solver_type: str = "lu",
        preconditioner_type: str = "default",
    ):
        """
        Projection class caching solver and matrix assembly
        Args:
            V (df.FunctionSpace): Function-space to project in to
            solver_type (str, optional): Type of solver. Defaults to "lu".
            preconditioner_type (str, optional): Type of preconditioner. Defaults to "default".
        Raises:
            RuntimeError: _description_
        """
        u = df.TrialFunction(V)
        self._v = df.TestFunction(V)
        self._dx = df.Measure("dx", domain=V.mesh())
        self._b = df.Function(V)
        self._A = df.assemble(ufl.inner(u, self._v) * self._dx)
        lu_methods = df.lu_solver_methods().keys()
        krylov_methods = df.krylov_solver_methods().keys()
        if solver_type == "lu" or solver_type in lu_methods:
            if preconditioner_type != "default":
                raise RuntimeError("LUSolver cannot be preconditioned")
            self.solver = df.LUSolver(self._A, "default")
        elif solver_type in krylov_methods:
            self.solver = df.PETScKrylovSolver(
                solver_type,
                preconditioner_type,
            )
            self.solver.set_operator(self._A)
        else:
            raise RuntimeError(
                f"Unknown solver type: {solver_type}, method has to be lu"
                + f", or {np.hstack(lu_methods, krylov_methods)}",
            )

    def project(self, u: df.Function, f: ufl.core.expr.Expr) -> None:
        """
        Project `f` into `u`.
        Args:
            u (df.Function): The function to project into
            f (ufl.core.expr.Expr): The ufl expression to project
        """
        df.assemble(ufl.inner(f, self._v) * self._dx, tensor=self._b.vector())
        self.solver.solve(u.vector(), self._b.vector())

    def __call__(self, u: df.Function, f: ufl.core.expr.Expr) -> None:
        self.project(u=u, f=f)
