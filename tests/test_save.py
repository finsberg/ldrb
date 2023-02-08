import os

import dolfin as df

import ldrb


def test_save():
    mesh = df.UnitSquareMesh(3, 3)
    # exit()
    spaces = ["DG_0", "DG_1", "CG_1", "CG_2", "R_0", "Quadrature_2", "Quadrature_4"]

    finite_elements = [
        df.FiniteElement(
            s.split("_")[0],
            mesh.ufl_cell(),
            int(s.split("_")[1]),
            quad_scheme="default",
        )
        for s in spaces
    ]
    scalar_spaces = [df.FunctionSpace(mesh, el) for el in finite_elements]
    scalar_functions = [
        df.Function(V, name="Scalar_{}".format(s))
        for (V, s) in zip(scalar_spaces, spaces)
    ]

    vector_elements = [
        df.VectorElement(
            s.split("_")[0],
            mesh.ufl_cell(),
            int(s.split("_")[1]),
            quad_scheme="default",
        )
        for s in spaces
    ]
    vector_spaces = [df.FunctionSpace(mesh, el) for el in vector_elements]
    vector_functions = [
        df.Function(V, name="Vector_{}".format(s))
        for (V, s) in zip(vector_spaces, spaces)
    ]

    for f, space in zip(scalar_functions, spaces):
        name = "test_scalar_fun_{space}".format(space=space)
        ldrb.fun_to_xdmf(f, name)
        os.remove(name + ".xdmf")
        os.remove(name + ".h5")

    for f, space in zip(vector_functions, spaces):
        name = "test_vector_fun_{space}".format(space=space)
        ldrb.fun_to_xdmf(f, name)
        os.remove(name + ".xdmf")
        os.remove(name + ".h5")
        name = "test_vector_fiber_{space}".format(space=space)
        ldrb.fiber_to_xdmf(f, name)
        os.remove(name + ".xdmf")
        os.remove(name + ".h5")


if __name__ == "__main__":
    test_save()
