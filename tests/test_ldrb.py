import numpy as np
import pytest

import ldrb


def norm(v):
    return np.linalg.norm(v)


def test_axis():

    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 0.5, 0.0])
    Q = ldrb.calculus.axis(u, v)

    e0, e1, e2 = Q.T

    assert np.all(e1 == u)
    assert np.all(v == np.linalg.norm(v) * e2)
    assert np.dot(e0, e1) == 0
    assert np.dot(e0, e2) == 0


def test_orient():
    pass


@pytest.fixture(scope="session")
def biv_geometry():
    return ldrb.create_biv_mesh()


@pytest.fixture(scope="session")
def lv_geometry():
    return ldrb.create_lv_mesh()


def test_lv_angles_alpha():

    data = {}
    eps = 0
    tol = 1e-12
    data["lv_scalar"] = np.array([1.0, 0.5, eps])
    data["lv_gradient"] = np.zeros(3 * len(data["lv_scalar"]))
    data["lv_gradient"][::3] = 1.0

    data["epi_scalar"] = np.flipud(data["lv_scalar"])
    data["epi_gradient"] = -1 * data["lv_gradient"]

    data["apex_gradient"] = np.zeros_like(data["lv_gradient"])
    data["apex_gradient"][1::3] = 1.0

    sheet_normal = np.zeros(9)
    sheet_normal[::3] = -1

    fib = ldrb.ldrb.compute_fiber_sheet_system(
        alpha_endo_lv=90,
        alpha_epi_lv=-90,
        beta_endo_lv=0,
        beta_epi_lv=0,
        **data,
    )

    assert norm(fib.fiber[:3] - np.array([0, 1, 0])) < tol
    assert norm(fib.fiber[3:6] - np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)])) < tol
    assert norm(fib.fiber[6:] - np.array([0, -1, 0])) < tol
    assert norm(fib.sheet_normal - sheet_normal) < tol

    fib = ldrb.ldrb.compute_fiber_sheet_system(
        alpha_endo_lv=-90,
        alpha_epi_lv=90,
        beta_endo_lv=0,
        beta_epi_lv=0,
        **data,
    )
    assert norm(fib.fiber[:3] - np.array([0, -1, 0])) < tol
    assert norm(fib.fiber[3:6] - np.array([0, -1 / np.sqrt(2), 1 / np.sqrt(2)])) < tol
    assert norm(fib.fiber[6:] - np.array([0, 1, 0])) < tol
    assert norm(fib.sheet_normal - sheet_normal) < tol

    for alpha in [60, -60]:
        a = np.radians(alpha)
        fib = ldrb.ldrb.compute_fiber_sheet_system(
            alpha_endo_lv=alpha,
            alpha_epi_lv=-alpha,
            beta_endo_lv=0,
            beta_epi_lv=0,
            **data,
        )
        assert norm(fib.fiber[:3] - np.array([0, np.sin(a), np.cos(a)])) < tol
        assert (
            norm(fib.fiber[3:6] - np.sign(alpha) * np.array([0, np.cos(a), np.sin(a)]))
            < tol
        )
        assert norm(fib.fiber[6:] - np.array([0, np.sin(-a), np.cos(-a)])) < tol

        assert norm(fib.sheet_normal - sheet_normal) < tol

        assert norm(fib.sheet[:3] - np.cross(fib.sheet_normal[:3], fib.fiber[:3])) < tol
        assert (
            norm(fib.sheet[3:6] - np.cross(fib.sheet_normal[3:6], fib.fiber[3:6])) < tol
        )
        assert norm(fib.sheet[6:] - np.cross(fib.sheet_normal[6:], fib.fiber[6:])) < tol

    for alpha in [30, 40, 50, -30, -40, -50]:
        a = np.radians(alpha)
        fib = ldrb.ldrb.compute_fiber_sheet_system(
            alpha_endo_lv=alpha,
            alpha_epi_lv=-alpha,
            beta_endo_lv=0,
            beta_epi_lv=0,
            **data,
        )
        assert norm(fib.fiber[:3] - np.array([0, np.sin(a), np.cos(a)])) < tol
        assert norm(fib.fiber[6:] - np.array([0, np.sin(-a), np.cos(-a)])) < tol

        assert norm(fib.sheet_normal - sheet_normal) < tol

        assert norm(fib.sheet[:3] - np.cross(fib.sheet_normal[:3], fib.fiber[:3])) < tol
        assert (
            norm(fib.sheet[3:6] - np.cross(fib.sheet_normal[3:6], fib.fiber[3:6])) < tol
        )
        assert norm(fib.sheet[6:] - np.cross(fib.sheet_normal[6:], fib.fiber[6:])) < tol


def test_lv_angles_beta():

    data = {}
    eps = 0
    tol = 1e-12
    data["lv_scalar"] = np.array([1.0, 0.5, eps])
    data["lv_gradient"] = np.zeros(3 * len(data["lv_scalar"]))
    data["lv_gradient"][::3] = 1.0

    data["epi_scalar"] = np.flipud(data["lv_scalar"])
    data["epi_gradient"] = -1 * data["lv_gradient"]

    data["apex_gradient"] = np.zeros_like(data["lv_gradient"])
    data["apex_gradient"][1::3] = 1.0

    sheet_normal = np.array([0, 1, 0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, -1, 0])
    fiber = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])

    fib = ldrb.ldrb.compute_fiber_sheet_system(
        alpha_endo_lv=0,
        alpha_epi_lv=0,
        beta_endo_lv=90,
        beta_epi_lv=-90,
        **data,
    )

    assert np.linalg.norm(fib.fiber - fiber) < tol
    assert norm(fib.sheet[:3] - np.array([1, 0, 0])) < tol
    assert norm(fib.sheet[3:6] - np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0])) < tol
    assert norm(fib.sheet[6:] - np.array([-1, 0, 0])) < tol
    assert norm(fib.sheet_normal - sheet_normal) < tol

    fib = ldrb.ldrb.compute_fiber_sheet_system(
        alpha_endo_lv=0, alpha_epi_lv=0, beta_endo_lv=-90, beta_epi_lv=90, **data
    )

    assert np.linalg.norm(fib.fiber - fiber) < tol
    assert norm(fib.sheet[:3] - np.array([-1, 0, 0])) < tol
    assert norm(fib.sheet[3:6] - np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0])) < tol
    assert norm(fib.sheet[6:] - np.array([1, 0, 0])) < tol
    sheet_normal *= -1
    sheet_normal[3] = -1 / np.sqrt(2)
    assert norm(fib.sheet_normal - sheet_normal) < tol

    for beta in [30, 40, 50, -30, -40, -50]:
        a = np.radians(beta)
        fib = ldrb.ldrb.compute_fiber_sheet_system(
            alpha_endo_lv=0,
            alpha_epi_lv=-0,
            beta_endo_lv=beta,
            beta_epi_lv=beta,
            **data,
        )
        assert np.linalg.norm(fib.fiber - fiber) < tol

        assert norm(fib.sheet[:3] - np.array([np.sin(a), np.cos(a), 0])) < tol
        assert norm(fib.sheet[3:6] - np.array([np.sin(a), np.cos(a), 0])) < tol
        assert norm(fib.sheet[6:] - np.array([np.sin(a), np.cos(a), 0])) < tol

        assert norm(fib.sheet_normal[:3] - np.cross(fib.fiber[:3], fib.sheet[:3])) < tol
        assert (
            norm(fib.sheet_normal[3:6] - np.cross(fib.fiber[3:6], fib.sheet[3:6])) < tol
        )
        assert norm(fib.sheet_normal[6:] - np.cross(fib.fiber[6:], fib.sheet[6:])) < tol


def test_ldrb_without_correct_markers_raises_RuntimeError(lv_geometry):
    with pytest.raises(RuntimeError):
        ldrb.dolfin_ldrb(mesh=lv_geometry.mesh)


@pytest.mark.parametrize("use_krylov_solver", [True, False])
def test_biv_regression(biv_geometry, use_krylov_solver):
    ldrb.dolfin_ldrb(
        mesh=biv_geometry.mesh,
        ffun=biv_geometry.ffun,
        markers=biv_geometry.markers,
        use_krylov_solver=use_krylov_solver,
    )


@pytest.mark.parametrize("use_krylov_solver", [True, False])
def test_lv_regression(lv_geometry, use_krylov_solver):
    ldrb.dolfin_ldrb(
        mesh=lv_geometry.mesh,
        ffun=lv_geometry.ffun,
        markers=lv_geometry.markers,
        use_krylov_solver=use_krylov_solver,
    )


@pytest.mark.xfail(
    raises=RuntimeError,
    strict=True,
    reason="Krylov solvers are not accrate enough?",
)
def test_krylov_laplace(lv_geometry):
    ldrb.ldrb.laplace(
        mesh=lv_geometry.mesh,
        ffun=lv_geometry.ffun,
        markers=lv_geometry.markers,
        use_krylov_solver=True,
        strict=True,
    )


def test_regular_laplace(lv_geometry):
    ldrb.ldrb.laplace(
        mesh=lv_geometry.mesh,
        ffun=lv_geometry.ffun,
        markers=lv_geometry.markers,
        use_krylov_solver=False,
        strict=True,
    )


if __name__ == "__main__":
    # test_axis()
    # test_lv_angles()
    # test_biv_regression()
    # test_lv_regression()
    # m = lv_geometry()
    # test_markers(m)
    pass
