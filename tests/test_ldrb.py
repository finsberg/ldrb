import cardiac_geometries
import numpy as np
import pytest

import ldrb


tol = 1e-12


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


def test_bislerp():
    Qa = np.zeros((3, 3))
    Qb = np.zeros((3, 3))

    Qa = np.array(
        [
            [-0.1145090439760228, -0.67883412599201054, -0.72530814709084901],
            [0.52470693935981916, 0.57863132132197759, -0.62439444405986955],
            [0.84354626822442202, -0.45207302880601674, 0.28993045716310317],
        ],
    )

    Qb = np.array(
        [
            [-0.04821968810650662, -0.68803994883005382, 0.72406898186073976],
            [-0.51030589763757894, 0.64013348825587579, 0.57429696851861345],
            [-0.85864005992919767, -0.34180425103749279, -0.38197788084846285],
        ],
    )
    t = 0.42738328729654113
    Qab = ldrb.calculus.bislerp(Qa, Qb, t)

    expected = np.array(
        [
            [0.04510129, -0.68529013, 0.72687228],
            [-0.52014838, 0.60509228, 0.60275119],
            [-0.85288424, -0.4052663, -0.32916211],
        ],
    )

    assert np.isclose(Qab, expected).all()


@pytest.fixture(scope="session")
def biv_geometry():
    geo = cardiac_geometries.create_biv_ellipsoid()
    markers = {
        "base": geo.markers["BASE"][0],
        "epi": geo.markers["EPI"][0],
        "lv": geo.markers["ENDO_LV"][0],
        "rv": geo.markers["ENDO_RV"][0],
    }
    geo.markers = markers
    return geo


@pytest.fixture(scope="session")
def lv_geometry():
    geo = cardiac_geometries.create_lv_ellipsoid()
    markers = {
        "base": geo.markers["BASE"][0],
        "epi": geo.markers["EPI"][0],
        "lv": geo.markers["ENDO"][0],
    }
    geo.markers = markers
    return geo


@pytest.fixture(name="data")
def angle_data():
    data = {}
    eps = 0
    data["lv_scalar"] = np.array([1.0, 0.5, eps])
    data["lv_gradient"] = np.zeros(3 * len(data["lv_scalar"]))
    data["lv_gradient"][::3] = 1.0

    data["epi_scalar"] = np.flipud(data["lv_scalar"])
    data["epi_gradient"] = -1 * data["lv_gradient"]

    data["apex_gradient"] = np.zeros_like(data["lv_gradient"])
    data["apex_gradient"][1::3] = 1.0
    return data


@pytest.mark.parametrize("beta", (90, -90, 30, -30, 40, -40, 50, -50))
def test_lv_angles_beta(beta, data):
    a = np.radians(beta)
    fib = ldrb.ldrb.compute_fiber_sheet_system(
        alpha_endo_lv=0,
        alpha_epi_lv=-0,
        beta_endo_lv=beta,
        beta_epi_lv=beta,
        **data,
    )
    fiber = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
    sheet = np.array(
        [np.sin(a), np.cos(a), 0, np.sin(a), np.cos(a), 0, np.sin(a), np.cos(a), 0],
    )
    sheet_normal = np.array(
        [
            -np.cos(a),
            np.sin(a),
            0,
            -np.cos(a),
            np.sin(a),
            0,
            -np.cos(a),
            np.sin(a),
            0,
        ],
    )

    assert norm(fib.fiber - fiber) < tol

    assert norm(fib.sheet - sheet) < tol
    assert norm(fib.sheet_normal - sheet_normal) < tol


@pytest.mark.parametrize("alpha", (90, -90, 60, -60, 30, -30, 40, -40, 50, -50))
def test_lv_angles_alpha(alpha, data):
    a = np.radians(alpha)
    fib = ldrb.ldrb.compute_fiber_sheet_system(
        alpha_endo_lv=alpha,
        alpha_epi_lv=-alpha,
        beta_endo_lv=0,
        beta_epi_lv=-0,
        **data,
    )

    fiber = np.array(
        [
            0,
            np.sin(a),
            np.cos(a),
            0,
            np.sin(a / 2),
            np.cos(a / 2),
            0,
            -np.sin(a),
            np.cos(a),
        ],
    )
    sheet = np.array(
        [
            0,
            np.cos(a),
            -np.sin(a),
            0,
            np.cos(a / 2),
            -np.sin(a / 2),
            0,
            np.cos(a),
            np.sin(a),
        ],
    )
    sheet_normal = np.array([-1, 0, 0, -1, 0, 0, -1, 0, 0])
    assert norm(fib.fiber - fiber) < tol
    assert norm(fib.sheet - sheet) < tol
    assert norm(fib.sheet_normal - sheet_normal) < tol


def test_ldrb_without_correct_markers_raises_RuntimeError(lv_geometry):
    with pytest.raises(RuntimeError):
        ldrb.dolfin_ldrb(mesh=lv_geometry.mesh)


def test_biv_regression(biv_geometry):
    ldrb.dolfin_ldrb(
        mesh=biv_geometry.mesh,
        ffun=biv_geometry.ffun,
        markers=biv_geometry.markers,
    )


def test_lv_regression(lv_geometry):
    ldrb.dolfin_ldrb(
        mesh=lv_geometry.mesh,
        ffun=lv_geometry.ffun,
        markers=lv_geometry.markers,
    )
