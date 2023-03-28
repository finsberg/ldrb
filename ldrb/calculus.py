import numba
import numpy as np


_float_1D_array = numba.types.Array(numba.float64, 1, "C")
_float_2D_array = numba.types.Array(numba.float64, 2, "C")


@numba.njit(numba.float64(_float_1D_array, _float_1D_array))
def quat_dot(q1: np.ndarray, q2: np.ndarray) -> float:
    """Quaternion dot product

    Parameters
    ----------
    q1 : np.ndarray
        First quaternion
    q2 : np.ndarray
        Second quaternion

    Returns
    -------
    float
        The quaternion dot product
    """
    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]


@numba.njit(_float_1D_array(_float_1D_array))
def normalize(u: np.ndarray) -> np.ndarray:
    """L2-Normalize vector with

    Parameters
    ----------
    u : np.ndarray
        Vector

    Returns
    -------
    np.ndarray
        Normalized vector
    """
    return u / np.linalg.norm(u)


@numba.njit(_float_1D_array(_float_1D_array, _float_1D_array, numba.float64))
def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation from `q1` to `q2` at `t`

    Parameters
    ----------
    q1 : np.ndarray
        Source quaternion
    q2 : np.ndarray
        Target quaternion
    t : float
        Interpolation factor, between 0 and 1

    Returns
    -------
    np.ndarray
        The spherical linear interpolation between `q1` and `q2` at `t`
    """
    dot = quat_dot(q1, q2)
    if dot > 1 - 1e-12:
        angle = np.arccos(dot)
        a = np.sin(angle * (1 - t)) / np.sin(angle)
        b = np.sin(angle * t) / np.sin(angle)

        q = q2
        q *= b
        q1a = q1
        q1a *= a
        q += q1a
    else:
        q = q1
        q *= 1 - t
        q2t = q2
        q2t *= t
        q += q2t
    return q


@numba.njit(_float_1D_array(_float_2D_array))
def rot2quat(Q: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion

    Parameters
    ----------
    Q : np.ndarray
        Rotation matrix

    Returns
    -------
    np.ndarray
        Quaternion
    """
    M11 = Q[0][0]
    M12 = Q[1][0]
    M13 = Q[2][0]
    M21 = Q[0][1]
    M22 = Q[1][1]
    M23 = Q[2][1]
    M31 = Q[0][2]
    M32 = Q[1][2]
    M33 = Q[2][2]

    w2 = 0.25 * (1 + M11 + M22 + M33)
    err = 1e-15

    if w2 > err:
        w = np.sqrt(w2)
        x = (M23 - M32) / (4.0 * w)
        y = (M31 - M13) / (4.0 * w)
        z = (M12 - M21) / (4.0 * w)
    else:
        w = 0.0
        x2 = -0.5 * (M22 + M33)
        if x2 > err:
            x = np.sqrt(x2)
            y = M12 / (2.0 * x)
            z = M13 / (2.0 * x)
        else:
            x = 0.0
            y2 = 0.5 * (1 - M33)
            if y2 > err:
                y = np.sqrt(y2)
                z = M23 / (2.0 * y)
            else:
                y = 0.0
                z = 1.0
    return normalize(np.array([w, x, y, z]))


@numba.njit(_float_2D_array(_float_1D_array))
def quat2rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix

    Parameters
    ----------
    q : np.ndarray
        Quaternion

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    Q = np.zeros((3, 3))
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z

    wx = w * x
    wy = w * y
    wz = w * z

    xy = x * y
    xz = x * z

    yz = y * z

    Q[0][0] = 1.0 - 2.0 * y2 - 2.0 * z2
    Q[1][0] = 2.0 * xy + 2.0 * wz
    Q[2][0] = 2.0 * xz - 2.0 * wy
    Q[0][1] = 2.0 * xy - 2.0 * wz
    Q[1][1] = 1.0 - 2.0 * x2 - 2.0 * z2
    Q[2][1] = 2.0 * yz + 2.0 * wx
    Q[0][2] = 2.0 * xz + 2.0 * wy
    Q[1][2] = 2.0 * yz - 2.0 * wx
    Q[2][2] = 1.0 - 2.0 * x2 - 2.0 * y2

    return Q


@numba.njit(_float_2D_array(_float_2D_array, _float_2D_array, numba.float64))
def bislerp(
    Qa: np.ndarray,
    Qb: np.ndarray,
    t: float,
) -> np.ndarray:
    r"""
    Linear interpolation of two orthogonal matrices.
    Assume that :math:`Q_a` and :math:`Q_b` refers to
    timepoint :math:`0` and :math:`1` respectively.
    Using spherical linear interpolation (slerp) find the
    orthogonal matrix at timepoint :math:`t`.
    """

    if ~Qa.any() and ~Qb.any():
        return np.zeros((3, 3))
    if ~Qa.any():
        return Qb
    if ~Qb.any():
        return Qa

    tol = 1e-12
    qa = rot2quat(Qb)
    qb = rot2quat(Qb)

    quat_i = np.array([0, 1, 0, 0])
    quat_j = np.array([0, 0, 1, 0])
    quat_k = np.array([0, 0, 0, 1])

    quat_array = [
        qa,
        -qa,
        quat_i * qa,
        -quat_i * qa,
        quat_j * qa,
        -quat_j * qa,
        quat_k * qa,
        -quat_k * qa,
    ]

    max_idx = 0
    max_dot = 0.0

    idx = 0
    for qm in [
        qa,
        -qa,
        quat_i * qa,
        -quat_i * qa,
        quat_j * qa,
        -quat_j * qa,
        quat_k * qa,
        -quat_k * qa,
    ]:
        dot = quat_dot(qm, qb)
        if dot > max_dot:
            max_dot = dot
            max_idx = idx
        idx += 1

    if max_dot > 1 - tol:
        return Qb

    qm = quat_array[max_idx]
    qm_slerp = slerp(qm, qb, t)
    qm_norm = normalize(qm_slerp)

    return quat2rot(qm_norm)


@numba.njit(_float_2D_array(_float_1D_array, _float_1D_array))
def axis(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""
    Construct the fiber orientation coordinate system.

    Given two vectors :math:`u` and :math:`v` in the apicobasal
    and transmural direction respectively return a matrix that
    represents an orthonormal basis in the circumferential (first),
    apicobasal (second) and transmural (third) direction.
    """

    e1 = normalize(u)

    # Create an initial guess for e0
    e2 = normalize(v)
    e2 -= e1.dot(e2) * e1
    e2 = normalize(e2)

    e0 = np.cross(e1, e2)
    e0 = normalize(e0)

    Q = np.zeros((3, 3))
    Q[:, 0] = e0
    Q[:, 1] = e1
    Q[:, 2] = e2

    return Q


@numba.njit(_float_2D_array(_float_2D_array, numba.float64, numba.float64))
def orient(Q: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    r"""
    Define the orthotropic fiber orientations.

    Given a coordinate system :math:`Q`, in the canonical
    basis, rotate it in order to align with the fiber, sheet
    and sheet-normal axis determine by the angles :math:`\alpha`
    (fiber) and :math:`\beta` (sheets).
    """
    A = np.zeros((3, 3))
    A[0, :] = [np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)), 0]
    A[1, :] = [np.sin(np.radians(alpha)), np.cos(np.radians(alpha)), 0]
    A[2, :] = [0, 0, 1]

    B = np.zeros((3, 3))
    B[0, :] = [1, 0, 0]
    B[1, :] = [0, np.cos(np.radians(beta)), np.sin(np.radians(beta))]
    B[2, :] = [0, -np.sin(np.radians(beta)), np.cos(np.radians(beta))]

    C = np.dot(Q.real, A).dot(B)
    return C


@numba.njit(
    _float_2D_array(
        numba.float64,
        numba.float64,
        numba.float64,
        _float_1D_array,
        _float_1D_array,
        _float_1D_array,
        _float_1D_array,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
        numba.float64,
    ),
)
def system_at_dof(
    lv: float,
    rv: float,
    epi: float,
    grad_lv: np.ndarray,
    grad_rv: np.ndarray,
    grad_epi: np.ndarray,
    grad_ab: np.ndarray,
    alpha_endo: float,
    alpha_epi: float,
    beta_endo: float,
    beta_epi: float,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Compte the fiber, sheet and sheet normal at a
    single degree of freedom

    Arguments
    ---------
    lv : float
        Value of the Laplace solution for the LV at the dof.
    rv : float
        Value of the Laplace solution for the RV at the dof.
    epi : float
        Value of the Laplace solution for the EPI at the dof.
    grad_lv : np.ndarray
        Gradient of the Laplace solution for the LV at the dof.
    grad_rv : np.ndarray
        Gradient of the Laplace solution for the RV at the dof.
    grad_epi : np.ndarray
        Gradient of the Laplace solution for the EPI at the dof.
    grad_epi : np.ndarray
        Gradient of the Laplace solution for the apex to base.
        at the dof
    alpha_endo : scalar
        Fiber angle at the endocardium.
    alpha_epi : scalar
        Fiber angle at the epicardium.
    beta_endo : scalar
        Sheet angle at the endocardium.
    beta_epi : scalar
        Sheet angle at the epicardium.
    tol : scalar
        Tolerance for whether to consider the scalar values.
        Default: 1e-7
    """

    if lv + rv < tol:
        depth = 0.5
    else:
        depth = rv / (lv + rv)

    alpha_s = alpha_endo * (1 - depth) - alpha_endo * depth
    alpha_w = alpha_endo * (1 - epi) + alpha_epi * epi
    beta_s = beta_endo * (1 - depth) - beta_endo * depth
    beta_w = beta_endo * (1 - epi) + beta_epi * epi

    Q_lv = np.zeros((3, 3))
    if lv > tol:
        Q_lv = axis(grad_ab, -1 * grad_lv)
        Q_lv = orient(Q_lv, alpha_s, beta_s)

    Q_rv = np.zeros((3, 3))
    if rv > tol:
        Q_rv = axis(grad_ab, grad_rv)
        Q_rv = orient(Q_rv, alpha_s, beta_s)

    Q_endo = bislerp(Q_lv, Q_rv, depth)

    Q_epi = np.zeros((3, 3))
    if epi > tol:
        Q_epi = axis(grad_ab, grad_epi)
        Q_epi = orient(Q_epi, alpha_w, beta_w)

    Q_fiber = bislerp(Q_endo, Q_epi, epi)
    return Q_fiber


@numba.njit
def compute_fiber_sheet_system(
    f0,
    s0,
    n0,
    xdofs,
    ydofs,
    zdofs,
    sdofs,
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
):
    grad_lv = np.zeros(3)
    grad_rv = np.zeros(3)
    grad_epi = np.zeros(3)
    grad_ab = np.zeros(3)

    for i in range(len(xdofs)):
        lv = lv_scalar[sdofs[i]]
        rv = rv_scalar[sdofs[i]]
        epi = epi_scalar[sdofs[i]]
        lv_rv = lv_rv_scalar[sdofs[i]]

        grad_lv[0] = lv_gradient[xdofs[i]]
        grad_lv[1] = lv_gradient[ydofs[i]]
        grad_lv[2] = lv_gradient[zdofs[i]]

        grad_rv[0] = rv_gradient[xdofs[i]]
        grad_rv[1] = rv_gradient[ydofs[i]]
        grad_rv[2] = rv_gradient[zdofs[i]]

        grad_epi[0] = epi_gradient[xdofs[i]]
        grad_epi[1] = epi_gradient[ydofs[i]]
        grad_epi[2] = epi_gradient[zdofs[i]]

        grad_ab[0] = apex_gradient[xdofs[i]]
        grad_ab[1] = apex_gradient[ydofs[i]]
        grad_ab[2] = apex_gradient[zdofs[i]]

        if epi > 0.5:
            if lv_rv >= 0.5:
                # We are in the LV region
                marker_scalar[sdofs[i]] = 1
                alpha_endo = alpha_endo_lv
                beta_endo = beta_endo_lv
                alpha_epi = alpha_epi_lv
                beta_epi = beta_epi_lv
            else:
                # We are in the RV region
                marker_scalar[sdofs[i]] = 2
                alpha_endo = alpha_endo_rv
                beta_endo = beta_endo_rv
                alpha_epi = alpha_epi_rv
                beta_epi = beta_epi_rv
        else:
            if lv_rv >= 1 - tol:
                # We are in the LV region
                marker_scalar[sdofs[i]] = 1
                alpha_endo = alpha_endo_lv
                beta_endo = beta_endo_lv
                alpha_epi = alpha_epi_lv
                beta_epi = beta_epi_lv
            elif lv_rv <= tol:
                # We are in the RV region
                marker_scalar[sdofs[i]] = 2
                alpha_endo = alpha_endo_rv
                beta_endo = beta_endo_rv
                alpha_epi = alpha_epi_rv
                beta_epi = beta_epi_rv
            else:
                # We are in the septum
                marker_scalar[sdofs[i]] = 3
                alpha_endo = alpha_endo_sept
                beta_endo = beta_endo_sept
                alpha_epi = alpha_epi_sept
                beta_epi = beta_epi_sept

        Q_fiber = system_at_dof(
            lv=lv,
            rv=rv,
            epi=epi,
            grad_lv=grad_lv,
            grad_rv=grad_rv,
            grad_epi=grad_epi,
            grad_ab=grad_ab,
            alpha_endo=alpha_endo,
            alpha_epi=alpha_epi,
            beta_endo=beta_endo,
            beta_epi=beta_epi,
            tol=tol,
        )

        f0[xdofs[i]] = Q_fiber[0, 0]
        f0[ydofs[i]] = Q_fiber[1, 0]
        f0[zdofs[i]] = Q_fiber[2, 0]

        s0[xdofs[i]] = Q_fiber[0, 1]
        s0[ydofs[i]] = Q_fiber[1, 1]
        s0[zdofs[i]] = Q_fiber[2, 1]

        n0[xdofs[i]] = Q_fiber[0, 2]
        n0[ydofs[i]] = Q_fiber[1, 2]
        n0[zdofs[i]] = Q_fiber[2, 2]
