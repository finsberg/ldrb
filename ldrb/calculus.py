import numba
import numpy as np

_float_1D_array = numba.types.Array(numba.float64, 1, "C")
_float_2D_array = numba.types.Array(numba.float64, 2, "C")


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
    u_norm = np.linalg.norm(u)
    if u_norm > 0.0:
        return u / u_norm
    return u


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
    dot = q1.dot(q2)
    q3 = q2
    if dot < 0.0:
        dot = -dot
        q3 = -q2

    if dot < 0.9999:
        angle = np.arccos(dot)
        a = np.sin(angle * (1 - t)) / np.sin(angle)
        b = np.sin(angle * t) / np.sin(angle)
        return a * q1 + b * q3

    # Angle is close to zero - do linear interpolation
    return q1 * (1 - t) + q3 * t


@numba.njit(_float_1D_array(_float_2D_array))
def rot2quat(Q: np.ndarray) -> np.ndarray:
    t = Q[0][0] + Q[1][1] + Q[2][2]

    if t > 0:
        r = np.sqrt(1.0 + t)
        s = 0.5 / r

        a = 0.5 * r
        b = (Q[2][1] - Q[1][2]) * s
        c = (Q[0][2] - Q[2][0]) * s
        d = (Q[1][0] - Q[0][1]) * s

    elif Q[0][0] > Q[1][1] and Q[0][0] > Q[2][2]:
        s = 2 * np.sqrt(1.0 + Q[0][0] - Q[1][1] - Q[2][2])

        a = (Q[2][1] - Q[1][2]) / s
        b = 0.25 * s
        c = (Q[0][1] + Q[1][0]) / s
        d = (Q[0][2] + Q[2][0]) / s

    elif Q[1][1] > Q[0][0] and Q[1][1] > Q[2][2]:
        s = 2 * np.sqrt(1.0 + Q[1][1] - Q[0][0] - Q[2][2])

        a = (Q[0][2] - Q[2][0]) / s
        b = (Q[0][1] + Q[1][0]) / s
        c = 0.25 * s
        d = (Q[1][2] + Q[2][1]) / s

    else:
        s = 2 * np.sqrt(1.0 + Q[2][2] - Q[0][0] - Q[1][1])

        a = (Q[1][0] - Q[0][1]) / s
        b = (Q[0][2] + Q[2][0]) / s
        c = (Q[1][2] + Q[2][1]) / s
        d = 0.25 * s

    return normalize(np.array([a, b, c, d]))


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

    # if ~Qa.any() and ~Qb.any():
    #     return np.zeros((3, 3))
    # if ~Qa.any():
    #     return Qb
    # if ~Qb.any():
    #     return Qa

    # tol = 1e-5

    # if t < tol:
    #     return Qa
    # if t > 1 - tol:
    #     return Qb

    qa = rot2quat(Qa)
    qb = rot2quat(Qb)

    a = qa[0]
    b = qa[1]
    c = qa[2]
    d = qa[3]

    # i_qa = np.array([-b, a, -d, c])
    # j_qa = np.array([-c, d, a, -b])
    # k_qa = np.array([-d, -c, b, a])

    qa_i = np.array([-b, a, d, -c])
    qa_j = np.array([-c, -d, a, b])
    qa_k = np.array([-d, c, -b, a])

    # quat_array = [
    #     qa,
    #     i_qa,
    #     j_qa,
    #     k_qa,
    # ]

    quat_array = [
        qa,
        qa_i,
        qa_j,
        qa_k,
    ]

    qm = quat_array[0]
    max_dot = abs(qm.dot(qb))

    for v in quat_array[1:]:
        dot = abs(v.dot(qb))
        if dot > max_dot:
            max_dot = dot
            qm = v

    qm_slerp = slerp(qm, qb, t)
    return quat2rot(qm_slerp)


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

    e2 = v - (e1.dot(v)) * e1
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
            # We are not in the septum
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
            if lv_rv >= 0.9:
                # We are in the LV region
                marker_scalar[sdofs[i]] = 1
                alpha_endo = alpha_endo_lv
                beta_endo = beta_endo_lv
                alpha_epi = alpha_epi_lv
                beta_epi = beta_epi_lv
            elif lv_rv <= 0.1:
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

        if lv + rv < 1e-12:
            depth = 0.5
        else:
            depth = rv / (lv + rv)

        alpha_s = alpha_endo * (1 - depth) - alpha_endo * depth
        alpha_w = alpha_endo * (1 - epi) + alpha_epi * epi
        beta_s = beta_endo * (1 - depth) - beta_endo * depth
        beta_w = beta_endo * (1 - epi) + beta_epi * epi

        Q_lv = axis(grad_ab, -grad_lv)
        Q_lv = orient(Q_lv, alpha_s, beta_s)

        Q_rv = axis(grad_ab, grad_rv)
        Q_rv = orient(Q_rv, alpha_s, beta_s)

        Q_epi = axis(grad_ab, grad_epi)
        Q_epi = orient(Q_epi, alpha_w, beta_w)

        Q_endo = bislerp(Q_lv, Q_rv, depth)
        Q_fiber = bislerp(Q_endo, Q_epi, epi)

        f0[xdofs[i]] = Q_fiber[0, 0]
        f0[ydofs[i]] = Q_fiber[1, 0]
        f0[zdofs[i]] = Q_fiber[2, 0]

        s0[xdofs[i]] = Q_fiber[0, 1]
        s0[ydofs[i]] = Q_fiber[1, 1]
        s0[zdofs[i]] = Q_fiber[2, 1]

        n0[xdofs[i]] = Q_fiber[0, 2]
        n0[ydofs[i]] = Q_fiber[1, 2]
        n0[zdofs[i]] = Q_fiber[2, 2]
