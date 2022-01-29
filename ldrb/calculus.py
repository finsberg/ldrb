import numba
import numpy as np


@numba.njit
def from_rotation_matrix(rot: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to a quaternion

    Parameters
    ----------
    rot : np.ndarray
        The rotation matrix

    Returns
    -------
    np.ndarray
        The quaternion
    """
    diagonals = np.empty(4)
    diagonals[0] = rot[0, 0]
    diagonals[1] = rot[1, 1]
    diagonals[2] = rot[2, 2]
    diagonals[3] = rot[0, 0] + rot[1, 1] + rot[2, 2]
    indices = np.argmax(diagonals)

    q = diagonals  # reuse storage space

    if indices == 0:

        q[0] = rot[2, 1] - rot[1, 2]
        q[1] = 1 + rot[0, 0] - rot[1, 1] - rot[2, 2]
        q[2] = rot[0, 1] + rot[1, 0]
        q[3] = rot[0, 2] + rot[2, 0]

    if indices == 1:
        q[0] = rot[0, 2] - rot[2, 0]
        q[1] = rot[1, 0] + rot[0, 1]
        q[2] = 1 - rot[0, 0] + rot[1, 1] - rot[2, 2]
        q[3] = rot[1, 2] + rot[2, 1]

    if indices == 2:
        q[0] = rot[1, 0] - rot[0, 1]
        q[1] = rot[2, 0] + rot[0, 2]
        q[2] = rot[2, 1] + rot[1, 2]
        q[3] = 1 - rot[0, 0] - rot[1, 1] + rot[2, 2]

    if indices == 3:
        q[0] = 1 + rot[0, 0] + rot[1, 1] + rot[2, 2]
        q[1] = rot[2, 1] - rot[1, 2]
        q[2] = rot[0, 2] - rot[2, 0]
        q[3] = rot[1, 0] - rot[0, 1]

    q /= np.linalg.norm(q)
    return q


@numba.njit
def as_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to a rotatoin matrix

    Parameters
    ----------
    q : np.ndarray
        The quaternion

    Returns
    -------
    np.ndarray
        The rotation matrix
    """
    n = np.linalg.norm(q)
    if abs(n - 1.0) < 1e-14:  # Input q is basically normalized
        return np.array(
            [
                [
                    1 - 2 * (q[2] ** 2 + q[3] ** 2),
                    2 * (q[1] * q[2] - q[3] * q[0]),
                    2 * (q[1] * q[3] + q[2] * q[0]),
                ],
                [
                    2 * (q[1] * q[2] + q[3] * q[0]),
                    1 - 2 * (q[1] ** 2 + q[3] ** 2),
                    2 * (q[2] * q[3] - q[1] * q[0]),
                ],
                [
                    2 * (q[1] * q[3] - q[2] * q[0]),
                    2 * (q[2] * q[3] + q[1] * q[0]),
                    1 - 2 * (q[1] ** 2 + q[2] ** 2),
                ],
            ],
        )
    else:  # Input q is not normalized
        return np.array(
            [
                [
                    1 - 2 * (q[2] ** 2 + q[3] ** 2) / n,
                    2 * (q[1] * q[2] - q[3] * q[0]) / n,
                    2 * (q[1] * q[3] + q[2] * q[0]) / n,
                ],
                [
                    2 * (q[1] * q[2] + q[3] * q[0]) / n,
                    1 - 2 * (q[1] ** 2 + q[3] ** 2) / n,
                    2 * (q[2] * q[3] - q[1] * q[0]) / n,
                ],
                [
                    2 * (q[1] * q[3] - q[2] * q[0]) / n,
                    2 * (q[2] * q[3] + q[1] * q[0]) / n,
                    1 - 2 * (q[1] ** 2 + q[2] ** 2) / n,
                ],
            ],
        )


@numba.njit
def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaterion multiplication

    Parameters
    ----------
    q1 : np.ndarray
        First quaternion
    q2 : np.ndarray
        Second quaternion

    Returns
    -------
    np.ndarray
        The product of q1 and q2
    """
    q = np.zeros(4)
    q[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    q[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    q[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    q[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return q


@numba.njit
def slerp(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    """Sperical linear interpolation between two quaternions

    Parameters
    ----------
    p0 : np.ndarray
        Quaternion corresponding to t=0
    p1 : np.ndarray
        Quaternions corresponding to t=1
    t : float
        Interpolation factor

    Returns
    -------
    np.ndarray
        Quaternion corresponding to t=t
    """

    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


@numba.njit
def bislerp(
    Qa: np.ndarray,
    Qb: np.ndarray,
    t: float,
) -> np.ndarray:
    r"""
    Linear interpolation of two orthogonal matrices.
    Assiume that :math:`Q_a` and :math:`Q_b` refers to
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
    qa = from_rotation_matrix(Qa)
    qb = from_rotation_matrix(Qb)

    quat_i = np.array([0, 1, 0, 0])
    quat_j = np.array([0, 0, 1, 0])
    quat_k = np.array([0, 0, 0, 1])

    quat_array = [
        qa,
        -qa,
        quat_multiply(qa, quat_i),
        -quat_multiply(qa, quat_i),
        quat_multiply(qa, quat_j),
        -quat_multiply(qa, quat_j),
        quat_multiply(qa, quat_k),
        -quat_multiply(qa, quat_k),
    ]
    dot_arr = np.array([np.abs((qi * qb).sum()) for qi in quat_array])
    max_idx = int(np.argmax(dot_arr))
    max_dot = dot_arr[max_idx]
    qm = quat_array[max_idx]

    if max_dot > 1 - tol:
        return Qb

    qm_norm = slerp(qm, qb, t)

    return as_rotation_matrix(qm_norm)


@numba.njit
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
    single degre of freedom

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
def normalize(u: np.ndarray) -> np.ndarray:
    """
    Normalize vector
    """
    return u / np.linalg.norm(u)


@numba.njit
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


@numba.njit
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
def _compute_fiber_sheet_system(
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
        if Q_fiber is None:
            continue

        f0[xdofs[i]] = Q_fiber[0, 0]
        f0[ydofs[i]] = Q_fiber[1, 0]
        f0[zdofs[i]] = Q_fiber[2, 0]

        s0[xdofs[i]] = Q_fiber[0, 1]
        s0[ydofs[i]] = Q_fiber[1, 1]
        s0[zdofs[i]] = Q_fiber[2, 1]

        n0[xdofs[i]] = Q_fiber[0, 2]
        n0[ydofs[i]] = Q_fiber[1, 2]
        n0[zdofs[i]] = Q_fiber[2, 2]
