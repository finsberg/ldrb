import numba
import numpy as np
import quaternion


@numba.njit
def from_rotation_matrix(rot):
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
def quat_multiply(q1, q2):
    q = np.zeros(4)
    q[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    q[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    q[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    q[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return q


# @numba.jit
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

    qa = np.quaternion(*qa)
    qb = np.quaternion(*qb)

    quat_i = np.quaternion(0, 1, 0, 0)
    quat_j = np.quaternion(0, 0, 1, 0)
    quat_k = np.quaternion(0, 0, 0, 1)

    quat_array = [
        qa,
        -qa,
        qa * quat_i,
        -qa * quat_i,
        qa * quat_j,
        -qa * quat_j,
        qa * quat_k,
        -qa * quat_k,
    ]
    dot_arr = [abs((qi.components * qb.components).sum()) for qi in quat_array]
    max_idx = int(np.argmax(dot_arr))
    max_dot = dot_arr[max_idx]
    qm = quat_array[max_idx]

    if max_dot > 1 - tol:
        return Qb

    qm_slerp = quaternion.slerp(qm, qb, 0, 1, t)
    qm_norm = qm_slerp.normalized()
    Qab = quaternion.as_rotation_matrix(qm_norm)
    return Qab


# @numba.jit
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


#
# @numba.jit
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
    lv_gradient,
    rv_gradient,
    epi_gradient,
    apex_gradient,
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
    for i in range(len(xdofs)):

        lv = lv_scalar[sdofs[i]]
        rv = rv_scalar[sdofs[i]]
        epi = epi_scalar[sdofs[i]]
        grad_lv = lv_gradient[[xdofs[i], ydofs[i], zdofs[i]]]
        grad_rv = rv_gradient[[xdofs[i], ydofs[i], zdofs[i]]]
        grad_epi = epi_gradient[[xdofs[i], ydofs[i], zdofs[i]]]
        grad_ab = apex_gradient[[xdofs[i], ydofs[i], zdofs[i]]]

        if lv > tol and rv < tol:
            # We are in the LV region
            alpha_endo = alpha_endo_lv
            beta_endo = beta_endo_lv
            alpha_epi = alpha_epi_lv
            beta_epi = beta_epi_lv
        elif lv < tol and rv > tol:
            # We are in the RV region
            alpha_endo = alpha_endo_rv
            beta_endo = beta_endo_rv
            alpha_epi = alpha_epi_rv
            beta_epi = beta_epi_rv
        elif lv > tol and rv > tol:
            # We are in the septum
            alpha_endo = alpha_endo_sept
            beta_endo = beta_endo_sept
            alpha_epi = alpha_epi_sept
            beta_epi = beta_epi_sept
        else:
            alpha_endo = alpha_endo_lv
            beta_endo = beta_endo_lv
            alpha_epi = alpha_epi_lv
            beta_epi = beta_epi_lv

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
        f0[[xdofs[i], ydofs[i], zdofs[i]]] = Q_fiber.T[0]
        s0[[xdofs[i], ydofs[i], zdofs[i]]] = Q_fiber.T[1]
        n0[[xdofs[i], ydofs[i], zdofs[i]]] = Q_fiber.T[2]
