import numpy as np
from scipy.spatial import cKDTree
from common.transforms import make_T, apply_T


def best_fit_transform(P, Q):
    # find R, t that best moves points P onto points Q (rows are matched pairs)
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    H = (P - p_mean).T @ (Q - q_mean)     # 3x3
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        # got a mirror flip, not a real rotation, so fix it
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = q_mean - R @ p_mean
    return make_T(R, t)


def icp_point_to_point(src, tgt, T_init=None, max_iter=50, max_dist=1.0, tol=1e-6):
    # moves src onto tgt. returns the 4x4 T, number of iterations, final mean match distance
    tree = cKDTree(tgt)                     # build once, reuse every iteration
    T = np.eye(4) if T_init is None else T_init.copy()
    for it in range(max_iter):
        moved = apply_T(T, src)             # 1. move source with current guess
        d, idx = tree.query(moved)          # 2. nearest target point for each source point
        keep = d < max_dist                 # 3. drop pairs that are too far apart
        if keep.sum() < 10:
            break
        dT = best_fit_transform(moved[keep], tgt[idx[keep]])   # 4. best small correction
        T = dT @ T                          # 5. add correction on top of current guess
        if abs(dT - np.eye(4)).max() < tol:  # correction is tiny -> converged
            break
    return T, it + 1, d[keep].mean()


def skew(w):
    # w x v == skew(w) @ v
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def exp_so3(w):
    # rotation vector -> rotation matrix (Rodrigues formula)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def icp_point_to_plane(src, tgt, tgt_normals, T_init=None, max_iter=50, max_dist=1.0, tol=1e-6):
    tree = cKDTree(tgt)
    T = np.eye(4) if T_init is None else T_init.copy()
    for it in range(max_iter):
        moved = apply_T(T, src)
        d, idx = tree.query(moved)
        keep = d < max_dist
        if keep.sum() < 10:
            break
        p = moved[keep]
        q = tgt[idx[keep]]
        n = tgt_normals[idx[keep]]

        A = np.hstack([np.cross(p, n), n])      # N x 6, one row a_i per pair
        b = np.sum(n * (p - q), axis=1)         # N, distance of each p to its plane
        x = np.linalg.solve(A.T @ A, -A.T @ b)  # 6 numbers: [w, t]

        dT = make_T(exp_so3(x[:3]), x[3:])
        T = dT @ T
        if abs(x).max() < tol:
            break
    return T, it + 1, d[keep].mean()


import open3d as o3d


def icp_gicp(src, tgt, T_init=None, max_dist=1.0, max_iter=50):
    # generalized icp from open3d. it builds a covariance "pancake" for every point in both clouds
    s = o3d.geometry.PointCloud()
    s.points = o3d.utility.Vector3dVector(src)
    t = o3d.geometry.PointCloud()
    t.points = o3d.utility.Vector3dVector(tgt)
    T0 = np.eye(4) if T_init is None else T_init
    reg = o3d.pipelines.registration
    res = reg.registration_generalized_icp(
        s, t, max_dist, T0,
        reg.TransformationEstimationForGeneralizedICP(),
        reg.ICPConvergenceCriteria(max_iteration=max_iter))
    # open3d doesn't report the iteration count, so return None for it
    return res.transformation.copy(), None, res.inlier_rmse
