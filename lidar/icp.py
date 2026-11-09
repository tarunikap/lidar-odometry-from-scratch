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
