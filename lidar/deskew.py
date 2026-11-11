import numpy as np
from common.transforms import rot_to_quat


def timestamps_from_angle(pts):
    # kitti .bin has no per-point time. a spinning lidar sweeps once per scan,
    # so horizontal angle -> fraction of the spin (0 = start, 1 = end)
    yaw = -np.arctan2(pts[:, 1], pts[:, 0])
    return 0.5 * (yaw / np.pi + 1.0)


def axis_angle(R):
    # rotation matrix -> (unit axis, angle), via the quaternion
    q = rot_to_quat(R)
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return v / n, 2 * np.arctan2(n, q[0])


def rotate(pts, k, th):
    # rotate each point about axis k by its own angle th (Rodrigues, one angle per point)
    c = np.cos(th)[:, None]
    s = np.sin(th)[:, None]
    return pts * c + np.cross(k, pts) * s + k[None, :] * (pts @ k)[:, None] * (1 - c)


def deskew(pts, s, T_motion):
    # pts: Nx3 measured points, s: when each was measured (0..1), T_motion: car motion over one spin
    # moves every point into the frame of the START of the spin
    k, ang = axis_angle(T_motion[:3, :3])
    return rotate(pts, k, s * ang) + s[:, None] * T_motion[:3, 3]


def skew(pts, s, T_motion):
    # the opposite: pretend points were measured during motion (used only to test deskew)
    k, ang = axis_angle(T_motion[:3, :3])
    return rotate(pts - s[:, None] * T_motion[:3, 3], k, -s * ang)
