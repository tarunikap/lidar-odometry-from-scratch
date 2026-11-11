import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lidar.io_kitti import list_scans, load_scan, load_velo_poses
from kiss_icp.kiss_icp import KissICP
from kiss_icp.config import load_config

DRIVE = 'data/2011_09_26/2011_09_26_drive_0005_sync'
CALIB = 'data/2011_09_26'


def timestamps(pts, flip):
    yaw = -np.arctan2(pts[:, 1], pts[:, 0])
    t = 0.5 * (yaw / np.pi + 1.0)
    return 1.0 - t if flip else t       # flip = assume the lidar spins the other way


def correct_kitti(pts, deg=0.205):
    # tilt every point up by 0.205 deg (fix used by IMLS-SLAM / CT-ICP for kitti)
    # axis is perpendicular to the point and to z, so the point rotates "upward"
    k = np.cross(pts, [0.0, 0.0, 1.0])
    k = k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-12)
    th = np.deg2rad(deg)
    return pts * np.cos(th) + np.cross(k, pts) * np.sin(th)


def run(deskew, flip=False, correct=False):
    cfg = load_config(None)
    cfg.data.deskew = deskew
    cfg.data.max_range = 100.0
    odom = KissICP(config=cfg)
    poses = []
    for path in list_scans(DRIVE):
        pts = load_scan(path)[:, :3].astype(np.float64)
        if correct:
            pts = correct_kitti(pts)
        odom.register_frame(pts, timestamps(pts, flip))
        poses.append(np.array(odom.last_pose).copy())
    return poses


gt = load_velo_poses(DRIVE, CALIB)
def length(P):
    return sum(np.linalg.norm(P[i + 1][:3, 3] - P[i][:3, 3]) for i in range(len(P) - 1))

print(f'GPS path length: {length(gt):.2f} m\n')
for name, kw in [('no deskew', dict(deskew=False)),
                 ('deskew', dict(deskew=True)),
                 ('deskew, flipped time', dict(deskew=True, flip=True)),
                 ('no deskew + 0.205 deg fix', dict(deskew=False, correct=True)),
                 ('deskew flipped + 0.205 deg fix', dict(deskew=True, flip=True, correct=True))]:
    P = run(**kw)
    err = np.linalg.norm(P[-1][:3, 3] - gt[-1][:3, 3])
    print(f'{name:32s} length {length(P):.2f} m   final error {err:.2f} m')
