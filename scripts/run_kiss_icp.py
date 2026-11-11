import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lidar.io_kitti import list_scans, load_scan
from kiss_icp.kiss_icp import KissICP
from kiss_icp.config import load_config

DRIVE = 'data/2011_09_26/2011_09_26_drive_0005_sync'


def save_kitti(path, poses):
    with open(path, 'w') as f:
        for T in poses:
            f.write(' '.join(f'{v:.6f}' for v in T[:3, :].ravel()) + '\n')


def fake_timestamps(pts):
    # kitti .bin has no per-point time. a spinning lidar sweeps around once per scan,
    # so the horizontal angle tells us roughly when each point was measured (0 = start, 1 = end)
    yaw = -np.arctan2(pts[:, 1], pts[:, 0])
    return 0.5 * (yaw / np.pi + 1.0)


def run(deskew):
    cfg = load_config(None)
    cfg.data.deskew = deskew
    cfg.data.max_range = 100.0
    odom = KissICP(config=cfg)
    poses = []
    t0 = time.time()
    for path in list_scans(DRIVE):
        pts = load_scan(path)[:, :3].astype(np.float64)
        odom.register_frame(pts, fake_timestamps(pts))
        # newer versions keep last_pose, older ones keep a list of poses
        pose = odom.last_pose if hasattr(odom, 'last_pose') else odom.poses[-1]
        poses.append(np.array(pose).copy())
    sec = time.time() - t0
    print(f'kiss-icp deskew={deskew}: {len(poses)} scans in {sec:.1f} s ({1000 * sec / len(poses):.0f} ms per scan)')
    return poses


save_kitti('results/kiss.txt', run(deskew=True))
save_kitti('results/kiss_nodeskew.txt', run(deskew=False))
print('saved results/kiss.txt and results/kiss_nodeskew.txt')
