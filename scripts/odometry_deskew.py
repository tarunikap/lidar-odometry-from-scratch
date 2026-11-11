import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lidar.io_kitti import list_scans, load_scan, load_velo_poses
from lidar.preprocess import voxel_downsample, estimate_normals
from lidar.icp import icp_point_to_plane
from lidar.deskew import deskew, timestamps_from_angle

DRIVE = 'data/2011_09_26/2011_09_26_drive_0005_sync'
CALIB = 'data/2011_09_26'


def run(mode):
    scans = list_scans(DRIVE)
    poses = [np.eye(4)]
    T_rel = np.eye(4)
    steps = []
    prev = voxel_downsample(load_scan(scans[0])[:, :3], 0.5)
    for i in range(1, len(scans)):
        raw = load_scan(scans[i])[:, :3].astype(np.float64)
        s = timestamps_from_angle(raw)
        if mode == 'start':
            raw = deskew(raw, s, T_rel)            # to the start of the spin
        elif mode == 'start, flipped':
            raw = deskew(raw, 1 - s, T_rel)        # assume it spins the other way
        elif mode == 'middle':
            raw = deskew(raw, s - 0.5, T_rel)      # to the middle of the spin (kiss-icp style)
        elif mode == 'middle, flipped':
            raw = deskew(raw, 0.5 - s, T_rel)
        cur = voxel_downsample(raw, 0.5)
        T_rel, _, _ = icp_point_to_plane(cur, prev, estimate_normals(prev), T_init=T_rel)
        poses.append(poses[-1] @ T_rel)
        steps.append(np.linalg.norm(T_rel[:3, 3]))
        prev = cur
    return poses, steps


def length(P):
    return sum(np.linalg.norm(P[i + 1][:3, 3] - P[i][:3, 3]) for i in range(len(P) - 1))


gt = load_velo_poses(DRIVE, CALIB)
print('GPS length 69.38 m, speed x time 71.32 m\n')
for mode in ['none', 'start', 'start, flipped', 'middle', 'middle, flipped']:
    P, steps = run(mode)
    first = ' '.join(f'{x:.2f}' for x in steps[:8])
    print(f'{mode:16s} length {length(P):7.2f} m | first steps (m): {first} | biggest step {max(steps):.2f} m')
