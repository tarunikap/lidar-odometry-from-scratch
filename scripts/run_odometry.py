import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lidar.io_kitti import list_scans, load_scan, load_velo_poses
from lidar.preprocess import voxel_downsample, estimate_normals
from lidar.icp import icp_point_to_point, icp_point_to_plane

DRIVE = 'data/2011_09_26/2011_09_26_drive_0005_sync'
CALIB = 'data/2011_09_26'
VOXEL = 0.5


def save_kitti(path, poses):
    # kitti format: each pose as one line of 12 numbers (top 3 rows of the 4x4)
    with open(path, 'w') as f:
        for T in poses:
            f.write(' '.join(f'{v:.6f}' for v in T[:3, :].ravel()) + '\n')


def run(method):
    scans = list_scans(DRIVE)
    poses = [np.eye(4)]          # start at the origin
    T_rel = np.eye(4)            # first guess: no motion
    iters_total = 0
    t0 = time.time()

    prev = voxel_downsample(load_scan(scans[0])[:, :3], VOXEL)
    for i in range(1, len(scans)):
        cur = voxel_downsample(load_scan(scans[i])[:, :3], VOXEL)
        # last step's motion is the starting guess (constant velocity)
        if method == 'p2p':
            T_rel, it, _ = icp_point_to_point(cur, prev, T_init=T_rel)
        else:
            T_rel, it, _ = icp_point_to_plane(cur, prev, estimate_normals(prev), T_init=T_rel)
        poses.append(poses[-1] @ T_rel)    # chain the new step onto the path
        iters_total += it
        prev = cur                          # current scan is the target next time

    sec = time.time() - t0
    n = len(scans) - 1
    print(f'{method}: {len(scans)} scans in {sec:.1f} s '
          f'({1000 * sec / n:.0f} ms per scan, avg {iters_total / n:.1f} iters)')
    return poses


gt = load_velo_poses(DRIVE, CALIB)
save_kitti('results/gt.txt', gt)

results = {}
for m in ['p2p', 'p2plane']:
    results[m] = run(m)
    save_kitti(f'results/{m}.txt', results[m])

# drift = how far the last pose is from the gps last pose, as % of distance driven
dist = sum(np.linalg.norm(gt[i + 1][:3, 3] - gt[i][:3, 3]) for i in range(len(gt) - 1))
print(f'\ndrive length: {dist:.1f} m')
for m, poses in results.items():
    e = np.linalg.norm(poses[-1][:3, 3] - gt[-1][:3, 3])
    print(f'{m}: final position error {e:.2f} m ({100 * e / dist:.2f}% of distance)')

# top-down plot of all three paths
plt.figure(figsize=(8, 6))
g = np.array([T[:3, 3] for T in gt])
plt.plot(g[:, 0], g[:, 1], 'k-', lw=3, label='GPS truth')
for m, c in [('p2p', 'tab:red'), ('p2plane', 'tab:blue')]:
    p = np.array([T[:3, 3] for T in results[m]])
    plt.plot(p[:, 0], p[:, 1], color=c, lw=1.5, label=m)
plt.axis('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.title('LiDAR odometry vs GPS, KITTI 2011_09_26_drive_0005')
plt.tight_layout()
plt.savefig('docs/trajectory.png', dpi=120)
print('saved docs/trajectory.png')
