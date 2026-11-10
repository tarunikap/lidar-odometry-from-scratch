import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.transforms import inv_T, rot_to_euler
from lidar.io_kitti import list_scans, load_scan, load_velo_poses
from lidar.preprocess import voxel_downsample, estimate_normals
from lidar.icp import icp_point_to_point, icp_point_to_plane

DRIVE = 'data/2011_09_26/2011_09_26_drive_0005_sync'
CALIB = 'data/2011_09_26'

i = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # compare scan i and i+1
scans = list_scans(DRIVE)
tgt = voxel_downsample(load_scan(scans[i])[:, :3], 0.5)
src = voxel_downsample(load_scan(scans[i + 1])[:, :3], 0.5)
normals = estimate_normals(tgt)

poses = load_velo_poses(DRIVE, CALIB)
T_true = inv_T(poses[i]) @ poses[i + 1]
yaw_true = np.degrees(rot_to_euler(T_true[:3, :3])[2])
print(f'scans {i} -> {i + 1}, true motion {T_true[:3, 3].round(3)} m, yaw {yaw_true:.3f} deg\n')

for name, fn, args in [('point-to-point', icp_point_to_point, (src, tgt)),
                       ('point-to-plane', icp_point_to_plane, (src, tgt, normals))]:
    t0 = time.time()
    T, iters, err = fn(*args)
    ms = (time.time() - t0) * 1000
    terr = 100 * np.linalg.norm(T[:3, 3] - T_true[:3, 3])
    yaw = np.degrees(rot_to_euler(T[:3, :3])[2])
    print(f'{name}: {ms:.0f} ms, {iters} iters, motion {T[:3, 3].round(3)}, '
          f'error {terr:.1f} cm, yaw {yaw:.3f} deg')
