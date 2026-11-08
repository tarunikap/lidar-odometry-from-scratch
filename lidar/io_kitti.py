import os
import numpy as np
from common.transforms import euler_to_rot, make_T, inv_T


def load_scan(path):
    # each point is 4 float32 numbers: x, y, z, intensity
    pts = np.fromfile(path, dtype=np.float32)
    return pts.reshape(-1, 4)


def list_scans(drive_dir):
    folder = os.path.join(drive_dir, 'velodyne_points', 'data')
    files = sorted(f for f in os.listdir(folder) if f.endswith('.bin'))
    return [os.path.join(folder, f) for f in files]


def read_calib_file(path):
    # lines look like "R: 1 0 0 0 1 0 0 0 1"
    data = {}
    with open(path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            try:
                data[key] = np.array([float(v) for v in val.split()])
            except ValueError:
                pass  # skip lines like calib_time that are not numbers
    return data


def load_rigid_calib(path):
    # calib_imu_to_velo.txt and calib_velo_to_cam.txt both store R (3x3) and T (3x1)
    c = read_calib_file(path)
    return make_T(c['R'].reshape(3, 3), c['T'])


def load_oxts_poses(drive_dir):
    # GPS/IMU readings -> 4x4 poses of the IMU in a local world frame
    folder = os.path.join(drive_dir, 'oxts', 'data')
    files = sorted(os.listdir(folder))
    er = 6378137.0  # earth radius in m
    scale = None
    poses = []
    for fname in files:
        vals = np.loadtxt(os.path.join(folder, fname))
        lat, lon, alt, roll, pitch, yaw = vals[:6]
        if scale is None:
            scale = np.cos(lat * np.pi / 180)
        # mercator projection to turn lat/lon into meters
        tx = scale * lon * np.pi * er / 180
        ty = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))
        poses.append(make_T(euler_to_rot(roll, pitch, yaw), [tx, ty, alt]))
    T0_inv = inv_T(poses[0])
    return [T0_inv @ T for T in poses]


def load_velo_poses(drive_dir, calib_dir):
    # poses of the lidar itself, using the imu -> velo calibration
    T_velo_imu = load_rigid_calib(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
    T_imu_velo = inv_T(T_velo_imu)
    velo = [T @ T_imu_velo for T in load_oxts_poses(drive_dir)]
    # shift so the lidar starts at the origin
    T0_inv = inv_T(velo[0])
    return [T0_inv @ T for T in velo]
