import numpy as np
import open3d as o3d


def voxel_downsample(pts, voxel=0.5):
    # which cube is each point in? e.g. point (1.2, 0.3, 0.9) with voxel 0.5 -> cube (2, 0, 1)
    keys = np.floor(pts / voxel).astype(np.int64)
    # find the unique cubes, and for every point which cube number it belongs to
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    # add up all points in each cube, then divide by how many -> average point per cube
    out = np.zeros((len(counts), 3))
    np.add.at(out, inv, pts)
    return out / counts[:, None]


def voxel_downsample_o3d(pts, voxel=0.5):
    # same thing using open3d, to compare against ours
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return np.asarray(pcd.voxel_down_sample(voxel).points)


def estimate_normals(pts, radius=1.0, max_nn=30):
    # for each point, fit a plane to its neighbors within radius, the plane's direction is the normal
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pcd.normals)
