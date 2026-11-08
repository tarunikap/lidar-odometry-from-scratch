import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# so python can find common/ and lidar/ when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lidar.io_kitti import list_scans, load_scan

DRIVE = 'data/2011_09_26/2011_09_26_drive_0005_sync'


def color_by_height(z, zmin=-2.0, zmax=2.0):
    # low points dark blue, high points yellow
    t = np.clip((z - zmin) / (zmax - zmin), 0, 1)
    return plt.get_cmap('viridis')(t)[:, :3]


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    pts = load_scan(list_scans(DRIVE)[idx])
    print(f'scan {idx}: {len(pts)} points')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(color_by_height(pts[:, 2]))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(f'KITTI scan {idx}', 1280, 720)
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))
    opt = vis.get_render_option()
    opt.point_size = 1.5
    opt.background_color = np.array([0.05, 0.05, 0.05])

    def save(v):
        out = f'docs/scan_{idx}.png'
        v.capture_screen_image(out)
        print('saved', out)
        return False

    vis.register_key_callback(ord('S'), save)
    print('drag to rotate, scroll to zoom, press S to save a screenshot, Q to quit')
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
