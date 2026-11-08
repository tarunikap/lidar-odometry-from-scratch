import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lidar.io_kitti import list_scans, load_scan
from lidar.range_image import to_range_image, from_range_image

DRIVE = 'data/2011_09_26/2011_09_26_drive_0005_sync'

pts = load_scan(list_scans(DRIVE)[0])[:, :3]
img = to_range_image(pts)

filled = (img > 0).sum()
print('image shape:', img.shape)
print('filled pixels:', filled, 'of', img.size, f'({100 * filled / img.size:.1f}%)')
print('points that got merged into a shared pixel:', len(pts) - filled)

# go back to 3D and see how far each point moved
back = from_range_image(img)
d, _ = cKDTree(pts).query(back)
print(f'back-projection error: median {100 * np.median(d):.1f} cm, 95% {100 * np.percentile(d, 95):.1f} cm')

shown = np.ma.masked_where(img < 0, img)
plt.figure(figsize=(16, 2.5))
plt.imshow(shown, cmap='turbo', vmin=0, vmax=40, aspect='auto')
plt.colorbar(label='range (m)')
plt.title('KITTI scan 0 as a range image (64 beams x 1024 angles)')
plt.tight_layout()
plt.savefig('docs/range_image.png', dpi=120)
print('saved docs/range_image.png')
