import numpy as np


def to_range_image(pts, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
    # pts is Nx3. returns HxW image where each pixel = range in m, -1 = empty
    r = np.linalg.norm(pts, axis=1)
    pts, r = pts[r > 0.1], r[r > 0.1]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    yaw = np.arctan2(y, x)      # horizontal angle, -pi to pi
    pitch = np.arcsin(z / r)    # vertical angle
    fu, fd = np.deg2rad(fov_up), np.deg2rad(fov_down)

    # angle -> pixel. column 0 is behind the car, middle column is straight ahead
    u = 0.5 * (1 - yaw / np.pi) * W
    v = (fu - pitch) / (fu - fd) * H   # row 0 = highest beam
    u = np.clip(np.floor(u), 0, W - 1).astype(int)
    v = np.clip(np.floor(v), 0, H - 1).astype(int)

    img = np.full((H, W), -1.0)
    # write far points first, so if two points hit the same pixel the closer one wins
    order = np.argsort(-r)
    img[v[order], u[order]] = r[order]
    return img


def from_range_image(img, fov_up=3.0, fov_down=-25.0):
    # back to Nx3 points, using the center angle of each pixel
    H, W = img.shape
    v, u = np.nonzero(img > 0)
    r = img[v, u]
    fu, fd = np.deg2rad(fov_up), np.deg2rad(fov_down)
    yaw = np.pi * (1 - 2 * (u + 0.5) / W)
    pitch = fu - (v + 0.5) / H * (fu - fd)
    x = r * np.cos(pitch) * np.cos(yaw)
    y = r * np.cos(pitch) * np.sin(yaw)
    z = r * np.sin(pitch)
    return np.stack([x, y, z], axis=1)
