import numpy as np
from common.transforms import make_T, euler_to_rot
from lidar.deskew import deskew, skew, timestamps_from_angle


def test_deskew_undoes_skew():
    rng = np.random.default_rng(0)
    # fake scene: points on a ring of walls 5 to 30 m away
    ang = rng.uniform(-np.pi, np.pi, 5000)
    r = rng.uniform(5, 30, 5000)
    pts = np.stack([r * np.cos(ang), r * np.sin(ang), rng.uniform(-2, 2, 5000)], axis=1)
    s = timestamps_from_angle(pts)

    # car drives 0.45 m forward and turns 1 deg during the spin
    T = make_T(euler_to_rot(0, 0, np.deg2rad(1.0)), [0.45, 0.02, 0.0])

    smeared = skew(pts, s, T)
    fixed = deskew(smeared, s, T)

    assert np.abs(smeared - pts).max() > 0.1     # the smear is real (more than 10 cm somewhere)
    assert np.abs(fixed - pts).max() < 1e-9      # and deskew removes it completely


def test_no_motion_no_change():
    pts = np.random.default_rng(1).normal(size=(100, 3)) * 10
    s = timestamps_from_angle(pts)
    assert np.allclose(deskew(pts, s, np.eye(4)), pts)
