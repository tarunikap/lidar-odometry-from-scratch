import numpy as np
from common.transforms import *


def test_euler_roundtrip():
    r, p, y = 0.1, -0.4, 2.0
    R = euler_to_rot(r, p, y)
    assert np.allclose(rot_to_euler(R), (r, p, y))


def test_rotation_is_valid():
    R = euler_to_rot(0.3, 0.2, -1.0)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1)


def test_quat_roundtrip_random():
    rng = np.random.default_rng(0)
    for i in range(100):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        q2 = rot_to_quat(quat_to_rot(q))
        assert np.allclose(q, q2)


def test_quat_180_deg():
    # trace is -1 here, so these hit the other 3 branches of rot_to_quat
    for R in [rot_x(np.pi), rot_y(np.pi), rot_z(np.pi)]:
        assert np.allclose(quat_to_rot(rot_to_quat(R)), R)


def test_inverse():
    T = make_T(euler_to_rot(0.1, 0.2, 0.3), [1, 2, 3])
    assert np.allclose(T @ inv_T(T), np.eye(4))
    assert np.allclose(inv_T(T), np.linalg.inv(T))


def test_apply_T():
    T = make_T(rot_z(np.pi / 2), [1, 0, 0])
    pts = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    out = apply_T(T, pts)
    assert np.allclose(out, [[1, 1, 0], [0, 0, 0]])


def test_slerp():
    q0 = rot_to_quat(np.eye(3))
    q1 = rot_to_quat(rot_z(np.pi / 2))
    assert np.allclose(slerp(q0, q1, 0), q0)
    assert np.allclose(slerp(q0, q1, 1), q1)
    # halfway between 0 and 90 deg should be 45 deg
    qm = slerp(q0, q1, 0.5)
    assert np.allclose(quat_to_rot(qm), rot_z(np.pi / 4))