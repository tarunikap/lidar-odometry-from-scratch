#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using namespace Eigen;

Matrix3d rot_x(double a) {
    Matrix3d R;
    R << 1, 0, 0,
         0, std::cos(a), -std::sin(a),
         0, std::sin(a), std::cos(a);
    return R;
}

Matrix3d rot_y(double a) {
    Matrix3d R;
    R << std::cos(a), 0, std::sin(a),
         0, 1, 0,
         -std::sin(a), 0, std::cos(a);
    return R;
}

Matrix3d rot_z(double a) {
    Matrix3d R;
    R << std::cos(a), -std::sin(a), 0,
         std::sin(a), std::cos(a), 0,
         0, 0, 1;
    return R;
}

Matrix3d euler_to_rot(double roll, double pitch, double yaw) {
    // ZYX order, same as the python version
    return rot_z(yaw) * rot_y(pitch) * rot_x(roll);
}

Vector3d rot_to_euler(const Matrix3d& R) {
    double pitch = std::asin(-R(2, 0));
    double roll = std::atan2(R(2, 1), R(2, 2));
    double yaw = std::atan2(R(1, 0), R(0, 0));
    return Vector3d(roll, pitch, yaw);
}

Matrix3d quat_to_rot(Vector4d q) {
    // q = [w, x, y, z]
    q.normalize();
    double w = q(0), x = q(1), y = q(2), z = q(3);
    Matrix3d R;
    R << 1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
         2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
         2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y);
    return R;
}

Vector4d rot_to_quat(const Matrix3d& R) {
    // let Eigen do the 4 branch part this time
    Quaterniond qe(R);
    Vector4d q(qe.w(), qe.x(), qe.y(), qe.z());
    if (q(0) < 0) q = -q;
    return q;
}

Matrix4d make_T(const Matrix3d& R, const Vector3d& t) {
    Matrix4d T = Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

Matrix4d inv_T(const Matrix4d& T) {
    Matrix3d R = T.block<3, 3>(0, 0);
    Vector3d t = T.block<3, 1>(0, 3);
    Matrix4d Ti = Matrix4d::Identity();
    Ti.block<3, 3>(0, 0) = R.transpose();
    Ti.block<3, 1>(0, 3) = -R.transpose() * t;
    return Ti;
}

MatrixX3d apply_T(const Matrix4d& T, const MatrixX3d& pts) {
    // pts is Nx3
    Matrix3d R = T.block<3, 3>(0, 0);
    Vector3d t = T.block<3, 1>(0, 3);
    MatrixX3d out = pts * R.transpose();
    out.rowwise() += t.transpose();
    return out;
}

int main() {
    Matrix3d R = euler_to_rot(0.1, 0.2, 0.3);
    std::cout << "euler back: " << rot_to_euler(R).transpose() << "\n";

    Vector4d q = rot_to_quat(R);
    std::cout << "quat [w x y z]: " << q.transpose() << "\n";
    std::cout << "quat roundtrip err: " << (quat_to_rot(q) - R).cwiseAbs().maxCoeff() << "\n";

    Matrix4d T = make_T(R, Vector3d(1, 2, 3));
    std::cout << "T*inv(T) err: " << (T * inv_T(T) - Matrix4d::Identity()).cwiseAbs().maxCoeff() << "\n";

    Matrix4d T2 = make_T(rot_z(M_PI / 2), Vector3d(1, 0, 0));
    MatrixX3d pts(2, 3);
    pts << 1, 0, 0,
           0, 1, 0;
    std::cout << "apply_T:\n" << apply_T(T2, pts) << "\n";
    return 0;
}
