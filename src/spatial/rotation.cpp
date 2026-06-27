// 3-D rotations: Rotation (quat/matrix/euler/rotvec) + Slerp.
#include "scypp/spatial/spatial.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::spatial {
namespace {
namespace sd = scypp::linalg::detail;
constexpr double kPi = 3.141592653589793238462643383279502884;
int axis_index(char c) { return c == 'x' || c == 'X' ? 0 : (c == 'y' || c == 'Y' ? 1 : 2); }

Rotation make(double x, double y, double z, double w) {
  double n = std::sqrt(x * x + y * y + z * z + w * w);
  Rotation r; r.x_ = x / n; r.y_ = y / n; r.z_ = z / n; r.w_ = w / n;
  return r;
}
Rotation elementary(int ax, double angle) {
  double h = angle / 2, s = std::sin(h), c = std::cos(h);
  double q[3] = {0, 0, 0}; q[ax] = s;
  return make(q[0], q[1], q[2], c);
}
}  // namespace

Rotation Rotation::from_quat(const ndarray& q) {
  auto v = sd::to_vec(q);
  return make(v[0], v[1], v[2], v[3]);
}

Rotation Rotation::operator*(const Rotation& o) const {
  return make(w_ * o.x_ + x_ * o.w_ + y_ * o.z_ - z_ * o.y_,
              w_ * o.y_ - x_ * o.z_ + y_ * o.w_ + z_ * o.x_,
              w_ * o.z_ + x_ * o.y_ - y_ * o.x_ + z_ * o.w_,
              w_ * o.w_ - x_ * o.x_ - y_ * o.y_ - z_ * o.z_);
}
Rotation Rotation::inv() const { Rotation r; r.x_ = -x_; r.y_ = -y_; r.z_ = -z_; r.w_ = w_; return r; }

ndarray Rotation::as_quat() const { return sd::from_vec({x_, y_, z_, w_}); }

ndarray Rotation::as_matrix() const {
  double x = x_, y = y_, z = z_, w = w_;
  std::vector<double> m{
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
      2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
  return sd::from_mat(m, 3, 3);
}

Rotation Rotation::from_matrix(const ndarray& mat) {
  auto m = sd::to_vec(mat);  // 3x3 row-major
  double tr = m[0] + m[4] + m[8];
  double x, y, z, w;
  if (tr > 0) {
    double s = std::sqrt(tr + 1.0) * 2;
    w = 0.25 * s; x = (m[7] - m[5]) / s; y = (m[2] - m[6]) / s; z = (m[3] - m[1]) / s;
  } else if (m[0] > m[4] && m[0] > m[8]) {
    double s = std::sqrt(1.0 + m[0] - m[4] - m[8]) * 2;
    w = (m[7] - m[5]) / s; x = 0.25 * s; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s;
  } else if (m[4] > m[8]) {
    double s = std::sqrt(1.0 + m[4] - m[0] - m[8]) * 2;
    w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = 0.25 * s; z = (m[5] + m[7]) / s;
  } else {
    double s = std::sqrt(1.0 + m[8] - m[0] - m[4]) * 2;
    w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = 0.25 * s;
  }
  return make(x, y, z, w);
}

Rotation Rotation::from_rotvec(const ndarray& v, bool degrees) {
  auto r = sd::to_vec(v);
  double scale = degrees ? kPi / 180.0 : 1.0;
  double angle = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) * scale;
  if (angle < 1e-12) return make(r[0] * scale / 2, r[1] * scale / 2, r[2] * scale / 2, 1.0);
  double s = std::sin(angle / 2) / (angle / scale);
  return make(r[0] * s, r[1] * s, r[2] * s, std::cos(angle / 2));
}

ndarray Rotation::as_rotvec(bool degrees) const {
  double w = w_, x = x_, y = y_, z = z_;
  if (w < 0) { w = -w; x = -x; y = -y; z = -z; }  // canonical (angle in [0, pi])
  double nv = std::sqrt(x * x + y * y + z * z);
  double angle = 2.0 * std::atan2(nv, w);
  double scale = (nv < 1e-12) ? 2.0 : angle / nv;
  if (degrees) scale *= 180.0 / kPi;
  return sd::from_vec({x * scale, y * scale, z * scale});
}

double Rotation::magnitude() const {
  double nv = std::sqrt(x_ * x_ + y_ * y_ + z_ * z_);
  return 2.0 * std::atan2(nv, std::fabs(w_));
}

ndarray Rotation::apply(const ndarray& v) const {
  numpp::ndarray V = v.astype(numpp::kFloat64).ascontiguousarray();
  auto m = sd::to_vec(as_matrix());
  bool single = (V.ndim() == 1);
  int64_t rows = single ? 1 : V.shape()[0];
  const double* vp = V.typed_data<double>();
  std::vector<double> out(rows * 3);
  for (int64_t r = 0; r < rows; ++r)
    for (int i = 0; i < 3; ++i)
      out[r * 3 + i] = m[i * 3] * vp[r * 3] + m[i * 3 + 1] * vp[r * 3 + 1] + m[i * 3 + 2] * vp[r * 3 + 2];
  return single ? sd::from_vec(out) : sd::from_mat(out, rows, 3);
}

Rotation Rotation::from_euler(const std::string& seq, const ndarray& angles, bool degrees) {
  auto a = sd::to_vec(angles);
  bool extrinsic = std::islower(static_cast<unsigned char>(seq[0]));
  double scale = degrees ? kPi / 180.0 : 1.0;
  Rotation q;  // identity
  for (size_t i = 0; i < seq.size(); ++i) {
    Rotation e = elementary(axis_index(seq[i]), a[i] * scale);
    q = extrinsic ? e * q : q * e;
  }
  return q;
}

ndarray Rotation::as_euler(const std::string& seq, bool degrees) const {
  // General quaternion-based extraction (Bernardes & Viollet 2022), as in SciPy.
  bool extrinsic = std::islower(static_cast<unsigned char>(seq[0]));
  std::string s = seq;
  if (!extrinsic) std::reverse(s.begin(), s.end());
  int i = axis_index(s[0]), j = axis_index(s[1]), k = axis_index(s[2]);
  bool symmetric = (i == k);
  if (symmetric) k = 3 - i - j;
  double sign = ((i - j) * (j - k) * (k - i)) / 2.0;  // ±1
  double quat[4] = {x_, y_, z_, w_};
  double a, b, c, dd;
  if (symmetric) { a = quat[3]; b = quat[i]; c = quat[j]; dd = quat[k] * sign; }
  else { a = quat[3] - quat[j]; b = quat[i] + quat[k] * sign; c = quat[j] + quat[3]; dd = quat[k] * sign - quat[i]; }

  double angles[3] = {0, 0, 0};
  angles[1] = 2.0 * std::atan2(std::hypot(c, dd), std::hypot(a, b));
  double half_sum = std::atan2(b, a), half_diff = std::atan2(dd, c);
  double eps = 1e-7;
  bool lock0 = std::fabs(angles[1]) <= eps, lockpi = std::fabs(angles[1] - kPi) <= eps;
  if (!lock0 && !lockpi) {
    angles[0] = half_sum - half_diff;
    angles[2] = half_sum + half_diff;
  } else {  // gimbal lock: third angle set to 0
    angles[2] = 0;
    angles[0] = lock0 ? 2.0 * half_sum : 2.0 * half_diff * (extrinsic ? -1.0 : 1.0);
  }
  if (!symmetric) { angles[2] *= sign; angles[1] -= kPi / 2.0; }
  if (!extrinsic) std::swap(angles[0], angles[2]);
  for (double& ang : angles) { if (ang < -kPi) ang += 2 * kPi; else if (ang > kPi) ang -= 2 * kPi; }
  double oscale = degrees ? 180.0 / kPi : 1.0;
  return sd::from_vec({angles[0] * oscale, angles[1] * oscale, angles[2] * oscale});
}

// ---- Slerp ----
Slerp::Slerp(const ndarray& times, const std::vector<Rotation>& rotations)
    : times_(sd::to_vec(times)), rots_(rotations) {}

Rotation Slerp::operator()(double t) const {
  size_t i = 0;
  while (i + 2 < times_.size() && t > times_[i + 1]) ++i;
  double u = (t - times_[i]) / (times_[i + 1] - times_[i]);
  Rotation q0 = rots_[i], q1 = rots_[i + 1];
  double dot = q0.x_ * q1.x_ + q0.y_ * q1.y_ + q0.z_ * q1.z_ + q0.w_ * q1.w_;
  if (dot < 0) { q1.x_ = -q1.x_; q1.y_ = -q1.y_; q1.z_ = -q1.z_; q1.w_ = -q1.w_; dot = -dot; }
  if (dot > 0.9995) return make(q0.x_ + u * (q1.x_ - q0.x_), q0.y_ + u * (q1.y_ - q0.y_),
                                q0.z_ + u * (q1.z_ - q0.z_), q0.w_ + u * (q1.w_ - q0.w_));
  double theta = std::acos(dot), s0 = std::sin((1 - u) * theta) / std::sin(theta), s1 = std::sin(u * theta) / std::sin(theta);
  return make(s0 * q0.x_ + s1 * q1.x_, s0 * q0.y_ + s1 * q1.y_, s0 * q0.z_ + s1 * q1.z_, s0 * q0.w_ + s1 * q1.w_);
}

}  // namespace scypp::spatial
