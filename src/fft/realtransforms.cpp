// Discrete cosine/sine transforms (types I-IV, norm=None) via direct summation,
// applied along a selectable axis. idct/idst use the inverse-type relations.
#include "scypp/fft/fft.hpp"

#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/error.hpp"

namespace scypp::fft {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
using numpp::ndarray;
using numpp::Shape;

std::vector<double> dct_line(const std::vector<double>& x, int type) {
  int N = static_cast<int>(x.size());
  std::vector<double> y(N, 0.0);
  if (type == 1) {
    for (int k = 0; k < N; ++k) {
      double s = x[0] + ((k % 2 == 0) ? x[N - 1] : -x[N - 1]);
      for (int n = 1; n < N - 1; ++n) s += 2.0 * x[n] * std::cos(kPi * k * n / (N - 1));
      y[k] = s;
    }
  } else if (type == 2) {
    for (int k = 0; k < N; ++k) {
      double s = 0.0;
      for (int n = 0; n < N; ++n) s += x[n] * std::cos(kPi * k * (2 * n + 1) / (2.0 * N));
      y[k] = 2.0 * s;
    }
  } else if (type == 3) {
    for (int k = 0; k < N; ++k) {
      double s = 0.5 * x[0];
      for (int n = 1; n < N; ++n) s += x[n] * std::cos(kPi * n * (2 * k + 1) / (2.0 * N));
      y[k] = 2.0 * s;
    }
  } else if (type == 4) {
    for (int k = 0; k < N; ++k) {
      double s = 0.0;
      for (int n = 0; n < N; ++n)
        s += x[n] * std::cos(kPi * (2 * k + 1) * (2 * n + 1) / (4.0 * N));
      y[k] = 2.0 * s;
    }
  } else {
    throw scypp::value_error("dct: type must be 1..4");
  }
  return y;
}

std::vector<double> dst_line(const std::vector<double>& x, int type) {
  int N = static_cast<int>(x.size());
  std::vector<double> y(N, 0.0);
  if (type == 1) {
    for (int k = 0; k < N; ++k) {
      double s = 0.0;
      for (int n = 0; n < N; ++n) s += x[n] * std::sin(kPi * (n + 1) * (k + 1) / (N + 1.0));
      y[k] = 2.0 * s;
    }
  } else if (type == 2) {
    for (int k = 0; k < N; ++k) {
      double s = 0.0;
      for (int n = 0; n < N; ++n) s += x[n] * std::sin(kPi * (k + 1) * (2 * n + 1) / (2.0 * N));
      y[k] = 2.0 * s;
    }
  } else if (type == 3) {
    for (int k = 0; k < N; ++k) {
      double s = ((k % 2 == 0) ? x[N - 1] : -x[N - 1]) * 0.5;
      for (int n = 0; n < N - 1; ++n) s += x[n] * std::sin(kPi * (n + 1) * (2 * k + 1) / (2.0 * N));
      y[k] = 2.0 * s;
    }
  } else if (type == 4) {
    for (int k = 0; k < N; ++k) {
      double s = 0.0;
      for (int n = 0; n < N; ++n)
        s += x[n] * std::sin(kPi * (2 * k + 1) * (2 * n + 1) / (4.0 * N));
      y[k] = 2.0 * s;
    }
  } else {
    throw scypp::value_error("dst: type must be 1..4");
  }
  return y;
}

int inverse_type(int type) {  // 1<->1, 2<->3, 4<->4
  switch (type) {
    case 1: return 1;
    case 2: return 3;
    case 3: return 2;
    case 4: return 4;
    default: throw scypp::value_error("idct/idst: type must be 1..4");
  }
}

std::vector<double> idct_line(const std::vector<double>& x, int type) {
  int N = static_cast<int>(x.size());
  double scale = (type == 1) ? (2.0 * (N - 1)) : (2.0 * N);
  std::vector<double> y = dct_line(x, inverse_type(type));
  for (double& v : y) v /= scale;
  return y;
}
std::vector<double> idst_line(const std::vector<double>& x, int type) {
  int N = static_cast<int>(x.size());
  double scale = (type == 1) ? (2.0 * (N + 1)) : (2.0 * N);
  std::vector<double> y = dst_line(x, inverse_type(type));
  for (double& v : y) v /= scale;
  return y;
}

std::vector<int64_t> c_strides(const Shape& sh) {
  int nd = static_cast<int>(sh.size());
  std::vector<int64_t> st(nd, 1);
  for (int i = nd - 2; i >= 0; --i) st[i] = st[i + 1] * sh[i + 1];
  return st;
}

// Apply a 1-D real transform along `axis` of `a`, preserving shape.
template <class LineFn>
ndarray apply_axis(const ndarray& a, int64_t axis, int type, LineFn fn) {
  ndarray ac = a.astype(numpp::kFloat64).ascontiguousarray();
  Shape sh = ac.shape();
  int nd = static_cast<int>(sh.size());
  if (axis < 0) axis += nd;
  if (axis < 0 || axis >= nd) throw scypp::value_error("axis out of range");
  auto st = c_strides(sh);
  int64_t L = sh[axis], stride = st[axis];
  ndarray out(sh, numpp::kFloat64);
  const double* in = ac.typed_data<double>();
  double* o = out.typed_data<double>();
  int64_t outer = ac.size() / L;
  std::vector<int> dims;
  for (int i = 0; i < nd; ++i) if (i != axis) dims.push_back(i);

  std::vector<double> line(L);
  for (int64_t oi = 0; oi < outer; ++oi) {
    int64_t base = 0, rem = oi;
    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
      int64_t dimsz = sh[dims[d]];
      base += (rem % dimsz) * st[dims[d]];
      rem /= dimsz;
    }
    for (int64_t l = 0; l < L; ++l) line[l] = in[base + l * stride];
    std::vector<double> r = fn(line, type);
    for (int64_t l = 0; l < L; ++l) o[base + l * stride] = r[l];
  }
  return out;
}

void check_norm(const std::string& norm) {
  if (norm != "backward")
    throw scypp::not_implemented_error("dct/dst: only norm=\"backward\" is implemented");
}

}  // namespace

ndarray dct(const ndarray& a, int type, int64_t axis, const std::string& norm) {
  check_norm(norm);
  return apply_axis(a, axis, type, dct_line);
}
ndarray idct(const ndarray& a, int type, int64_t axis, const std::string& norm) {
  check_norm(norm);
  return apply_axis(a, axis, type, idct_line);
}
ndarray dst(const ndarray& a, int type, int64_t axis, const std::string& norm) {
  check_norm(norm);
  return apply_axis(a, axis, type, dst_line);
}
ndarray idst(const ndarray& a, int type, int64_t axis, const std::string& norm) {
  check_norm(norm);
  return apply_axis(a, axis, type, idst_line);
}

}  // namespace scypp::fft
