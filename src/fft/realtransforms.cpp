// Discrete cosine/sine transforms (types I-IV) via direct summation, applied
// along selectable axes. The unnormalized ("backward") line kernels match
// scipy.fft; normalization and the orthogonalized variant are layered on top so
// norm in {"backward", "ortho", "forward"} reproduce scipy.fft bit-for-bit.
//
// norm semantics (scipy.fft): map norm -> inorm code (backward=0, ortho=1,
// forward=2). A forward transform applies that code; an inverse applies
// (2 - code) and swaps DCT/DST type 2<->3. inorm scales the whole line by 1
// (0), sqrt(1/S) (1) or 1/S (2), where S is the type's "logical" factor
// (2(N-1) for DCT-I, 2(N+1) for DST-I, otherwise 2N). When norm=="ortho" the
// transform is also orthogonalized: boundary samples are scaled by sqrt(2) so
// the coefficient matrix becomes orthonormal.
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

// Map a norm string to scipy's inorm code (0 backward, 1 ortho, 2 forward).
int norm_code(const std::string& norm) {
  if (norm == "backward") return 0;
  if (norm == "ortho") return 1;
  if (norm == "forward") return 2;
  throw scypp::value_error("dct/dst: norm must be \"backward\", \"ortho\" or \"forward\"");
}

// Inverse swaps DCT/DST type 2<->3 (1 and 4 are their own inverse type).
int inverse_type(int type) {
  switch (type) {
    case 1: return 1;
    case 2: return 3;
    case 3: return 2;
    case 4: return 4;
    default: throw scypp::value_error("idct/idst: type must be 1..4");
  }
}

// Full normalized line transform: input-boundary orthogonalization, unnormalized
// kernel, output-boundary orthogonalization, then the inorm scale factor.
std::vector<double> r2r_line(std::vector<double> x, int type, int inorm, bool ortho, bool is_dst) {
  int N = static_cast<int>(x.size());
  const double s2 = std::sqrt(2.0);
  if (ortho && N > 1) {
    if (!is_dst) {
      if (type == 1) { x[0] *= s2; x[N - 1] *= s2; }
      else if (type == 3) { x[0] *= s2; }
    } else {
      if (type == 3) { x[N - 1] *= s2; }
    }
  }
  std::vector<double> y = is_dst ? dst_line(x, type) : dct_line(x, type);
  if (ortho && N > 1) {
    if (!is_dst) {
      if (type == 1) { y[0] /= s2; y[N - 1] /= s2; }
      else if (type == 2) { y[0] /= s2; }
    } else {
      if (type == 2) { y[N - 1] /= s2; }
    }
  }
  if (inorm != 0) {
    double S;
    if (!is_dst) S = (type == 1) ? 2.0 * (N - 1) : 2.0 * N;
    else         S = (type == 1) ? 2.0 * (N + 1) : 2.0 * N;
    double f = (inorm == 1) ? std::sqrt(1.0 / S) : (1.0 / S);
    for (double& v : y) v *= f;
  }
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
ndarray apply_axis(const ndarray& a, int64_t axis, LineFn fn) {
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
    std::vector<double> r = fn(line);
    for (int64_t l = 0; l < L; ++l) o[base + l * stride] = r[l];
  }
  return out;
}

// Resolve the axis list for an N-D transform (default: every axis, in order).
std::vector<int64_t> resolve_axes(const ndarray& a, const std::optional<std::vector<int64_t>>& axes) {
  int nd = static_cast<int>(a.shape().size());
  if (!axes) {
    std::vector<int64_t> all(nd);
    for (int i = 0; i < nd; ++i) all[i] = i;
    return all;
  }
  std::vector<int64_t> out;
  for (int64_t ax : *axes) {
    int64_t a2 = ax < 0 ? ax + nd : ax;
    if (a2 < 0 || a2 >= nd) throw scypp::value_error("axis out of range");
    out.push_back(a2);
  }
  return out;
}

// One forward/inverse 1-D transform, applied along a single axis.
ndarray r2r(const ndarray& a, int type, int64_t axis, int inorm, bool ortho, bool is_dst) {
  return apply_axis(a, axis, [&](const std::vector<double>& line) {
    return r2r_line(line, type, inorm, ortho, is_dst);
  });
}

// One forward/inverse 1-D transform, applied successively over several axes.
ndarray r2rn(const ndarray& a, int type, const std::optional<std::vector<int64_t>>& axes,
             int inorm, bool ortho, bool is_dst) {
  std::vector<int64_t> ax = resolve_axes(a, axes);
  ndarray out = a.astype(numpp::kFloat64).ascontiguousarray();
  for (int64_t axis : ax) out = r2r(out, type, axis, inorm, ortho, is_dst);
  return out;
}

}  // namespace

ndarray dct(const ndarray& a, int type, int64_t axis, const std::string& norm) {
  int code = norm_code(norm);
  return r2r(a, type, axis, code, code == 1, /*is_dst=*/false);
}
ndarray idct(const ndarray& a, int type, int64_t axis, const std::string& norm) {
  int code = norm_code(norm);
  return r2r(a, inverse_type(type), axis, 2 - code, code == 1, /*is_dst=*/false);
}
ndarray dst(const ndarray& a, int type, int64_t axis, const std::string& norm) {
  int code = norm_code(norm);
  return r2r(a, type, axis, code, code == 1, /*is_dst=*/true);
}
ndarray idst(const ndarray& a, int type, int64_t axis, const std::string& norm) {
  int code = norm_code(norm);
  return r2r(a, inverse_type(type), axis, 2 - code, code == 1, /*is_dst=*/true);
}

ndarray dctn(const ndarray& a, int type, std::optional<std::vector<int64_t>> axes,
             const std::string& norm) {
  int code = norm_code(norm);
  return r2rn(a, type, axes, code, code == 1, /*is_dst=*/false);
}
ndarray idctn(const ndarray& a, int type, std::optional<std::vector<int64_t>> axes,
              const std::string& norm) {
  int code = norm_code(norm);
  return r2rn(a, inverse_type(type), axes, 2 - code, code == 1, /*is_dst=*/false);
}
ndarray dstn(const ndarray& a, int type, std::optional<std::vector<int64_t>> axes,
             const std::string& norm) {
  int code = norm_code(norm);
  return r2rn(a, type, axes, code, code == 1, /*is_dst=*/true);
}
ndarray idstn(const ndarray& a, int type, std::optional<std::vector<int64_t>> axes,
              const std::string& norm) {
  int code = norm_code(norm);
  return r2rn(a, inverse_type(type), axes, 2 - code, code == 1, /*is_dst=*/true);
}

}  // namespace scypp::fft
