// Numerically stable reductions: logsumexp, softmax, log_softmax. All use the
// max-shift trick. Axis handling operates on a contiguous float64 copy with
// explicit C-order stride arithmetic.

#include "scypp/special/special.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/error.hpp"

namespace scypp::special {
namespace {

using numpp::ndarray;
using numpp::Shape;

double logsumexp_line(const std::vector<double>& v) {
  double m = -INFINITY;
  for (double e : v) m = std::max(m, e);
  if (std::isinf(m)) return m;  // all -inf -> -inf
  double s = 0.0;
  for (double e : v) s += std::exp(e - m);
  return m + std::log(s);
}

int norm_axis(int axis, int nd) {
  if (axis < 0) axis += nd;
  if (axis < 0 || axis >= nd) throw scypp::value_error("axis out of range");
  return axis;
}

// C-order element strides for a shape.
std::vector<int64_t> c_strides(const Shape& sh) {
  int nd = static_cast<int>(sh.size());
  std::vector<int64_t> st(nd, 1);
  for (int i = nd - 2; i >= 0; --i) st[i] = st[i + 1] * sh[i + 1];
  return st;
}

}  // namespace

double logsumexp(const ndarray& a) {
  ndarray ac = a.astype(numpp::kFloat64).ascontiguousarray();
  const double* p = ac.typed_data<double>();
  std::vector<double> all(p, p + ac.size());
  return logsumexp_line(all);
}

ndarray logsumexp(const ndarray& a, int axis, bool keepdims) {
  ndarray ac = a.astype(numpp::kFloat64).ascontiguousarray();
  Shape sh = ac.shape();
  int nd = static_cast<int>(sh.size());
  axis = norm_axis(axis, nd);
  auto st = c_strides(sh);
  int64_t L = sh[axis];
  int64_t stride = st[axis];

  Shape outsh;
  for (int i = 0; i < nd; ++i) {
    if (i == axis) { if (keepdims) outsh.push_back(1); }
    else outsh.push_back(sh[i]);
  }
  if (outsh.empty()) outsh.push_back(1);
  ndarray out(outsh, numpp::kFloat64);

  const double* in = ac.typed_data<double>();
  double* o = out.typed_data<double>();
  int64_t outer = ac.size() / L;

  // Non-axis dims, in original order, for decoding the linear output index.
  std::vector<int> dims;
  for (int i = 0; i < nd; ++i) if (i != axis) dims.push_back(i);

  std::vector<double> line(L);
  for (int64_t oi = 0; oi < outer; ++oi) {
    int64_t base = 0, rem = oi;
    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
      int64_t dimsz = sh[dims[d]];
      int64_t idx = rem % dimsz;
      rem /= dimsz;
      base += idx * st[dims[d]];
    }
    for (int64_t l = 0; l < L; ++l) line[l] = in[base + l * stride];
    o[oi] = logsumexp_line(line);
  }
  return out;
}

ndarray softmax(const ndarray& a, std::optional<int> axis) {
  ndarray ac = a.astype(numpp::kFloat64).ascontiguousarray();
  ndarray out(ac.shape(), numpp::kFloat64);
  const double* in = ac.typed_data<double>();
  double* o = out.typed_data<double>();

  if (!axis.has_value()) {
    int64_t n = ac.size();
    double m = -INFINITY;
    for (int64_t i = 0; i < n; ++i) m = std::max(m, in[i]);
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) { o[i] = std::exp(in[i] - m); s += o[i]; }
    for (int64_t i = 0; i < n; ++i) o[i] /= s;
    return out;
  }

  Shape sh = ac.shape();
  int nd = static_cast<int>(sh.size());
  int ax = norm_axis(*axis, nd);
  auto st = c_strides(sh);
  int64_t L = sh[ax], stride = st[ax];
  int64_t outer = ac.size() / L;
  std::vector<int> dims;
  for (int i = 0; i < nd; ++i) if (i != ax) dims.push_back(i);

  for (int64_t oi = 0; oi < outer; ++oi) {
    int64_t base = 0, rem = oi;
    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
      int64_t dimsz = sh[dims[d]];
      base += (rem % dimsz) * st[dims[d]];
      rem /= dimsz;
    }
    double m = -INFINITY;
    for (int64_t l = 0; l < L; ++l) m = std::max(m, in[base + l * stride]);
    double s = 0.0;
    for (int64_t l = 0; l < L; ++l) { double e = std::exp(in[base + l * stride] - m); o[base + l * stride] = e; s += e; }
    for (int64_t l = 0; l < L; ++l) o[base + l * stride] /= s;
  }
  return out;
}

ndarray log_softmax(const ndarray& a, std::optional<int> axis) {
  ndarray sm = softmax(a, axis);
  double* p = sm.typed_data<double>();
  int64_t n = sm.size();
  for (int64_t i = 0; i < n; ++i) p[i] = std::log(p[i]);
  return sm;
}

}  // namespace scypp::special
