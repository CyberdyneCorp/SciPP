#pragma once
// Internal helpers for scypp::ndimage: boundary index mapping and 2-D image
// <-> ndarray conversion.

#include <cstdint>
#include <string>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"

namespace scypp::ndimage::detail {

// Map an index to a source index per SciPy's boundary mode; -1 means the
// "constant" out-of-range case (caller substitutes cval).
inline int bidx(int i, int n, const std::string& mode) {
  if (i >= 0 && i < n) return i;
  if (mode == "constant") return -1;
  if (mode == "nearest") return i < 0 ? 0 : n - 1;
  if (mode == "wrap") { int r = i % n; if (r < 0) r += n; return r; }
  bool reflect = (mode == "reflect");  // else mirror
  while (i < 0 || i >= n) {
    if (i < 0) i = reflect ? -i - 1 : -i;
    if (i >= n) i = reflect ? 2 * n - 1 - i : 2 * n - 2 - i;
  }
  return i;
}

struct Img { std::vector<double> d; int64_t r = 0, c = 0; };

inline Img to_img(const numpp::ndarray& a) {
  numpp::ndarray ac = a.astype(numpp::kFloat64).ascontiguousarray();
  Img im;
  if (ac.ndim() == 1) { im.r = 1; im.c = ac.shape()[0]; }  // 1-D image is a single row
  else { im.r = ac.shape()[0]; im.c = ac.shape()[1]; }
  const double* p = ac.typed_data<double>();
  im.d.assign(p, p + im.r * im.c);
  return im;
}
inline numpp::ndarray to_nd(const Img& im) {
  numpp::ndarray a(numpp::Shape{im.r, im.c}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int64_t i = 0; i < im.r * im.c; ++i) p[i] = im.d[i];
  return a;
}

}  // namespace scypp::ndimage::detail
