// Geometric transforms: map_coordinates, affine_transform, shift, zoom, rotate
// (interpolation orders 0 and 1).
#include "scypp/ndimage/ndimage.hpp"

#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/linalg/detail.hpp"
#include "scypp/ndimage/detail.hpp"

namespace scypp::ndimage {
namespace d = detail;
namespace {
namespace ld = scypp::linalg::detail;
constexpr double kPi = 3.141592653589793238462643383279502884;

double sample(const d::Img& im, double y, double x, int order, const std::string& mode, double cval) {
  int R = static_cast<int>(im.r), C = static_cast<int>(im.c);
  // SciPy: in constant mode a coordinate outside [0, n-1] yields cval (no blend).
  if (mode == "constant" && (y < 0 || y > R - 1 || x < 0 || x > C - 1)) return cval;
  if (order == 0) {
    int yi = d::bidx(static_cast<int>(std::lround(y)), R, mode);
    int xi = d::bidx(static_cast<int>(std::lround(x)), C, mode);
    return (yi < 0 || xi < 0) ? cval : im.d[yi * C + xi];
  }
  int y0 = static_cast<int>(std::floor(y)), x0 = static_cast<int>(std::floor(x));
  double fy = y - y0, fx = x - x0, v = 0;
  for (int dy = 0; dy < 2; ++dy)
    for (int dx = 0; dx < 2; ++dx) {
      int yi = d::bidx(y0 + dy, R, mode), xi = d::bidx(x0 + dx, C, mode);
      double val = (yi < 0 || xi < 0) ? cval : im.d[yi * C + xi];
      v += (dy ? fy : 1 - fy) * (dx ? fx : 1 - fx) * val;
    }
  return v;
}
}  // namespace

ndarray map_coordinates(const ndarray& input, const ndarray& coordinates, int order,
                        const std::string& mode, double cval) {
  d::Img im = d::to_img(input);
  numpp::ndarray Co = coordinates.astype(numpp::kFloat64).ascontiguousarray();
  int64_t npts = Co.shape()[1];
  const double* cp = Co.typed_data<double>();  // row 0 = y, row 1 = x
  std::vector<double> out(npts);
  for (int64_t k = 0; k < npts; ++k) out[k] = sample(im, cp[k], cp[npts + k], order, mode, cval);
  return ld::from_vec(out);
}

ndarray affine_transform(const ndarray& input, const ndarray& matrix, const ndarray& offset,
                         int order, const std::string& mode, double cval) {
  d::Img im = d::to_img(input);
  auto m = ld::to_vec(matrix), o = ld::to_vec(offset);
  d::Img out{std::vector<double>(im.r * im.c), im.r, im.c};
  for (int64_t r = 0; r < im.r; ++r)
    for (int64_t c = 0; c < im.c; ++c) {
      double iy = m[0] * r + m[1] * c + o[0];
      double ix = m[2] * r + m[3] * c + o[1];
      out.d[r * im.c + c] = sample(im, iy, ix, order, mode, cval);
    }
  return d::to_nd(out);
}

ndarray shift(const ndarray& input, const std::vector<double>& sh, int order, const std::string& mode, double cval) {
  d::Img im = d::to_img(input);
  d::Img out{std::vector<double>(im.r * im.c), im.r, im.c};
  for (int64_t r = 0; r < im.r; ++r)
    for (int64_t c = 0; c < im.c; ++c)
      out.d[r * im.c + c] = sample(im, r - sh[0], c - sh[1], order, mode, cval);
  return d::to_nd(out);
}

ndarray zoom(const ndarray& input, double factor, int order, const std::string& mode, double cval) {
  d::Img im = d::to_img(input);
  int64_t R = im.r, C = im.c;
  int64_t OR = static_cast<int64_t>(std::round(R * factor)), OC = static_cast<int64_t>(std::round(C * factor));
  d::Img out{std::vector<double>(OR * OC), OR, OC};
  double sr = (R - 1.0) / (OR - 1.0), sc = (C - 1.0) / (OC - 1.0);  // SciPy grid alignment
  for (int64_t r = 0; r < OR; ++r)
    for (int64_t c = 0; c < OC; ++c) out.d[r * OC + c] = sample(im, r * sr, c * sc, order, mode, cval);
  return d::to_nd(out);
}

ndarray rotate(const ndarray& input, double angle, int order, const std::string& mode, double cval) {
  d::Img im = d::to_img(input);
  int64_t R = im.r, C = im.c;
  double th = angle * kPi / 180.0, ct = std::cos(th), st = std::sin(th);
  double cy = (R - 1) / 2.0, cx = (C - 1) / 2.0;  // reshape=False: same-size output
  d::Img out{std::vector<double>(R * C), R, C};
  for (int64_t r = 0; r < R; ++r)
    for (int64_t c = 0; c < C; ++c) {
      double dy = r - cy, dx = c - cx;
      double iy = ct * dy + st * dx + cy;  // inverse rotation maps output -> input
      double ix = -st * dy + ct * dx + cx;
      out.d[r * C + c] = sample(im, iy, ix, order, mode, cval);
    }
  return d::to_nd(out);
}

}  // namespace scypp::ndimage
