// Morphology: binary/grey erosion & dilation, opening/closing, and the exact
// Euclidean distance transform.
#include "scypp/ndimage/ndimage.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "scypp/ndimage/detail.hpp"

namespace scypp::ndimage {
namespace d = detail;
namespace {
// 3x3 connectivity-1 structuring element (center + 4 edge neighbours).
const int kStruct[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

d::Img binary_step(const d::Img& im, bool dilate) {
  d::Img out{std::vector<double>(im.r * im.c, 0.0), im.r, im.c};
  for (int64_t r = 0; r < im.r; ++r)
    for (int64_t c = 0; c < im.c; ++c) {
      bool acc = dilate ? false : true;
      for (auto& s : kStruct) {
        int64_t rr = r + s[0], cc = c + s[1];
        bool v = (rr < 0 || rr >= im.r || cc < 0 || cc >= im.c) ? false : (im.d[rr * im.c + cc] != 0);
        if (dilate) acc = acc || v; else acc = acc && v;
      }
      out.d[r * im.c + c] = acc ? 1.0 : 0.0;
    }
  return out;
}
d::Img repeat(d::Img im, bool dilate, int iters) {
  for (int i = 0; i < iters; ++i) im = binary_step(im, dilate);
  return im;
}
}  // namespace

ndarray binary_erosion(const ndarray& input, int iterations) { return d::to_nd(repeat(d::to_img(input), false, iterations)); }
ndarray binary_dilation(const ndarray& input, int iterations) { return d::to_nd(repeat(d::to_img(input), true, iterations)); }
ndarray binary_opening(const ndarray& input, int iterations) {
  return d::to_nd(repeat(repeat(d::to_img(input), false, iterations), true, iterations));
}
ndarray binary_closing(const ndarray& input, int iterations) {
  return d::to_nd(repeat(repeat(d::to_img(input), true, iterations), false, iterations));
}

ndarray grey_erosion(const ndarray& input, int size) { return minimum_filter(input, size, "reflect", 0.0); }
ndarray grey_dilation(const ndarray& input, int size) { return maximum_filter(input, size, "reflect", 0.0); }

namespace {
// 1-D squared-distance transform (Felzenszwalb & Huttenlocher).
void edt_1d(const std::vector<double>& f, std::vector<double>& out, int n) {
  std::vector<int> v(n);
  std::vector<double> z(n + 1);
  int kk = 0;
  v[0] = 0; z[0] = -1e20; z[1] = 1e20;
  for (int q = 1; q < n; ++q) {
    double s;
    while (true) {
      s = ((f[q] + q * q) - (f[v[kk]] + v[kk] * v[kk])) / (2.0 * q - 2.0 * v[kk]);
      if (s <= z[kk]) --kk; else break;
    }
    ++kk; v[kk] = q; z[kk] = s; z[kk + 1] = 1e20;
  }
  kk = 0;
  for (int q = 0; q < n; ++q) {
    while (z[kk + 1] < q) ++kk;
    double dx = q - v[kk];
    out[q] = dx * dx + f[v[kk]];
  }
}
}  // namespace

ndarray distance_transform_edt(const ndarray& input) {
  d::Img im = d::to_img(input);
  int64_t R = im.r, C = im.c;
  double INF = 1e20;
  std::vector<double> g(R * C);
  for (int64_t i = 0; i < R * C; ++i) g[i] = (im.d[i] != 0) ? INF : 0.0;
  // transform along columns then rows
  std::vector<double> col(R), colo(R);
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t r = 0; r < R; ++r) col[r] = g[r * C + c];
    edt_1d(col, colo, static_cast<int>(R));
    for (int64_t r = 0; r < R; ++r) g[r * C + c] = colo[r];
  }
  std::vector<double> row(C), rowo(C);
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) row[c] = g[r * C + c];
    edt_1d(row, rowo, static_cast<int>(C));
    for (int64_t c = 0; c < C; ++c) g[r * C + c] = std::sqrt(rowo[c]);
  }
  d::Img out{g, R, C};
  return d::to_nd(out);
}

}  // namespace scypp::ndimage
