// Filters: correlate1d/convolve1d, correlate/convolve, uniform/gaussian,
// median/minimum/maximum, sobel/prewitt/laplace.
#include "scypp/ndimage/ndimage.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "numpp/backend/backend.hpp"
#include "scypp/linalg/detail.hpp"
#include "scypp/ndimage/detail.hpp"

namespace scypp::ndimage {
namespace d = detail;
namespace {
namespace ld = scypp::linalg::detail;
thread_local Backend g_last = Backend::Cpu;

// Correlate a 1-D weight set along `axis` of a 2-D image.
d::Img corr1d(const d::Img& im, const std::vector<double>& w, int axis, const std::string& mode,
              double cval, int origin) {
  int L = static_cast<int>(w.size());
  int center = L / 2 + origin;
  d::Img out{std::vector<double>(im.r * im.c), im.r, im.c};
  auto at = [&](int64_t r, int64_t c) { return im.d[r * im.c + c]; };
  for (int64_t r = 0; r < im.r; ++r)
    for (int64_t c = 0; c < im.c; ++c) {
      double s = 0;
      for (int k = 0; k < L; ++k) {
        int64_t rr = r, cc = c;
        if (axis == 0) { int idx = d::bidx(static_cast<int>(r) + k - center, im.r, mode); if (idx < 0) { s += w[k] * cval; continue; } rr = idx; }
        else { int idx = d::bidx(static_cast<int>(c) + k - center, im.c, mode); if (idx < 0) { s += w[k] * cval; continue; } cc = idx; }
        s += w[k] * at(rr, cc);
      }
      out.d[r * im.c + c] = s;
    }
  return out;
}
int real_axis(int axis, int /*ndim*/) { return axis < 0 ? 1 : axis; }
}  // namespace

Backend last_backend() { return g_last; }

ndarray correlate1d(const ndarray& input, const ndarray& weights, int axis, const std::string& mode,
                    double cval, int origin) {
  d::Img im = d::to_img(input);
  return d::to_nd(corr1d(im, ld::to_vec(weights), real_axis(axis, 2), mode, cval, origin));
}
ndarray convolve1d(const ndarray& input, const ndarray& weights, int axis, const std::string& mode,
                   double cval, int origin) {
  std::vector<double> w = ld::to_vec(weights);
  std::reverse(w.begin(), w.end());
  int L = static_cast<int>(w.size());
  // convolve = correlate with reversed weights; origin shifts for the reversal.
  int conv_origin = -origin + (L % 2 == 0 ? -1 : 0);
  d::Img im = d::to_img(input);
  return d::to_nd(corr1d(im, w, real_axis(axis, 2), mode, cval, conv_origin));
}

ndarray correlate(const ndarray& input, const ndarray& weights, const std::string& mode, double cval) {
  d::Img im = d::to_img(input);
  d::Img w = d::to_img(weights);
  int wr = static_cast<int>(w.r), wc = static_cast<int>(w.c), cr = wr / 2, cc = wc / 2;
  d::Img out{std::vector<double>(im.r * im.c), im.r, im.c};
  for (int64_t r = 0; r < im.r; ++r)
    for (int64_t c = 0; c < im.c; ++c) {
      double s = 0;
      for (int a = 0; a < wr; ++a)
        for (int b = 0; b < wc; ++b) {
          int ri = d::bidx(static_cast<int>(r) + a - cr, im.r, mode);
          int ci = d::bidx(static_cast<int>(c) + b - cc, im.c, mode);
          double v = (ri < 0 || ci < 0) ? cval : im.d[ri * im.c + ci];
          s += w.d[a * wc + b] * v;
        }
      out.d[r * im.c + c] = s;
    }
  return d::to_nd(out);
}
ndarray convolve(const ndarray& input, const ndarray& weights, const std::string& mode, double cval) {
  d::Img w = d::to_img(weights);
  std::reverse(w.d.begin(), w.d.end());  // flip kernel
  return correlate(input, d::to_nd(w), mode, cval);
}

ndarray uniform_filter1d(const ndarray& input, int size, int axis, const std::string& mode, double cval) {
  std::vector<double> w(size, 1.0 / size);
  return correlate1d(input, ld::from_vec(w), axis, mode, cval, 0);
}
ndarray uniform_filter(const ndarray& input, int size, const std::string& mode, double cval, Backend forced) {
  const auto& reg = numpp::CapabilityRegistry::instance();
  g_last = (forced == Backend::Device || reg.gpu_available(numpp::Backend::Device)) && forced == Backend::Device ? Backend::Device : Backend::Cpu;
  std::vector<double> w(size, 1.0 / size);
  d::Img im = d::to_img(input);
  im = corr1d(im, w, 0, mode, cval, 0);
  im = corr1d(im, w, 1, mode, cval, 0);
  return d::to_nd(im);
}

namespace {
std::vector<double> gaussian_kernel(double sigma, double truncate) {
  int radius = static_cast<int>(truncate * sigma + 0.5);
  std::vector<double> w(2 * radius + 1);
  double sum = 0;
  for (int x = -radius; x <= radius; ++x) { double v = std::exp(-0.5 * x * x / (sigma * sigma)); w[x + radius] = v; sum += v; }
  for (double& v : w) v /= sum;
  return w;
}
}  // namespace

ndarray gaussian_filter1d(const ndarray& input, double sigma, int axis, double truncate,
                          const std::string& mode, double cval) {
  return correlate1d(input, ld::from_vec(gaussian_kernel(sigma, truncate)), axis, mode, cval, 0);
}
ndarray gaussian_filter(const ndarray& input, double sigma, double truncate, const std::string& mode,
                        double cval, Backend forced) {
  g_last = forced;
  std::vector<double> w = gaussian_kernel(sigma, truncate);
  d::Img im = d::to_img(input);
  im = corr1d(im, w, 0, mode, cval, 0);
  im = corr1d(im, w, 1, mode, cval, 0);
  return d::to_nd(im);
}

namespace {
ndarray rank_filter(const ndarray& input, int size, const std::string& mode, double cval, int which) {
  d::Img im = d::to_img(input);
  int rad = size / 2;
  d::Img out{std::vector<double>(im.r * im.c), im.r, im.c};
  std::vector<double> win;
  for (int64_t r = 0; r < im.r; ++r)
    for (int64_t c = 0; c < im.c; ++c) {
      win.clear();
      for (int a = -rad; a <= rad; ++a)
        for (int b = -rad; b <= rad; ++b) {
          int ri = d::bidx(static_cast<int>(r) + a, im.r, mode), ci = d::bidx(static_cast<int>(c) + b, im.c, mode);
          win.push_back((ri < 0 || ci < 0) ? cval : im.d[ri * im.c + ci]);
        }
      if (which == 0) out.d[r * im.c + c] = *std::min_element(win.begin(), win.end());
      else if (which == 2) out.d[r * im.c + c] = *std::max_element(win.begin(), win.end());
      else { std::sort(win.begin(), win.end()); out.d[r * im.c + c] = win[win.size() / 2]; }
    }
  return d::to_nd(out);
}
}  // namespace

ndarray minimum_filter(const ndarray& input, int size, const std::string& mode, double cval) { return rank_filter(input, size, mode, cval, 0); }
ndarray median_filter(const ndarray& input, int size, const std::string& mode, double cval) { return rank_filter(input, size, mode, cval, 1); }
ndarray maximum_filter(const ndarray& input, int size, const std::string& mode, double cval) { return rank_filter(input, size, mode, cval, 2); }

ndarray sobel(const ndarray& input, int axis, const std::string& mode, double cval) {
  int ax = real_axis(axis, 2), other = 1 - ax;
  ndarray t = correlate1d(input, ld::from_vec({-1, 0, 1}), ax, mode, cval, 0);
  return correlate1d(t, ld::from_vec({1, 2, 1}), other, mode, cval, 0);
}
ndarray prewitt(const ndarray& input, int axis, const std::string& mode, double cval) {
  int ax = real_axis(axis, 2), other = 1 - ax;
  ndarray t = correlate1d(input, ld::from_vec({-1, 0, 1}), ax, mode, cval, 0);
  return correlate1d(t, ld::from_vec({1, 1, 1}), other, mode, cval, 0);
}
ndarray laplace(const ndarray& input, const std::string& mode, double cval) {
  d::Img a = d::to_img(correlate1d(input, ld::from_vec({1, -2, 1}), 0, mode, cval, 0));
  d::Img b = d::to_img(correlate1d(input, ld::from_vec({1, -2, 1}), 1, mode, cval, 0));
  for (int64_t i = 0; i < a.r * a.c; ++i) a.d[i] += b.d[i];
  return d::to_nd(a);
}

}  // namespace scypp::ndimage
