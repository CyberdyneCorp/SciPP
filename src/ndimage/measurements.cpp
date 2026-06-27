// Measurements: label, center_of_mass, sum/mean/maximum/minimum, find_objects.
#include "scypp/ndimage/ndimage.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <vector>

#include "scypp/linalg/detail.hpp"
#include "scypp/ndimage/detail.hpp"

namespace scypp::ndimage {
namespace d = detail;
namespace ld = scypp::linalg::detail;

LabelResult label(const ndarray& input) {
  d::Img im = d::to_img(input);
  int64_t R = im.r, C = im.c, n = R * C;
  std::vector<int> uf(n, -1);
  std::function<int(int)> find = [&](int x) { while (uf[x] >= 0) x = uf[x]; return x; };
  auto unite = [&](int a, int b) { a = find(a); b = find(b); if (a != b) uf[a] = b; };
  auto fg = [&](int64_t r, int64_t c) { return im.d[r * C + c] != 0; };
  for (int64_t r = 0; r < R; ++r)
    for (int64_t c = 0; c < C; ++c) {
      if (!fg(r, c)) continue;
      uf[r * C + c] = (uf[r * C + c] < 0) ? -2 : uf[r * C + c];  // mark as foreground root
      if (r > 0 && fg(r - 1, c)) unite(static_cast<int>(r * C + c), static_cast<int>((r - 1) * C + c));
      if (c > 0 && fg(r, c - 1)) unite(static_cast<int>(r * C + c), static_cast<int>(r * C + c - 1));
    }
  std::vector<double> lab(n, 0.0);
  std::map<int, int> root_to_label;
  int next = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (im.d[i] == 0) continue;
    int rt = find(static_cast<int>(i));
    auto it = root_to_label.find(rt);
    if (it == root_to_label.end()) { root_to_label[rt] = ++next; lab[i] = next; }
    else lab[i] = it->second;
  }
  d::Img out{lab, R, C};
  return {d::to_nd(out), next};
}

ndarray center_of_mass(const ndarray& input) {
  d::Img im = d::to_img(input);
  double sr = 0, sc = 0, tot = 0;
  for (int64_t r = 0; r < im.r; ++r)
    for (int64_t c = 0; c < im.c; ++c) { double v = im.d[r * im.c + c]; sr += r * v; sc += c * v; tot += v; }
  return ld::from_vec({sr / tot, sc / tot});
}

double sum_labels(const ndarray& input) { auto v = ld::to_vec(input); double s = 0; for (double x : v) s += x; return s; }
double mean(const ndarray& input) { auto v = ld::to_vec(input); double s = 0; for (double x : v) s += x; return s / v.size(); }
double maximum(const ndarray& input) { auto v = ld::to_vec(input); return *std::max_element(v.begin(), v.end()); }
double minimum(const ndarray& input) { auto v = ld::to_vec(input); return *std::min_element(v.begin(), v.end()); }

namespace {
ndarray labeled(const ndarray& input, const ndarray& labels, const std::vector<int>& index, bool meanmode) {
  auto v = ld::to_vec(input), lab = ld::to_vec(labels);
  std::vector<double> out(index.size());
  for (size_t k = 0; k < index.size(); ++k) {
    double s = 0; int cnt = 0;
    for (size_t i = 0; i < v.size(); ++i) if (static_cast<int>(lab[i]) == index[k]) { s += v[i]; ++cnt; }
    out[k] = meanmode ? (cnt ? s / cnt : 0.0) : s;
  }
  return ld::from_vec(out);
}
}  // namespace

ndarray sum_labels(const ndarray& input, const ndarray& labels, const std::vector<int>& index) { return labeled(input, labels, index, false); }
ndarray mean(const ndarray& input, const ndarray& labels, const std::vector<int>& index) { return labeled(input, labels, index, true); }

}  // namespace scypp::ndimage
