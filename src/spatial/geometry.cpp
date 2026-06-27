// 2-D computational geometry: ConvexHull (monotone chain) and Delaunay
// (Bowyer-Watson incremental).
#include "scypp/spatial/spatial.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>
#include <set>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::spatial {
namespace {
namespace sd = scypp::linalg::detail;
double cross(double ox, double oy, double ax, double ay, double bx, double by) {
  return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}
}  // namespace

ConvexHull::ConvexHull(const ndarray& points) {
  numpp::ndarray P = points.astype(numpp::kFloat64).ascontiguousarray();
  int64_t n = P.shape()[0];
  const double* p = P.typed_data<double>();
  auto X = [&](int64_t i) { return p[i * 2]; };
  auto Y = [&](int64_t i) { return p[i * 2 + 1]; };
  std::vector<int64_t> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  std::sort(ids.begin(), ids.end(), [&](int64_t a, int64_t b) { return X(a) < X(b) || (X(a) == X(b) && Y(a) < Y(b)); });

  std::vector<int64_t> hull(2 * n);
  int kk = 0;
  for (int64_t ii = 0; ii < n; ++ii) {  // lower hull
    int64_t i = ids[ii];
    while (kk >= 2 && cross(X(hull[kk - 2]), Y(hull[kk - 2]), X(hull[kk - 1]), Y(hull[kk - 1]), X(i), Y(i)) <= 0) --kk;
    hull[kk++] = i;
  }
  int lower = kk + 1;
  for (int64_t ii = n - 2; ii >= 0; --ii) {  // upper hull
    int64_t i = ids[ii];
    while (kk >= lower && cross(X(hull[kk - 2]), Y(hull[kk - 2]), X(hull[kk - 1]), Y(hull[kk - 1]), X(i), Y(i)) <= 0) --kk;
    hull[kk++] = i;
  }
  hull.resize(kk - 1);  // last == first

  std::vector<double> verts(hull.begin(), hull.end());
  vertices = sd::from_vec(verts);
  int64_t h = static_cast<int64_t>(hull.size());
  std::vector<double> simp(h * 2);
  double per = 0, ar = 0;
  for (int64_t i = 0; i < h; ++i) {
    int64_t a = hull[i], b = hull[(i + 1) % h];
    simp[i * 2] = static_cast<double>(a); simp[i * 2 + 1] = static_cast<double>(b);
    per += std::hypot(X(b) - X(a), Y(b) - Y(a));
    ar += X(a) * Y(b) - X(b) * Y(a);  // shoelace
  }
  simplices = sd::from_mat(simp, h, 2);
  area = per;
  volume = std::fabs(ar) / 2.0;
}

// ---- Delaunay ----
namespace {
struct Tri { int a, b, c; };
bool in_circumcircle(const std::vector<double>& pt, int a, int b, int c, double px, double py) {
  double ax = pt[a * 2] - px, ay = pt[a * 2 + 1] - py;
  double bx = pt[b * 2] - px, by = pt[b * 2 + 1] - py;
  double cx = pt[c * 2] - px, cy = pt[c * 2 + 1] - py;
  double det = (ax * ax + ay * ay) * (bx * cy - cx * by) -
               (bx * bx + by * by) * (ax * cy - cx * ay) +
               (cx * cx + cy * cy) * (ax * by - bx * ay);
  return det > 1e-12;  // triangle is CCW
}
void ccw(const std::vector<double>& pt, int& a, int& b, int& c) {
  double o = cross(pt[a * 2], pt[a * 2 + 1], pt[b * 2], pt[b * 2 + 1], pt[c * 2], pt[c * 2 + 1]);
  if (o < 0) std::swap(b, c);
}
}  // namespace

Delaunay::Delaunay(const ndarray& points) {
  numpp::ndarray P = points.astype(numpp::kFloat64).ascontiguousarray();
  int64_t n = P.shape()[0];
  const double* p = P.typed_data<double>();
  pts_.assign(p, p + n * 2);
  double minx = 1e300, miny = 1e300, maxx = -1e300, maxy = -1e300;
  for (int64_t i = 0; i < n; ++i) { minx = std::min(minx, pts_[i * 2]); maxx = std::max(maxx, pts_[i * 2]); miny = std::min(miny, pts_[i * 2 + 1]); maxy = std::max(maxy, pts_[i * 2 + 1]); }
  double dx = maxx - minx, dy = maxy - miny, dm = std::max(dx, dy), mx = (minx + maxx) / 2, my = (miny + maxy) / 2;
  // super-triangle vertices at indices n, n+1, n+2
  pts_.push_back(mx - 20 * dm); pts_.push_back(my - dm);
  pts_.push_back(mx); pts_.push_back(my + 20 * dm);
  pts_.push_back(mx + 20 * dm); pts_.push_back(my - dm);
  int s0 = static_cast<int>(n), s1 = static_cast<int>(n + 1), s2 = static_cast<int>(n + 2);
  ccw(pts_, s0, s1, s2);  // the in-circle test requires CCW orientation
  std::vector<Tri> tris{{s0, s1, s2}};

  for (int64_t i = 0; i < n; ++i) {
    double px = pts_[i * 2], py = pts_[i * 2 + 1];
    std::vector<Tri> bad;
    std::vector<Tri> keep;
    for (const Tri& t : tris) (in_circumcircle(pts_, t.a, t.b, t.c, px, py) ? bad : keep).push_back(t);
    // boundary = edges that appear once among bad triangles
    std::map<std::pair<int, int>, int> edge_count;
    auto add = [&](int u, int v) { edge_count[{std::min(u, v), std::max(u, v)}]++; };
    for (const Tri& t : bad) { add(t.a, t.b); add(t.b, t.c); add(t.c, t.a); }
    tris = keep;
    for (const Tri& t : bad) {
      int e[3][2] = {{t.a, t.b}, {t.b, t.c}, {t.c, t.a}};
      for (auto& ed : e)
        if (edge_count[{std::min(ed[0], ed[1]), std::max(ed[0], ed[1])}] == 1) {
          int a = ed[0], b = ed[1], c = static_cast<int>(i);
          ccw(pts_, a, b, c);
          tris.push_back({a, b, c});
        }
    }
  }
  for (const Tri& t : tris)
    if (t.a < n && t.b < n && t.c < n) tris_.push_back({t.a, t.b, t.c});
}

ndarray Delaunay::simplices() const {
  int64_t m = static_cast<int64_t>(tris_.size());
  std::vector<double> out(m * 3);
  for (int64_t i = 0; i < m; ++i) { out[i * 3] = tris_[i][0]; out[i * 3 + 1] = tris_[i][1]; out[i * 3 + 2] = tris_[i][2]; }
  return sd::from_mat(out, m, 3);
}

int64_t Delaunay::find_simplex(const ndarray& p) const {
  std::vector<double> q = sd::to_vec(p);
  for (size_t t = 0; t < tris_.size(); ++t) {
    int a = tris_[t][0], b = tris_[t][1], c = tris_[t][2];
    double d1 = cross(pts_[a * 2], pts_[a * 2 + 1], pts_[b * 2], pts_[b * 2 + 1], q[0], q[1]);
    double d2 = cross(pts_[b * 2], pts_[b * 2 + 1], pts_[c * 2], pts_[c * 2 + 1], q[0], q[1]);
    double d3 = cross(pts_[c * 2], pts_[c * 2 + 1], pts_[a * 2], pts_[a * 2 + 1], q[0], q[1]);
    bool neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    if (!(neg && pos)) return static_cast<int64_t>(t);
  }
  return -1;
}

}  // namespace scypp::spatial
