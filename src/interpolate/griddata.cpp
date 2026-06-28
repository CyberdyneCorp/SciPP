// griddata — scattered 2-D interpolation (nearest / linear), mirroring
// scipy.interpolate.griddata. "nearest" uses a KD-tree; "linear" interpolates
// barycentrically over the Delaunay triangulation and returns fill_value outside
// the convex hull.
#include "scipp/interpolate/interpolate.hpp"

#include <stdexcept>
#include <vector>

#include "numpp/core/creation.hpp"
#include "numpp/core/dtype.hpp"
#include "scipp/spatial/spatial.hpp"

namespace scipp::interpolate {
namespace {
numpp::ndarray make_row(double x, double y) {
  numpp::ndarray p(numpp::Shape{1, 2}, numpp::kFloat64);
  double* d = p.typed_data<double>();
  d[0] = x;
  d[1] = y;
  return p;
}
}  // namespace

ndarray griddata(const ndarray& points, const ndarray& values, const ndarray& xi,
                 std::string method, double fill_value) {
  const auto& ps = points.shape();
  const auto& xs = xi.shape();
  if (ps.size() != 2 || ps[1] != 2)
    throw std::invalid_argument("griddata: points must be (n, 2)");
  if (xs.size() != 2 || xs[1] != 2)
    throw std::invalid_argument("griddata: xi must be (m, 2)");
  const int64_t n = ps[0], m = xs[0];
  const double* P = points.typed_data<double>();
  const double* V = values.typed_data<double>();
  const double* X = xi.typed_data<double>();

  numpp::ndarray out(numpp::Shape{m}, numpp::kFloat64);
  double* O = out.typed_data<double>();

  if (method == "nearest") {
    scipp::spatial::KDTree tree(points);
    auto q = tree.query(xi, 1);
    const double* idx = q.indices.typed_data<double>();
    for (int64_t i = 0; i < m; ++i) O[i] = V[static_cast<int64_t>(idx[i])];
    return out;
  }

  if (method != "linear")
    throw std::invalid_argument("griddata: method must be 'nearest' or 'linear'");

  scipp::spatial::Delaunay tri(points);
  numpp::ndarray simp = tri.simplices();
  const double* S = simp.typed_data<double>();
  (void)n;

  for (int64_t i = 0; i < m; ++i) {
    double px = X[i * 2], py = X[i * 2 + 1];
    int64_t t = tri.find_simplex(make_row(px, py));
    if (t < 0) {
      O[i] = fill_value;
      continue;
    }
    int64_t ia = static_cast<int64_t>(S[t * 3]);
    int64_t ib = static_cast<int64_t>(S[t * 3 + 1]);
    int64_t ic = static_cast<int64_t>(S[t * 3 + 2]);
    double ax = P[ia * 2], ay = P[ia * 2 + 1];
    double bx = P[ib * 2], by = P[ib * 2 + 1];
    double cx = P[ic * 2], cy = P[ic * 2 + 1];
    // Barycentric coordinates of (px,py) in triangle (a,b,c).
    double v0x = bx - ax, v0y = by - ay;
    double v1x = cx - ax, v1y = cy - ay;
    double v2x = px - ax, v2y = py - ay;
    double d00 = v0x * v0x + v0y * v0y;
    double d01 = v0x * v1x + v0y * v1y;
    double d11 = v1x * v1x + v1y * v1y;
    double d20 = v2x * v0x + v2y * v0y;
    double d21 = v2x * v1x + v2y * v1y;
    double denom = d00 * d11 - d01 * d01;
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;
    O[i] = u * V[ia] + v * V[ib] + w * V[ic];
  }
  return out;
}

}  // namespace scipp::interpolate
