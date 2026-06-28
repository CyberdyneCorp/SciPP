// Oracle tests for scipp::spatial against frozen SciPy golden data.
#include <algorithm>
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/spatial/spatial.hpp"
#include "scipp_test.hpp"

namespace sa = scipp::spatial;
namespace {
numpp::ndarray vec(std::vector<double> v) {
  numpp::ndarray a(numpp::Shape{(int64_t)v.size()}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
  return a;
}
numpp::ndarray mat(const double* d, int r, int c) {
  numpp::ndarray a(numpp::Shape{r, c}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < r * c; ++i) p[i] = d[i];
  return a;
}
std::vector<double> tov(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kFloat64).ascontiguousarray();
  const double* p = c.typed_data<double>();
  return std::vector<double>(p, p + c.size());
}
void cv(const numpp::ndarray& got, const double* exp, int n, double rtol = 1e-9, double atol = 1e-11) {
  auto g = tov(got);
  for (int i = 0; i < n && i < (int)g.size(); ++i) CHECK_CLOSE(g[i], exp[i], rtol, atol);
}
}  // namespace
#define M(name) mat(golden::name##_d, golden::name##_r, golden::name##_c)
#define G(name) golden::name, golden::name##_n

TEST_CASE("distances") {
  auto P = M(sa_P), Q = M(sa_Q);
  cv(sa::pdist(P, "euclidean"), G(sa_pdist_euclidean));
  cv(sa::pdist(P, "cityblock"), G(sa_pdist_cityblock));
  cv(sa::pdist(P, "chebyshev"), G(sa_pdist_chebyshev));
  cv(sa::pdist(P, "cosine"), G(sa_pdist_cosine));
  cv(sa::pdist(P, "correlation"), G(sa_pdist_correlation));
  cv(sa::pdist(P, "hamming"), G(sa_pdist_hamming));
  cv(sa::cdist(P, Q, "euclidean"), golden::sa_cdist_eucl_d, golden::sa_cdist_eucl_r * golden::sa_cdist_eucl_c);
  cv(sa::squareform(sa::pdist(P)), golden::sa_squareform_d, golden::sa_squareform_r * golden::sa_squareform_c);
  // Backend dispatch equivalence: cdist euclidean now delegates to
  // numpp::cdist_euclidean, which owns device offload + CPU fallback. Forcing
  // Cpu pins the CPU path; Device lets NumPP choose (it falls back to CPU here
  // since NumPP is CPU-only locally, so last_backend() may report Cpu). The
  // numeric result is identical regardless of backend.
  auto a = tov(sa::cdist(P, Q, "euclidean", 2.0, sa::Backend::Cpu));
  CHECK(sa::last_backend() == sa::Backend::Cpu);
  auto b = tov(sa::cdist(P, Q, "euclidean", 2.0, sa::Backend::Device));
  auto bk = sa::last_backend();
  CHECK(bk == sa::Backend::Cpu || bk == sa::Backend::Device);
  for (size_t i = 0; i < a.size(); ++i) CHECK_CLOSE(a[i], b[i], 1e-12, 1e-12);
}

TEST_CASE("cdist euclidean via numpp delegation") {
  auto P = M(sa_P), Q = M(sa_Q);
  // (a) Offloaded euclidean/sqeuclidean cdist still matches the SciPy golden.
  cv(sa::cdist(P, Q, "euclidean"), golden::sa_cdist_eucl_d,
     golden::sa_cdist_eucl_r * golden::sa_cdist_eucl_c);
  auto eucl = tov(sa::cdist(P, Q, "euclidean"));
  auto sq = tov(sa::cdist(P, Q, "sqeuclidean"));
  for (size_t i = 0; i < eucl.size(); ++i) CHECK_CLOSE(sq[i], eucl[i] * eucl[i], 1e-12, 1e-12);
  // pdist euclidean delegates through the same kernel and matches the golden.
  cv(sa::pdist(P, "euclidean"), G(sa_pdist_euclidean));
  // (b) last_backend() is reported after the delegated call.
  (void)sa::cdist(P, Q, "euclidean");
  auto bk = sa::last_backend();
  CHECK(bk == sa::Backend::Cpu || bk == sa::Backend::Device);

  // (c) Small known case: rows of A at (0,0) and (3,4); B at origin and (1,1).
  std::vector<double> av{0, 0, 3, 4}, bv{0, 0, 1, 1};
  numpp::ndarray Asm = mat(av.data(), 2, 2), Bsm = mat(bv.data(), 2, 2);
  auto d = tov(sa::cdist(Asm, Bsm, "euclidean"));
  // D[0][0]=0, D[0][1]=sqrt(2), D[1][0]=5, D[1][1]=sqrt(4+9)=sqrt(13).
  CHECK_CLOSE(d[0], 0.0, 1e-12, 1e-12);
  CHECK_CLOSE(d[1], std::sqrt(2.0), 1e-12, 1e-12);
  CHECK_CLOSE(d[2], 5.0, 1e-12, 1e-12);
  CHECK_CLOSE(d[3], std::sqrt(13.0), 1e-12, 1e-12);
}

TEST_CASE("KDTree") {
  auto P = M(sa_P), Q = M(sa_Q);
  sa::KDTree tree(P);
  auto qr = tree.query(Q, 2);
  cv(qr.distances, golden::sa_kd_dist_d, golden::sa_kd_dist_r * golden::sa_kd_dist_c);
  cv(qr.indices, golden::sa_kd_idx_d, golden::sa_kd_idx_r * golden::sa_kd_idx_c);
}

TEST_CASE("ConvexHull and Delaunay") {
  auto P = M(sa_P);
  sa::ConvexHull hull(P);
  CHECK_CLOSE(hull.area, golden::sa_hull_area, 1e-9, 1e-11);
  CHECK_CLOSE(hull.volume, golden::sa_hull_volume, 1e-9, 1e-11);
  auto vs = tov(hull.vertices);
  std::sort(vs.begin(), vs.end());
  for (int i = 0; i < golden::sa_hull_vertices_n; ++i) CHECK_CLOSE(vs[i], golden::sa_hull_vertices[i], 0, 0);

  sa::Delaunay d(M(sa_DP));  // general-position points -> unique triangulation
  // sort each triangle's vertices and lexsort the triangle list to compare sets
  auto raw = tov(d.simplices());
  int nt = (int)raw.size() / 3;
  std::vector<std::array<double, 3>> tris(nt);
  for (int i = 0; i < nt; ++i) { tris[i] = {raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]}; std::sort(tris[i].begin(), tris[i].end()); }
  std::sort(tris.begin(), tris.end());
  CHECK(nt == golden::sa_delaunay_r);
  for (int i = 0; i < nt && i < golden::sa_delaunay_r; ++i)
    for (int j = 0; j < 3; ++j) CHECK_CLOSE(tris[i][j], golden::sa_delaunay_d[i * 3 + j], 0, 0);
  CHECK((d.find_simplex(vec({0.5, 0.5})) >= 0) == (golden::sa_find_simplex_in > 0.5));
}

TEST_CASE("Rotation") {
  auto r = sa::Rotation::from_euler("xyz", vec({30, 45, 60}), true);
  cv(r.as_matrix(), golden::sa_rot_matrix_d, 9, 1e-9, 1e-11);
  cv(r.apply(vec({1, 2, 3})), G(sa_rot_apply), 1e-9, 1e-11);
  cv(r.as_euler("xyz", true), G(sa_rot_euler), 1e-7, 1e-9);
  cv(r.as_rotvec(), G(sa_rot_rotvec), 1e-9, 1e-11);
  CHECK_CLOSE(r.magnitude(), golden::sa_rot_mag, 1e-9, 1e-11);
  // quaternion up to sign
  auto q = tov(r.as_quat());
  bool same = true, neg = true;
  for (int i = 0; i < 4; ++i) { same &= std::fabs(q[i] - golden::sa_rot_quat[i]) < 1e-9; neg &= std::fabs(q[i] + golden::sa_rot_quat[i]) < 1e-9; }
  CHECK(same || neg);
  // round-trip via matrix
  cv(sa::Rotation::from_matrix(r.as_matrix()).apply(vec({1, 2, 3})), G(sa_rot_apply), 1e-9, 1e-11);
  // composition / inverse
  auto r2 = sa::Rotation::from_quat(vec({0.1, 0.2, 0.3, 0.9}));
  cv((r * r2).apply(vec({1, 0, 0})), G(sa_rot_compose_apply), 1e-9, 1e-11);
  cv(r.inv().apply(vec({1, 2, 3})), G(sa_rot_inv_apply), 1e-9, 1e-11);
}

TEST_CASE("Slerp") {
  std::vector<sa::Rotation> key{sa::Rotation::from_euler("z", vec({0}), true),
                                sa::Rotation::from_euler("z", vec({90}), true),
                                sa::Rotation::from_euler("z", vec({180}), true)};
  sa::Slerp sl(vec({0, 1, 2}), key);
  cv(sl(0.7).apply(vec({1, 0, 0})), G(sa_slerp_apply), 1e-9, 1e-11);
}
