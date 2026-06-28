// Oracle tests for scipp::cluster and scipp::io against frozen SciPy golden data.
#include <string>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/cluster/cluster.hpp"
#include "scipp/io/io.hpp"
#include "scipp_test.hpp"

namespace cl = scipp::cluster;
namespace io = scipp::io;
namespace {
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
std::string gp(const char* name) { return std::string(SCIPP_GOLDEN_DIR) + "/" + name; }
}  // namespace
#define M(name) mat(golden::name##_d, golden::name##_r, golden::name##_c)
#define MN(name) (golden::name##_r * golden::name##_c)
#define G(name) golden::name, golden::name##_n

TEST_CASE("cluster.vq") {
  auto obs = M(cl_obs);
  cv(cl::whiten(obs), golden::cl_whiten_d, MN(cl_whiten));
  auto r = cl::vq(obs, M(cl_cb));
  cv(r.code, G(cl_vq_code));
  cv(r.dist, G(cl_vq_dist));
  auto km = cl::kmeans2(obs, M(cl_cb), 10);
  cv(km.centroids, golden::cl_km_centroids_d, MN(cl_km_centroids));
  cv(km.labels, G(cl_km_labels));
}

TEST_CASE("cluster.hierarchy") {
  auto pts = M(cl_pts);
  cv(cl::linkage(pts, "single"), golden::cl_link_single_d, MN(cl_link_single), 1e-9, 1e-11);
  cv(cl::linkage(pts, "complete"), golden::cl_link_complete_d, MN(cl_link_complete), 1e-9, 1e-11);
  cv(cl::linkage(pts, "average"), golden::cl_link_average_d, MN(cl_link_average), 1e-9, 1e-11);
  cv(cl::linkage(pts, "ward"), golden::cl_link_ward_d, MN(cl_link_ward), 1e-9, 1e-11);
  auto Z = cl::linkage(pts, "average");
  cv(cl::cophenet(Z), G(cl_cophenet), 1e-9, 1e-11);
  // fcluster up to relabeling
  auto got = tov(cl::fcluster(Z, 3, "maxclust"));
  std::vector<double> exp(golden::cl_fcluster, golden::cl_fcluster + golden::cl_fcluster_n);
  for (size_t i = 0; i < got.size(); ++i)
    for (size_t j = 0; j < got.size(); ++j) CHECK((got[i] == got[j]) == (exp[i] == exp[j]));
}

TEST_CASE("io: Matrix Market") {
  auto a = io::mmread(gp("sample.mtx"));
  cv(a, golden::io_mm_d, MN(io_mm));
  // round-trip
  std::string tmp = std::string(SCIPP_GOLDEN_DIR) + "/_rt.mtx";
  io::mmwrite(tmp, a);
  cv(io::mmread(tmp), golden::io_mm_d, MN(io_mm));
  std::remove(tmp.c_str());
}

TEST_CASE("io: WAV") {
  auto w = io::wavread(gp("sample.wav"));
  CHECK(w.rate == static_cast<int>(golden::io_wav_rate));
  cv(w.data, G(io_wav), 0, 0);
  // round-trip
  std::string tmp = std::string(SCIPP_GOLDEN_DIR) + "/_rt.wav";
  io::wavwrite(tmp, w.rate, w.data);
  auto w2 = io::wavread(tmp);
  CHECK(w2.rate == w.rate);
  cv(w2.data, G(io_wav), 0, 0);
  std::remove(tmp.c_str());
}

TEST_CASE("io: ARFF") {
  cv(io::loadarff(gp("sample.arff")), golden::io_arff_d, MN(io_arff));
}
