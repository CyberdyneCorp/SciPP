// Oracle tests for scipp::stats discrete distributions and rank tests.
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/stats/stats.hpp"
#include "scipp_test.hpp"

namespace st = scipp::stats;
namespace {
numpp::ndarray vec(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
}  // namespace
#define V(name) vec(golden::name, golden::name##_n)

TEST_CASE("discrete distributions") {
  for (int k = 0; k <= 10; ++k) {
    CHECK_CLOSE(st::binom::pmf(k, 10, 0.35), golden::dd_binom_pmf[k], 1e-10, 1e-12);
    CHECK_CLOSE(st::binom::cdf(k, 10, 0.35), golden::dd_binom_cdf[k], 1e-9, 1e-11);
  }
  CHECK_CLOSE(st::binom::ppf(0.7, 10, 0.35), golden::dd_binom_ppf, 0, 0);
  for (int k = 0; k < 10; ++k) {
    CHECK_CLOSE(st::poisson::pmf(k, 3.2), golden::dd_poisson_pmf[k], 1e-10, 1e-12);
    CHECK_CLOSE(st::poisson::cdf(k, 3.2), golden::dd_poisson_cdf[k], 1e-9, 1e-11);
    CHECK_CLOSE(st::nbinom::pmf(k, 5, 0.4), golden::dd_nbinom_pmf[k], 1e-10, 1e-12);
    CHECK_CLOSE(st::nbinom::cdf(k, 5, 0.4), golden::dd_nbinom_cdf[k], 1e-9, 1e-11);
  }
  for (int k = 1; k <= 7; ++k) {
    CHECK_CLOSE(st::geom::pmf(k, 0.3), golden::dd_geom_pmf[k - 1], 1e-10, 1e-12);
    CHECK_CLOSE(st::geom::cdf(k, 0.3), golden::dd_geom_cdf[k - 1], 1e-10, 1e-12);
  }
  for (int k = 0; k < 8; ++k) {
    CHECK_CLOSE(st::hypergeom::pmf(k, 20, 7, 12), golden::dd_hypergeom_pmf[k], 1e-10, 1e-12);
    CHECK_CLOSE(st::hypergeom::cdf(k, 20, 7, 12), golden::dd_hypergeom_cdf[k], 1e-9, 1e-11);
  }
  CHECK_CLOSE(st::binom::mean(10, 0.35), golden::dd_binom_mean, 1e-12, 1e-12);
  CHECK_CLOSE(st::nbinom::var(5, 0.4), golden::dd_nbinom_var, 1e-10, 1e-12);
}

TEST_CASE("rank tests") {
  auto rx = V(dd_rx), ry = V(dd_ry);
  auto mw = st::mannwhitneyu(rx, ry);
  CHECK_CLOSE(mw.statistic, golden::dd_mw_stat, 1e-9, 1e-11);
  CHECK_CLOSE(mw.pvalue, golden::dd_mw_p, 1e-6, 1e-8);
  auto wl = st::wilcoxon(rx, ry);
  CHECK_CLOSE(wl.statistic, golden::dd_wl_stat, 1e-9, 1e-11);
  CHECK_CLOSE(wl.pvalue, golden::dd_wl_p, 1e-6, 1e-8);
  std::vector<numpp::ndarray> groups{V(dd_g1), V(dd_g2), V(dd_g3)};
  auto kr = st::kruskal(groups);
  CHECK_CLOSE(kr.statistic, golden::dd_kr_stat, 1e-9, 1e-11);
  CHECK_CLOSE(kr.pvalue, golden::dd_kr_p, 1e-7, 1e-9);
  auto kt = st::kendalltau(rx, ry);
  CHECK_CLOSE(kt.statistic, golden::dd_kt_stat, 1e-9, 1e-11);
  CHECK_CLOSE(kt.pvalue, golden::dd_kt_p, 1e-6, 1e-8);
}
