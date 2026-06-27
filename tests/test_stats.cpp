// Oracle tests for scypp::stats against frozen SciPy golden data.
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/stats/stats.hpp"
#include "scypp_test.hpp"

namespace st = scypp::stats;

namespace {
constexpr double R = 1e-7, A = 1e-9;
numpp::ndarray vec(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
numpp::ndarray m2(const double* d, int r, int c) {
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
}  // namespace
#define V(name) vec(golden::name, golden::name##_n)

TEST_CASE("continuous distributions") {
  CHECK_CLOSE(st::norm::pdf(0.7, 0.2, 1.5), golden::st_norm_pdf, R, A);
  CHECK_CLOSE(st::norm::cdf(0.7, 0.2, 1.5), golden::st_norm_cdf, R, A);
  CHECK_CLOSE(st::norm::ppf(0.83, 0.2, 1.5), golden::st_norm_ppf, R, A);
  CHECK_CLOSE(st::gamma::cdf(3.0, 2.5, 0, 1.3), golden::st_gamma_cdf, R, A);
  CHECK_CLOSE(st::gamma::ppf(0.6, 2.5, 0, 1.3), golden::st_gamma_ppf, 1e-7, 1e-8);
  CHECK_CLOSE(st::chi2::cdf(5.0, 3), golden::st_chi2_cdf, R, A);
  CHECK_CLOSE(st::chi2::sf(5.0, 3), golden::st_chi2_sf, R, A);
  CHECK_CLOSE(st::beta::cdf(0.4, 2.0, 3.0), golden::st_beta_cdf, R, A);
  CHECK_CLOSE(st::beta::ppf(0.7, 2.0, 3.0), golden::st_beta_ppf, 1e-7, 1e-8);
  CHECK_CLOSE(st::t::cdf(1.5, 8), golden::st_t_cdf, R, A);
  CHECK_CLOSE(st::t::ppf(0.975, 8), golden::st_t_ppf, 1e-7, 1e-8);
  CHECK_CLOSE(st::f::cdf(2.0, 5, 10), golden::st_f_cdf, R, A);
  CHECK_CLOSE(st::f::sf(2.0, 5, 10), golden::st_f_sf, R, A);
  CHECK_CLOSE(st::expon::cdf(2.0, 0, 1.5), golden::st_expon_cdf, R, A);
  // ppf inverts cdf
  CHECK_CLOSE(st::gamma::cdf(st::gamma::ppf(0.42, 2.5, 0, 1.3), 2.5, 0, 1.3), 0.42, 1e-7, 1e-8);
  CHECK_CLOSE(st::norm::sf(0.7, 0.2, 1.5), 1.0 - st::norm::cdf(0.7, 0.2, 1.5), R, A);
}

TEST_CASE("summary statistics") {
  auto d = V(st_data);
  CHECK_CLOSE(st::gmean(d), golden::st_gmean, R, A);
  CHECK_CLOSE(st::hmean(d), golden::st_hmean, R, A);
  CHECK_CLOSE(st::skew(d), golden::st_skew, R, A);
  CHECK_CLOSE(st::kurtosis(d), golden::st_kurtosis, R, A);
  CHECK_CLOSE(st::sem(d), golden::st_sem, R, A);
  CHECK_CLOSE(st::variation(d), golden::st_variation, R, A);
  CHECK_CLOSE(st::iqr(d), golden::st_iqr, R, A);
  auto rk = tov(st::rankdata(V(st_tied)));
  for (int i = 0; i < golden::st_rankdata_n; ++i) CHECK_CLOSE(rk[i], golden::st_rankdata[i], R, A);
  auto z = tov(st::zscore(d));
  for (int i = 0; i < golden::st_zscore_n; ++i) CHECK_CLOSE(z[i], golden::st_zscore[i], R, A);
}

TEST_CASE("correlation and regression") {
  auto x = V(st_cx), y = V(st_cy);
  auto pr = st::pearsonr(x, y);
  CHECK_CLOSE(pr.statistic, golden::st_pearson_r, R, A);
  CHECK_CLOSE(pr.pvalue, golden::st_pearson_p, 1e-6, 1e-10);
  auto sp = st::spearmanr(x, y);
  CHECK_CLOSE(sp.statistic, golden::st_spearman_r, R, A);
  CHECK_CLOSE(sp.pvalue, golden::st_spearman_p, 1e-6, 1e-9);
  auto lr = st::linregress(x, y);
  CHECK_CLOSE(lr.slope, golden::st_lr_slope, R, A);
  CHECK_CLOSE(lr.intercept, golden::st_lr_intercept, R, A);
  CHECK_CLOSE(lr.rvalue, golden::st_lr_r, R, A);
  CHECK_CLOSE(lr.pvalue, golden::st_lr_p, 1e-6, 1e-10);
  CHECK_CLOSE(lr.stderr, golden::st_lr_stderr, R, A);
}

TEST_CASE("hypothesis tests") {
  auto ga = V(st_ga), gb = V(st_gb);
  auto t1 = st::ttest_1samp(ga, 5.0);
  CHECK_CLOSE(t1.statistic, golden::st_t1_stat, R, A);
  CHECK_CLOSE(t1.pvalue, golden::st_t1_p, 1e-6, 1e-9);
  auto ti = st::ttest_ind(ga, gb);
  CHECK_CLOSE(ti.statistic, golden::st_ti_stat, R, A);
  CHECK_CLOSE(ti.pvalue, golden::st_ti_p, 1e-6, 1e-9);
  auto tr = st::ttest_rel(ga, gb);
  CHECK_CLOSE(tr.statistic, golden::st_tr_stat, R, A);
  CHECK_CLOSE(tr.pvalue, golden::st_tr_p, 1e-6, 1e-9);
  double ga2[] = {5.5, 5.8, 6.1, 5.9, 6.0, 5.7, 6.2};
  std::vector<numpp::ndarray> groups{ga, gb, vec(ga2, 7)};
  auto fo = st::f_oneway(groups);
  CHECK_CLOSE(fo.statistic, golden::st_fo_stat, R, A);
  CHECK_CLOSE(fo.pvalue, golden::st_fo_p, 1e-6, 1e-9);
  auto ks = st::ks_2samp(ga, gb);
  CHECK_CLOSE(ks.statistic, golden::st_ks_stat, R, A);
  CHECK_CLOSE(ks.pvalue, golden::st_ks_p, 1e-5, 1e-7);
  auto c2 = st::chi2_contingency(m2(golden::st_ct_d, golden::st_ct_r, golden::st_ct_c));
  CHECK_CLOSE(c2.statistic, golden::st_chi2c_stat, R, A);
  CHECK_CLOSE(c2.pvalue, golden::st_chi2c_p, 1e-6, 1e-9);
  auto ce = tov(c2.expected);
  for (int i = 0; i < golden::st_chi2c_exp_r * golden::st_chi2c_exp_c; ++i)
    CHECK_CLOSE(ce[i], golden::st_chi2c_exp_d[i], R, A);
  auto nt = st::normaltest(V(st_nt_data));
  CHECK_CLOSE(nt.statistic, golden::st_nt_stat, 1e-6, 1e-8);
  CHECK_CLOSE(nt.pvalue, golden::st_nt_p, 1e-6, 1e-8);
}

TEST_CASE("gaussian_kde") {
  st::gaussian_kde scott(V(st_kde_data), "scott");
  auto s = tov(scott(V(st_kde_q)));
  for (int i = 0; i < golden::st_kde_scott_n; ++i) CHECK_CLOSE(s[i], golden::st_kde_scott[i], 1e-7, 1e-9);
  st::gaussian_kde silv(V(st_kde_data), "silverman");
  auto sv = tov(silv(V(st_kde_q)));
  for (int i = 0; i < golden::st_kde_silverman_n; ++i) CHECK_CLOSE(sv[i], golden::st_kde_silverman[i], 1e-7, 1e-9);
}
