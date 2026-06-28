// Oracle tests for scypp::stats normality tests (shapiro, anderson).
#include <stdexcept>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/stats/stats.hpp"
#include "scypp_test.hpp"

namespace st = scypp::stats;

namespace {
numpp::ndarray vecn(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
}  // namespace
#define VN(name) vecn(golden::name, golden::name##_n)

TEST_CASE("shapiro-wilk") {
  auto rn = st::shapiro(VN(nm_normal));
  CHECK_CLOSE(rn.statistic, golden::nm_normal_sw_W, 1e-4, 1e-6);
  CHECK_CLOSE(rn.pvalue, golden::nm_normal_sw_p, 1e-3, 1e-4);

  auto rs = st::shapiro(VN(nm_skew));
  CHECK_CLOSE(rs.statistic, golden::nm_skew_sw_W, 1e-4, 1e-6);
  CHECK_CLOSE(rs.pvalue, golden::nm_skew_sw_p, 1e-3, 1e-6);

  // n == 3 exact-p branch
  auto r3 = st::shapiro(VN(nm_small));
  CHECK_CLOSE(r3.statistic, golden::nm_small_sw_W, 1e-4, 1e-6);
  CHECK_CLOSE(r3.pvalue, golden::nm_small_sw_p, 1e-3, 1e-4);

  // fewer than 3 observations is an error
  const double two[2] = {1.0, 2.0};
  CHECK_THROWS_AS(st::shapiro(vecn(two, 2)), std::invalid_argument);
}

TEST_CASE("anderson-darling normal") {
  auto an = st::anderson(VN(nm_normal));
  CHECK_CLOSE(an.statistic, golden::nm_normal_ad_A2, 1e-4, 1e-6);
  CHECK(static_cast<int>(an.critical_values.size()) == golden::nm_normal_ad_crit_n);
  for (int i = 0; i < golden::nm_normal_ad_crit_n; ++i) {
    CHECK_CLOSE(an.critical_values[i], golden::nm_normal_ad_crit[i], 0, 1e-9);
    CHECK_CLOSE(an.significance_level[i], golden::nm_normal_ad_sig[i], 0, 1e-9);
  }

  auto as = st::anderson(VN(nm_skew));
  CHECK_CLOSE(as.statistic, golden::nm_skew_ad_A2, 1e-4, 1e-6);
  for (int i = 0; i < golden::nm_skew_ad_crit_n; ++i)
    CHECK_CLOSE(as.critical_values[i], golden::nm_skew_ad_crit[i], 0, 1e-9);
}
