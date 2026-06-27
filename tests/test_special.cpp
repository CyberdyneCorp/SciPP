// Oracle tests for scypp::special against frozen SciPy golden data.
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/special/special.hpp"
#include "scypp_test.hpp"

namespace sp = scypp::special;

namespace {
numpp::ndarray make1d(const std::vector<double>& v) {
  numpp::ndarray a(numpp::Shape{static_cast<int64_t>(v.size())}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
  return a;
}
numpp::ndarray make2d(int64_t r, int64_t c, const std::vector<double>& v) {
  numpp::ndarray a(numpp::Shape{r, c}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
  return a;
}
constexpr double R = 1e-10, A = 1e-12;
}  // namespace

TEST_CASE("gamma family") {
  CHECK_ARR(gamma, [](double x) { return sp::gamma(x); }, R, A);
  CHECK_ARR(gammaln, [](double x) { return sp::gammaln(x); }, R, A);
  CHECK_ARR(digamma, [](double x) { return sp::digamma(x); }, 1e-9, 1e-11);
  CHECK_ARR(polygamma1, [](double x) { return sp::polygamma(1, x); }, 1e-9, 1e-11);
  CHECK_ARR(polygamma2, [](double x) { return sp::polygamma(2, x); }, 1e-8, 1e-10);
  CHECK_CLOSE(sp::beta(2, 3), golden::beta_2_3, R, A);
  CHECK_CLOSE(sp::beta(0.5, 2.5), golden::beta_05_25, R, A);
  CHECK_CLOSE(sp::betaln(2, 3), golden::betaln_2_3, R, A);
  CHECK_CLOSE(sp::betaln(10, 20), golden::betaln_10_20, R, A);
  // recurrence gamma(x+1) = x*gamma(x)
  CHECK_CLOSE(sp::gamma(6.0), 5.0 * sp::gamma(5.0), R, A);
  // pole at a non-positive integer
  CHECK(std::isnan(sp::digamma(-3.0)) || std::isinf(sp::gamma(-3.0)));
}

TEST_CASE("error functions") {
  CHECK_ARR(erf, [](double x) { return sp::erf(x); }, R, A);
  CHECK_ARR(erfc, [](double x) { return sp::erfc(x); }, R, A);
  CHECK_ARR(erfinv, [](double x) { return sp::erfinv(x); }, 1e-9, 1e-11);
  CHECK_ARR(erfcinv, [](double x) { return sp::erfcinv(x); }, 1e-9, 1e-11);
  // complementary identity and inverse round-trip
  CHECK_CLOSE(sp::erf(0.7) + sp::erfc(0.7), 1.0, R, A);
  CHECK_CLOSE(sp::erf(sp::erfinv(0.42)), 0.42, 1e-12, 1e-12);
}

TEST_CASE("exponential integrals") {
  CHECK_ARR(exp1, [](double x) { return sp::exp1(x); }, 1e-9, 1e-11);
  CHECK_ARR(expi, [](double x) { return sp::expi(x); }, 1e-9, 1e-11);
  CHECK_ARR(expn2, [](double x) { return sp::expn(2, x); }, 1e-9, 1e-11);
  CHECK_ARR(exprel, [](double x) { return sp::exprel(x); }, R, A);
  CHECK_CLOSE(sp::exprel(0.0), 1.0, R, A);
}

TEST_CASE("bessel") {
  CHECK_ARR(jv0, [](double x) { return sp::jv(0.0, x); }, 1e-10, 1e-12);
  CHECK_ARR(jv1, [](double x) { return sp::jv(1.0, x); }, 1e-10, 1e-12);
  CHECK_ARR(jvhalf, [](double x) { return sp::jv(0.5, x); }, 1e-10, 1e-12);
  CHECK_ARR(yv0, [](double x) { return sp::yv(0.0, x); }, 1e-9, 1e-11);
  CHECK_ARR(yv1, [](double x) { return sp::yv(1.0, x); }, 1e-9, 1e-11);
  CHECK_ARR(iv0, [](double x) { return sp::iv(0.0, x); }, 1e-10, 1e-12);
  CHECK_ARR(iv1, [](double x) { return sp::iv(1.0, x); }, 1e-10, 1e-12);
  CHECK_ARR(kv0, [](double x) { return sp::kv(0.0, x); }, 1e-9, 1e-11);
  CHECK_ARR(kv1, [](double x) { return sp::kv(1.0, x); }, 1e-9, 1e-11);
  CHECK_ARR(i0, [](double x) { return sp::i0(x); }, 1e-10, 1e-12);
  CHECK_ARR(i1, [](double x) { return sp::i1(x); }, 1e-10, 1e-12);
  CHECK_ARR(k0, [](double x) { return sp::k0(x); }, 1e-9, 1e-11);
  CHECK_ARR(k1, [](double x) { return sp::k1(x); }, 1e-9, 1e-11);
  // specialized vs general agree
  CHECK_CLOSE(sp::i0(2.0), sp::iv(0.0, 2.0), R, A);
}

TEST_CASE("orthogonal evaluators") {
  CHECK_ARR(legendre3, [](double x) { return sp::eval_legendre(3, x); }, R, A);
  CHECK_ARR(legendre10, [](double x) { return sp::eval_legendre(10, x); }, 1e-10, 1e-12);
  CHECK_ARR(chebyt4, [](double x) { return sp::eval_chebyt(4, x); }, R, A);
  CHECK_ARR(chebyt7, [](double x) { return sp::eval_chebyt(7, x); }, R, A);
  CHECK_ARR(hermite3, [](double x) { return sp::eval_hermite(3, x); }, R, A);
  CHECK_ARR(laguerre3, [](double x) { return sp::eval_laguerre(3, x); }, R, A);
  CHECK_ARR(genlag3, [](double x) { return sp::eval_genlaguerre(3, 1.5, x); }, R, A);
}

TEST_CASE("combinatorics") {
  CHECK_CLOSE(sp::comb(10, 3, true), golden::comb_10_3, R, A);
  CHECK_CLOSE(sp::comb(10, 3, false), golden::comb_10_3, 1e-12, 1e-9);
  CHECK_CLOSE(sp::comb(52, 5, true), golden::comb_52_5, R, A);
  CHECK_CLOSE(sp::perm(10, 3, true), golden::perm_10_3, R, A);
  CHECK_CLOSE(sp::factorial(5, true), golden::fact_5, R, A);
  CHECK_CLOSE(sp::factorial(10, true), golden::fact_10, R, A);
  CHECK(sp::comb(3, 5) == 0.0);
}

TEST_CASE("ndarray overloads match scalar path") {
  auto x = make1d({0.5, 1.0, 2.5, 5.0});
  auto g = sp::gamma(x);
  const double* gp = g.typed_data<double>();
  CHECK(g.shape() == (numpp::Shape{4}));
  for (int i = 0; i < 4; ++i) CHECK_CLOSE(gp[i], sp::gamma(x.typed_data<double>()[i]), R, A);

  auto b = sp::jv(0.0, x);
  const double* bp = b.typed_data<double>();
  for (int i = 0; i < 4; ++i) CHECK_CLOSE(bp[i], sp::jv(0.0, x.typed_data<double>()[i]), R, A);
}

TEST_CASE("reductions") {
  auto a1 = make1d({1.0, 2.0, 3.0, 4.0});
  CHECK_CLOSE(sp::logsumexp(a1), golden::lse_a1, R, A);
  auto big = make1d({1000.0, 1001.0, 1002.0});
  CHECK_CLOSE(sp::logsumexp(big), golden::lse_big, R, A);  // overflow-safe

  auto sm = sp::softmax(a1);
  const double* sp_ = sm.typed_data<double>();
  double s = 0.0;
  for (int i = 0; i < 4; ++i) { CHECK_CLOSE(sp_[i], golden::softmax_a1[i], R, A); s += sp_[i]; }
  CHECK_CLOSE(s, 1.0, R, A);

  auto a2 = make2d(2, 3, {1.0, 2.0, 3.0, 4.0, 6.0, 8.0});
  auto lse1 = sp::logsumexp(a2, 1);
  const double* l = lse1.typed_data<double>();
  CHECK_CLOSE(l[0], golden::lse_a2_ax1[0], R, A);
  CHECK_CLOSE(l[1], golden::lse_a2_ax1[1], R, A);

  auto sm2 = sp::softmax(a2, 1);
  const double* s2 = sm2.typed_data<double>();
  for (int i = 0; i < 6; ++i) CHECK_CLOSE(s2[i], golden::softmax_a2_ax1[i], R, A);
}
