// Oracle tests for scypp::fft against frozen SciPy golden data.
#include <vector>

#include "golden.hpp"
#include "numpp/core/creation.hpp"  // numpp::arange
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "numpp/umath/ufunc.hpp"  // numpp::real / numpp::imag
#include "scypp/fft/fft.hpp"
#include "scypp_test.hpp"

namespace ff = scypp::fft;

namespace {
constexpr double R = 1e-9, A = 1e-11;

numpp::ndarray vec(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
std::vector<double> contig(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kFloat64).ascontiguousarray();
  const double* p = c.typed_data<double>();
  return std::vector<double>(p, p + c.size());
}
void check_vec(const numpp::ndarray& got, const double* exp, int n,
               double rtol = R, double atol = A) {
  auto g = contig(got);
  CHECK(static_cast<int>(g.size()) == n);
  for (int i = 0; i < n && i < static_cast<int>(g.size()); ++i) CHECK_CLOSE(g[i], exp[i], rtol, atol);
}
}  // namespace

#define V(name) golden::name, golden::name##_n

TEST_CASE("fft / rfft match SciPy") {
  auto x = vec(V(fft_x));
  auto F = ff::fft(x);
  check_vec(numpp::real(F), V(fft_re));
  check_vec(numpp::imag(F), V(fft_im));
  auto Rf = ff::rfft(x);
  check_vec(numpp::real(Rf), V(rfft_re));
  check_vec(numpp::imag(Rf), V(rfft_im));
  auto Fo = ff::fft(x, std::nullopt, -1, "ortho");
  check_vec(numpp::real(Fo), V(fft_ortho_re));
  check_vec(numpp::imag(Fo), V(fft_ortho_im));
}

TEST_CASE("fft inverse round-trips") {
  auto x = vec(V(fft_x));
  auto back = ff::ifft(ff::fft(x));
  check_vec(numpp::real(back), V(fft_x), 1e-10, 1e-11);
  auto rback = ff::irfft(ff::rfft(x), golden::fft_x_n);
  check_vec(rback, V(fft_x), 1e-10, 1e-11);
}

TEST_CASE("DCT / DST match SciPy") {
  auto x = vec(V(fft_x));
  check_vec(ff::dct(x, 1), V(dct1), 1e-9, 1e-10);
  check_vec(ff::dct(x, 2), V(dct2), 1e-9, 1e-10);
  check_vec(ff::dct(x, 3), V(dct3), 1e-9, 1e-10);
  check_vec(ff::dct(x, 4), V(dct4), 1e-9, 1e-10);
  check_vec(ff::dst(x, 1), V(dst1), 1e-9, 1e-10);
  check_vec(ff::dst(x, 2), V(dst2), 1e-9, 1e-10);
  check_vec(ff::dst(x, 3), V(dst3), 1e-9, 1e-10);
  check_vec(ff::dst(x, 4), V(dst4), 1e-9, 1e-10);
  check_vec(ff::idct(x, 2), V(idct2), 1e-9, 1e-10);
  check_vec(ff::idst(x, 2), V(idst2), 1e-9, 1e-10);
}

TEST_CASE("DCT / DST inverses round-trip") {
  auto x = vec(V(fft_x));
  for (int t = 1; t <= 4; ++t) {
    check_vec(ff::idct(ff::dct(x, t), t), V(fft_x), 1e-9, 1e-10);
    check_vec(ff::idst(ff::dst(x, t), t), V(fft_x), 1e-9, 1e-10);
  }
}

TEST_CASE("DCT along an axis") {
  auto X2 = [] {
    numpp::ndarray a(numpp::Shape{golden::fft_X2_r, golden::fft_X2_c}, numpp::kFloat64);
    double* p = a.typed_data<double>();
    for (int i = 0; i < golden::fft_X2_r * golden::fft_X2_c; ++i) p[i] = golden::fft_X2_d[i];
    return a;
  }();
  auto d = ff::dct(X2, 2, 0);
  auto g = contig(d);
  for (int i = 0; i < golden::fft_dct2_ax0_r * golden::fft_dct2_ax0_c; ++i)
    CHECK_CLOSE(g[i], golden::fft_dct2_ax0_d[i], 1e-9, 1e-10);
}

TEST_CASE("helpers and next_fast_len") {
  check_vec(ff::fftfreq(8, 0.1), V(fftfreq8));
  check_vec(ff::rfftfreq(8, 0.1), V(rfftfreq8));
  check_vec(ff::fftshift(numpp::arange(0.0, 8.0, 1.0, numpp::kFloat64)), V(fftshift8));
  CHECK(ff::next_fast_len(1) == static_cast<int64_t>(golden::nfl_1));
  CHECK(ff::next_fast_len(7) == static_cast<int64_t>(golden::nfl_7));
  CHECK(ff::next_fast_len(8) == static_cast<int64_t>(golden::nfl_8));
  CHECK(ff::next_fast_len(13) == static_cast<int64_t>(golden::nfl_13));
  CHECK(ff::next_fast_len(100) == static_cast<int64_t>(golden::nfl_100));
  CHECK(ff::next_fast_len(1000) == static_cast<int64_t>(golden::nfl_1000));
}

TEST_CASE("fftpack delegates to fft") {
  auto x = vec(V(fft_x));
  auto a = contig(numpp::real(scypp::fftpack::fft(x)));
  auto b = contig(numpp::real(ff::fft(x)));
  for (size_t i = 0; i < a.size(); ++i) CHECK_CLOSE(a[i], b[i], 1e-12, 1e-12);
  auto c = contig(scypp::fftpack::dct(x, 2));
  auto e = contig(ff::dct(x, 2));
  for (size_t i = 0; i < c.size(); ++i) CHECK_CLOSE(c[i], e[i], 1e-12, 1e-12);
}
