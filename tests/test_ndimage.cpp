// Oracle tests for scypp::ndimage against frozen SciPy golden data.
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scypp/ndimage/ndimage.hpp"
#include "scypp_test.hpp"

namespace ni = scypp::ndimage;
namespace {
numpp::ndarray mat(const double* d, int r, int c) {
  numpp::ndarray a(numpp::Shape{r, c}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < r * c; ++i) p[i] = d[i];
  return a;
}
numpp::ndarray vec(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
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
#define MN(name) (golden::name##_r * golden::name##_c)
#define G(name) golden::name, golden::name##_n

TEST_CASE("filters across modes") {
  auto I = M(ni_I);
  cv(ni::gaussian_filter(I, 1.0, 4.0, "reflect"), golden::ni_gauss_reflect_d, MN(ni_gauss_reflect));
  cv(ni::gaussian_filter(I, 1.0, 4.0, "nearest"), golden::ni_gauss_nearest_d, MN(ni_gauss_nearest));
  cv(ni::gaussian_filter(I, 1.0, 4.0, "mirror"), golden::ni_gauss_mirror_d, MN(ni_gauss_mirror));
  cv(ni::gaussian_filter(I, 1.0, 4.0, "wrap"), golden::ni_gauss_wrap_d, MN(ni_gauss_wrap));
  cv(ni::gaussian_filter(I, 1.0, 4.0, "constant"), golden::ni_gauss_constant_d, MN(ni_gauss_constant));
  cv(ni::uniform_filter(I, 3), golden::ni_uniform_d, MN(ni_uniform));
  cv(ni::median_filter(I, 3), golden::ni_median_d, MN(ni_median));
  cv(ni::minimum_filter(I, 3), golden::ni_minimum_d, MN(ni_minimum));
  cv(ni::maximum_filter(I, 3), golden::ni_maximum_d, MN(ni_maximum));
  cv(ni::sobel(I, 0), golden::ni_sobel0_d, MN(ni_sobel0));
  cv(ni::sobel(I, 1), golden::ni_sobel1_d, MN(ni_sobel1));
  cv(ni::laplace(I), golden::ni_laplace_d, MN(ni_laplace));
  double w121[] = {1, 2, 1};
  cv(ni::correlate1d(vec(golden::ni_I_d, 5), vec(w121, 3)), G(ni_corr1d));
  // dispatch equivalence
  auto a = tov(ni::gaussian_filter(I, 1.0, 4.0, "reflect", 0.0, ni::Backend::Cpu));
  CHECK(ni::last_backend() == ni::Backend::Cpu);
  auto b = tov(ni::gaussian_filter(I, 1.0, 4.0, "reflect", 0.0, ni::Backend::Device));
  CHECK(ni::last_backend() == ni::Backend::Device);
  for (size_t i = 0; i < a.size(); ++i) CHECK_CLOSE(a[i], b[i], 1e-12, 1e-12);
}

TEST_CASE("morphology") {
  auto B = M(ni_B);
  cv(ni::binary_erosion(B), golden::ni_erosion_d, MN(ni_erosion));
  cv(ni::binary_dilation(B), golden::ni_dilation_d, MN(ni_dilation));
  cv(ni::binary_opening(B), golden::ni_opening_d, MN(ni_opening));
  cv(ni::grey_erosion(M(ni_I), 3), golden::ni_grey_erosion_d, MN(ni_grey_erosion));
  cv(ni::distance_transform_edt(B), golden::ni_edt_d, MN(ni_edt), 1e-9, 1e-11);
}

TEST_CASE("measurements") {
  auto L = ni::label(M(ni_Limg));
  CHECK(L.num_features == static_cast<int>(golden::ni_label_num));
  // labels match up to relabeling: compare label-equality structure
  auto got = tov(L.labels), exp = tov(M(ni_label));
  for (size_t i = 0; i < got.size(); ++i)
    for (size_t j = 0; j < got.size(); ++j)
      CHECK((got[i] == got[j]) == (exp[i] == exp[j]));
  cv(ni::center_of_mass(M(ni_I)), G(ni_com));
  CHECK_CLOSE(ni::sum_labels(M(ni_I)), golden::ni_sum, 1e-9, 1e-11);
}

TEST_CASE("geometric transforms") {
  auto I = M(ni_I);
  cv(ni::shift(I, {1.0, 0.5}, 1, "constant", 0.0), golden::ni_shift1_d, MN(ni_shift1), 1e-9, 1e-11);
  double amat[] = {1, 0.2, -0.1, 1}, aoff[] = {0.3, -0.2};
  cv(ni::affine_transform(I, mat(amat, 2, 2), vec(aoff, 2), 1, "constant", 0.0),
     golden::ni_affine1_d, MN(ni_affine1), 1e-9, 1e-11);
  cv(ni::zoom(I, 1.5, 1, "constant", 0.0), golden::ni_zoom1_d, MN(ni_zoom1), 1e-9, 1e-11);
  cv(ni::rotate(I, 30.0, 1, "constant", 0.0), golden::ni_rotate1_d, MN(ni_rotate1), 1e-9, 1e-11);
  cv(ni::map_coordinates(I, M(ni_coords), 1, "constant", 0.0), G(ni_mapcoord), 1e-9, 1e-11);
}
