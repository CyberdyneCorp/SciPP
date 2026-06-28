// Oracle tests for hypergeometric functions: hyp0f1, hyp1f1, hyp2f1, hyperu.
#include "golden.hpp"
#include "scipp/special/special.hpp"
#include "scipp_test.hpp"

namespace sp = scipp::special;

namespace {
constexpr double R = 1e-9, A = 1e-11;
}  // namespace

TEST_CASE("hyp0f1 confluent limit") {
  for (int i = 0; i < golden::h0f1_x_n; ++i) {
    CHECK_CLOSE(sp::hyp0f1(2.0, golden::h0f1_x[i]), golden::h0f1_b2[i], R, A);
    CHECK_CLOSE(sp::hyp0f1(0.5, golden::h0f1_x[i]), golden::h0f1_bhalf[i], R, A);
  }
  // 0F1(;b;0) = 1; non-positive integer b is a pole -> nan.
  CHECK_CLOSE(sp::hyp0f1(3.0, 0.0), 1.0, R, A);
  CHECK(std::isnan(sp::hyp0f1(-1.0, 1.0)));
}

TEST_CASE("hyp1f1 Kummer confluent") {
  for (int i = 0; i < golden::h1f1_x_n; ++i) {
    CHECK_CLOSE(sp::hyp1f1(1.5, 2.5, golden::h1f1_x[i]), golden::h1f1_a[i], R, A);
    CHECK_CLOSE(sp::hyp1f1(-2.0, 3.0, golden::h1f1_x[i]), golden::h1f1_b[i], R, A);
    CHECK_CLOSE(sp::hyp1f1(0.5, 1.3, golden::h1f1_x[i]), golden::h1f1_c[i], R, A);
  }
  // M(a,b,0) = 1; non-positive integer b -> nan.
  CHECK_CLOSE(sp::hyp1f1(2.0, 4.0, 0.0), 1.0, R, A);
  CHECK(std::isnan(sp::hyp1f1(1.0, -2.0, 0.5)));
}

TEST_CASE("hyp2f1 Gauss") {
  for (int i = 0; i < golden::h2f1_z_n; ++i) {
    CHECK_CLOSE(sp::hyp2f1(0.5, 1.0, 1.5, golden::h2f1_z[i]), golden::h2f1_a[i], R, A);
    CHECK_CLOSE(sp::hyp2f1(1.0, 2.0, 3.5, golden::h2f1_z[i]), golden::h2f1_b[i], R, A);
    CHECK_CLOSE(sp::hyp2f1(-1.5, 0.7, 2.2, golden::h2f1_z[i]), golden::h2f1_c[i], R, A);
  }
  // 2F1(a,b;c;0) = 1; Gauss summation theorem at z = 1.
  CHECK_CLOSE(sp::hyp2f1(0.3, 1.2, 2.0, 0.0), 1.0, R, A);
  CHECK_CLOSE(sp::hyp2f1(0.5, 1.0, 3.0, 1.0), golden::h2f1_one, R, A);
  // c a non-positive integer -> nan.
  CHECK(std::isnan(sp::hyp2f1(1.0, 1.0, -1.0, 0.3)));
}

TEST_CASE("hyperu Tricomi") {
  // The connection formula subtracts two 1F1 solutions, so a handful of points
  // lose a couple of digits to cancellation; 1e-8 stays inside the band.
  constexpr double RU = 1e-8, AU = 1e-11;
  for (int i = 0; i < golden::hu_x_n; ++i) {
    CHECK_CLOSE(sp::hyperu(0.5, 1.3, golden::hu_x[i]), golden::hu_a[i], RU, AU);
    CHECK_CLOSE(sp::hyperu(2.0, 0.4, golden::hu_x[i]), golden::hu_b[i], RU, AU);
    CHECK_CLOSE(sp::hyperu(1.5, 2.7, golden::hu_x[i]), golden::hu_c[i], RU, AU);
  }
  // x <= 0 outside the delivered region -> nan.
  CHECK(std::isnan(sp::hyperu(0.5, 1.3, 0.0)));
  CHECK(std::isnan(sp::hyperu(0.5, 1.3, -1.0)));
}
