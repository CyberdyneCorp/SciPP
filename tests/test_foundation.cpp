// Foundation regression tests: the scipp exception model is aligned with
// numpp::error so a single catch site spans both layers.
#include <stdexcept>

#include "numpp/core/error.hpp"
#include "scipp/error.hpp"
#include "scipp_test.hpp"

TEST_CASE("error hierarchy is aligned with numpp::error") {
  // Each SciPP error is catchable as scipp::error AND numpp::error.
  CHECK_THROWS_AS(throw scipp::value_error("bad arg"), scipp::error);
  CHECK_THROWS_AS(throw scipp::value_error("bad arg"), numpp::error);
  CHECK_THROWS_AS(throw scipp::linalg_error("singular"), scipp::error);
  CHECK_THROWS_AS(throw scipp::linalg_error("singular"), numpp::error);
  CHECK_THROWS_AS(throw scipp::not_implemented_error("todo"), scipp::error);
  CHECK_THROWS_AS(throw scipp::not_implemented_error("todo"), numpp::error);

  // The shared base also spans NumPP's own linalg failures.
  CHECK_THROWS_AS(throw numpp::linalg_error("from numpp"), numpp::error);

  // Distinct types: a value_error is not a linalg_error.
  bool distinct = false;
  try {
    throw scipp::value_error("v");
  } catch (const scipp::linalg_error&) {
    distinct = false;
  } catch (const scipp::error&) {
    distinct = true;
  }
  CHECK(distinct);
}
