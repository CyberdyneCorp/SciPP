// Foundation regression tests: the scypp exception model is aligned with
// numpp::error so a single catch site spans both layers.
#include <stdexcept>

#include "numpp/core/error.hpp"
#include "scypp/error.hpp"
#include "scypp_test.hpp"

TEST_CASE("error hierarchy is aligned with numpp::error") {
  // Each ScyPP error is catchable as scypp::error AND numpp::error.
  CHECK_THROWS_AS(throw scypp::value_error("bad arg"), scypp::error);
  CHECK_THROWS_AS(throw scypp::value_error("bad arg"), numpp::error);
  CHECK_THROWS_AS(throw scypp::linalg_error("singular"), scypp::error);
  CHECK_THROWS_AS(throw scypp::linalg_error("singular"), numpp::error);
  CHECK_THROWS_AS(throw scypp::not_implemented_error("todo"), scypp::error);
  CHECK_THROWS_AS(throw scypp::not_implemented_error("todo"), numpp::error);

  // The shared base also spans NumPP's own linalg failures.
  CHECK_THROWS_AS(throw numpp::linalg_error("from numpp"), numpp::error);

  // Distinct types: a value_error is not a linalg_error.
  bool distinct = false;
  try {
    throw scypp::value_error("v");
  } catch (const scypp::linalg_error&) {
    distinct = false;
  } catch (const scypp::error&) {
    distinct = true;
  }
  CHECK(distinct);
}
