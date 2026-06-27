#pragma once
// ScyPP exception model, aligned with numpp::error so a single catch site spans
// both layers. SciPy's special functions do NOT throw on domain edges (they
// return nan/inf); these exceptions are reserved for genuine misuse such as an
// unknown constant name or a shape mismatch surfaced from NumPP.

#include <stdexcept>
#include <string>

#include "numpp/core/error.hpp"

namespace scypp {

// Base ScyPP error. Derives from numpp::error so `catch (const numpp::error&)`
// handles both NumPP and ScyPP failures.
struct error : numpp::error {
  explicit error(const std::string& what) : numpp::error(what) {}
};

// Invalid value / argument (mirrors SciPy ValueError, e.g. unknown CODATA key).
struct value_error : error {
  explicit value_error(const std::string& what) : error(what) {}
};

// Requested behavior is not implemented in this build/phase.
struct not_implemented_error : error {
  explicit not_implemented_error(const std::string& what) : error(what) {}
};

}  // namespace scypp
