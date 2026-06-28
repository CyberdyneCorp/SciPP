#pragma once
// SciPP exception model, aligned with numpp::error so a single catch site spans
// both layers. SciPy's special functions do NOT throw on domain edges (they
// return nan/inf); these exceptions are reserved for genuine misuse such as an
// unknown constant name or a shape mismatch surfaced from NumPP.

#include <stdexcept>
#include <string>

#include "numpp/core/error.hpp"

namespace scipp {

// Base SciPP error. Derives from numpp::error so `catch (const numpp::error&)`
// handles both NumPP and SciPP failures.
struct error : numpp::error {
  explicit error(const std::string& what) : numpp::error(what) {}
};

// Invalid value / argument (mirrors SciPy ValueError, e.g. unknown CODATA key).
struct value_error : error {
  explicit value_error(const std::string& what) : error(what) {}
};

// Numerical-linear-algebra failure (mirrors scipy.linalg.LinAlgError and
// numpp::linalg_error — e.g. a singular system surfaced from a SciPP routine).
// A `catch (const numpp::error&)` site still spans NumPP's own linalg_error.
struct linalg_error : error {
  explicit linalg_error(const std::string& what) : error(what) {}
};

// Requested behavior is not implemented in this build/phase.
struct not_implemented_error : error {
  explicit not_implemented_error(const std::string& what) : error(what) {}
};

}  // namespace scipp
