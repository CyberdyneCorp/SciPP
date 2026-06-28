#pragma once
// Internal helpers for scipp::linalg: convert between numpp::ndarray and plain
// row-major double buffers for the hand-written kernels (LU, special matrices).

#include <cstdint>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/error.hpp"

namespace scipp::linalg::detail {

// Copy a 2-D array into a row-major float64 buffer; outputs its dimensions.
inline std::vector<double> to_mat(const numpp::ndarray& a, int64_t& rows, int64_t& cols) {
  if (a.ndim() != 2) throw scipp::value_error("expected a 2-D matrix");
  numpp::ndarray ac = a.astype(numpp::kFloat64).ascontiguousarray();
  rows = ac.shape()[0];
  cols = ac.shape()[1];
  const double* p = ac.typed_data<double>();
  return std::vector<double>(p, p + rows * cols);
}

// Copy a 1-D array into a float64 buffer.
inline std::vector<double> to_vec(const numpp::ndarray& a) {
  numpp::ndarray ac = a.astype(numpp::kFloat64).ascontiguousarray();
  const double* p = ac.typed_data<double>();
  return std::vector<double>(p, p + ac.size());
}

// Build an (rows x cols) float64 matrix from a row-major buffer.
inline numpp::ndarray from_mat(const std::vector<double>& d, int64_t rows, int64_t cols) {
  numpp::ndarray out(numpp::Shape{rows, cols}, numpp::kFloat64);
  double* o = out.typed_data<double>();
  for (int64_t i = 0; i < rows * cols; ++i) o[i] = d[i];
  return out;
}

inline numpp::ndarray from_vec(const std::vector<double>& d) {
  numpp::ndarray out(numpp::Shape{static_cast<int64_t>(d.size())}, numpp::kFloat64);
  double* o = out.typed_data<double>();
  for (size_t i = 0; i < d.size(); ++i) o[i] = d[i];
  return out;
}

}  // namespace scipp::linalg::detail
