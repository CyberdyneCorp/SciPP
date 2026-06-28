#pragma once
// Internal helpers for scipp::sparse: int64 / double array <-> std::vector.

#include <cstdint>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::sparse::detail {

namespace ld = scipp::linalg::detail;

inline std::vector<int64_t> iv(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kInt64).ascontiguousarray();
  const int64_t* p = c.typed_data<int64_t>();
  return std::vector<int64_t>(p, p + c.size());
}
inline numpp::ndarray from_iv(const std::vector<int64_t>& v) {
  numpp::ndarray a(numpp::Shape{static_cast<int64_t>(v.size())}, numpp::kInt64);
  int64_t* p = a.typed_data<int64_t>();
  for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
  return a;
}
inline std::vector<double> dv(const numpp::ndarray& a) { return ld::to_vec(a); }
inline numpp::ndarray from_dv(const std::vector<double>& v) { return ld::from_vec(v); }

}  // namespace scipp::sparse::detail
