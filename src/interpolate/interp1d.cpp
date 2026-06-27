// 1-D interpolation (linear / nearest / previous / next).
#include "scypp/interpolate/interpolate.hpp"

#include <algorithm>
#include <cmath>

#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::interpolate {
namespace {
namespace sd = scypp::linalg::detail;
}

Interp1d::Interp1d(const ndarray& x, const ndarray& y, std::string kind,
                   std::optional<double> fill_value)
    : x_(sd::to_vec(x)), y_(sd::to_vec(y)), kind_(std::move(kind)), fill_(fill_value) {
  if (x_.size() != y_.size() || x_.size() < 2)
    throw scypp::value_error("Interp1d: x and y must have equal length >= 2");
}

double Interp1d::operator()(double xq) const {
  int n = static_cast<int>(x_.size());
  if (xq < x_[0] || xq > x_[n - 1]) {
    if (fill_) return *fill_;
    throw scypp::value_error("Interp1d: query out of bounds");
  }
  int i = static_cast<int>(std::upper_bound(x_.begin(), x_.end(), xq) - x_.begin()) - 1;
  i = std::clamp(i, 0, n - 2);
  if (kind_ == "linear") {
    double t = (xq - x_[i]) / (x_[i + 1] - x_[i]);
    return y_[i] + t * (y_[i + 1] - y_[i]);
  }
  if (kind_ == "nearest") {
    return (xq - x_[i] <= x_[i + 1] - xq) ? y_[i] : y_[i + 1];
  }
  if (kind_ == "previous") return y_[i];
  if (kind_ == "next") return (xq == x_[i]) ? y_[i] : y_[i + 1];
  throw scypp::value_error("Interp1d: unknown kind " + kind_);
}

ndarray Interp1d::operator()(const ndarray& xq) const {
  std::vector<double> q = sd::to_vec(xq);
  std::vector<double> out(q.size());
  for (size_t i = 0; i < q.size(); ++i) out[i] = (*this)(q[i]);
  return sd::from_vec(out);
}

}  // namespace scypp::interpolate
