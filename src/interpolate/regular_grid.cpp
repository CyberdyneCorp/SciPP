// N-D rectilinear-grid interpolation (linear / nearest) and interpn.
#include "scipp/interpolate/interpolate.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::interpolate {
namespace {
namespace sd = scipp::linalg::detail;
}

RegularGridInterpolator::RegularGridInterpolator(std::vector<ndarray> points, const ndarray& values,
                                                 std::string method)
    : method_(std::move(method)) {
  for (auto& p : points) grids_.push_back(sd::to_vec(p));
  numpp::ndarray v = values.astype(numpp::kFloat64).ascontiguousarray();
  shape_.assign(v.shape().begin(), v.shape().end());
  const double* vp = v.typed_data<double>();
  values_.assign(vp, vp + v.size());
  if (shape_.size() != grids_.size())
    throw scipp::value_error("RegularGridInterpolator: values rank must match number of axes");
}

ndarray RegularGridInterpolator::operator()(const ndarray& xi) const {
  numpp::ndarray q = xi.astype(numpp::kFloat64).ascontiguousarray();
  int ndim = static_cast<int>(grids_.size());
  int64_t m = (q.ndim() == 1) ? 1 : q.shape()[0];
  const double* qp = q.typed_data<double>();

  std::vector<int64_t> stride(ndim, 1);  // C-order strides into values_
  for (int d = ndim - 2; d >= 0; --d) stride[d] = stride[d + 1] * shape_[d + 1];

  std::vector<double> out(m);
  std::vector<int> idx(ndim);
  std::vector<double> t(ndim);
  for (int64_t r = 0; r < m; ++r) {
    const double* x = qp + r * ndim;
    for (int d = 0; d < ndim; ++d) {
      const auto& g = grids_[d];
      int n = static_cast<int>(g.size());
      int i = static_cast<int>(std::upper_bound(g.begin(), g.end(), x[d]) - g.begin()) - 1;
      i = std::clamp(i, 0, n - 2);
      idx[d] = i;
      t[d] = (x[d] - g[i]) / (g[i + 1] - g[i]);
    }
    if (method_ == "nearest") {
      int64_t flat = 0;
      for (int d = 0; d < ndim; ++d) flat += (idx[d] + (t[d] > 0.5 ? 1 : 0)) * stride[d];
      out[r] = values_[flat];
    } else {  // multilinear: blend the 2^ndim corners
      double acc = 0.0;
      for (int corner = 0; corner < (1 << ndim); ++corner) {
        double w = 1.0;
        int64_t flat = 0;
        for (int d = 0; d < ndim; ++d) {
          int bit = (corner >> d) & 1;
          w *= bit ? t[d] : (1.0 - t[d]);
          flat += (idx[d] + bit) * stride[d];
        }
        acc += w * values_[flat];
      }
      out[r] = acc;
    }
  }
  return sd::from_vec(out);
}

ndarray interpn(std::vector<ndarray> points, const ndarray& values, const ndarray& xi,
                std::string method) {
  return RegularGridInterpolator(std::move(points), values, std::move(method))(xi);
}

}  // namespace scipp::interpolate
