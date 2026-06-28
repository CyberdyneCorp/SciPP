// Gaussian kernel density estimation (Scott/Silverman bandwidth).
#include "scipp/stats/stats.hpp"

#include <cmath>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/linalg/linalg.hpp"
#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::stats {
namespace {
namespace sd = scipp::linalg::detail;
constexpr double kTwoPi = 6.283185307179586;
}  // namespace

gaussian_kde::gaussian_kde(const ndarray& dataset, const std::string& bw_method) {
  numpp::ndarray D = dataset.astype(numpp::kFloat64).ascontiguousarray();
  if (D.ndim() == 1) { d_ = 1; n_ = static_cast<int>(D.size()); }
  else { d_ = static_cast<int>(D.shape()[0]); n_ = static_cast<int>(D.shape()[1]); }
  const double* dp = D.typed_data<double>();
  data_.assign(dp, dp + static_cast<size_t>(d_) * n_);

  if (bw_method == "scott") factor_ = std::pow(n_, -1.0 / (d_ + 4));
  else if (bw_method == "silverman") factor_ = std::pow(n_ * (d_ + 2.0) / 4.0, -1.0 / (d_ + 4));
  else throw scipp::value_error("gaussian_kde: unknown bw_method " + bw_method);

  std::vector<double> mean(d_, 0.0);
  for (int i = 0; i < d_; ++i) { for (int k = 0; k < n_; ++k) mean[i] += data_[i * n_ + k]; mean[i] /= n_; }
  std::vector<double> cov(static_cast<size_t>(d_) * d_, 0.0);
  for (int i = 0; i < d_; ++i)
    for (int j = 0; j < d_; ++j) {
      double s = 0;
      for (int k = 0; k < n_; ++k) s += (data_[i * n_ + k] - mean[i]) * (data_[j * n_ + k] - mean[j]);
      cov[i * d_ + j] = s / (n_ - 1) * factor_ * factor_;  // data covariance * factor^2
    }

  if (d_ == 1) {
    inv_cov_ = {1.0 / cov[0]};
    norm_ = std::sqrt(kTwoPi * cov[0]);
  } else {
    inv_cov_ = sd::to_vec(numpp::linalg::inv(sd::from_mat(cov, d_, d_)));
    double det = std::fabs(numpp::linalg::det(sd::from_mat(cov, d_, d_)).astype(numpp::kFloat64)
                               .ascontiguousarray().typed_data<double>()[0]);
    norm_ = std::pow(kTwoPi, d_ / 2.0) * std::sqrt(det);
  }
}

ndarray gaussian_kde::evaluate(const ndarray& points) const {
  numpp::ndarray P = points.astype(numpp::kFloat64).ascontiguousarray();
  int64_t m = (d_ == 1) ? P.size() : P.shape()[1];
  const double* pp = P.typed_data<double>();
  std::vector<double> out(m, 0.0);
  std::vector<double> diff(d_);
  for (int64_t j = 0; j < m; ++j) {
    double acc = 0.0;
    for (int k = 0; k < n_; ++k) {
      for (int i = 0; i < d_; ++i) {
        double pj = (d_ == 1) ? pp[j] : pp[i * m + j];
        diff[i] = pj - data_[i * n_ + k];
      }
      double energy = 0.0;  // diff^T inv_cov diff
      for (int i = 0; i < d_; ++i)
        for (int l = 0; l < d_; ++l) energy += diff[i] * inv_cov_[i * d_ + l] * diff[l];
      acc += std::exp(-0.5 * energy);
    }
    out[j] = acc / (n_ * norm_);
  }
  return sd::from_vec(out);
}

ndarray gaussian_kde::operator()(const ndarray& points) const { return evaluate(points); }

}  // namespace scipp::stats
