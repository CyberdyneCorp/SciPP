// Vector quantization: whiten, vq, kmeans2 (explicit-init Lloyd iteration).
#include "scypp/cluster/cluster.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::cluster {
namespace {
namespace sd = scypp::linalg::detail;
struct Mat { std::vector<double> d; int64_t r, c; };
Mat to_mat(const ndarray& a) {
  numpp::ndarray ac = a.astype(numpp::kFloat64).ascontiguousarray();
  int64_t r = ac.shape()[0], c = ac.ndim() == 1 ? 1 : ac.shape()[1];
  const double* p = ac.typed_data<double>();
  return {std::vector<double>(p, p + r * c), r, c};
}
}  // namespace

ndarray whiten(const ndarray& obs) {
  Mat m = to_mat(obs);
  std::vector<double> std_(m.c, 0.0), mean(m.c, 0.0);
  for (int64_t j = 0; j < m.c; ++j) {
    for (int64_t i = 0; i < m.r; ++i) mean[j] += m.d[i * m.c + j];
    mean[j] /= m.r;
    for (int64_t i = 0; i < m.r; ++i) { double e = m.d[i * m.c + j] - mean[j]; std_[j] += e * e; }
    std_[j] = std::sqrt(std_[j] / m.r);
  }
  std::vector<double> out(m.r * m.c);
  for (int64_t i = 0; i < m.r; ++i)
    for (int64_t j = 0; j < m.c; ++j) out[i * m.c + j] = m.d[i * m.c + j] / std_[j];
  return sd::from_mat(out, m.r, m.c);
}

namespace {
void assign(const Mat& obs, const Mat& cb, std::vector<int>& code, std::vector<double>& dist) {
  code.assign(obs.r, 0); dist.assign(obs.r, 0.0);
  for (int64_t i = 0; i < obs.r; ++i) {
    double best = std::numeric_limits<double>::infinity(); int bk = 0;
    for (int64_t k = 0; k < cb.r; ++k) {
      double s = 0; for (int64_t j = 0; j < obs.c; ++j) { double e = obs.d[i * obs.c + j] - cb.d[k * cb.c + j]; s += e * e; }
      if (s < best) { best = s; bk = static_cast<int>(k); }
    }
    code[i] = bk; dist[i] = std::sqrt(best);
  }
}
}  // namespace

VQResult vq(const ndarray& obs, const ndarray& code_book) {
  Mat o = to_mat(obs), cb = to_mat(code_book);
  std::vector<int> code; std::vector<double> dist;
  assign(o, cb, code, dist);
  std::vector<double> codef(code.begin(), code.end());
  return {sd::from_vec(codef), sd::from_vec(dist)};
}

KMeans2Result kmeans2(const ndarray& data, const ndarray& init, int iter) {
  Mat o = to_mat(data), cb = to_mat(init);
  std::vector<int> code; std::vector<double> dist;
  for (int it = 0; it < iter; ++it) {
    assign(o, cb, code, dist);
    std::vector<double> sum(cb.r * cb.c, 0.0); std::vector<int> cnt(cb.r, 0);
    for (int64_t i = 0; i < o.r; ++i) { int k = code[i]; ++cnt[k]; for (int64_t j = 0; j < o.c; ++j) sum[k * cb.c + j] += o.d[i * o.c + j]; }
    for (int64_t k = 0; k < cb.r; ++k)
      if (cnt[k] > 0) for (int64_t j = 0; j < cb.c; ++j) cb.d[k * cb.c + j] = sum[k * cb.c + j] / cnt[k];
  }
  assign(o, cb, code, dist);
  std::vector<double> labelf(code.begin(), code.end());
  return {sd::from_mat(cb.d, cb.r, cb.c), sd::from_vec(labelf)};
}

}  // namespace scypp::cluster
