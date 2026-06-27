// Hierarchical clustering: linkage (Lance-Williams), fcluster, cophenet.
#include "scypp/cluster/cluster.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::cluster {
namespace {
namespace sd = scypp::linalg::detail;
constexpr double kInf = std::numeric_limits<double>::infinity();

// Build the n×n distance matrix from observations or a condensed vector.
std::vector<double> distances(const ndarray& y, int64_t& n) {
  numpp::ndarray Y = y.astype(numpp::kFloat64).ascontiguousarray();
  if (Y.ndim() == 2) {  // observations -> euclidean pdist
    int64_t r = Y.shape()[0], c = Y.shape()[1];
    const double* p = Y.typed_data<double>();
    n = r;
    std::vector<double> D(r * r, 0.0);
    for (int64_t i = 0; i < r; ++i)
      for (int64_t j = i + 1; j < r; ++j) {
        double s = 0; for (int64_t k = 0; k < c; ++k) { double e = p[i * c + k] - p[j * c + k]; s += e * e; }
        D[i * r + j] = D[j * r + i] = std::sqrt(s);
      }
    return D;
  }
  std::vector<double> cond = sd::to_vec(Y);  // condensed
  int64_t k = cond.size();
  n = static_cast<int64_t>((1 + std::sqrt(1.0 + 8.0 * k)) / 2);
  std::vector<double> D(n * n, 0.0);
  int64_t idx = 0;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = i + 1; j < n; ++j) { D[i * n + j] = D[j * n + i] = cond[idx++]; }
  return D;
}
}  // namespace

ndarray linkage(const ndarray& y, const std::string& method, const std::string&) {
  int64_t n;
  std::vector<double> D = distances(y, n);
  std::vector<int> size(n, 1), id(n), active(n, 1);
  for (int64_t i = 0; i < n; ++i) id[i] = static_cast<int>(i);
  std::vector<double> Z;  // (n-1) x 4
  for (int64_t step = 0; step < n - 1; ++step) {
    double dmin = kInf; int a = -1, b = -1;
    for (int64_t i = 0; i < n; ++i) if (active[i])
      for (int64_t j = i + 1; j < n; ++j) if (active[j])
        if (D[i * n + j] < dmin) { dmin = D[i * n + j]; a = static_cast<int>(i); b = static_cast<int>(j); }
    int ia = id[a], ib = id[b];
    Z.push_back(std::min(ia, ib)); Z.push_back(std::max(ia, ib)); Z.push_back(dmin);
    Z.push_back(size[a] + size[b]);
    for (int64_t k = 0; k < n; ++k) {
      if (!active[k] || k == a || k == b) continue;
      double dak = D[a * n + k], dbk = D[b * n + k], nd;
      if (method == "single") nd = std::min(dak, dbk);
      else if (method == "complete") nd = std::max(dak, dbk);
      else if (method == "average") nd = (size[a] * dak + size[b] * dbk) / (size[a] + size[b]);
      else if (method == "ward") {
        double T = size[a] + size[b] + size[k];
        nd = std::sqrt(((size[a] + size[k]) * dak * dak + (size[b] + size[k]) * dbk * dbk -
                        size[k] * dmin * dmin) / T);
      } else nd = std::min(dak, dbk);
      D[a * n + k] = D[k * n + a] = nd;
    }
    active[b] = 0; size[a] += size[b]; id[a] = static_cast<int>(n + step);
  }
  return sd::from_mat(Z, n - 1, 4);
}

namespace {
// Members of each cluster id (0..n-1 singletons, n.. merges).
std::vector<std::vector<int>> cluster_members(const std::vector<double>& Z, int64_t n) {
  std::vector<std::vector<int>> mem(2 * n - 1);
  for (int64_t i = 0; i < n; ++i) mem[i] = {static_cast<int>(i)};
  for (int64_t s = 0; s < n - 1; ++s) {
    int c1 = static_cast<int>(Z[s * 4]), c2 = static_cast<int>(Z[s * 4 + 1]);
    auto& dst = mem[n + s];
    dst = mem[c1];
    dst.insert(dst.end(), mem[c2].begin(), mem[c2].end());
  }
  return mem;
}
}  // namespace

ndarray cophenet(const ndarray& Zin) {
  std::vector<double> Z = sd::to_vec(Zin);
  int64_t n = Z.size() / 4 + 1;
  auto mem = cluster_members(Z, n);
  std::vector<double> coph(n * n, 0.0);
  for (int64_t s = 0; s < n - 1; ++s) {
    int c1 = static_cast<int>(Z[s * 4]), c2 = static_cast<int>(Z[s * 4 + 1]);
    double d = Z[s * 4 + 2];
    for (int x : mem[c1]) for (int y : mem[c2]) { coph[x * n + y] = d; coph[y * n + x] = d; }
  }
  std::vector<double> out;
  for (int64_t i = 0; i < n; ++i) for (int64_t j = i + 1; j < n; ++j) out.push_back(coph[i * n + j]);
  return sd::from_vec(out);
}

ndarray fcluster(const ndarray& Zin, double t, const std::string& criterion) {
  std::vector<double> Z = sd::to_vec(Zin);
  int64_t n = Z.size() / 4 + 1;
  std::vector<int> uf(2 * n - 1);
  for (size_t i = 0; i < uf.size(); ++i) uf[i] = static_cast<int>(i);
  std::function<int(int)> find = [&](int x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; };
  int64_t merges = (criterion == "maxclust") ? (n - static_cast<int64_t>(t)) : n - 1;
  for (int64_t s = 0; s < n - 1; ++s) {
    if (criterion == "distance" && Z[s * 4 + 2] > t) continue;
    if (criterion == "maxclust" && s >= merges) break;
    int c1 = static_cast<int>(Z[s * 4]), c2 = static_cast<int>(Z[s * 4 + 1]);
    uf[find(c1)] = find(static_cast<int>(n + s));
    uf[find(c2)] = find(static_cast<int>(n + s));
  }
  // assign labels 1..k by first appearance over observations 0..n-1
  std::vector<double> lab(n);
  std::vector<int> root_to_label(2 * n - 1, 0);
  int next = 0;
  for (int64_t i = 0; i < n; ++i) {
    int r = find(static_cast<int>(i));
    if (root_to_label[r] == 0) root_to_label[r] = ++next;
    lab[i] = root_to_label[r];
  }
  return sd::from_vec(lab);
}

}  // namespace scypp::cluster
