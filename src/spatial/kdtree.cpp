// KD-tree: build (median split) + k-NN query + radius query.
#include "scipp/spatial/spatial.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::spatial {
namespace {
namespace sd = scipp::linalg::detail;
}

int KDTree::build(std::vector<int64_t>& ids, int lo, int hi, int depth) {
  if (lo >= hi) return -1;
  int axis = depth % static_cast<int>(d_);
  int mid = (lo + hi) / 2;
  std::nth_element(ids.begin() + lo, ids.begin() + mid, ids.begin() + hi,
                   [&](int64_t a, int64_t b) { return pts_[a * d_ + axis] < pts_[b * d_ + axis]; });
  int node = static_cast<int>(nodes_.size());
  nodes_.push_back({ids[mid], axis, -1, -1});
  int l = build(ids, lo, mid, depth + 1);
  int r = build(ids, mid + 1, hi, depth + 1);
  nodes_[node].left = l;
  nodes_[node].right = r;
  return node;
}

KDTree::KDTree(const ndarray& points) {
  numpp::ndarray P = points.astype(numpp::kFloat64).ascontiguousarray();
  n_ = P.shape()[0];
  d_ = P.shape()[1];
  const double* p = P.typed_data<double>();
  pts_.assign(p, p + n_ * d_);
  std::vector<int64_t> ids(n_);
  for (int64_t i = 0; i < n_; ++i) ids[i] = i;
  root_ = build(ids, 0, static_cast<int>(n_), 0);
}

namespace {
double dist2(const double* a, const double* b, int64_t d) {
  double s = 0; for (int64_t i = 0; i < d; ++i) { double e = a[i] - b[i]; s += e * e; } return s;
}
}  // namespace

KDTree::QueryResult KDTree::query(const ndarray& x, int k) const {
  numpp::ndarray X = x.astype(numpp::kFloat64).ascontiguousarray();
  bool single = (X.ndim() == 1);
  int64_t q = single ? 1 : X.shape()[0];
  const double* xp = X.typed_data<double>();

  std::vector<double> dout(q * k);
  std::vector<double> iout(q * k);
  for (int64_t qi = 0; qi < q; ++qi) {
    const double* xq = xp + qi * d_;
    // max-heap of (dist2, idx), size <= k
    std::priority_queue<std::pair<double, int64_t>> heap;
    std::function<void(int)> rec = [&](int node) {
      if (node < 0) return;
      const Node& nd = nodes_[node];
      double dd = dist2(xq, &pts_[nd.idx * d_], d_);
      if (static_cast<int>(heap.size()) < k) heap.push({dd, nd.idx});
      else if (dd < heap.top().first) { heap.pop(); heap.push({dd, nd.idx}); }
      double diff = xq[nd.axis] - pts_[nd.idx * d_ + nd.axis];
      int near = diff < 0 ? nd.left : nd.right;
      int far = diff < 0 ? nd.right : nd.left;
      rec(near);
      if (static_cast<int>(heap.size()) < k || diff * diff < heap.top().first) rec(far);
    };
    rec(root_);
    std::vector<std::pair<double, int64_t>> res;
    while (!heap.empty()) { res.push_back(heap.top()); heap.pop(); }
    std::sort(res.begin(), res.end());  // ascending distance
    for (int j = 0; j < k; ++j) { dout[qi * k + j] = std::sqrt(res[j].first); iout[qi * k + j] = static_cast<double>(res[j].second); }
  }
  ndarray dist = single ? sd::from_vec(dout) : sd::from_mat(dout, q, k);
  ndarray idx = single ? sd::from_vec(iout) : sd::from_mat(iout, q, k);
  return {dist, idx};
}

std::vector<int64_t> KDTree::query_ball_point(const ndarray& x, double r) const {
  std::vector<double> xq = sd::to_vec(x);
  double r2 = r * r;
  std::vector<int64_t> hits;
  std::function<void(int)> rec = [&](int node) {
    if (node < 0) return;
    const Node& nd = nodes_[node];
    if (dist2(xq.data(), &pts_[nd.idx * d_], d_) <= r2) hits.push_back(nd.idx);
    double diff = xq[nd.axis] - pts_[nd.idx * d_ + nd.axis];
    rec(diff < 0 ? nd.left : nd.right);
    if (diff * diff <= r2) rec(diff < 0 ? nd.right : nd.left);
  };
  rec(root_);
  std::sort(hits.begin(), hits.end());
  return hits;
}

}  // namespace scipp::spatial
