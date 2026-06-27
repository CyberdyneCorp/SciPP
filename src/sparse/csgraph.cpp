// Compressed-sparse graph algorithms: shortest paths, components, MST.
#include "scypp/sparse/sparse.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <functional>
#include <vector>

#include "scypp/sparse/detail.hpp"

namespace scypp::sparse::csgraph {
namespace d = scypp::sparse::detail;
namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Edge { int64_t to; double w; };
std::vector<std::vector<Edge>> adjacency(const CsrMatrix& g, bool directed) {
  auto ip = d::iv(g.indptr()), id = d::iv(g.indices());
  auto da = d::dv(g.data());
  int64_t n = g.rows();
  std::vector<std::vector<Edge>> adj(n);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t k = ip[i]; k < ip[i + 1]; ++k) {
      adj[i].push_back({id[k], da[k]});
      if (!directed) adj[id[k]].push_back({i, da[k]});
    }
  return adj;
}
ndarray dist_matrix(std::vector<double>& D, int64_t n) { return d::ld::from_mat(D, n, n); }
}  // namespace

ndarray dijkstra(const CsrMatrix& graph, bool directed) {
  auto adj = adjacency(graph, directed);
  int64_t n = graph.rows();
  std::vector<double> D(n * n, kInf);
  for (int64_t s = 0; s < n; ++s) {
    std::vector<double> dist(n, kInf);
    dist[s] = 0;
    std::priority_queue<std::pair<double, int64_t>, std::vector<std::pair<double, int64_t>>,
                        std::greater<>> pq;
    pq.push({0.0, s});
    while (!pq.empty()) {
      auto [du, u] = pq.top(); pq.pop();
      if (du > dist[u]) continue;
      for (const Edge& e : adj[u]) if (du + e.w < dist[e.to]) { dist[e.to] = du + e.w; pq.push({dist[e.to], e.to}); }
    }
    for (int64_t j = 0; j < n; ++j) D[s * n + j] = dist[j];
  }
  return dist_matrix(D, n);
}

ndarray bellman_ford(const CsrMatrix& graph, bool directed) {
  auto adj = adjacency(graph, directed);
  int64_t n = graph.rows();
  std::vector<double> D(n * n, kInf);
  for (int64_t s = 0; s < n; ++s) {
    std::vector<double> dist(n, kInf);
    dist[s] = 0;
    for (int64_t it = 0; it < n - 1; ++it)
      for (int64_t u = 0; u < n; ++u)
        if (dist[u] != kInf)
          for (const Edge& e : adj[u]) if (dist[u] + e.w < dist[e.to]) dist[e.to] = dist[u] + e.w;
    for (int64_t j = 0; j < n; ++j) D[s * n + j] = dist[j];
  }
  return dist_matrix(D, n);
}

ndarray floyd_warshall(const CsrMatrix& graph, bool directed) {
  auto adj = adjacency(graph, directed);
  int64_t n = graph.rows();
  std::vector<double> D(n * n, kInf);
  for (int64_t i = 0; i < n; ++i) D[i * n + i] = 0;
  for (int64_t i = 0; i < n; ++i)
    for (const Edge& e : adj[i]) D[i * n + e.to] = std::min(D[i * n + e.to], e.w);
  for (int64_t k = 0; k < n; ++k)
    for (int64_t i = 0; i < n; ++i)
      if (D[i * n + k] != kInf)
        for (int64_t j = 0; j < n; ++j)
          if (D[k * n + j] != kInf && D[i * n + k] + D[k * n + j] < D[i * n + j])
            D[i * n + j] = D[i * n + k] + D[k * n + j];
  return dist_matrix(D, n);
}

ComponentsResult connected_components(const CsrMatrix& graph, bool directed,
                                      const std::string&) {
  auto adj = adjacency(graph, /*directed=*/false);  // weak connectivity (undirected)
  (void)directed;
  int64_t n = graph.rows();
  std::vector<int> parent(n, -1);
  std::vector<int64_t> labels(n, -1);
  int ncomp = 0;
  for (int64_t s = 0; s < n; ++s) {
    if (labels[s] != -1) continue;
    std::vector<int64_t> stack{s};
    labels[s] = ncomp;
    while (!stack.empty()) {
      int64_t u = stack.back(); stack.pop_back();
      for (const Edge& e : adj[u]) if (labels[e.to] == -1) { labels[e.to] = ncomp; stack.push_back(e.to); }
    }
    ++ncomp;
  }
  std::vector<double> lab(labels.begin(), labels.end());
  return {ncomp, d::from_dv(lab)};
}

CsrMatrix minimum_spanning_tree(const CsrMatrix& graph) {
  auto ip = d::iv(graph.indptr()), id = d::iv(graph.indices());
  auto da = d::dv(graph.data());
  int64_t n = graph.rows();
  struct E { double w; int64_t u, v; };
  std::vector<E> edges;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t k = ip[i]; k < ip[i + 1]; ++k) edges.push_back({da[k], i, id[k]});
  std::sort(edges.begin(), edges.end(), [](const E& a, const E& b) { return a.w < b.w; });
  std::vector<int64_t> uf(n); for (int64_t i = 0; i < n; ++i) uf[i] = i;
  std::function<int64_t(int64_t)> find = [&](int64_t x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; };
  std::vector<int64_t> row, col; std::vector<double> val;
  for (const E& e : edges) {
    int64_t ru = find(e.u), rv = find(e.v);
    if (ru != rv) { uf[ru] = rv; int64_t a = std::min(e.u, e.v), b = std::max(e.u, e.v); row.push_back(a); col.push_back(b); val.push_back(e.w); }
  }
  return CsrMatrix::from_coo({d::from_dv(val), d::from_iv(row), d::from_iv(col), n, n});
}

}  // namespace scypp::sparse::csgraph
