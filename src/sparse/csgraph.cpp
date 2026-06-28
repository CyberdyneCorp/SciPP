// Compressed-sparse graph algorithms: shortest paths, components, MST.
#include "scipp/sparse/sparse.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <functional>
#include <vector>

#include "scipp/sparse/detail.hpp"

namespace scipp::sparse::csgraph {
namespace d = scipp::sparse::detail;
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

namespace {
// Neighbour lists in CSR column order. For undirected graphs each node lists
// its out-neighbours (CSR order) followed by its in-neighbours (ascending
// source order), matching SciPy's symmetrised traversal convention.
std::vector<std::vector<int64_t>> traversal_adj(const CsrMatrix& g, bool directed) {
  auto ip = d::iv(g.indptr()), id = d::iv(g.indices());
  int64_t n = g.rows();
  std::vector<std::vector<int64_t>> adj(n);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t k = ip[i]; k < ip[i + 1]; ++k) adj[i].push_back(id[k]);
  if (!directed)
    for (int64_t i = 0; i < n; ++i)
      for (int64_t k = ip[i]; k < ip[i + 1]; ++k) adj[id[k]].push_back(i);
  return adj;
}
}  // namespace

TraversalResult breadth_first_order(const CsrMatrix& graph, int64_t i_start, bool directed) {
  auto adj = traversal_adj(graph, directed);
  int64_t n = graph.rows();
  std::vector<char> seen(n, 0);
  std::vector<double> order, pred(n, -9999.0);
  std::queue<int64_t> q;
  q.push(i_start); seen[i_start] = 1;
  while (!q.empty()) {
    int64_t u = q.front(); q.pop();
    order.push_back(static_cast<double>(u));
    for (int64_t v : adj[u])
      if (!seen[v]) { seen[v] = 1; pred[v] = static_cast<double>(u); q.push(v); }
  }
  return {d::from_dv(order), d::from_dv(pred)};
}

TraversalResult depth_first_order(const CsrMatrix& graph, int64_t i_start, bool directed) {
  auto adj = traversal_adj(graph, directed);
  int64_t n = graph.rows();
  std::vector<char> seen(n, 0);
  std::vector<double> order, pred(n, -9999.0);
  std::vector<std::pair<int64_t, size_t>> stack;
  seen[i_start] = 1;
  order.push_back(static_cast<double>(i_start));
  stack.push_back({i_start, 0});
  while (!stack.empty()) {
    auto& [u, ki] = stack.back();
    bool descended = false;
    while (ki < adj[u].size()) {
      int64_t v = adj[u][ki++];
      if (!seen[v]) {
        seen[v] = 1; pred[v] = static_cast<double>(u);
        order.push_back(static_cast<double>(v));
        stack.push_back({v, 0});
        descended = true;
        break;
      }
    }
    if (!descended) stack.pop_back();
  }
  return {d::from_dv(order), d::from_dv(pred)};
}

ndarray johnson(const CsrMatrix& graph, bool directed) {
  auto adj = adjacency(graph, directed);
  int64_t n = graph.rows();
  // Bellman-Ford from a virtual source (0-weight edge to every node) -> h[].
  std::vector<double> h(n, 0.0);
  for (int64_t it = 0; it < n; ++it) {
    bool changed = false;
    for (int64_t u = 0; u < n; ++u)
      for (const Edge& e : adj[u])
        if (h[u] + e.w < h[e.to]) { h[e.to] = h[u] + e.w; changed = true; }
    if (!changed) break;
  }
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
      for (const Edge& e : adj[u]) {
        double w = e.w + h[u] - h[e.to];  // reweighted, non-negative
        if (du + w < dist[e.to]) { dist[e.to] = du + w; pq.push({dist[e.to], e.to}); }
      }
    }
    for (int64_t j = 0; j < n; ++j)
      D[s * n + j] = (dist[j] == kInf) ? kInf : dist[j] - h[s] + h[j];
  }
  return dist_matrix(D, n);
}

namespace {
struct FlowEdge { int64_t to, cap, rev; };
}  // namespace

MaximumFlowResult maximum_flow(const CsrMatrix& graph, int64_t source, int64_t sink) {
  auto ip = d::iv(graph.indptr()), id = d::iv(graph.indices());
  auto da = d::dv(graph.data());
  int64_t n = graph.rows();
  std::vector<std::vector<FlowEdge>> g(n);
  struct Orig { int64_t u, v, cap; size_t fidx; };
  std::vector<Orig> orig;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t k = ip[i]; k < ip[i + 1]; ++k) {
      int64_t v = id[k], c = static_cast<int64_t>(std::llround(da[k]));
      size_t fi = g[i].size();
      g[i].push_back({v, c, static_cast<int64_t>(g[v].size())});
      g[v].push_back({i, 0, static_cast<int64_t>(fi)});
      orig.push_back({i, v, c, fi});
    }
  int64_t flow_value = 0;
  while (true) {  // Edmonds-Karp: BFS for shortest augmenting path
    std::vector<int64_t> peU(n, -1), peI(n, -1);
    std::vector<char> seen(n, 0);
    std::queue<int64_t> q;
    q.push(source); seen[source] = 1;
    while (!q.empty()) {
      int64_t u = q.front(); q.pop();
      for (size_t k = 0; k < g[u].size(); ++k) {
        const FlowEdge& e = g[u][k];
        if (!seen[e.to] && e.cap > 0) {
          seen[e.to] = 1; peU[e.to] = u; peI[e.to] = static_cast<int64_t>(k);
          q.push(e.to);
        }
      }
    }
    if (!seen[sink]) break;
    int64_t bottleneck = std::numeric_limits<int64_t>::max();
    for (int64_t v = sink; v != source; v = peU[v])
      bottleneck = std::min(bottleneck, g[peU[v]][peI[v]].cap);
    for (int64_t v = sink; v != source; v = peU[v]) {
      FlowEdge& e = g[peU[v]][peI[v]];
      e.cap -= bottleneck;
      g[e.to][e.rev].cap += bottleneck;
    }
    flow_value += bottleneck;
  }
  std::vector<int64_t> row, col; std::vector<double> val;
  for (const Orig& o : orig) {
    int64_t f = o.cap - g[o.u][o.fidx].cap;
    if (f != 0) { row.push_back(o.u); col.push_back(o.v); val.push_back(static_cast<double>(f)); }
  }
  CsrMatrix flow = CsrMatrix::from_coo({d::from_dv(val), d::from_iv(row), d::from_iv(col), n, n});
  return {flow_value, flow};
}

ndarray maximum_bipartite_matching(const CsrMatrix& graph, const std::string& perm_type) {
  auto ip = d::iv(graph.indptr()), id = d::iv(graph.indices());
  int64_t nrows = graph.rows(), ncols = graph.cols();
  std::vector<std::vector<int64_t>> cols_of_row(nrows);
  for (int64_t i = 0; i < nrows; ++i)
    for (int64_t k = ip[i]; k < ip[i + 1]; ++k) cols_of_row[i].push_back(id[k]);
  std::vector<int64_t> match_col(ncols, -1);  // column -> row
  std::function<bool(int64_t, std::vector<char>&)> augment =
      [&](int64_t row, std::vector<char>& used) {
        for (int64_t c : cols_of_row[row]) {
          if (used[c]) continue;
          used[c] = 1;
          if (match_col[c] == -1 || augment(match_col[c], used)) {
            match_col[c] = row;
            return true;
          }
        }
        return false;
      };
  for (int64_t i = 0; i < nrows; ++i) {
    std::vector<char> used(ncols, 0);
    augment(i, used);
  }
  if (perm_type == "row") {
    std::vector<double> out(ncols);
    for (int64_t c = 0; c < ncols; ++c) out[c] = static_cast<double>(match_col[c]);
    return d::from_dv(out);
  }
  std::vector<double> out(nrows, -1.0);
  for (int64_t c = 0; c < ncols; ++c)
    if (match_col[c] != -1) out[match_col[c]] = static_cast<double>(c);
  return d::from_dv(out);
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

}  // namespace scipp::sparse::csgraph
