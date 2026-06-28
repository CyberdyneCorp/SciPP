# Add csgraph traversal + flow algorithms to scypp::sparse

## Why

The `scypp::sparse::csgraph` submodule already ships the distance/structure
algorithms (`dijkstra`, `bellman_ford`, `floyd_warshall`, `connected_components`,
`minimum_spanning_tree`), but the traversal and flow family from
`scipy.sparse.csgraph` was deferred to the `add-sparse-extras` backlog. These are
the standard building blocks for graph search, all-pairs shortest paths with
negative edges, network flow, and assignment problems, and they reuse the same
CSR-adjacency machinery already present in `csgraph.cpp` — no NumPP changes are
required.

## What changes

Extends the **sparse** capability — `scypp::sparse::csgraph`, validated against
`scipy.sparse.csgraph` on small fixed graphs:

- **`breadth_first_order` / `depth_first_order`**: return the node visitation
  order plus a predecessor array (sentinel `-9999` for the start and unreachable
  nodes). Neighbour iteration follows SciPy's CSR column-order convention; for
  `directed=false` each node's out-neighbours (CSR order) are followed by its
  in-neighbours (ascending source order), exactly reproducing SciPy's symmetrised
  traversal order.
- **`johnson`**: all-pairs shortest paths via Johnson's algorithm
  (Bellman-Ford reweighting from a virtual source, then Dijkstra per source),
  correct in the presence of negative edge weights.
- **`maximum_flow`**: max flow on an integer-capacity directed graph via
  Edmonds-Karp (BFS shortest augmenting paths); returns the integer flow value
  and a residual/flow matrix on the original edges.
- **`maximum_bipartite_matching`**: maximum-cardinality matching via Kuhn's
  augmenting-path algorithm; returns the `perm_type="row"` (length-cols) or
  `perm_type="column"` (length-rows) matching array with `-1` for unmatched
  vertices, matching SciPy's convention.

## Impact

- Affected specs: **modifies** the `sparse` capability (adds one requirement).
- Affected code: extends `src/sparse/csgraph.cpp` and
  `include/scypp/sparse/sparse.hpp` (four new entry points plus `TraversalResult`
  and `MaximumFlowResult` result structs), `tests/test_sparse_csgraph.cpp`, and
  the oracle generator.
- Trims the `csgraph` traversal/flow line from the `add-sparse-extras` backlog.

## Non-goals

- `breadth_first_tree` / `depth_first_tree` (the spanning-tree CSR variants) and
  `structural_rank` — separate backlog items.
- Strong-component mode for `connected_components` (the existing weak mode is
  unchanged); tracked separately in the backlog.
- Returning SciPy's exact `maximum_flow.flow` matrix structure or a particular
  maximum matching when several exist: the flow *value* is unique and is matched
  exactly, and the matching graphs in the oracle are chosen to have a unique
  maximum matching so the returned arrays agree element-for-element.
