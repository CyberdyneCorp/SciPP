# sparse (delta)

## ADDED Requirements

### Requirement: csgraph traversal and flow algorithms
The system SHALL provide `breadth_first_order`, `depth_first_order`, `johnson`,
`maximum_flow`, and `maximum_bipartite_matching` in `scypp::sparse::csgraph`,
operating on CSR graphs and matching `scipy.sparse.csgraph` to integer exactness
for orders/matchings/flow values and to `allclose` tolerance (~1e-12) for
distances.

#### Scenario: Breadth- and depth-first order
- GIVEN a directed CSR graph and a start node
- WHEN `breadth_first_order` (resp. `depth_first_order`) is called
- THEN the returned node array SHALL list the reachable nodes in the same order
  as SciPy (neighbours visited in CSR column order), and the predecessor array
  SHALL give each node's parent in the traversal tree with `-9999` for the start
  node and unreachable nodes.

#### Scenario: Undirected traversal symmetrisation
- GIVEN an asymmetric CSR graph traversed with `directed=false`
- WHEN `breadth_first_order` or `depth_first_order` is called
- THEN each node's neighbours SHALL be its out-neighbours (CSR order) followed by
  its in-neighbours (ascending source order), reproducing SciPy's symmetrised
  traversal order and predecessors.

#### Scenario: Johnson all-pairs shortest paths with negative edges
- GIVEN a directed CSR graph containing negative edge weights but no negative
  cycle
- WHEN `johnson` is called
- THEN the returned distance matrix SHALL equal `scipy.sparse.csgraph.johnson`
  to `allclose` tolerance, with `inf` for unreachable pairs.

#### Scenario: Maximum flow value
- GIVEN a directed CSR graph with integer capacities, a source and a sink
- WHEN `maximum_flow` is called
- THEN the returned `flow_value` SHALL equal SciPy's `.flow_value` exactly, and
  the net flow leaving the source in the returned flow matrix SHALL equal that
  value.

#### Scenario: Maximum bipartite matching
- GIVEN a CSR biadjacency matrix
- WHEN `maximum_bipartite_matching` is called with `perm_type="row"` (length
  cols) or `perm_type="column"` (length rows)
- THEN the returned matching array SHALL equal SciPy's, with `-1` for unmatched
  vertices, for graphs whose maximum matching is unique.
