# Tasks — csgraph traversal + flow

- [x] `breadth_first_order` / `depth_first_order` (node order + predecessors, CSR column-order neighbours, undirected out-then-in symmetrisation) in `src/sparse/csgraph.cpp`
- [x] `johnson` all-pairs shortest paths (Bellman-Ford reweighting + per-source Dijkstra)
- [x] `maximum_flow` (Edmonds-Karp on integer capacities; flow value + flow matrix)
- [x] `maximum_bipartite_matching` (Kuhn augmenting paths; row/column perm with -1 sentinel)
- [x] header decl + result structs in `include/scipp/sparse/sparse.hpp`
- [x] extend oracle generator with fixed graphs; regenerate `tests/golden/golden.hpp`
- [x] `tests/test_sparse_csgraph.cpp` vs `scipy.sparse.csgraph`; register in CMake
- [x] full suite green; `openspec validate add-sparse-csgraph-traversal --strict`
- [x] trim the `csgraph` traversal/flow line from `add-sparse-extras` backlog
