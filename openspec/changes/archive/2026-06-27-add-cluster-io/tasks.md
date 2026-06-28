# Tasks — cluster + io (Phase 12)

## 1. cluster.vq
- [x] `include/scipp/cluster/cluster.hpp`; `src/cluster/*.cpp` in build; export from umbrella
- [x] `whiten`, `vq`, `kmeans2` (explicit init, Lloyd iteration)

## 2. cluster.hierarchy
- [x] `linkage` (single/complete/average/ward via Lance–Williams)
- [x] `fcluster` (distance/maxclust), `cophenet`

## 3. io
- [x] `include/scipp/io/io.hpp`; `src/io/io.cpp` in build; export
- [x] Matrix Market `mmread`/`mmwrite`
- [x] WAV `wavread`/`wavwrite` (16/32-bit PCM)
- [x] ARFF `loadarff` (numeric)

## 4. Oracle + validation
- [x] Generator writes golden `sample.mtx`/`sample.wav`/`sample.arff` + expected values; `SCIPP_GOLDEN_DIR` define
- [x] `tests/test_cluster_io.cpp`: vq/kmeans2/linkage/fcluster/cophenet + io read/round-trip vs SciPy
- [x] CPU build green; full suite green; `openspec validate add-cluster-io --strict`
- [x] Check off Phase 12 in `bootstrap-scipp-foundation/tasks.md`; update README to v1.0
