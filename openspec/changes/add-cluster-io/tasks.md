# Tasks — cluster + io (Phase 12)

## 1. cluster.vq
- [ ] `include/scypp/cluster/cluster.hpp`; `src/cluster/*.cpp` in build; export from umbrella
- [ ] `whiten`, `vq`, `kmeans2` (explicit init, Lloyd iteration)

## 2. cluster.hierarchy
- [ ] `linkage` (single/complete/average/ward via Lance–Williams)
- [ ] `fcluster` (distance/maxclust), `cophenet`

## 3. io
- [ ] `include/scypp/io/io.hpp`; `src/io/io.cpp` in build; export
- [ ] Matrix Market `mmread`/`mmwrite`
- [ ] WAV `wavread`/`wavwrite` (16/32-bit PCM)
- [ ] ARFF `loadarff` (numeric)

## 4. Oracle + validation
- [ ] Generator writes golden `sample.mtx`/`sample.wav`/`sample.arff` + expected values; `SCYPP_GOLDEN_DIR` define
- [ ] `tests/test_cluster_io.cpp`: vq/kmeans2/linkage/fcluster/cophenet + io read/round-trip vs SciPy
- [ ] CPU build green; full suite green; `openspec validate add-cluster-io --strict`
- [ ] Check off Phase 12 in `bootstrap-scypp-foundation/tasks.md`; update README to v1.0
