# Design — cluster + io (Phase 12)

## cluster.vq

- **whiten(obs)** — divide each column by its standard deviation (population,
  `ddof=0`), as SciPy.
- **vq(obs, code_book)** — assign each observation to the nearest code-book row
  (Euclidean), returning `(code, dist)`.
- **kmeans2(data, init_centroids, iter)** — Lloyd iteration from the given
  centroids: assign (via `vq`), then recompute each centroid as the mean of its
  members; repeat `iter` times. Returns `(centroids, labels)`. Deterministic given
  the initial centroids (SciPy's `minit="matrix"`).

## cluster.hierarchy

- **linkage(y, method, metric)** — `y` is either an observation matrix (the
  Euclidean `pdist` is computed) or a condensed distance vector. Agglomerative
  clustering merges the globally closest pair each step, updating inter-cluster
  distances via the **Lance–Williams** recurrence (`single`=min, `complete`=max,
  `average`=size-weighted mean, `ward`=variance, on squared distances). The result
  is the SciPy `(n−1)×4` linkage matrix `[c1, c2, dist, size]` with cluster ids
  `0..n−1` for singletons and `n, n+1, …` for merges (smaller id first). Validated
  on general-position points where the merge order is unique.
- **fcluster(Z, t, criterion)** — `distance`: cut the dendrogram so each flat
  cluster's cophenetic distance ≤ `t`; `maxclust`: the smallest height giving ≤ `t`
  clusters. Labels are `1..k`, matching SciPy up to relabeling.
- **cophenet(Z)** — the cophenetic distance for each pair (the merge height at
  which they first share a cluster), as a condensed vector.

## io

- **Matrix Market** — `mmread` parses the `%%MatrixMarket` header + `coordinate`
  (sparse) or `array` (dense) body into a dense `ndarray`; `mmwrite` writes the
  coordinate form of the nonzeros. 1-based indices, `general` symmetry.
- **WAV** — `wavread` parses the RIFF/`fmt `/`data` chunks (16-bit and 32-bit
  integer PCM, mono/stereo) returning `(rate, samples)`; `wavwrite` writes 16-bit
  PCM. Sample values match SciPy's integer arrays.
- **ARFF** — `loadarff` parses `@attribute … numeric` declarations and the `@data`
  rows into a numeric `ndarray` (nominal/string attributes are out of scope here).

## Testing without Python

The generator writes golden sample files (`sample.mtx`, `sample.wav`,
`sample.arff`) into `tests/golden/` (committed) and emits the expected parsed
values. The C++ tests read those files via a `SCIPP_GOLDEN_DIR` compile definition,
and round-trip writes through a temp path. Cluster routines compare to SciPy's
`whiten`/`vq`/`kmeans2`/`linkage`/`fcluster`/`cophenet` outputs.

## Oracle strategy

`whiten`/`vq`/`kmeans2` compared `allclose`; `linkage`/`cophenet` compared
`allclose` (general-position uniqueness); `fcluster` compared up to relabeling;
`mmread`/`wavread`/`loadarff` compared to the known file contents and round-tripped.

## Open questions

- `linkage` tie-breaking matches SciPy only for distinct merge distances; tests use
  general-position inputs. `fcluster` labelings are compared by the induced
  partition (relabeling-invariant).
