# Add cluster + io (Phase 12)

## Why

Phase 12 — the final roadmap phase — ports `scipy.cluster` (vector quantization
and hierarchical clustering) and `scipy.io` (file I/O), completing the
commonly-used SciPy surface for SciPP's v1.0.

## What changes

Adds the **cluster** and **io** capabilities — `scipp::cluster` and `scipp::io`,
validated against the SciPy oracle:

- **`cluster.vq`**: `whiten`, `vq` (vector quantization / nearest-centroid
  assignment), and `kmeans2` with explicit initial centroids (deterministic).
- **`cluster.hierarchy`**: `linkage` (`single`/`complete`/`average`/`ward`),
  `fcluster` (`distance`/`maxclust` criteria), and `cophenet`.
- **`io`**: Matrix Market `mmread`/`mmwrite`, WAV `wavread`/`wavwrite` (PCM), and
  ARFF `loadarff` (numeric attributes).

## Impact

- Affected specs: **adds** the `cluster` and `io` capabilities.
- Affected code: new `include/scipp/cluster/`, `include/scipp/io/`,
  `src/cluster/`, `src/io/`, `tests/test_cluster_io.cpp`, golden sample files
  (`.mtx`/`.wav`/`.arff`), extended oracle generator.
- Roadmap: checks off Phase 12 in `bootstrap-scipp-foundation/tasks.md`.

## Non-goals (deferred)

- **Stochastic `kmeans`/`kmeans2` random init** — needs a SciPy-matching RNG
  stream; the deterministic explicit-init path ships here.
- **`dendrogram`** (plotting), `fclusterdata`, `inconsistent`, `leaders`,
  `optimal_leaf_ordering`.
- **`scipy.datasets`** (`ascent`/`face`/`electrocardiogram`) — bundled sample
  data fetched via pooch; a thin data-loader follow-up, not algorithmic.
- **`io` extras**: MATLAB `.mat` (out of scope per project), NetCDF, IDL, Harwell-
  Boeing, WAV beyond integer PCM, ARFF nominal/string attributes.
