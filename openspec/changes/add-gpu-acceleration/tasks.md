# Tasks — wire SciPP hot kernels to NumPP device acceleration

- [x] Bump NumPP floor to 1.5.0 (`find_package(NumPP 1.5.0 ...)`); re-bootstrap `.deps/numpp`
- [x] sparse: `spmv`/`spmm` via `numpp::csr_spmv`; map `last_backend()`; cross-backend test
- [x] spatial: `cdist`/`pdist` euclidean/sqeuclidean via `numpp::cdist_euclidean`; test
- [x] ndimage: `correlate1d` + separable filters via `numpp::correlate1d` (mode mapping); test
- [x] build cleanup: drop `-DNUMPP_WARNINGS_AS_ERRORS=OFF` from bootstrap; umbrella includes; re-evaluate registry workaround
- [x] trim `add-sparse-extras` GPU item; `openspec validate --all --strict`; archive
