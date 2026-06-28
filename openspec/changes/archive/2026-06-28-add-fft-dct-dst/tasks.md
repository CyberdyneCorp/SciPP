# Tasks — DCT/DST normalizations and N-D transforms

- [x] shared `r2r_line` kernel in `src/fft/realtransforms.cpp` (input/output boundary orthogonalization + `inorm` scale factor)
- [x] map `norm` -> inorm code; forward applies code, inverse applies `2 - code` with type 2↔3 swap; reject unknown norm with `value_error`
- [x] `dctn`/`idctn`/`dstn`/`idstn` axis-loop driver (default all axes); declarations in `include/scypp/fft/fft.hpp`
- [x] extend oracle generator (all types × {ortho, forward}, inverse variants, 2-D `dctn`/`dstn`); regenerate `golden.hpp`
- [x] tests in `tests/test_fft.cpp` (norm match, every-norm round-trip, unknown-norm throw, N-D match + round-trip)
- [x] full suite green; `openspec validate add-fft-dct-dst --strict`
- [x] trim DCT/DST normalization and `dctn`/`idctn` items from `add-fft-extras` backlog
