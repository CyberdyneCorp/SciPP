# Add DCT/DST normalizations and N-D transforms to scipp::fft

## Why

`scipp::fft` already ships `dct`/`idct` and `dst`/`idst` for types I–IV, but only
under the default `norm="backward"` normalization. SciPy's `scipy.fft` exposes
three normalizations — `backward`, `ortho`, `forward` — and the `ortho` variant
is the orthogonalized transform widely used in signal/image processing (it is
what MATLAB's `dct` computes). SciPy also provides the N-D entry points
`dctn`/`idctn`/`dstn`/`idstn` that apply the 1-D transform over selectable axes.
Both were deferred to the `add-fft-extras` backlog. They need no NumPP changes —
the existing direct-summation line kernels already match SciPy bit-for-bit under
`backward`, so the missing pieces are the normalization/orthogonalization layer
and an axis-loop driver.

## What changes

Extends the **fft** capability — `scipp::fft` — validated against `scipy.fft`:

- **`norm="ortho"` and `norm="forward"`** for `dct`/`idct`/`dst`/`idst`, types
  I–IV. The norm is mapped to SciPy's `inorm` code (backward=0, ortho=1,
  forward=2); a forward transform applies that code and an inverse applies
  `2 - code` while swapping DCT/DST type 2↔3, matching the unnormalized
  type-2/3 inverse relationship. `inorm` scales each line by `1`, `sqrt(1/S)` or
  `1/S` where `S` is the type's logical factor (`2(N-1)` for DCT-I, `2(N+1)` for
  DST-I, otherwise `2N`). For `norm="ortho"` the transform is additionally
  orthogonalized — boundary samples are scaled by `sqrt(2)` per SciPy's
  `orthogonalize=True` default — so the coefficient matrix is orthonormal and the
  forward/inverse pair are exact transposes.
- **`dctn`/`idctn`/`dstn`/`idstn`**: apply the 1-D transform successively over a
  selectable list of axes (default: all axes), honoring the same `type` and
  `norm` arguments.

## Impact

- Affected specs: **modifies** the `fft` capability (extends the DCT/DST
  requirement with the three norms; adds an N-D requirement).
- Affected code: rewrites the normalization layer of `src/fft/realtransforms.cpp`
  (shared `r2r_line` kernel + axis-loop driver), adds the four N-D declarations to
  `include/scipp/fft/fft.hpp`, extends `tests/oracle/generate.py` and
  `tests/test_fft.cpp`. No change to the existing `backward` behavior or the
  `scipp::fftpack` re-exports.
- Trims the DCT/DST normalization and `dctn`/`idctn` items from the
  `add-fft-extras` backlog.

## Non-goals

- `fht`/`ifht` (Hankel transform) and `czt`/`zoom_fft`/`CZT` (chirp-z) remain in
  the `add-fft-extras` backlog.
- The `s`/`overwrite_x`/`workers`/explicit `orthogonalize` keyword arguments of
  SciPy's N-D entry points. The N-D transforms here take `type`, `axes` and
  `norm`; padding/truncation via `s` and the standalone `orthogonalize` toggle are
  out of scope (the `ortho` norm implies the orthogonalized variant, as in
  SciPy's default).
