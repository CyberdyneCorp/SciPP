# Design — fft + fftpack (Phase 3)

## Context

`scipp::fft` ports `scipy.fft` on top of NumPP. NumPP's FFTs already match SciPy's
API and `norm` conventions exactly, so the complex/real transforms are thin
forwarding wrappers. The work in this phase is the **DCT/DST family**, which NumPP
does not provide, plus `next_fast_len` and the `fftpack` alias surface.

## FFTs — delegation

`scipp::fft::fft(a, n, axis, norm)` forwards to `numpp::fft::fft` with identical
argument semantics; likewise `ifft`/`rfft`/`irfft`/`hfft`/`ihfft`, the N-D
variants, and `fftfreq`/`rfftfreq`/`fftshift`/`ifftshift`. GPU-accelerated FFT is
inherited from NumPP when present; this change adds no device code.

## DCT/DST — implementation

Implemented by the direct O(N²) summation formulas (exact, simple, and adequate
for the test sizes; an FFT-based O(N log N) path is a later optimization). The
default `norm=None` ("backward") definitions match SciPy:

- **DCT-I**: `y[k] = x[0] + (−1)ᵏ x[N−1] + 2 Σ_{n=1}^{N−2} x[n] cos(πkn/(N−1))`
- **DCT-II**: `y[k] = 2 Σ_{n=0}^{N−1} x[n] cos(πk(2n+1)/(2N))`
- **DCT-III**: `y[k] = x[0] + 2 Σ_{n=1}^{N−1} x[n] cos(πn(2k+1)/(2N))`
- **DCT-IV**: `y[k] = 2 Σ_{n=0}^{N−1} x[n] cos(π(2k+1)(2n+1)/(4N))`
- **DST-I…IV**: the analogous sine forms.

`idct`/`idst` use the inverse-type relations (verified against the SciPy oracle):

| forward type | inverse maps to | scale (norm=None) |
|---|---|---|
| I   | I   | 1 / (2(N−1))  (DST: 2(N+1)) |
| II  | III | 1 / (2N) |
| III | II  | 1 / (2N) |
| IV  | IV  | 1 / (2N) |

So e.g. `idct(x, type=2) = dct(x, type=3) / (2N)`.

### Axis handling

A `apply_axis(a, axis, line_fn)` helper (same C-order stride walk used by the
Phase 1 reductions) applies a 1-D `vector<double> → vector<double>` transform along
the requested axis of a contiguous float64 copy and scatters the result back,
preserving shape. DCT/DST are real-to-real, so no complex storage is needed.

## next_fast_len

`next_fast_len(n)` returns the smallest integer ≥ n whose only prime factors are
2, 3, 5, 7, 11 (SciPy's 11-smooth fast-length set), found by trial factoring
upward. The real-FFT variant uses the same set.

## fftpack

`scipp::fftpack` re-exports the common legacy entry points (`fft`/`ifft`/`rfft`/
`irfft`/`dct`/`idct`/`dst`/`idst`) as inline wrappers over `scipp::fft`, matching
SciPy's "`fftpack` is the legacy predecessor" relationship without duplicating
logic.

## Oracle strategy

Extend `tests/oracle/generate.py` to emit, for representative real and complex
signals: the real/imag parts of `fft`/`rfft` outputs, the `dct`/`dst`/`idct`/`idst`
results for each type, and `fftfreq`/`fftshift`/`next_fast_len` values. Tests assert
`allclose` and verify round-trips (`ifft(fft(x)) ≈ x`, `idct(dct(x,t),t) ≈ x`).

## Open questions

- DCT/DST `norm="ortho"` requires per-type endpoint scalings that are easy to get
  subtly wrong; deferred until needed, with the FFT family still covering all norms.
