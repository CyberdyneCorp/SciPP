# fft Specification

## Purpose
TBD - created by archiving change add-fft. Update Purpose after archive.
## Requirements
### Requirement: Discrete Fourier transforms

`scypp::fft` SHALL provide `fft`/`ifft`, `rfft`/`irfft`, `hfft`/`ihfft` and the
N-D variants `fft2`/`ifft2`, `fftn`/`ifftn`, `rfftn`/`irfftn`, with `n`/`s`,
`axis`/`axes` and `norm` (`backward`/`ortho`/`forward`) arguments matching SciPy.
(oracle: scipy/fft)

#### Scenario: Forward transform matches SciPy
- GIVEN a real or complex input signal
- WHEN `fft` (or `rfft`) is computed
- THEN the real and imaginary parts of the result are `allclose` to
  `scipy.fft.fft` (resp. `rfft`)

#### Scenario: Inverse round-trips
- GIVEN a signal `x`
- WHEN `ifft(fft(x))` (and `irfft(rfft(x), n=len(x))`) are computed
- THEN the result is `allclose` to `x`

#### Scenario: Orthonormal norm is energy-preserving
- GIVEN a signal `x`
- WHEN `fft(x, norm="ortho")` is computed
- THEN it matches `scipy.fft.fft(x, norm="ortho")` and preserves the L2 norm

### Requirement: Discrete cosine and sine transforms

`scypp::fft` SHALL provide `dct`/`idct` and `dst`/`idst` for types I–IV over a
selectable `axis`, with `norm` in {`"backward"` (default), `"ortho"`,
`"forward"`} matching SciPy. `"ortho"` SHALL produce the orthogonalized variant
(SciPy's `orthogonalize=True` default). An unknown `norm` SHALL raise
`value_error`. (oracle: scipy/fft/_realtransforms.py)

#### Scenario: DCT/DST match SciPy for each type
- GIVEN a real signal and a type in {1,2,3,4}
- WHEN `dct(x, type)` and `dst(x, type)` are computed with the default norm
- THEN the results are `allclose` to `scipy.fft.dct`/`dst`

#### Scenario: Orthonormal and forward norms match SciPy
- GIVEN a real signal and a type in {1,2,3,4}
- WHEN `dct`/`dst`/`idct`/`idst` are computed with `norm="ortho"` or
  `norm="forward"`
- THEN each result is `allclose` to the corresponding `scipy.fft` call with the
  same `norm`

#### Scenario: Inverse cosine/sine transforms round-trip
- GIVEN a real signal `x`, a type in {1,2,3,4} and a norm in
  {`backward`, `ortho`, `forward`}
- WHEN `idct(dct(x, type, norm), type, norm)` and the DST analogue are computed
- THEN each is `allclose` to `x`, and `idct`/`idst` individually match SciPy

#### Scenario: Transform along a chosen axis
- GIVEN a 2-D array and an axis
- WHEN `dct(x, type=2, axis=…)` is computed
- THEN each line along that axis is transformed, matching SciPy's `axis` behavior

#### Scenario: Unknown norm is rejected
- GIVEN a real signal
- WHEN `dct(x, type=2, axis=-1, norm="bogus")` is called
- THEN `value_error` is raised

### Requirement: Helper functions

`scypp::fft` SHALL provide `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` and
`next_fast_len`, matching SciPy. `next_fast_len(n)` SHALL return the smallest
11-smooth integer ≥ n. (oracle: scipy/fft/_helper.py)

#### Scenario: Frequencies and shifts match SciPy
- GIVEN a length `n` and sample spacing `d`
- WHEN `fftfreq(n, d)` / `rfftfreq(n, d)` are computed and `fftshift`/`ifftshift`
  are applied
- THEN the results are `allclose` to SciPy

#### Scenario: next_fast_len returns an 11-smooth length
- GIVEN an integer `n`
- WHEN `next_fast_len(n)` is called
- THEN it returns the smallest integer ≥ n whose prime factors are all ≤ 11,
  matching `scipy.fft.next_fast_len`

### Requirement: Legacy fftpack namespace

`scypp::fftpack` SHALL re-export `fft`/`ifft`/`rfft`/`irfft`/`dct`/`idct`/`dst`/
`idst` as the legacy API surface, producing results identical to the
corresponding `scypp::fft` functions. (oracle: scipy/fftpack)

#### Scenario: fftpack delegates to fft
- GIVEN a signal
- WHEN a `scypp::fftpack` function and its `scypp::fft` counterpart are called
- THEN they return identical results

### Requirement: N-D cosine and sine transforms

`scypp::fft` SHALL provide `dctn`/`idctn` and `dstn`/`idstn` that apply the 1-D
DCT/DST of a given `type` and `norm` over a selectable list of `axes`, defaulting
to all axes, matching SciPy. (oracle: scipy/fft/_realtransforms.py)

#### Scenario: dctn/dstn over all axes match SciPy
- GIVEN a 2-D array and a type in {1,2,3,4}
- WHEN `dctn(x, type)` / `dstn(x, type)` are computed with the default (all) axes
- THEN the result is `allclose` to `scipy.fft.dctn`/`dstn`, including under
  `norm="ortho"`

#### Scenario: N-D inverse round-trips
- GIVEN a 2-D array, a type in {1,2,3,4} and a norm
- WHEN `idctn(dctn(x, type, norm), type, norm)` and the DST analogue are computed
- THEN each is `allclose` to the original array

