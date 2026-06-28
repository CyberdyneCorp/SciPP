# fft Specification

## ADDED Requirements

### Requirement: Orthonormal and forward DCT/DST normalization
`scipp::fft` SHALL support the `ortho` and `forward` normalizations for
`dct`/`idct`/`dst`/`idst` (types I–IV), matching SciPy. (oracle: scipy/fft/_realtransforms.py)

#### Scenario: Orthonormal DCT is energy-preserving
- GIVEN a real signal and a type in {1,2,3,4}
- WHEN `dct(x, type, norm="ortho")` is computed
- THEN it matches SciPy and the transform preserves the L2 norm

### Requirement: N-D cosine/sine transforms
`scipp::fft` SHALL provide `dctn`, `idctn`, `dstn` and `idstn` over multiple axes,
matching SciPy. (oracle: scipy/fft/_realtransforms.py)

#### Scenario: N-D DCT round-trips
- GIVEN an N-D real array
- WHEN `idctn(dctn(x))` is computed
- THEN the result is `allclose` to `x` and `dctn` matches SciPy

### Requirement: Hankel and chirp-z transforms
`scipp::fft` SHALL provide `fht`/`ifht` (Hankel) and `czt`/`zoom_fft` (chirp-z),
matching SciPy within documented tolerance. (oracle: scipy/fft/_fftlog.py, scipy/signal/_czt.py)

#### Scenario: Chirp-z matches the DFT on the unit circle
- GIVEN a signal
- WHEN `czt` is evaluated over a full unit-circle contour
- THEN the result is `allclose` to the FFT, matching SciPy
