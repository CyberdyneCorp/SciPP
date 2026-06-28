# fft (delta)

## MODIFIED Requirements

### Requirement: Discrete cosine and sine transforms

`scipp::fft` SHALL provide `dct`/`idct` and `dst`/`idst` for types I–IV over a
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

## ADDED Requirements

### Requirement: N-D cosine and sine transforms

`scipp::fft` SHALL provide `dctn`/`idctn` and `dstn`/`idstn` that apply the 1-D
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
