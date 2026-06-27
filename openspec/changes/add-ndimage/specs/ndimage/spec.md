# ndimage Specification

## ADDED Requirements

### Requirement: Filters

`scypp::ndimage` SHALL provide `correlate1d`/`convolve1d`, `correlate`/`convolve`,
`uniform_filter`(`1d`), `gaussian_filter`(`1d`), `median_filter`,
`minimum_filter`, `maximum_filter`, `sobel`, `prewitt` and `laplace`, with the
boundary modes `reflect`/`nearest`/`mirror`/`wrap`/`constant`, matching SciPy.
Separable filters SHALL select a backend with a portable CPU fallback. (oracle:
scipy/ndimage/_filters.py)

#### Scenario: Filters match SciPy across modes
- GIVEN an image and a filter (gaussian/uniform/median/sobel)
- WHEN the filter is applied with each boundary mode
- THEN the result is `allclose` to `scipy.ndimage`'s for that mode

#### Scenario: Separable filter dispatch
- GIVEN a `gaussian_filter` run on the CPU path and the device-reference path
- WHEN the selected backend is queried
- THEN the two results are equal within tolerance and the backend is reported, with
  the CPU kernel always available

### Requirement: Morphology

`scypp::ndimage` SHALL provide `binary_erosion`/`dilation`/`opening`/`closing`,
`grey_erosion`/`dilation` and `distance_transform_edt`, matching SciPy. (oracle:
scipy/ndimage/_morphology.py)

#### Scenario: Binary and grey morphology match SciPy
- GIVEN a binary (resp. greyscale) image and a structuring element
- WHEN erosion/dilation/opening/closing are applied
- THEN the results match `scipy.ndimage`'s

#### Scenario: Exact Euclidean distance transform
- GIVEN a binary image
- WHEN `distance_transform_edt` is computed
- THEN the result is `allclose` to SciPy's exact EDT

### Requirement: Measurements

`scypp::ndimage` SHALL provide `label`, `center_of_mass`, `sum_labels`, `mean`,
`maximum`, `minimum` and `find_objects`, matching SciPy. (oracle:
scipy/ndimage/_measurements.py)

#### Scenario: Connected-component labeling
- GIVEN a binary image
- WHEN `label` is computed
- THEN the number of features matches SciPy and the labeling matches up to
  relabeling

#### Scenario: Label statistics
- GIVEN an image and a label array
- WHEN `center_of_mass`, `sum_labels`, `mean`, `maximum`, `minimum` are computed
- THEN the results are `allclose` to SciPy's

### Requirement: Geometric transforms

`scypp::ndimage` SHALL provide `shift`, `zoom`, `rotate`, `affine_transform` and
`map_coordinates` with interpolation orders 0 and 1, matching SciPy. (oracle:
scipy/ndimage/_interpolation.py)

#### Scenario: Transforms match SciPy at orders 0 and 1
- GIVEN an image and a transform (shift/zoom/rotate/affine)
- WHEN it is applied with `order=0` and `order=1`
- THEN the result is `allclose` to `scipy.ndimage`'s for that order

#### Scenario: map_coordinates samples the input
- GIVEN an image and a set of coordinates
- WHEN `map_coordinates(input, coords, order=1)` is computed
- THEN the sampled values are `allclose` to SciPy's
