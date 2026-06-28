# ndimage Specification

## ADDED Requirements

### Requirement: Device-accelerated separable correlation
`scipp::ndimage` SHALL compute separable 1-D correlation (`correlate1d`, and the
gaussian/uniform/derivative filters built on it) through NumPP's `correlate1d`
device kernel, mapping SciPP boundary modes to `numpp::FilterMode`, falling back to
the CPU path and reporting the chosen backend via `last_backend()`. (oracle:
scipy.ndimage.correlate1d)

#### Scenario: correlate1d is backend-independent and matches SciPy
- GIVEN a 1-D input, a weight kernel, and a boundary mode
- WHEN `correlate1d` runs on the CPU and any present device backend
- THEN the results are `allclose` to each other and to `scipy.ndimage.correlate1d`
