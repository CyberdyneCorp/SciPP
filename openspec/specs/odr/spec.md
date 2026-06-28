# odr Specification

## Purpose
TBD - created by archiving change add-odr. Update Purpose after archive.
## Requirements
### Requirement: Orthogonal distance regression
`scipp::odr` SHALL provide an `ODR` fit over a `Data` (x, y with optional per-point
standard deviations sx, sy) and a `Model` (`f(beta, x)`) that minimizes the
orthogonal distance between the data and the model curve over both the parameters
`beta` and the per-point x-offsets `delta`, returning the estimated parameters,
their standard errors, the parameter covariance, the residual variance, and the sum
of squares, matching `scipy.odr`. (oracle: scipy/odr)

#### Scenario: Linear ODR matches SciPy
- GIVEN x, y data generated from a linear model with noise in both variables
- WHEN `ODR(Data(x, y), Model(linear), beta0).run()` is executed
- THEN the estimated parameters are `allclose` to `scipy.odr` to 1e-5 and the
  residual variance and sum of squares to 1e-4 on the same inputs

#### Scenario: Nonlinear ODR matches SciPy
- GIVEN x, y data from a nonlinear model `b0·exp(b1·x)` with noise in both variables
- WHEN `ODR(Data(x, y), Model(fcn), beta0).run()` is executed
- THEN the estimated parameters, residual variance, and sum of squares are
  `allclose` to `scipy.odr` on the same inputs

#### Scenario: Weighted ODR matches SciPy
- GIVEN x, y data with per-point standard deviations sx and sy
- WHEN `ODR(Data(x, y, sx, sy), Model(linear), beta0).run()` is executed
- THEN the estimated parameters and residual variance match `scipy.odr` run with the
  equivalent inverse-variance weights `wd = 1/sx^2`, `we = 1/sy^2`

