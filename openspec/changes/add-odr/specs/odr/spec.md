# odr Specification

## ADDED Requirements

### Requirement: Orthogonal distance regression
`scypp::odr` SHALL provide an `ODR` fit over `Data` and a `Model` that minimizes
the orthogonal distance between the data and the model curve, returning the
estimated parameters and their standard errors, matching `scipy.odr`. (oracle:
scipy/odr)

#### Scenario: Linear ODR matches SciPy
- GIVEN x, y data generated from a linear model with noise in both variables
- WHEN `ODR(Data(x, y), Model(linear), beta0).run()` is executed
- THEN the estimated parameters and residual variance are `allclose` to
  `scipy.odr` on the same inputs
