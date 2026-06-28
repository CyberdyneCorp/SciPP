# integrate (delta)

## ADDED Requirements

### Requirement: Nested and extended quadrature
The system SHALL provide `romberg`, `quad_vec`, `dblquad`, `tplquad`, and `nquad`,
composing the adaptive `quad` to integrate vector-valued and multidimensional
integrands (including variable inner bounds) to the requested tolerance.

#### Scenario: Romberg integration
- GIVEN `f(x) = sin(x)` on `[0, π]`
- WHEN `romberg` is called
- THEN the result SHALL equal `2` to `allclose` tolerance.

#### Scenario: Vector-valued quadrature
- GIVEN `f(x) = [1, x, x², sin x]` on `[0, 1]`
- WHEN `quad_vec` is called
- THEN each component SHALL match `scipy.integrate.quad_vec` to `allclose`
  tolerance.

#### Scenario: Double integral with variable bounds
- GIVEN `func(y, x) = x + y`, `x ∈ [0, 1]`, `y ∈ [0, x]`
- WHEN `dblquad` is called
- THEN the result SHALL match `scipy.integrate.dblquad` to `allclose` tolerance.

#### Scenario: Triple and N-dimensional integrals
- GIVEN a separable integrand over a hyper-rectangle
- WHEN `tplquad` and `nquad` are called
- THEN both SHALL return the same value, matching SciPy to `allclose` tolerance.
