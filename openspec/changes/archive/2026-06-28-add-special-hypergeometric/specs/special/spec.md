# special (delta)

## ADDED Requirements

### Requirement: Confluent hypergeometric limit 0F1

`scipp::special` SHALL provide `hyp0f1(b, x)` returning the confluent limit
`sum_n x^n / ((b)_n n!)` for real `b` and `x` via term-recurrence summation,
matching `scipy.special.hyp0f1(b, x)` within documented tolerance (~1e-9). A
non-positive integer `b` (a pole) returns `nan` (no throw); `hyp0f1(b, 0) = 1`.

#### Scenario: 0F1 values match SciPy
- GIVEN orders `b = 2, 1/2` and an array of real arguments spanning negative and
  positive `x`
- WHEN `hyp0f1(b, x)` is evaluated
- THEN each is `allclose` to `scipy.special.hyp0f1(b, x)`

#### Scenario: Origin and pole
- GIVEN `x = 0` and a non-positive integer `b`
- WHEN `hyp0f1` is evaluated
- THEN `hyp0f1(b, 0) = 1` and `hyp0f1(-1, x) = nan`

### Requirement: Kummer confluent hypergeometric 1F1

`scipp::special` SHALL provide `hyp1f1(a, b, x)` returning Kummer's confluent
`M(a, b, x) = sum_n (a)_n/((b)_n n!) x^n` for real `a, b, x`, using Kummer's
transformation `M(a,b,x) = e^x M(b-a,b,-x)` for `x < 0`, matching
`scipy.special.hyp1f1(a, b, x)` within documented tolerance (~1e-9) over
moderate `|a|, |b|, |x|`. A non-positive integer `b` returns `nan` (no throw);
`hyp1f1(a, b, 0) = 1`.

#### Scenario: 1F1 values match SciPy
- GIVEN parameter sets `(a,b) = (3/2, 5/2), (-2, 3), (1/2, 13/10)` and an array
  of arguments spanning both signs of `x`
- WHEN `hyp1f1(a, b, x)` is evaluated
- THEN each is `allclose` to `scipy.special.hyp1f1(a, b, x)`

#### Scenario: Origin and pole
- GIVEN `x = 0` and a non-positive integer `b`
- WHEN `hyp1f1` is evaluated
- THEN `hyp1f1(a, b, 0) = 1` and `hyp1f1(a, -2, x) = nan`

### Requirement: Gauss hypergeometric 2F1

`scipp::special` SHALL provide `hyp2f1(a, b, c, z)` returning the Gauss
`sum_n (a)_n(b)_n/((c)_n n!) z^n` for real `a, b, c` and real `z` in `(-1, 1]`,
using the direct series, the Pfaff transformation for `z <= -1/2`, the `1 - z`
reflection for `z` in `(1/2, 1)` with non-integer `c - a - b`, and the Gauss
summation theorem at `z = 1` (when `c - a - b > 0`), matching
`scipy.special.hyp2f1(a, b, c, z)` within documented tolerance (~1e-9). A
non-positive integer `c` returns `nan`; `z` outside `(-1, 1]` returns `nan` (no
throw); `hyp2f1(a, b, c, 0) = 1`.

#### Scenario: 2F1 values match SciPy
- GIVEN parameter sets `(a,b,c) = (1/2, 1, 3/2), (1, 2, 7/2), (-3/2, 7/10, 11/5)`
  and an array of `z` in `(-1, 1)` exercising the direct series, the Pfaff
  branch and the `1 - z` reflection
- WHEN `hyp2f1(a, b, c, z)` is evaluated
- THEN each is `allclose` to `scipy.special.hyp2f1(a, b, c, z)`

#### Scenario: Gauss theorem and domain
- GIVEN `z = 0`, the boundary `z = 1` with `c - a - b > 0`, a non-positive
  integer `c`, and `z` outside `(-1, 1]`
- WHEN `hyp2f1` is evaluated
- THEN `hyp2f1(a, b, c, 0) = 1`, `hyp2f1(a, b, c, 1)` equals the Gauss-theorem
  value `Gamma(c)Gamma(c-a-b)/(Gamma(c-a)Gamma(c-b))`, and the out-of-domain
  cases return `nan`

### Requirement: Tricomi confluent hypergeometric U

`scipp::special` SHALL provide `hyperu(a, b, x)` returning Tricomi's confluent
`U(a, b, x)` for real `a`, non-integer real `b`, and `x > 0`, built from the two
`1F1` solutions
`U = pi/sin(pi b) [M(a,b,x)/(Gamma(a-b+1)Gamma(b)) - x^{1-b} M(a-b+1,2-b,x)/(Gamma(a)Gamma(2-b))]`,
matching `scipy.special.hyperu(a, b, x)` within documented tolerance (~1e-8;
a few points lose digits to subtractive cancellation). Integer `b` and
`x <= 0` return `nan` (no throw).

#### Scenario: U values match SciPy
- GIVEN parameter sets `(a,b) = (1/2, 13/10), (2, 2/5), (3/2, 27/10)` and an
  array of positive arguments `x`
- WHEN `hyperu(a, b, x)` is evaluated
- THEN each is `allclose` to `scipy.special.hyperu(a, b, x)`

#### Scenario: Domain
- GIVEN `x <= 0` or integer `b`
- WHEN `hyperu` is evaluated
- THEN the result is `nan`
