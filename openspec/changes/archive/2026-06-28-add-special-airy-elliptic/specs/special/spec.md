# special (delta)

## ADDED Requirements

### Requirement: Airy functions

`scipp::special` SHALL provide `airy(x)` returning `(Ai, Aip, Bi, Bip)` and the
exponentially-scaled `airye(x)`, evaluated for real `x` and matching SciPy within
documented tolerance (~1e-9). Out-of-domain behavior follows SciPy: `airye`
returns `nan` for the scaled `Ai`/`Aip` at `x < 0` while `Bi`/`Bip` are unscaled.
(oracle: scipy/special/_basic.py, scipy/special/airy.c)

#### Scenario: Airy and derivatives match SciPy
- GIVEN an array of real arguments spanning negative, zero and positive values
- WHEN `airy(x)` is evaluated
- THEN `Ai`, `Aip`, `Bi`, `Bip` are each `allclose` to `scipy.special.airy`

#### Scenario: Airy satisfies the Wronskian
- GIVEN a real argument `x`
- WHEN `airy(x)` is evaluated
- THEN `Ai*Bip - Aip*Bi` is `allclose` to `1/pi`

#### Scenario: Exponentially scaled Airy matches SciPy
- GIVEN an array of real arguments
- WHEN `airye(x)` is evaluated
- THEN for `x >= 0` the four outputs are `allclose` to `scipy.special.airye`, and
  for `x < 0` the scaled `Ai`/`Aip` are `nan` while `Bi`/`Bip` equal the unscaled
  values

### Requirement: Elliptic integrals

`scipp::special` SHALL provide the complete elliptic integrals `ellipk(m)`,
`ellipkm1(p)` (= `ellipk(1-p)`) and `ellipe(m)` via the arithmetic-geometric-mean
iteration, the incomplete integrals `ellipkinc(phi, m)` and `ellipeinc(phi, m)`
via the Carlson symmetric forms, and the Jacobi elliptic functions
`ellipj(u, m)` returning `(sn, cn, dn, ph)`, all parameterized by `m = k^2` and
matching SciPy within documented tolerance. Out-of-domain inputs return `nan`/`inf`
(no throw): `ellipk(1)`/`ellipkm1(0)` are `inf`, `ellipe(1)` is `1`, and `m > 1`
yields `nan`. (oracle: scipy/special/cephes/ellpk.c, ellpe.c, ellie.c, ellpj.c)

#### Scenario: Complete integrals match SciPy
- GIVEN an array of parameters `m <= 1` (including negative `m`)
- WHEN `ellipk(m)` and `ellipe(m)` are evaluated
- THEN both are `allclose` to `scipy.special.ellipk`/`ellipe`, with
  `ellipk(0) = ellipe(0) = pi/2` and `ellipe(1) = 1`

#### Scenario: ellipkm1 is accurate near m = 1
- GIVEN small `p` in `(0, 1]`
- WHEN `ellipkm1(p)` is evaluated
- THEN it is `allclose` to `scipy.special.ellipkm1(p)` (= `ellipk(1-p)`)

#### Scenario: Incomplete integrals match SciPy and reduce to complete
- GIVEN an amplitude `phi` (including `phi > pi/2` and negative `phi`) and a
  parameter `m`
- WHEN `ellipkinc(phi, m)` and `ellipeinc(phi, m)` are evaluated
- THEN both are `allclose` to `scipy.special.ellipkinc`/`ellipeinc`, and at
  `phi = pi/2` they equal `ellipk(m)`/`ellipe(m)`

#### Scenario: Jacobi functions match SciPy and satisfy identities
- GIVEN an argument `u` and a parameter `m`
- WHEN `ellipj(u, m)` is evaluated
- THEN `sn`, `cn`, `dn`, `ph` are `allclose` to `scipy.special.ellipj`, with
  `sn^2 + cn^2 = 1` and `dn^2 + m*sn^2 = 1`
