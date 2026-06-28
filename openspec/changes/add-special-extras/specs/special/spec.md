# special Specification

## ADDED Requirements

### Requirement: Airy functions
`scipp::special` SHALL provide `airy` (Ai, Ai', Bi, Bi') and `airye` (exponentially
scaled), matching SciPy. (oracle: scipy/special airy)

#### Scenario: Airy matches SciPy
- GIVEN real arguments spanning negative and positive ranges
- WHEN `airy(x)` is computed
- THEN Ai/Ai'/Bi/Bi' are `allclose` to `scipy.special.airy`

### Requirement: Elliptic integrals
`scipp::special` SHALL provide the complete and incomplete elliptic integrals
`ellipk`, `ellipkm1`, `ellipe`, `ellipkinc`, `ellipeinc`, and the Jacobi elliptic
functions `ellipj`, matching SciPy. (oracle: scipy/special elliptic)

#### Scenario: Complete elliptic integrals match SciPy
- GIVEN parameters m in [0, 1)
- WHEN `ellipk(m)` and `ellipe(m)` are computed
- THEN both are `allclose` to SciPy

### Requirement: Error-function relatives
`scipp::special` SHALL provide `erfcx`, `dawsn`, `wofz`/`voigt_profile`, and
`fresnel`, matching SciPy. (oracle: scipy/special erf family)

#### Scenario: Scaled complementary error function matches SciPy
- GIVEN real arguments including large positive values
- WHEN `erfcx(x)` is computed
- THEN it is `allclose` to `scipy.special.erfcx` without overflow

### Requirement: Additional Bessel functions
`scipp::special` SHALL provide spherical Bessel functions `spherical_jn`/`yn`/`in`/`kn`,
the Kelvin functions `kelvin`, and the Hankel functions `hankel1`/`hankel2`,
matching SciPy. (oracle: scipy/special bessel)

#### Scenario: Spherical Bessel matches SciPy
- GIVEN an order n and real arguments
- WHEN `spherical_jn(n, x)` is computed
- THEN it is `allclose` to `scipy.special.spherical_jn`

### Requirement: Sine and cosine integrals
`scipp::special` SHALL provide `sici` (Si, Ci) and `shichi` (Shi, Chi), matching
SciPy. (oracle: scipy/special sici)

#### Scenario: Sine/cosine integrals match SciPy
- GIVEN real arguments
- WHEN `sici(x)` is computed
- THEN Si and Ci are `allclose` to `scipy.special.sici`

### Requirement: Miscellaneous special functions
`scipp::special` SHALL provide `lambertw`, `zeta`/`zetac`, `struve`/`modstruve`,
and `spence`, matching SciPy. (oracle: scipy/special)

#### Scenario: Lambert W and zeta match SciPy
- GIVEN valid real arguments
- WHEN `lambertw(x)` and `zeta(x)` are computed
- THEN both are `allclose` to SciPy on the principal branch / real axis

### Requirement: Hypergeometric functions
`scipp::special` SHALL provide the confluent and Gauss hypergeometric functions
`hyp0f1`, `hyp1f1`, `hyp2f1`, and `hyperu`, matching SciPy within documented
convergence regions. (oracle: scipy/special hypergeometric)

#### Scenario: Confluent hypergeometric matches SciPy
- GIVEN parameters and arguments inside the convergent region
- WHEN `hyp1f1(a, b, x)` is computed
- THEN it is `allclose` to `scipy.special.hyp1f1`
