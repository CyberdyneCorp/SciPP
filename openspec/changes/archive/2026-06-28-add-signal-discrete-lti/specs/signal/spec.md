# signal Specification

## ADDED Requirements

### Requirement: Discrete-time LTI systems
`scipp::signal` SHALL provide discrete-time LTI support: `cont2discrete`,
`dstep`, `dimpulse`, `dlsim`, `dfreqresp` and `dbode`, matching SciPy.
(oracle: scipy.signal.cont2discrete / dstep / dimpulse / dlsim / dfreqresp / dbode)

#### Scenario: Continuous-to-discrete conversion
- GIVEN a continuous state-space `(A,B,C,D)` and a sample time `dt`
- WHEN `cont2discrete` is called with method `"zoh"`, `"bilinear"` or `"euler"`
- THEN the returned discrete matrices `Ad,Bd,Cd,Dd` and `dt` SHALL be `allclose`
  to SciPy's (~1e-8), with `"zoh"` using the matrix-exponential block formula and
  `"bilinear"`/`"euler"` the alpha-parameterized linear formulas

#### Scenario: Discrete step and impulse responses
- GIVEN a discrete system obtained from a continuous one via `zoh`
- WHEN `dstep(system, n)` and `dimpulse(system, n)` are evaluated over `n` samples
- THEN the output sequences SHALL match SciPy's `dstep`/`dimpulse` to `allclose`
  tolerance (~1e-8), produced by iterating `x[k+1]=Ad x[k]+Bd u[k]`,
  `y[k]=Cd x[k]+Dd u[k]`

#### Scenario: Discrete simulation
- GIVEN a discrete system and an input sequence `u`
- WHEN `dlsim(system, u)` is evaluated
- THEN the output sequence SHALL match SciPy's `dlsim` to `allclose` tolerance

#### Scenario: Discrete frequency response
- GIVEN a discrete system and frequencies `w` in radians/sample
- WHEN `dfreqresp(system, w)` and `dbode(system, w)` are evaluated
- THEN the complex response `H(e^{jw})`, the magnitude in dB and the unwrapped
  phase in degrees SHALL match SciPy's `dfreqresp`/`dbode` to `allclose`
  tolerance (~1e-8)
