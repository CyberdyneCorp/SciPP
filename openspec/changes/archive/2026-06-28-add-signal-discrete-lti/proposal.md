# Add discrete-time LTI systems to scipp::signal

## Why

`scipp::signal` already ships the continuous-time LTI surface (`tf2ss`,
`freqresp`, `bode`, `lsim`, `step`, `impulse`). The discrete-time companions and
the continuous-to-discrete bridge were deferred to the signal backlog. They are
the natural counterpart for sampled-data / control work and reuse the existing
`scipp::linalg::expm` and `numpp::linalg::solve`, so no NumPP changes are needed.

## What changes

Extends the **signal** capability with discrete-time LTI support, validated
against `scipy.signal`:

- **`cont2discrete(system, dt, method)`**: convert a continuous `(A,B,C,D)`
  state-space (or a transfer function via `tf2ss`) to a discrete
  `DiscreteStateSpace(Ad,Bd,Cd,Dd,dt)`. Methods: `"zoh"` (block matrix-exponential
  formula using `expm`), `"bilinear"`/`"tustin"` (Tustin, alpha=0.5), `"euler"`/
  `"forward_diff"` (alpha=0) and `"backward_diff"` (alpha=1), matching SciPy's
  alpha-parameterized linear discretization.
- **`dstep` / `dimpulse`**: discrete step/impulse response over `n` samples by
  iterating `x[k+1]=Ad x[k]+Bd u[k]`, `y[k]=Cd x[k]+Dd u[k]`.
- **`dlsim(system, u, t)`**: simulate the discrete system for an input sequence.
- **`dfreqresp(system, w)` / `dbode(system, w)`**: discrete frequency response
  `H(e^{jw}) = Cd (zI-Ad)^{-1} Bd + Dd` at `z = e^{jw}` (w in radians/sample),
  with magnitude (dB) and unwrapped phase (degrees) for `dbode`.

## Impact

- Affected specs: **modifies** the `signal` capability (adds one requirement).
- Affected code: new `src/signal/dlti.cpp`, header additions in
  `include/scipp/signal/signal.hpp` (`DiscreteStateSpace` plus the six entry
  points), `tests/test_signal_discrete.cpp`, oracle generator, and CMake lists.
- Trims the discrete-LTI and `cont2discrete` items from the
  `add-signal-remaining` backlog.

## Non-goals

- Multi-input/multi-output `dstep`/`dimpulse` looping over every input column —
  the responses simulate a single input/output (input 0, output 0), which covers
  the SISO systems produced by `tf2ss`. MIMO looping stays in the backlog.
- `dlti`/`StateSpace` class API, `dfreqz` whole-circle defaults, and automatic
  frequency-grid generation when `w` is omitted — callers pass the frequency
  vector explicitly.
