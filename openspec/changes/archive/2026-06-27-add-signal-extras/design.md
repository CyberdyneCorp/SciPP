# Design — signal extras

## Spectral (csd / coherence / spectrogram / stft / istft)

Generalize the Phase-8 `_spectral` segment loop into a helper that returns the
per-segment FFT matrix. `csd(x,y)` cross-multiplies the two segment spectra and
averages (complex); `coherence = |Pxy|² / (Pxx·Pyy)`. `spectrogram` returns the
per-segment PSD matrix `Sxx` plus segment times. `stft` returns the complex
short-time transform with SciPy's `boundary="zeros"` padding and
`scaling`/normalization; `istft` inverts via weighted overlap-add. Defaults match
SciPy (`window="hann"`, `nperseg`, `noverlap=nperseg//2`/`nperseg*3//4` for stft).

## Peak analysis

`find_peaks` locates local maxima with SciPy's plateau handling (a flat top's peak
is its midpoint), then applies the optional `height`, `distance` (keep highest
within distance), `prominence`, and `width` filters. `peak_prominences` walks left
and right to the higher enclosing valley; `peak_widths` measures at a relative
height between the peak and its prominence base — both ported from SciPy's
algorithms.

## LTI systems

`TransferFunction{num, den}` (continuous). `freqresp` evaluates `H(jω) =
polyval(num, jω)/polyval(den, jω)`; `bode` returns magnitude (dB) and phase
(deg). `tf2ss` builds the controllable canonical state space `(A,B,C,D)`. `lsim`
simulates by **zero-order-hold discretization**: `Ad = expm(A·dt)`,
`Bd = A⁻¹(Ad−I)B` (using `scipp::linalg::expm`/`solve`), then iterates the state and
outputs `y = C·x + D·u`. `step`/`impulse` are `lsim` with the step / impulse input.

## Resampling

- `resample(x, num)` — FFT, resize the spectrum to `num` (splitting the Nyquist
  bin), inverse FFT, scale by `num/N`.
- `upfirdn(h, x, up, down)` — insert `up−1` zeros, FIR-filter by `h`, take every
  `down`-th sample.
- `resample_poly(x, up, down)` — `firwin` anti-alias filter at `1/max(up,down)`
  (Kaiser, SciPy's default length), then `upfirdn` with the documented phase trim.
- `decimate(x, q)` — SciPy default: 8th-order `cheby1(8, 0.05, 0.8/q)` then
  `filtfilt`, then `x[::q]`.

## ellip / bessel design

- `ellipap(N, rp, rs)` — elliptic prototype: complete elliptic integral `K`
  (AGM) and Jacobi functions `sn/cn/dn` (descending Landen), the elliptic degree
  equation for the modulus, then the zeros/poles from the inverse Jacobi map and
  gain normalization — matching SciPy's `ellipap`.
- `besselap(N)` — reverse Bessel polynomial roots (companion eigenvalues of the
  Bessel polynomial) with SciPy's default `norm="phase"` delay normalization.
- `ellip`/`bessel` feed these prototypes through the Phase-8 frequency
  transform + bilinear pipeline.

## Other filtering

- `savgol_coeffs(window, polyorder)` — least-squares polynomial filter via the
  Vandermonde pseudo-inverse (`scipp::linalg::lstsq`); `savgol_filter` convolves
  with `mode="interp"` end handling (fit the boundary polynomial).
- `medfilt(x, k)` — sliding-window median (zero-padded edges).
- `wiener(x, size)` — local mean/variance adaptive filter.
- `convolve2d`/`correlate2d` — direct 2-D convolution with `full`/`same`/`valid`
  modes and `fill`/`wrap`/`symm` boundaries.

## Oracle strategy

Every routine compared `allclose` to `scipy.signal` on representative inputs:
spectral matrices, peak indices/properties, LTI responses, resampled signals,
`ellip`/`bessel` coefficients, and the 2-D/median/SG outputs. Tolerances tight
(`~1e-7`), looser where FFT resampling, ZOH simulation, or elliptic-function
iteration accumulates (`~1e-6`).

## Open questions

- `ellipap` is the hardest to match bit-for-bit (Jacobi-function iteration); if
  wing accuracy lags SciPy, the test compares the realized frequency response
  rather than raw coefficients, and the limitation is documented.
