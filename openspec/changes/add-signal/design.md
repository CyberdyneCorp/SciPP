# Design — signal (Phase 8)

## Context

`scypp::signal` ports the deterministic DSP core of `scipy.signal` over
`numpp::ndarray`, reusing `scypp::fft`/`numpp::fft` for the FFT-based routines.
The intricate parts are the IIR filter design (analog prototype → frequency
transform → bilinear) and `filtfilt`/`welch`, whose exact SciPy behavior is
reproduced.

## Convolution / windows / waveforms

- **convolve/correlate** — direct sums with `full`/`same`/`valid` cropping
  (correlate = convolve with the reversed, conjugated kernel). **fftconvolve**
  zero-pads to `next_fast_len`, multiplies spectra, inverse-transforms, crops.
- **windows** — closed-form coefficient formulas matching SciPy (symmetric vs
  periodic via `fftbins`); `get_window` dispatches by name. Kaiser uses `i0`
  (from Phase 1 `special`); tukey/flattop use SciPy's exact coefficients.
- **waveforms** — `chirp` (linear/quadratic/log/hyperbolic phase), `sawtooth`,
  `square` (with duty), `unit_impulse`.

## Time-domain filtering

- **lfilter** — Direct Form II transposed difference equation with the `a`/`b`
  coefficients normalized by `a[0]`; supports initial conditions `zi`.
- **lfilter_zi** — steady-state initial condition from the companion linear solve
  (`numpp::linalg::solve`), as SciPy computes it.
- **filtfilt** — zero-phase: odd-extension padding of length
  `3·max(len(a),len(b))` (SciPy default `padtype="odd"`), seed with
  `lfilter_zi·x[0]`, forward filter, reverse, filter again, reverse, unpad.
- **sosfilt** — cascade of second-order sections, each a DF2T biquad.
- **hilbert** — analytic signal via FFT (zero the negative frequencies, double the
  positive), matching `scipy.signal.hilbert`.
- **detrend** — `constant` (subtract mean) and `linear` (least-squares line).
- **freqz** — evaluate `B(e^{jω})/A(e^{jω})` at `worN` frequencies.

## Filter design

IIR (`butter`/`cheby1`/`cheby2`):
1. Analog lowpass **prototype** zeros/poles/gain (`buttap`/`cheb1ap`/`cheb2ap`).
2. **Frequency transform** to the requested band: `lp2lp_zpk`, `lp2hp_zpk`,
   `lp2bp_zpk`, `lp2bs_zpk`, with pre-warped critical frequencies.
3. **bilinear_zpk** (digital), then `zpk2tf` (→ `b,a`) or `zpk2sos`.
Frequencies are normalized to Nyquist (`Wn` in `[0,1]`) as in SciPy. Conversions
`tf2zpk`/`zpk2tf`/`zpk2sos`/`tf2sos` are standalone.

FIR (`firwin`): windowed-sinc — ideal band response sampled, multiplied by the
window, normalized at the band center; supports `lowpass`/`highpass`/`bandpass`/
`bandstop` via the `pass_zero`/cutoff convention.

## Spectral analysis

- **periodogram** — windowed FFT magnitude², scaled to a power-spectral-density
  (`scaling="density"`, dividing by `fs·Σw²`) or spectrum, one-sided for real input.
- **welch** — split into `nperseg` overlapping segments (`noverlap=nperseg/2`
  default), detrend each (`constant`), window, FFT, average the periodograms, with
  SciPy's density scaling and one-sided folding. Defaults match SciPy
  (`window="hann"`, `nperseg=256` capped at signal length).

## Oracle strategy

Every routine is compared `allclose` to `scipy.signal` on representative inputs:
convolution across modes; each window; `butter`/`cheby` coefficients (`b,a` and
`sos`) and `freqz` response; `lfilter`/`filtfilt`/`sosfilt` outputs; `firwin` taps;
`welch`/`periodogram` frequencies and PSD; waveform samples. Real-valued, so
tolerances are tight (`~1e-9`), looser where FFT or design accumulates (`~1e-7`).

## Open questions

- `welch`/`filtfilt` have many parameters; this change implements the common
  defaults (and the parameters needed to reach them) and validates those. The
  remaining knobs (`scaling="spectrum"`, alternate `padtype`, `average="median"`)
  are follow-ups.
