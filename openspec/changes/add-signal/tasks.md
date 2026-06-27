# Tasks — signal (Phase 8)

## 1. Module scaffold
- [x] `include/scypp/signal/signal.hpp` (result structs + decls); `src/signal/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`

## 2. Convolution + waveforms + windows
- [x] `convolve`, `correlate`, `fftconvolve` (full/same/valid)
- [x] windows: `boxcar`, `hann`, `hamming`, `blackman`, `bartlett`, `blackmanharris`, `flattop`, `kaiser`, `tukey`, `get_window`
- [x] waveforms: `chirp`, `sawtooth`, `square`, `unit_impulse`

## 3. Time-domain filtering
- [x] `lfilter` (DF2T), `lfilter_zi`, `sosfilt`
- [x] `filtfilt` (odd padding, zero-phase)
- [x] `detrend` (constant/linear), `hilbert` (analytic via FFT), `freqz`

## 4. Filter design
- [x] analog prototypes `buttap`/`cheb1ap`/`cheb2ap`; transforms `lp2lp`/`lp2hp`/`lp2bp`/`lp2bs`; `bilinear_zpk`
- [x] `butter`, `cheby1`, `cheby2` (lp/hp/bp/bs); `firwin`
- [x] `tf2zpk`, `zpk2tf`, `zpk2sos`, `tf2sos`

## 5. Spectral
- [x] `periodogram`, `welch` (density scaling, one-sided, Hann default)

## 6. Oracle + validation
- [x] Extend `tests/oracle/generate.py` with signal golden values; regenerate
- [x] `tests/test_signal.cpp`: convolution/windows/waveforms; lfilter/filtfilt/sosfilt; butter/cheby/firwin + freqz; welch/periodogram
- [x] CPU build green; full suite green
- [x] `openspec validate add-signal --strict` green
- [x] Check off Phase 8 in `bootstrap-scypp-foundation/tasks.md`; update README
