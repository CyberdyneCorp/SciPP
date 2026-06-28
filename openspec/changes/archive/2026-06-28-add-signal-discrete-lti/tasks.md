# Tasks — discrete-time LTI

- [x] `DiscreteStateSpace` struct + entry points in `include/scipp/signal/signal.hpp`
- [x] `cont2discrete` (zoh via expm block formula; bilinear/euler/backward_diff via alpha formula) in `src/signal/dlti.cpp`
- [x] `dstep`/`dimpulse`/`dlsim` discrete state recursion
- [x] `dfreqresp` (complex `(zI-Ad)^-1` solve) and `dbode` (dB magnitude + unwrapped phase)
- [x] wire `signal/dlti.cpp` into `src/CMakeLists.txt`
- [x] extend oracle generator (zoh/bilinear/euler matrices, step/impulse/lsim sequences, freqresp/bode); regenerate
- [x] `tests/test_signal_discrete.cpp` vs scipy; register in `tests/CMakeLists.txt`
- [x] full suite green; `openspec validate add-signal-discrete-lti --strict`
- [x] trim discrete-LTI and `cont2discrete` items from `add-signal-remaining` backlog
