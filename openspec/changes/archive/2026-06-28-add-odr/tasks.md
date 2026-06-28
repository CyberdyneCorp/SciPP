# Tasks — odr

- [x] `scipp::odr::Model`, `Data`, `ODR` driver (orthogonal-distance least squares)
      — `include/scipp/odr/odr.hpp`, `src/odr/odr.cpp`, wired into `src/CMakeLists.txt`
      and the `scipp.hpp` umbrella.
- [x] estimated params, std errors, parameter covariance, residual variance, and
      sum-of-squares outputs (`Output`).
- [x] augmented `[beta, delta]` Levenberg-Marquardt fit reusing
      `scipp::optimize::least_squares`; equal-weight TLS and per-point sx/sy weighting.
- [x] oracle tests vs `scipy.odr` on linear, nonlinear, and weighted models
      (`tests/test_odr.cpp`, golden block in `tests/oracle/generate.py`).
