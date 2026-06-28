# Tasks — odr

- [x] `scypp::odr::Model`, `Data`, `ODR` driver (orthogonal-distance least squares)
      — `include/scypp/odr/odr.hpp`, `src/odr/odr.cpp`, wired into `src/CMakeLists.txt`
      and the `scypp.hpp` umbrella.
- [x] estimated params, std errors, parameter covariance, residual variance, and
      sum-of-squares outputs (`Output`).
- [x] augmented `[beta, delta]` Levenberg-Marquardt fit reusing
      `scypp::optimize::least_squares`; equal-weight TLS and per-point sx/sy weighting.
- [x] oracle tests vs `scipy.odr` on linear, nonlinear, and weighted models
      (`tests/test_odr.cpp`, golden block in `tests/oracle/generate.py`).
