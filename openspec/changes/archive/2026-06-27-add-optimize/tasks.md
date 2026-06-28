# Tasks — optimize (Phase 4)

## 1. Module scaffold
- [x] `include/scipp/optimize/optimize.hpp` (callable typedefs + result structs); `src/optimize/*.cpp` in `src/CMakeLists.txt`; export from `scipp/scipp.hpp`
- [x] Shared finite-difference helpers `num_gradient`, `num_jacobian`

## 2. Scalar root finding
- [x] `brentq` (Brent bracketing; invalid-bracket raises), `bisect`, `newton` (Newton + secant fallback)

## 3. Scalar minimization
- [x] `minimize_scalar` methods `"brent"` (with bracketing) and `"bounded"`

## 4. Multivariate minimization
- [x] `minimize` `"Nelder-Mead"` (simplex, SciPy default coefficients)
- [x] `minimize` `"BFGS"` (inverse-Hessian update, FD gradient, Armijo line search)
- [x] `OptimizeResult` with x/fun/success/nit/nfev; non-convergence → success=false

## 5. Least squares + roots
- [x] `least_squares` (Levenberg–Marquardt, FD Jacobian via `numpp::linalg::solve`)
- [x] `curve_fit` (residual = model−y; `pcov = σ̂²(JᵀJ)⁻¹`)
- [x] `fsolve` (multivariate Newton, FD Jacobian)

## 6. Oracle + validation
- [x] Extend `tests/oracle/generate.py` with optimize golden solutions; regenerate
- [x] `tests/test_optimize.cpp`: roots, scalar/multivariate minima, curve_fit/least_squares/fsolve vs SciPy + identities (`f(root)≈0`, `‖grad‖≈0`)
- [x] CPU build green; full suite green
- [x] `openspec validate add-optimize --strict` green
- [x] Check off Phase 4 in `bootstrap-scipp-foundation/tasks.md`; update README
