# Tasks — optimize (Phase 4)

## 1. Module scaffold
- [ ] `include/scypp/optimize/optimize.hpp` (callable typedefs + result structs); `src/optimize/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`
- [ ] Shared finite-difference helpers `num_gradient`, `num_jacobian`

## 2. Scalar root finding
- [ ] `brentq` (Brent bracketing; invalid-bracket raises), `bisect`, `newton` (Newton + secant fallback)

## 3. Scalar minimization
- [ ] `minimize_scalar` methods `"brent"` (with bracketing) and `"bounded"`

## 4. Multivariate minimization
- [ ] `minimize` `"Nelder-Mead"` (simplex, SciPy default coefficients)
- [ ] `minimize` `"BFGS"` (inverse-Hessian update, FD gradient, Armijo line search)
- [ ] `OptimizeResult` with x/fun/success/nit/nfev; non-convergence → success=false

## 5. Least squares + roots
- [ ] `least_squares` (Levenberg–Marquardt, FD Jacobian via `numpp::linalg::solve`)
- [ ] `curve_fit` (residual = model−y; `pcov = σ̂²(JᵀJ)⁻¹`)
- [ ] `fsolve` (multivariate Newton, FD Jacobian)

## 6. Oracle + validation
- [ ] Extend `tests/oracle/generate.py` with optimize golden solutions; regenerate
- [ ] `tests/test_optimize.cpp`: roots, scalar/multivariate minima, curve_fit/least_squares/fsolve vs SciPy + identities (`f(root)≈0`, `‖grad‖≈0`)
- [ ] CPU build green; full suite green
- [ ] `openspec validate add-optimize --strict` green
- [ ] Check off Phase 4 in `bootstrap-scypp-foundation/tasks.md`; update README
