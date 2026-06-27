# Design — optimize (Phase 4)

## Context

`scypp::optimize` ports the unconstrained core of `scipy.optimize`. Unlike the
earlier phases it implements real iterative algorithms rather than wrapping
NumPP. Callables are passed as `std::function`; vectors are `numpp::ndarray`.
Linear systems inside Levenberg–Marquardt and Newton root finding use
`numpp::linalg::solve`.

## API shape

```cpp
namespace scypp::optimize {
using ScalarFn = std::function<double(double)>;
using ObjFn    = std::function<double(const ndarray&)>;     // Rⁿ → R
using VecFn    = std::function<ndarray(const ndarray&)>;     // Rⁿ → Rᵐ

struct OptimizeResult { ndarray x; double fun; bool success; int nit; int nfev; std::string message; };
struct RootResult     { double root; bool converged; int iterations; int function_calls; };
struct ScalarMinResult{ double x; double fun; bool success; int nit; };
struct LeastSquaresResult { ndarray x; ndarray fun; double cost; bool success; int nfev; };
struct CurveFitResult { ndarray popt; ndarray pcov; };
}
```

`OptimizeResult` mirrors SciPy's result object (`.x`, `.fun`, `.success`, `.nit`,
`.nfev`, `.message`); non-convergence is reported through `success=false`, not an
exception (matching SciPy).

## Algorithms

- **brentq** — Brent's method on a sign-changing bracket `[a,b]`: inverse
  quadratic interpolation with bisection fallback; converges to `xtol + rtol·|x|`.
  Raises `value_error` if `f(a)·f(b) > 0`.
- **bisect** — straight bisection on a bracket.
- **newton** — Newton–Raphson when a derivative is supplied, else the secant
  method from `x0` (and a second nearby point); matches SciPy's `newton`.
- **minimize_scalar** — `"brent"` (parabolic interpolation + golden-section
  safeguard, with an initial bracketing step) and `"bounded"` (Brent on a closed
  interval).
- **minimize / Nelder-Mead** — the standard reflection/expansion/
  contraction/shrink simplex with SciPy's default coefficients and the same
  initial-simplex construction (`x0` perturbed by 5%, or 0.00025 for zero
  components), terminating on the `xatol`/`fatol` simplex spread.
- **minimize / BFGS** — quasi-Newton with an inverse-Hessian update, a
  finite-difference gradient (forward differences, SciPy's default step), and an
  Armijo/backtracking line search; terminates on `‖grad‖∞ ≤ gtol`.
- **least_squares** — Levenberg–Marquardt: forward-difference Jacobian `J`, solve
  `(JᵀJ + λ diag(JᵀJ)) δ = −Jᵀr`, adapt `λ` on accept/reject; terminates on small
  step or gradient. Returns residual vector, `cost = ½‖r‖²`.
- **curve_fit** — minimizes `model(x, p) − y` via `least_squares`; parameter
  covariance `pcov = σ̂² (JᵀJ)⁻¹` with `σ̂²` from the residual variance.
- **fsolve** — multivariate Newton: forward-difference Jacobian, `solve(J, −f)`
  step, until `‖f‖∞` is small.

## Finite differences

A shared `num_gradient(f, x)` (forward differences, step `√εmach · max(|xᵢ|,1)`)
and `num_jacobian(F, x)` back the gradient/Jacobian-free entry points, matching
SciPy's default numerical-derivative behavior closely enough to agree on the
optimum.

## Oracle strategy

Optimizers reach the same optimum by different iterate paths, so tests assert the
**solution** (`x*`, `f*` / root) is `allclose` to SciPy on shared, well-posed
problems (Rosenbrock, quadratics, `sin`, exponential-decay fitting), with looser
tolerances (`~1e-5`) than the closed-form phases. Round-trip/identity checks
(e.g. `f(root) ≈ 0`, gradient ≈ 0 at the BFGS optimum) supplement the oracle.

## Open questions

- BFGS line search and FD-gradient details differ slightly from SciPy's; tests
  compare the located optimum, not iteration counts. If a problem proves
  path-sensitive, it is swapped for a better-conditioned one rather than loosening
  tolerance excessively.
