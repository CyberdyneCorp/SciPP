# Tasks — nested / extended quadrature

- [x] `romberg` (Richardson-extrapolated trapezoid) in `src/integrate/nested_quad.cpp`
- [x] `quad_vec` (componentwise adaptive `quad`)
- [x] `dblquad` / `tplquad` (nested `quad` with variable bounds)
- [x] `nquad` (recursive fixed-bound hyper-rectangle)
- [x] type aliases + decls in `include/scypp/integrate/integrate.hpp`; wire into build
- [x] extend oracle generator; regenerate
- [x] `tests/test_integrate_nested.cpp` vs SciPy
- [x] full suite green; `openspec validate add-integrate-nested-quad --strict`
- [x] trim multidim-quadrature items from `add-integrate-stiff-bvp` backlog
