// Sparse products (spmv/spmm) and the backend-dispatch hook.
#include "scipp/sparse/sparse.hpp"

#include <vector>

#include "numpp/backend/backend.hpp"
#include "numpp/core/dtype.hpp"
#include "scipp/sparse/detail.hpp"

namespace scipp::sparse {
namespace d = detail;
namespace {
thread_local Backend g_last = Backend::Cpu;
}

// Map the backend NumPP actually used onto the sparse module's public enum: any
// device-class backend collapses to Device, everything else to Cpu.
namespace {
Backend map_backend(numpp::Backend b) {
  return (b == numpp::Backend::Cpu || b == numpp::Backend::Auto) ? Backend::Cpu
                                                                 : Backend::Device;
}
}  // namespace

ndarray CsrMatrix::spmv(const ndarray& x) const {
  // Delegate the SpMV to NumPP's tiered kernel: it owns device dispatch and the
  // portable CPU fallback, so the result is identical to the old hand-rolled loop.
  return numpp::csr_spmv(indptr_, indices_, data_, x);
}

ndarray CsrMatrix::spmm(const ndarray& X) const {
  numpp::ndarray Xc = X.astype(numpp::kFloat64).ascontiguousarray();
  int64_t p = (Xc.ndim() == 1) ? 1 : Xc.shape()[1];
  const double* xp = Xc.typed_data<double>();
  std::vector<double> Y(rows_ * p, 0.0);
  // SpMM = one SpMV per RHS column, each offloaded through the NumPP kernel.
  numpp::ndarray xcol(numpp::Shape{cols_}, numpp::kFloat64);
  double* cp = xcol.typed_data<double>();
  for (int64_t c = 0; c < p; ++c) {
    for (int64_t r = 0; r < cols_; ++r) cp[r] = xp[r * p + c];
    numpp::ndarray ycol =
        numpp::csr_spmv(indptr_, indices_, data_, xcol).astype(numpp::kFloat64).ascontiguousarray();
    const double* ycp = ycol.typed_data<double>();
    for (int64_t i = 0; i < rows_; ++i) Y[i * p + c] = ycp[i];
  }
  return d::ld::from_mat(Y, rows_, p);
}

Backend last_backend() { return g_last; }

// Backend dispatch: NumPP's csr_spmv auto-selects a device backend above its
// size threshold and always falls back to the portable CPU kernel, so the result
// is provably equivalent. A `Cpu` request pins the CPU kernel; any other request
// lets NumPP choose. last_backend() then reflects NumPP's actual choice (CPU
// locally where NumPP is CPU-only; Device where a NumPP GPU variant is present).
ndarray spmv(const CsrMatrix& A, const ndarray& x, Backend forced) {
  numpp::Backend nb =
      (forced == Backend::Cpu) ? numpp::Backend::Cpu : numpp::Backend::Auto;
  numpp::ndarray y = numpp::csr_spmv(A.indptr(), A.indices(), A.data(), x, nb);
  g_last = map_backend(numpp::last_backend());
  return y;
}

}  // namespace scipp::sparse
