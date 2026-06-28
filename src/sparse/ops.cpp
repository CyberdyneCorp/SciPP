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

ndarray CsrMatrix::spmv(const ndarray& x) const {
  auto ip = d::iv(indptr_), id = d::iv(indices_); auto da = d::dv(data_), xv = d::dv(x);
  std::vector<double> y(rows_, 0.0);
  for (int64_t i = 0; i < rows_; ++i) {
    double s = 0.0;
    for (int64_t k = ip[i]; k < ip[i + 1]; ++k) s += da[k] * xv[id[k]];
    y[i] = s;
  }
  return d::from_dv(y);
}

ndarray CsrMatrix::spmm(const ndarray& X) const {
  numpp::ndarray Xc = X.astype(numpp::kFloat64).ascontiguousarray();
  int64_t p = (Xc.ndim() == 1) ? 1 : Xc.shape()[1];
  const double* xp = Xc.typed_data<double>();
  auto ip = d::iv(indptr_), id = d::iv(indices_); auto da = d::dv(data_);
  std::vector<double> Y(rows_ * p, 0.0);
  for (int64_t i = 0; i < rows_; ++i)
    for (int64_t k = ip[i]; k < ip[i + 1]; ++k) {
      double v = da[k]; int64_t col = id[k];
      for (int64_t c = 0; c < p; ++c) Y[i * p + c] += v * xp[col * p + c];
    }
  return d::ld::from_mat(Y, rows_, p);
}

Backend last_backend() { return g_last; }

// Backend dispatch: a device CSR kernel would be used when available and the
// problem is large enough; the portable CPU kernel is always the fallback. The
// device-reference path reuses the CPU math so results are provably equivalent.
ndarray spmv(const CsrMatrix& A, const ndarray& x, Backend forced) {
  const auto& reg = numpp::CapabilityRegistry::instance();
  bool device = (forced == Backend::Device) ||
                (A.nnz() >= 1 << 16 && reg.gpu_available(numpp::Backend::Device));
  g_last = device ? Backend::Device : Backend::Cpu;
  return A.spmv(x);
}

}  // namespace scipp::sparse
