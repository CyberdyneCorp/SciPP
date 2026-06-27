#pragma once
// scypp::linalg — port of scipy.linalg (Phase 2 subset). Built on numpp::ndarray;
// standard decompositions delegate to numpp::linalg with SciPy conventions, and
// the SciPy-only routines (lu/expm/polar/special matrices) are added here.

#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"

namespace scypp::linalg {

using numpp::ndarray;

// ---- result structs (SciPy-shaped) ----
struct LUResult { ndarray p, l, u; };          // A = p @ l @ u
struct LUFactor { ndarray lu; ndarray piv; };  // packed LU + 0-based row pivots
struct QRResult { ndarray q, r; };
struct SVDResult { ndarray u, s, vh; };
struct EigResult { ndarray eigenvalues, eigenvectors; };
struct EighResult { ndarray eigenvalues, eigenvectors; };
struct CholFactor { ndarray c; bool lower; };
struct PolarResult { ndarray u, p; };
struct LstsqResult { ndarray x; ndarray residues; ndarray rank; ndarray s; };

// ---- basic ----
ndarray inv(const ndarray& a);
double  det(const ndarray& a);
ndarray solve(const ndarray& a, const ndarray& b);
LstsqResult lstsq(const ndarray& a, const ndarray& b);
ndarray pinv(const ndarray& a);
ndarray pinvh(const ndarray& a);
double  norm(const ndarray& a);                       // 2-norm / Frobenius
ndarray norm(const ndarray& a, const std::string& ord);

// ---- decompositions ----
LUResult lu(const ndarray& a);
LUFactor lu_factor(const ndarray& a);
ndarray  lu_solve(const LUFactor& f, const ndarray& b);
QRResult qr(const ndarray& a, const std::string& mode = "economic");
SVDResult svd(const ndarray& a, bool full_matrices = true);
ndarray  svdvals(const ndarray& a);
ndarray  cholesky(const ndarray& a, bool lower = false);   // SciPy: upper default
CholFactor cho_factor(const ndarray& a, bool lower = false);
ndarray  cho_solve(const CholFactor& f, const ndarray& b);

// ---- eigenvalue problems ----
EigResult  eig(const ndarray& a);
ndarray    eigvals(const ndarray& a);
EighResult eigh(const ndarray& a);
ndarray    eigvalsh(const ndarray& a);

// ---- matrix functions ----
ndarray     expm(const ndarray& a);
PolarResult polar(const ndarray& a, const std::string& side = "right");

// ---- special matrices ----
ndarray toeplitz(const ndarray& c);
ndarray toeplitz(const ndarray& c, const ndarray& r);
ndarray circulant(const ndarray& c);
ndarray hankel(const ndarray& c);
ndarray hankel(const ndarray& c, const ndarray& r);
ndarray tri(int64_t n, int64_t m = -1, int64_t k = 0);
ndarray block_diag(const std::vector<ndarray>& blocks);
ndarray companion(const ndarray& a);
ndarray leslie(const ndarray& f, const ndarray& s);
ndarray kron(const ndarray& a, const ndarray& b);
ndarray hilbert(int64_t n);
ndarray hadamard(int64_t n);
ndarray pascal(int64_t n, const std::string& kind = "symmetric");

}  // namespace scypp::linalg
