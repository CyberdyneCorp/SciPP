// Eigenvalue problems — delegate to numpp::linalg.
#include "scypp/linalg/linalg.hpp"

#include "numpp/linalg/linalg.hpp"

namespace scypp::linalg {

EigResult eig(const ndarray& a) {
  numpp::linalg::EigResult r = numpp::linalg::eig(a);
  return {r.eigenvalues, r.eigenvectors};
}

ndarray eigvals(const ndarray& a) { return numpp::linalg::eigvals(a); }

EighResult eigh(const ndarray& a) {
  numpp::linalg::EighResult r = numpp::linalg::eigh(a);
  return {r.eigenvalues, r.eigenvectors};
}

ndarray eigvalsh(const ndarray& a) { return numpp::linalg::eigvalsh(a); }

}  // namespace scypp::linalg
