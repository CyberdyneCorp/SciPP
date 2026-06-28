// Sparse formats: COO/CSR/CSC, conversions, transpose, diagonal, arithmetic,
// and constructors (eye/identity/diags).
#include "scipp/sparse/sparse.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scipp/sparse/detail.hpp"

namespace scipp::sparse {
namespace d = detail;

CsrMatrix::CsrMatrix(ndarray data, ndarray indices, ndarray indptr, int64_t rows, int64_t cols)
    : data_(std::move(data)), indices_(std::move(indices)), indptr_(std::move(indptr)),
      rows_(rows), cols_(cols) {}

CsrMatrix CsrMatrix::from_coo(const CooMatrix& coo) {
  auto rr = d::iv(coo.row), cc = d::iv(coo.col);
  auto vv = d::dv(coo.data);
  int64_t n = coo.rows, m = coo.cols, nnz = static_cast<int64_t>(vv.size());
  std::vector<int64_t> indptr(n + 1, 0);
  for (int64_t k = 0; k < nnz; ++k) indptr[rr[k] + 1]++;
  for (int64_t i = 0; i < n; ++i) indptr[i + 1] += indptr[i];
  std::vector<int64_t> ind(nnz);
  std::vector<double> dat(nnz);
  std::vector<int64_t> next = indptr;
  for (int64_t k = 0; k < nnz; ++k) { int64_t r = rr[k], dst = next[r]++; ind[dst] = cc[k]; dat[dst] = vv[k]; }
  // canonicalize: sort each row by column and sum duplicates
  std::vector<int64_t> oind; std::vector<double> odat; std::vector<int64_t> optr(n + 1, 0);
  for (int64_t i = 0; i < n; ++i) {
    int64_t s = indptr[i], e = indptr[i + 1];
    std::vector<int64_t> order(e - s);
    std::iota(order.begin(), order.end(), s);
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return ind[a] < ind[b]; });
    int64_t last = -1;
    for (int64_t o : order) {
      if (ind[o] == last) odat.back() += dat[o];
      else { oind.push_back(ind[o]); odat.push_back(dat[o]); last = ind[o]; }
    }
    optr[i + 1] = static_cast<int64_t>(oind.size());
  }
  return CsrMatrix(d::from_dv(odat), d::from_iv(oind), d::from_iv(optr), n, m);
}

CsrMatrix CsrMatrix::from_dense(const ndarray& a) {
  numpp::ndarray ac = a.astype(numpp::kFloat64).ascontiguousarray();
  int64_t n = ac.shape()[0], m = ac.shape()[1];
  const double* p = ac.typed_data<double>();
  std::vector<int64_t> row, col; std::vector<double> val;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < m; ++j)
      if (p[i * m + j] != 0.0) { row.push_back(i); col.push_back(j); val.push_back(p[i * m + j]); }
  return from_coo({d::from_dv(val), d::from_iv(row), d::from_iv(col), n, m});
}

ndarray CsrMatrix::toarray() const {
  auto ip = d::iv(indptr_), id = d::iv(indices_); auto da = d::dv(data_);
  std::vector<double> dense(rows_ * cols_, 0.0);
  for (int64_t i = 0; i < rows_; ++i)
    for (int64_t k = ip[i]; k < ip[i + 1]; ++k) dense[i * cols_ + id[k]] += da[k];
  return d::ld::from_mat(dense, rows_, cols_);
}

int64_t CsrMatrix::nnz() const { return data_.size(); }

CscMatrix CsrMatrix::transpose() const {
  // CSR of A is exactly CSC of Aᵀ with the same arrays.
  return CscMatrix(data_, indices_, indptr_, cols_, rows_);
}

ndarray CsrMatrix::diagonal() const {
  auto ip = d::iv(indptr_), id = d::iv(indices_); auto da = d::dv(data_);
  int64_t n = std::min(rows_, cols_);
  std::vector<double> diag(n, 0.0);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t k = ip[i]; k < ip[i + 1]; ++k) if (id[k] == i) diag[i] = da[k];
  return d::from_dv(diag);
}

CsrMatrix CsrMatrix::scaled(double s) const {
  auto da = d::dv(data_);
  for (double& v : da) v *= s;
  return CsrMatrix(d::from_dv(da), indices_, indptr_, rows_, cols_);
}

CsrMatrix CsrMatrix::add(const CsrMatrix& b) const {
  // concatenate triplets and re-canonicalize (duplicates summed).
  auto aip = d::iv(indptr_), aid = d::iv(indices_); auto ada = d::dv(data_);
  auto bip = d::iv(b.indptr_), bid = d::iv(b.indices_); auto bda = d::dv(b.data_);
  std::vector<int64_t> row, col; std::vector<double> val;
  for (int64_t i = 0; i < rows_; ++i) for (int64_t k = aip[i]; k < aip[i + 1]; ++k) { row.push_back(i); col.push_back(aid[k]); val.push_back(ada[k]); }
  for (int64_t i = 0; i < b.rows_; ++i) for (int64_t k = bip[i]; k < bip[i + 1]; ++k) { row.push_back(i); col.push_back(bid[k]); val.push_back(bda[k]); }
  return from_coo({d::from_dv(val), d::from_iv(row), d::from_iv(col), rows_, cols_});
}

// ---- CSC ----
CscMatrix::CscMatrix(ndarray data, ndarray indices, ndarray indptr, int64_t rows, int64_t cols)
    : data_(std::move(data)), indices_(std::move(indices)), indptr_(std::move(indptr)),
      rows_(rows), cols_(cols) {}

int64_t CscMatrix::nnz() const { return data_.size(); }

ndarray CscMatrix::toarray() const {
  auto ip = d::iv(indptr_), id = d::iv(indices_); auto da = d::dv(data_);
  std::vector<double> dense(rows_ * cols_, 0.0);
  for (int64_t j = 0; j < cols_; ++j)
    for (int64_t k = ip[j]; k < ip[j + 1]; ++k) dense[id[k] * cols_ + j] += da[k];
  return d::ld::from_mat(dense, rows_, cols_);
}

CsrMatrix CscMatrix::tocsr() const { return CsrMatrix::from_dense(toarray()); }

// ---- constructors ----
CsrMatrix identity(int64_t n) {
  std::vector<double> data(n, 1.0);
  std::vector<int64_t> ind(n), ptr(n + 1);
  for (int64_t i = 0; i < n; ++i) { ind[i] = i; ptr[i] = i; }
  ptr[n] = n;
  return CsrMatrix(d::from_dv(data), d::from_iv(ind), d::from_iv(ptr), n, n);
}
CsrMatrix eye(int64_t n) { return identity(n); }

CsrMatrix diags(const std::vector<ndarray>& diagonals, const std::vector<int>& offsets, int64_t n) {
  std::vector<int64_t> row, col; std::vector<double> val;
  for (size_t d_ = 0; d_ < diagonals.size(); ++d_) {
    auto vals = d::dv(diagonals[d_]);
    int off = offsets[d_];
    for (size_t i = 0; i < vals.size(); ++i) {
      int64_t r = (off >= 0) ? static_cast<int64_t>(i) : static_cast<int64_t>(i) - off;
      int64_t c = (off >= 0) ? static_cast<int64_t>(i) + off : static_cast<int64_t>(i);
      if (r < n && c < n) { row.push_back(r); col.push_back(c); val.push_back(vals[i]); }
    }
  }
  return CsrMatrix::from_coo({d::from_dv(val), d::from_iv(row), d::from_iv(col), n, n});
}

}  // namespace scipp::sparse
