// File I/O: Matrix Market, WAV (integer PCM), ARFF (numeric).
#include "scypp/io/io.hpp"

#include <cstdint>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/error.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::io {
namespace sd = scypp::linalg::detail;

// ---- Matrix Market ----
ndarray mmread(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw scypp::value_error("mmread: cannot open " + path);
  std::string line;
  std::getline(f, line);
  bool coordinate = line.find("coordinate") != std::string::npos;
  while (std::getline(f, line)) if (!line.empty() && line[0] != '%') break;  // skip comments
  std::istringstream hs(line);
  int64_t rows, cols, nnz = 0;
  hs >> rows >> cols;
  if (coordinate) hs >> nnz;
  std::vector<double> dense(rows * cols, 0.0);
  if (coordinate) {
    for (int64_t e = 0; e < nnz; ++e) {
      int64_t i, j; double v;
      f >> i >> j >> v;
      dense[(i - 1) * cols + (j - 1)] = v;  // 1-based
    }
  } else {  // array (dense, column-major)
    for (int64_t j = 0; j < cols; ++j)
      for (int64_t i = 0; i < rows; ++i) f >> dense[i * cols + j];
  }
  return sd::from_mat(dense, rows, cols);
}

void mmwrite(const std::string& path, const ndarray& a) {
  numpp::ndarray ac = a.astype(numpp::kFloat64).ascontiguousarray();
  int64_t rows = ac.shape()[0], cols = ac.shape()[1];
  const double* p = ac.typed_data<double>();
  int64_t nnz = 0;
  for (int64_t i = 0; i < rows * cols; ++i) if (p[i] != 0.0) ++nnz;
  std::ofstream f(path);
  f << "%%MatrixMarket matrix coordinate real general\n";
  f << rows << " " << cols << " " << nnz << "\n";
  f.precision(16);
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j)
      if (p[i * cols + j] != 0.0) f << (i + 1) << " " << (j + 1) << " " << p[i * cols + j] << "\n";
}

// ---- WAV ----
namespace {
uint32_t rd32(std::ifstream& f) { unsigned char b[4]; f.read(reinterpret_cast<char*>(b), 4); return b[0] | b[1] << 8 | b[2] << 16 | uint32_t(b[3]) << 24; }
uint16_t rd16(std::ifstream& f) { unsigned char b[2]; f.read(reinterpret_cast<char*>(b), 2); return b[0] | b[1] << 8; }
void wr32(std::ofstream& f, uint32_t v) { unsigned char b[4] = {(unsigned char)(v), (unsigned char)(v >> 8), (unsigned char)(v >> 16), (unsigned char)(v >> 24)}; f.write(reinterpret_cast<char*>(b), 4); }
void wr16(std::ofstream& f, uint16_t v) { unsigned char b[2] = {(unsigned char)(v), (unsigned char)(v >> 8)}; f.write(reinterpret_cast<char*>(b), 2); }
}  // namespace

WavData wavread(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw scypp::value_error("wavread: cannot open " + path);
  char tag[4];
  f.read(tag, 4);  // RIFF
  rd32(f);
  f.read(tag, 4);  // WAVE
  int channels = 1, bits = 16, rate = 0;
  std::vector<double> samples;
  while (f.read(tag, 4)) {
    uint32_t sz = rd32(f);
    if (std::strncmp(tag, "fmt ", 4) == 0) {
      rd16(f);  // audio format
      channels = rd16(f);
      rate = static_cast<int>(rd32(f));
      rd32(f); rd16(f);  // byte rate, block align
      bits = rd16(f);
      for (uint32_t r = 16; r < sz; ++r) f.get();  // skip extra fmt bytes
    } else if (std::strncmp(tag, "data", 4) == 0) {
      int bytes = bits / 8;
      int64_t count = sz / bytes;
      samples.resize(count);
      for (int64_t i = 0; i < count; ++i) {
        if (bits == 16) { int16_t v = static_cast<int16_t>(rd16(f)); samples[i] = v; }
        else { int32_t v = static_cast<int32_t>(rd32(f)); samples[i] = v; }
      }
    } else {
      for (uint32_t r = 0; r < sz; ++r) f.get();  // skip unknown chunk
    }
  }
  WavData out;
  out.rate = rate;
  if (channels > 1) {
    int64_t n = static_cast<int64_t>(samples.size()) / channels;
    out.data = sd::from_mat(samples, n, channels);
  } else {
    out.data = sd::from_vec(samples);
  }
  return out;
}

void wavwrite(const std::string& path, int rate, const ndarray& data) {
  numpp::ndarray dc = data.astype(numpp::kFloat64).ascontiguousarray();
  int channels = dc.ndim() == 1 ? 1 : static_cast<int>(dc.shape()[1]);
  int64_t total = dc.size();
  const double* p = dc.typed_data<double>();
  int bits = 16, byte_rate = rate * channels * bits / 8, data_bytes = static_cast<int>(total * bits / 8);
  std::ofstream f(path, std::ios::binary);
  f.write("RIFF", 4); wr32(f, 36 + data_bytes); f.write("WAVE", 4);
  f.write("fmt ", 4); wr32(f, 16); wr16(f, 1); wr16(f, channels); wr32(f, rate);
  wr32(f, byte_rate); wr16(f, channels * bits / 8); wr16(f, bits);
  f.write("data", 4); wr32(f, data_bytes);
  for (int64_t i = 0; i < total; ++i) wr16(f, static_cast<uint16_t>(static_cast<int16_t>(p[i])));
}

// ---- ARFF ----
ndarray loadarff(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw scypp::value_error("loadarff: cannot open " + path);
  std::string line;
  int nattr = 0;
  bool in_data = false;
  std::vector<double> rows;
  int ncol = 0;
  while (std::getline(f, line)) {
    std::string lower = line;
    for (char& c : lower) c = std::tolower(static_cast<unsigned char>(c));
    if (!in_data) {
      if (lower.rfind("@attribute", 0) == 0) ++nattr;
      else if (lower.rfind("@data", 0) == 0) { in_data = true; ncol = nattr; }
      continue;
    }
    if (line.empty() || line[0] == '%') continue;
    std::istringstream ls(line);
    std::string tok;
    int got = 0;
    while (std::getline(ls, tok, ',')) { rows.push_back(std::stod(tok)); ++got; }
    (void)got;
  }
  int64_t nrows = ncol ? static_cast<int64_t>(rows.size()) / ncol : 0;
  return sd::from_mat(rows, nrows, ncol);
}

}  // namespace scypp::io
