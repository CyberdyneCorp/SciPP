#pragma once
// scypp::spatial — port of scipy.spatial (Phase 10): distances, KD-tree,
// 2-D convex hull / Delaunay, and 3-D rotations.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"

namespace scypp::spatial {

using numpp::ndarray;

// ---- distances ----
enum class Backend { Cpu, Device };
Backend last_backend();
ndarray pdist(const ndarray& X, const std::string& metric = "euclidean", double p = 2.0);
ndarray cdist(const ndarray& XA, const ndarray& XB, const std::string& metric = "euclidean",
              double p = 2.0, Backend forced = Backend::Cpu);
ndarray squareform(const ndarray& X);
ndarray distance_matrix(const ndarray& X, const ndarray& Y, double p = 2.0);

// ---- KD-tree ----
class KDTree {
 public:
  explicit KDTree(const ndarray& points);
  struct QueryResult { ndarray distances, indices; };
  QueryResult query(const ndarray& x, int k = 1) const;
  std::vector<int64_t> query_ball_point(const ndarray& x, double r) const;

 private:
  struct Node { int64_t idx; int axis; int left = -1, right = -1; };
  int build(std::vector<int64_t>& ids, int lo, int hi, int depth);
  std::vector<double> pts_;  // n*d
  int64_t n_ = 0, d_ = 0, root_ = -1;
  std::vector<Node> nodes_;
};

// ---- 2-D computational geometry ----
struct ConvexHull {
  explicit ConvexHull(const ndarray& points);
  ndarray vertices;   // hull vertex indices, counter-clockwise
  ndarray simplices;  // hull edges (nedges x 2)
  double area = 0;    // perimeter (2-D convention)
  double volume = 0;  // enclosed area (2-D convention)
};

class Delaunay {
 public:
  explicit Delaunay(const ndarray& points);
  ndarray simplices() const;  // ntri x 3 point indices
  int64_t find_simplex(const ndarray& p) const;

 private:
  std::vector<double> pts_;  // n*2
  std::vector<std::array<int, 3>> tris_;
};

// ---- 3-D rotations ----
class Rotation {
 public:
  Rotation() : x_(0), y_(0), z_(0), w_(1) {}
  static Rotation from_quat(const ndarray& q);                    // (x,y,z,w) scalar-last
  static Rotation from_matrix(const ndarray& m);
  static Rotation from_rotvec(const ndarray& v, bool degrees = false);
  static Rotation from_euler(const std::string& seq, const ndarray& angles, bool degrees = false);
  ndarray as_quat() const;
  ndarray as_matrix() const;
  ndarray as_rotvec(bool degrees = false) const;
  ndarray as_euler(const std::string& seq, bool degrees = false) const;
  ndarray apply(const ndarray& v) const;
  Rotation inv() const;
  Rotation operator*(const Rotation& other) const;
  double magnitude() const;

  double x_, y_, z_, w_;
};

class Slerp {
 public:
  Slerp(const ndarray& times, const std::vector<Rotation>& rotations);
  Rotation operator()(double t) const;

 private:
  std::vector<double> times_;
  std::vector<Rotation> rots_;
};

}  // namespace scypp::spatial
