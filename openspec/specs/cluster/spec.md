# cluster Specification

## Purpose
TBD - created by archiving change add-cluster-io. Update Purpose after archive.
## Requirements
### Requirement: Vector quantization

`scypp::cluster` SHALL provide `whiten`, `vq` and `kmeans2` (with explicit initial
centroids), matching SciPy. (oracle: scipy/cluster/vq.py)

#### Scenario: whiten and vq match SciPy
- GIVEN an observation matrix and a code book
- WHEN `whiten(obs)` and `vq(obs, code_book)` are computed
- THEN the whitened data, nearest-centroid codes and distances are `allclose` to
  SciPy's

#### Scenario: kmeans2 converges deterministically
- GIVEN data and explicit initial centroids
- WHEN `kmeans2(data, init, iter)` runs
- THEN the resulting centroids and labels are `allclose` to SciPy's
  `kmeans2(..., minit="matrix")`

### Requirement: Hierarchical clustering

`scypp::cluster` SHALL provide `linkage` (`single`/`complete`/`average`/`ward`),
`fcluster` (`distance`/`maxclust`) and `cophenet`, matching SciPy. (oracle:
scipy/cluster/hierarchy.py)

#### Scenario: Linkage matches SciPy
- GIVEN observations in general position and a linkage method
- WHEN `linkage` is computed
- THEN the `(n−1)×4` linkage matrix is `allclose` to SciPy's

#### Scenario: Flat clusters and cophenetic distances
- GIVEN a linkage matrix
- WHEN `fcluster(Z, t)` and `cophenet(Z)` are computed
- THEN the flat clustering matches SciPy up to relabeling and the cophenetic
  distances are `allclose` to SciPy's

