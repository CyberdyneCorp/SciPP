# spatial Specification

## ADDED Requirements

### Requirement: Device-accelerated pairwise euclidean distance
`scipp::spatial` SHALL compute `cdist` and `pdist` for the `euclidean` and
`sqeuclidean` metrics through NumPP's `cdist_euclidean` device kernel, falling back
to the CPU path and reporting the chosen backend via `last_backend()`. Other
metrics remain on the CPU kernel. (oracle: scipy.spatial.distance.cdist)

#### Scenario: Euclidean cdist is backend-independent and matches SciPy
- GIVEN two point sets A (m×d) and B (n×d)
- WHEN `cdist(A, B, "euclidean")` runs on the CPU and any present device backend
- THEN the results are `allclose` to each other and to `scipy.spatial.distance.cdist`
