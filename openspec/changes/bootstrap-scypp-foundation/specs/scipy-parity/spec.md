# scipy-parity Specification

## ADDED Requirements

### Requirement: Documented SciPy parity backlog

ScyPP SHALL maintain a prioritized, tiered backlog of SciPy subpackages not yet
implemented, with a suggested implementation order, so that parity work flows
through OpenSpec as discrete changes. Each subpackage below is a **target**; it
graduates into its own OpenSpec change (proposal/specs/tasks) when started, ported
against the SciPy oracle and its OpenSpec baseline at
`/home/leonardo/work/scipy/openspec/specs/<name>/spec.md`.

#### Scenario: Backlog is current and discoverable
- WHEN a contributor inspects this change and `openspec/project.md`
- THEN they find the target subpackages grouped by phase with the order in which
  they should be implemented

#### Scenario: A subpackage graduates to its own change
- WHEN a backlog subpackage (e.g. `optimize`) is started
- THEN it is created as its own OpenSpec change ported against the SciPy oracle, and
  GPU-accelerable kernels within it reuse the backend-acceleration substrate

### Requirement: special and constants parity (Phase 1)

ScyPP SHALL port `scipy.special` (gamma/`gammaln`/`beta`/`digamma`, error functions
`erf`/`erfc`/`erfinv`, Bessel `jv`/`yv`/`iv`/`kv`, `expn`/`exprel`, orthogonal
polynomials, `logsumexp`, `softmax`) and `scipy.constants` (CODATA constants,
unit-conversion helpers). (oracle: scipy/special, scipy/constants)

#### Scenario: Special-function values match SciPy
- GIVEN representative arguments across each function family
- WHEN the ScyPP special function is evaluated
- THEN the result is `allclose` to the SciPy reference within the function's
  documented tolerance

#### Scenario: Physical constants match CODATA values
- WHEN a constant or unit conversion is queried
- THEN it equals SciPy's value

### Requirement: linalg parity (Phase 2)

ScyPP SHALL port `scipy.linalg`: basic ops (`inv`/`det`/`solve`/`lstsq`/`pinv`/
`norm`), eigenproblems (`eig`/`eigh`/`eig_banded`), decompositions (`lu`/`qr`/`svd`/
`cholesky`/`ldl`/`schur`/`qz`/`polar`), matrix functions (`expm`/`logm`/`sqrtm`/
`funm`), matrix-equation solvers (`solve_sylvester`/`solve_lyapunov`/`solve_*_are`),
special-matrix constructors, and BLAS/LAPACK low-level access — with GEMM-backed and
GPU-accelerable hot paths. (oracle: scipy/linalg)

#### Scenario: Decomposition reconstructs the input
- GIVEN a matrix `A`
- WHEN a ScyPP decomposition (`lu`/`qr`/`svd`/`cholesky`) is computed
- THEN recombining the factors reconstructs `A` within tolerance and the result is
  `allclose` to SciPy's

#### Scenario: Accelerated linalg agrees with CPU
- GIVEN a build with a GPU/BLAS backend
- WHEN a GEMM-backed routine runs on a large input
- THEN the accelerated result equals the CPU result within tolerance

### Requirement: fft and fftpack parity (Phase 3)

ScyPP SHALL port `scipy.fft`/`scipy.fftpack`: complex and real transforms
(`fft`/`ifft`/`rfft`/`irfft`/`hfft`), N-D transforms (`fftn`/`rfftn`), the discrete
cosine/sine transforms (`dct`/`idct`/`dst`/`idst`, types I–IV), and helpers
(`fftfreq`/`fftshift`/`next_fast_len`) — reusing NumPP's FFT core and adding the
SciPy-only DCT/DST, with GPU-accelerable transforms. (oracle: scipy/fft,
scipy/fftpack)

#### Scenario: Transforms match SciPy including DCT/DST
- GIVEN representative signals
- WHEN ScyPP computes `fft`/`rfft`/`dct`/`dst` (and inverses)
- THEN the results are `allclose` to SciPy and round-trip to the input within
  tolerance

### Requirement: optimize parity (Phase 4)

ScyPP SHALL port `scipy.optimize`: scalar/multivariate minimization (`minimize` with
Nelder-Mead/BFGS/L-BFGS-B/Powell/CG/Newton-CG/trust-region methods), root finding
(`root`/`brentq`/`newton`/`fsolve`), least squares (`least_squares`/`curve_fit`/
`nnls`), and linear/mixed-integer programming (`linprog`/`milp`), returning
SciPy-shaped result objects. (oracle: scipy/optimize)

#### Scenario: Minimizer converges to SciPy's optimum
- GIVEN a benchmark objective and starting point
- WHEN a ScyPP method minimizes it
- THEN the located optimum and function value are `allclose` to SciPy's, and the
  result object reports `success` consistently

### Requirement: integrate and differentiate parity (Phase 5)

ScyPP SHALL port `scipy.integrate` (`quad`/`dblquad`/`simpson`/`trapezoid`,
`solve_ivp` with RK45/RK23/DOP853/Radau/BDF/LSODA, `solve_bvp`) and
`scipy.differentiate` (finite-difference derivative/jacobian/hessian). (oracle:
scipy/integrate, scipy/differentiate)

#### Scenario: ODE integration matches SciPy
- GIVEN an initial-value problem
- WHEN `solve_ivp` integrates it with a given method and tolerances
- THEN the solution at the evaluation points is `allclose` to SciPy's

### Requirement: interpolate parity (Phase 6)

ScyPP SHALL port `scipy.interpolate`: 1-D interpolation (`interp1d`,
`CubicSpline`/`PchipInterpolator`/`Akima1DInterpolator`), B-splines (`splrep`/
`splev`/`BSpline`), and N-D interpolation (`griddata`/`RegularGridInterpolator`/
`RBFInterpolator`). (oracle: scipy/interpolate)

#### Scenario: Interpolant matches SciPy
- GIVEN sample points and query points
- WHEN a ScyPP interpolant is evaluated at the query points
- THEN the values are `allclose` to SciPy's and reproduce the samples at the knots

### Requirement: stats parity (Phase 7)

ScyPP SHALL port `scipy.stats`: continuous and discrete distributions (pdf/cdf/ppf/
rvs/fit/moments), summary statistics, hypothesis tests (`ttest_*`/`ks_*`/`mannwhitneyu`/
`chi2_contingency`/`pearsonr`/`spearmanr`), quasi-Monte-Carlo (`qmc`), and KDE
(`gaussian_kde`), honoring shared `axis`/`nan_policy`/`alternative` conventions and
seeding through NumPP `random`. (oracle: scipy/stats)

#### Scenario: Distribution and test results match SciPy
- GIVEN distribution parameters and a sample
- WHEN ScyPP evaluates pdf/cdf/ppf and runs a hypothesis test
- THEN the values and test statistic/p-value are `allclose` to SciPy's, with
  stochastic sampling validated reproducibly or distributionally

### Requirement: signal parity (Phase 8)

ScyPP SHALL port `scipy.signal`: convolution/correlation (`convolve`/`fftconvolve`/
`correlate`), filtering (`lfilter`/`filtfilt`/`sosfilt`), filter design (`butter`/
`cheby1`/`cheby2`/`ellip`/`firwin`/`iirfilter`), spectral analysis (`welch`/
`periodogram`/`spectrogram`/`stft`), and LTI systems (`lti`/`bode`/`freqz`).
(oracle: scipy/signal)

#### Scenario: Filtering and spectra match SciPy
- GIVEN a signal and a designed filter
- WHEN ScyPP filters the signal and estimates its spectrum
- THEN the filtered output and spectral estimates are `allclose` to SciPy's

### Requirement: sparse parity (Phase 9)

ScyPP SHALL port `scipy.sparse`: sparse array/matrix formats (CSR/CSC/COO/DIA/LIL/
BSR) with construction and conversion, arithmetic and SpMV, `sparse.linalg`
(`spsolve`/`cg`/`gmres`/`eigsh`/`svds`), and `sparse.csgraph` (shortest paths,
connected components, MST) — with GPU-accelerable CSR SpMV. (oracle: scipy/sparse)

#### Scenario: Sparse operations match dense reference
- GIVEN a sparse matrix and a vector
- WHEN ScyPP computes SpMV and a sparse solve
- THEN results are `allclose` to SciPy's, and the GPU SpMV path agrees with the CPU
  path within tolerance

### Requirement: spatial parity (Phase 10)

ScyPP SHALL port `scipy.spatial`: `KDTree`/`cKDTree` nearest-neighbor queries,
distance functions and `cdist`/`pdist`/`squareform`, computational geometry
(`ConvexHull`/`Delaunay`/`Voronoi`), and `transform.Rotation`/`Slerp` — with
GPU-accelerable pairwise distance matrices. (oracle: scipy/spatial)

#### Scenario: Spatial queries and distances match SciPy
- GIVEN a point set
- WHEN ScyPP builds a KD-tree, queries neighbors, and computes pairwise distances
- THEN neighbor indices and distance matrices match SciPy's (distances `allclose`)

### Requirement: ndimage parity (Phase 11)

ScyPP SHALL port `scipy.ndimage`: filters (`gaussian_filter`/`uniform_filter`/
`median_filter`/`convolve`/`sobel`), morphology (`binary_*`/`grey_*`/
`distance_transform_edt`), measurements (`label`/`center_of_mass`/`find_objects`),
and geometric transforms (`zoom`/`rotate`/`affine_transform`/`map_coordinates`) —
with GPU-accelerable separable convolution. (oracle: scipy/ndimage)

#### Scenario: Image filters and measurements match SciPy
- GIVEN an N-D image array
- WHEN ScyPP applies a filter, a morphology op, and a measurement
- THEN the outputs are `allclose` (or label-equivalent) to SciPy's

### Requirement: cluster, io, and datasets parity (Phase 12)

ScyPP SHALL port `scipy.cluster` (`kmeans`/`kmeans2`/`vq`, hierarchical `linkage`/
`fcluster`/`dendrogram`), `scipy.io` (Matrix Market, WAV, ARFF), and
`scipy.datasets` (sample datasets). MATLAB `.mat` read/write (`loadmat`/`savemat`)
is out of scope. (oracle: scipy/cluster, scipy/io, scipy/datasets)

#### Scenario: Clustering and I/O round-trip match SciPy
- GIVEN sample data
- WHEN ScyPP runs k-means/linkage and reads/writes a supported file format
- THEN cluster assignments match SciPy (up to label permutation) and I/O round-trips
  preserve the data
