# io Specification

## Purpose
TBD - created by archiving change add-cluster-io. Update Purpose after archive.
## Requirements
### Requirement: Matrix Market I/O

`scipp::io` SHALL provide `mmread` and `mmwrite` for the Matrix Market format,
matching SciPy. (oracle: scipy/io/_mmio.py)

#### Scenario: Read and round-trip Matrix Market
- GIVEN a Matrix Market file
- WHEN `mmread` parses it
- THEN the dense matrix equals SciPy's `mmread(...).toarray()`, and writing it with
  `mmwrite` then reading it back reproduces the matrix

### Requirement: WAV audio I/O

`scipp::io` SHALL provide `wavread` and `wavwrite` for integer-PCM WAV files,
matching SciPy. (oracle: scipy/io/wavfile.py)

#### Scenario: Read and round-trip WAV
- GIVEN a PCM WAV file
- WHEN `wavread` parses it
- THEN the sample rate and samples equal SciPy's `wavfile.read(...)`, and
  `wavwrite` then `wavread` round-trips the samples

### Requirement: ARFF loading

`scipp::io` SHALL provide `loadarff` for the numeric attributes of an ARFF file,
matching SciPy. (oracle: scipy/io/arff)

#### Scenario: Load ARFF numeric data
- GIVEN an ARFF file with numeric attributes
- WHEN `loadarff` parses it
- THEN the numeric data matrix equals SciPy's `loadarff(...)` values

