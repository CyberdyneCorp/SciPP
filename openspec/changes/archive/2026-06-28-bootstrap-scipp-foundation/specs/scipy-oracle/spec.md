# scipy-oracle Specification

## ADDED Requirements

### Requirement: SciPy as the numerical oracle

SciPP tests SHALL validate numeric results against real Python SciPy
(`/home/leonardo/work/scipy`) as the oracle, asserting the SciPP result is
`allclose` to the SciPy reference within per-domain tolerances, mirroring how NumPP
validates against NumPy and SymPP against SymPy.

#### Scenario: Result matches SciPy reference
- GIVEN a SciPP routine and a set of inputs
- WHEN the test runs the corresponding Python SciPy call to produce reference values
- THEN the SciPP result is asserted `allclose` to the SciPy result within the
  documented tolerance for that domain

#### Scenario: Specs cite the ported source
- WHEN a SciPP requirement is written
- THEN it cites the SciPy source it ports as a breadcrumb (e.g.
  `(oracle: scipy/linalg/_decomp_lu.py)`)

### Requirement: Frozen oracle data for Python-free CI

The oracle harness SHALL support a checked/frozen mode that serializes SciPy
reference values into `tests/golden/`, so CI can run the comparison suite without a
Python or SciPy installation.

#### Scenario: CI runs without Python
- GIVEN frozen golden data committed under `tests/golden/`
- WHEN the test suite runs in an environment with no Python/SciPy
- THEN the oracle comparisons run against the frozen values and pass

#### Scenario: Regenerate frozen data
- WHEN the golden data is regenerated against the SciPy oracle
- THEN any divergence from the previously frozen values is surfaced for review
  before being committed

### Requirement: Reproducible validation of stochastic routines

The harness SHALL pin seeds via NumPP's bit-exact `random` for stochastic routines
(`stats` sampling, QMC, stochastic optimizers) so comparisons are reproducible.
Where SciPy's RNG stream cannot be matched bit-for-bit, the harness SHALL instead
assert documented distributional/statistical equivalence.

#### Scenario: Seeded sampling is reproducible
- GIVEN a stochastic routine seeded through NumPP's `random`
- WHEN the test runs twice with the same seed
- THEN it produces identical results both times

#### Scenario: Distributional equivalence when streams differ
- GIVEN a sampler whose stream cannot be made bit-identical to SciPy's
- WHEN its output is validated
- THEN the test asserts statistical/distributional agreement (e.g. moments,
  goodness-of-fit) rather than element-wise equality, and the choice is documented

### Requirement: Regression test for every fixed divergence

A regression test reproducing the divergent case SHALL be added whenever an oracle
comparison reveals a divergence from SciPy that is then fixed, so the fix is locked
in.

#### Scenario: Bug fix ships with a regression test
- GIVEN a divergence from the SciPy oracle that has been corrected
- WHEN the fix is committed
- THEN a regression test covering the previously divergent input is included
