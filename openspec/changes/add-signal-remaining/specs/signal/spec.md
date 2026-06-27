# signal Specification

## ADDED Requirements

### Requirement: Discrete-time LTI systems
`scypp::signal` SHALL provide discrete-time LTI analysis `dstep`, `dimpulse`,
`dlsim`, `dbode`, `dfreqresp` and `cont2discrete`, matching SciPy. (oracle: scipy/signal/_ltisys.py)

#### Scenario: Discrete step response
- GIVEN a discrete transfer function and a sample time
- WHEN `dstep` is computed
- THEN the response matches SciPy

### Requirement: Continuous-to-discrete conversion
`scypp::signal::cont2discrete` SHALL convert a continuous system to discrete via
the supported methods (`zoh`, `bilinear`, `euler`), matching SciPy. (oracle: scipy/signal/_lti_conversion.py)

#### Scenario: ZOH discretization
- GIVEN a continuous transfer function and a sample time
- WHEN `cont2discrete(..., method="zoh")` is called
- THEN the discrete system matches SciPy

### Requirement: Separable 2-D filtering
`scypp::signal::sepfir2d` SHALL apply a separable 2-D FIR filter, matching SciPy. (oracle: scipy/signal/_spline.pyx)

#### Scenario: Separable filter matches the full 2-D convolution
- GIVEN an image and separable row/column kernels
- WHEN `sepfir2d` is applied
- THEN the result matches the equivalent 2-D convolution and SciPy
