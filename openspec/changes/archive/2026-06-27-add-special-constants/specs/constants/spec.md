# constants Specification

## ADDED Requirements

### Requirement: Mathematical and SI scale constants

`scipp::constants` SHALL expose the standard mathematical and SI physical
constants as `constexpr double` values matching SciPy's `scipy.constants`,
including at least `pi`, `golden`, `c` (speed of light), `h`, `hbar`, `G`, `e`
(elementary charge), `k` (Boltzmann), `N_A` (Avogadro), `R` (gas), `g`
(standard gravity) and `atm`. (oracle: scipy/constants/_constants.py)

#### Scenario: Scale constants match SciPy
- WHEN a constant such as `c`, `h`, `hbar`, `k` or `N_A` is read
- THEN its value equals SciPy's `scipy.constants` value exactly

### Requirement: CODATA physical-constants table

`scipp::constants` SHALL provide a `physical_constants` table mapping a constant's
name to its value, unit and relative precision, with `value(name)`, `unit(name)`
and `precision(name)` accessors. The table SHALL be transcribed from the same
CODATA release SciPy uses (recorded in the table). Unknown names SHALL raise
`scipp::value_error`. (oracle: scipy/constants/_codata.py)

#### Scenario: Lookup matches SciPy
- GIVEN a valid CODATA constant name (e.g. "electron mass")
- WHEN `value(name)`, `unit(name)` and `precision(name)` are queried
- THEN they equal SciPy's `scipy.constants.physical_constants[name]` triple

#### Scenario: Unknown constant name
- GIVEN a name not present in the table
- WHEN `value(name)` is queried
- THEN `scipp::value_error` is raised (mirroring SciPy's `KeyError`)

### Requirement: Named unit scale factors

`scipp::constants` SHALL expose the named unit scale factors as `constexpr double`
matching SciPy, including SI prefixes (`kilo`, `mega`, `milli`, `micro`, `nano`,
…), and common units (`minute`, `hour`, `day`, `inch`, `foot`, `mile`, `bar`,
`atm`, `eV`, `gram`, `lb`). (oracle: scipy/constants/_constants.py)

#### Scenario: Unit factors match SciPy
- WHEN a named unit factor (e.g. `hour`, `inch`, `bar`, `eV`) is read
- THEN its value equals SciPy's corresponding `scipy.constants` value

### Requirement: Unit-conversion helpers

`scipp::constants` SHALL provide `convert_temperature`, `lambda2nu` and
`nu2lambda`, operating element-wise over `numpp::ndarray` with scalar overloads,
matching SciPy. (oracle: scipy/constants/_constants.py)

#### Scenario: Temperature conversion matches SciPy
- GIVEN an array of temperatures and a source/target scale among
  Celsius/Kelvin/Fahrenheit/Rankine
- WHEN `convert_temperature(arr, from, to)` is called
- THEN the result is `allclose` to SciPy, and converting there-and-back returns
  the original within tolerance

#### Scenario: Wavelength/frequency conversion round-trips
- GIVEN an array of wavelengths
- WHEN `lambda2nu` then `nu2lambda` are applied
- THEN the result is `allclose` to the input and to SciPy's `lambda2nu`/`nu2lambda`
