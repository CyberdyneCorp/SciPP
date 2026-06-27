#pragma once
// scypp::constants — port of scipy.constants (Phase 1 subset).
// Scale constants and unit factors are constexpr; the CODATA table is a runtime
// lookup. Values match the CODATA release used by the reference SciPy.

#include <optional>
#include <string>
#include <string_view>

#include "numpp/core/ndarray.hpp"

namespace scypp::constants {

using numpp::ndarray;

// ---- mathematical constants ----
inline constexpr double pi     = 3.141592653589793238462643383279502884;
inline constexpr double golden = 1.618033988749894848204586834365638118;
inline constexpr double golden_ratio = golden;

// ---- SI defining / physical constants (CODATA, matches scipy.constants) ----
inline constexpr double c            = 299792458.0;            // speed of light, m/s
inline constexpr double speed_of_light = c;
inline constexpr double mu_0         = 1.25663706212e-6;       // vacuum permeability
inline constexpr double epsilon_0    = 8.8541878128e-12;       // vacuum permittivity
inline constexpr double h            = 6.62607015e-34;         // Planck, J*s
inline constexpr double hbar         = 1.0545718176461565e-34; // h/(2*pi)
inline constexpr double G            = 6.6743e-11;             // gravitation
inline constexpr double g            = 9.80665;                // standard gravity
inline constexpr double e            = 1.602176634e-19;        // elementary charge, C
inline constexpr double R            = 8.3144626181532395;     // molar gas constant (N_A*k)
inline constexpr double alpha        = 7.2973525693e-3;        // fine-structure
inline constexpr double N_A          = 6.02214076e23;          // Avogadro
inline constexpr double Avogadro     = N_A;
inline constexpr double k            = 1.380649e-23;           // Boltzmann, J/K
inline constexpr double Boltzmann    = k;
inline constexpr double sigma        = 5.670374419e-8;         // Stefan-Boltzmann
inline constexpr double Wien         = 2.897771955e-3;
inline constexpr double Rydberg      = 10973731.568160;
inline constexpr double m_e          = 9.1093837015e-31;       // electron mass
inline constexpr double m_p          = 1.67262192369e-27;      // proton mass
inline constexpr double m_n          = 1.67492749804e-27;      // neutron mass

// ---- SI prefixes (scale factors) ----
inline constexpr double quetta = 1e30; inline constexpr double ronna = 1e27;
inline constexpr double yotta = 1e24;  inline constexpr double zetta = 1e21;
inline constexpr double exa   = 1e18;  inline constexpr double peta  = 1e15;
inline constexpr double tera  = 1e12;  inline constexpr double giga  = 1e9;
inline constexpr double mega  = 1e6;   inline constexpr double kilo  = 1e3;
inline constexpr double hecto = 1e2;   inline constexpr double deka  = 1e1;
inline constexpr double deci  = 1e-1;  inline constexpr double centi = 1e-2;
inline constexpr double milli = 1e-3;  inline constexpr double micro = 1e-6;
inline constexpr double nano  = 1e-9;  inline constexpr double pico  = 1e-12;
inline constexpr double femto = 1e-15; inline constexpr double atto  = 1e-18;
inline constexpr double zepto = 1e-21; inline constexpr double yocto = 1e-24;

// ---- common units (in SI) ----
inline constexpr double minute = 60.0;
inline constexpr double hour   = 3600.0;
inline constexpr double day    = 86400.0;
inline constexpr double week   = 604800.0;
inline constexpr double inch   = 0.0254;
inline constexpr double foot   = 0.3048;
inline constexpr double yard   = 0.9144;
inline constexpr double mile   = 1609.344;
inline constexpr double atm    = 101325.0;
inline constexpr double bar    = 1e5;
inline constexpr double mmHg   = 133.322387415;
inline constexpr double eV     = 1.602176634e-19;
inline constexpr double gram   = 1e-3;
inline constexpr double lb     = 0.45359237;
inline constexpr double pound  = lb;

// ---- CODATA physical-constants table ----
struct ConstantEntry {
  double value;
  std::string unit;
  double uncertainty;  // absolute uncertainty (0.0 for exact)
};

// CODATA release string the table is transcribed from.
std::string_view codata_version();

// Lookup helpers. Unknown names raise scypp::value_error.
const ConstantEntry& physical_constant(const std::string& name);
double      value(const std::string& name);
std::string unit(const std::string& name);
double      precision(const std::string& name);  // relative uncertainty

// ---- unit-conversion helpers ----
// scales: "Celsius"/"C", "Kelvin"/"K", "Fahrenheit"/"F", "Rankine"/"R".
double  convert_temperature(double value, std::string_view from, std::string_view to);
ndarray convert_temperature(const ndarray& value, std::string_view from, std::string_view to);

double  lambda2nu(double lambda);   // wavelength (m) -> frequency (Hz)
double  nu2lambda(double nu);       // frequency (Hz) -> wavelength (m)
ndarray lambda2nu(const ndarray& lambda);
ndarray nu2lambda(const ndarray& nu);

}  // namespace scypp::constants
