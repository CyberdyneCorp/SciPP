// Oracle tests for Airy functions and elliptic integrals against SciPy golden.
#include "golden.hpp"
#include "scypp/special/special.hpp"
#include "scypp_test.hpp"

namespace sp = scypp::special;

namespace {
constexpr double R = 1e-10, A = 1e-12;
constexpr double RA = 1e-9, AA = 1e-11;  // Airy: slightly looser per task

void check_nan(double v) { CHECK(std::isnan(v)); }
}  // namespace

TEST_CASE("airy") {
  for (int i = 0; i < golden::ai_x_n; ++i) {
    sp::airy_t a = sp::airy(golden::ai_x[i]);
    CHECK_CLOSE(a.Ai, golden::ai_Ai[i], RA, AA);
    CHECK_CLOSE(a.Aip, golden::ai_Aip[i], RA, AA);
    CHECK_CLOSE(a.Bi, golden::ai_Bi[i], RA, AA);
    CHECK_CLOSE(a.Bip, golden::ai_Bip[i], RA, AA);
  }
  // Wronskian identity Ai*Bi' - Ai'*Bi = 1/pi.
  sp::airy_t a = sp::airy(1.3);
  CHECK_CLOSE(a.Ai * a.Bip - a.Aip * a.Bi, 1.0 / 3.14159265358979323846, 1e-10, 1e-12);
}

TEST_CASE("airye") {
  for (int i = 0; i < golden::aie_x_n; ++i) {
    double x = golden::aie_x[i];
    sp::airy_t a = sp::airye(x);
    if (x < 0.0) {
      check_nan(a.Ai);
      check_nan(a.Aip);
    } else {
      CHECK_CLOSE(a.Ai, golden::aie_Ai[i], RA, AA);
      CHECK_CLOSE(a.Aip, golden::aie_Aip[i], RA, AA);
    }
    CHECK_CLOSE(a.Bi, golden::aie_Bi[i], RA, AA);
    CHECK_CLOSE(a.Bip, golden::aie_Bip[i], RA, AA);
  }
}

TEST_CASE("complete elliptic integrals") {
  CHECK_ARR(ellipk, [](double m) { return sp::ellipk(m); }, R, A);
  CHECK_ARR(ellipe, [](double m) { return sp::ellipe(m); }, R, A);
  CHECK_ARR(ellipkm1, [](double p) { return sp::ellipkm1(p); }, R, A);
  // K(1) = inf, E(1) = 1, E(0) = K(0) = pi/2.
  CHECK(std::isinf(sp::ellipk(1.0)));
  CHECK_CLOSE(sp::ellipe(1.0), 1.0, R, A);
  CHECK_CLOSE(sp::ellipk(0.0), 1.5707963267948966, R, A);
  // out-of-domain
  CHECK(std::isnan(sp::ellipk(1.5)));
  CHECK(std::isnan(sp::ellipe(2.0)));
}

TEST_CASE("incomplete elliptic integrals") {
  struct Grid { const double* phi; const double* F; const double* E; double m; };
  const Grid grids[] = {
      {golden::eiF_a_phi, golden::eiF_a_out, golden::eiE_a_out, golden::ei_m_a},
      {golden::eiF_b_phi, golden::eiF_b_out, golden::eiE_b_out, golden::ei_m_b},
      {golden::eiF_c_phi, golden::eiF_c_out, golden::eiE_c_out, golden::ei_m_c}};
  for (const auto& g : grids) {
    for (int i = 0; i < golden::eiF_a_phi_n; ++i) {
      CHECK_CLOSE(sp::ellipkinc(g.phi[i], g.m), g.F[i], R, A);
      CHECK_CLOSE(sp::ellipeinc(g.phi[i], g.m), g.E[i], R, A);
    }
  }
  // F(0|m) = E(0|m) = 0; reduces to complete at phi = pi/2.
  CHECK_CLOSE(sp::ellipkinc(0.0, 0.5), 0.0, R, A);
  CHECK_CLOSE(sp::ellipkinc(1.5707963267948966, 0.5), sp::ellipk(0.5), 1e-9, 1e-11);
  CHECK_CLOSE(sp::ellipeinc(1.5707963267948966, 0.5), sp::ellipe(0.5), 1e-9, 1e-11);
}

TEST_CASE("jacobi elliptic functions") {
  struct Grid {
    const double* u; const double* sn; const double* cn; const double* dn;
    const double* ph; double m;
  };
  const Grid grids[] = {
      {golden::ej_a_u, golden::ej_a_sn, golden::ej_a_cn, golden::ej_a_dn,
       golden::ej_a_ph, golden::ej_m_a},
      {golden::ej_b_u, golden::ej_b_sn, golden::ej_b_cn, golden::ej_b_dn,
       golden::ej_b_ph, golden::ej_m_b},
      {golden::ej_c_u, golden::ej_c_sn, golden::ej_c_cn, golden::ej_c_dn,
       golden::ej_c_ph, golden::ej_m_c}};
  for (const auto& g : grids) {
    for (int i = 0; i < golden::ej_a_u_n; ++i) {
      sp::ellipj_t e = sp::ellipj(g.u[i], g.m);
      CHECK_CLOSE(e.sn, g.sn[i], R, A);
      CHECK_CLOSE(e.cn, g.cn[i], R, A);
      CHECK_CLOSE(e.dn, g.dn[i], R, A);
      CHECK_CLOSE(e.ph, g.ph[i], R, A);
      // identities sn^2 + cn^2 = 1, dn^2 + m sn^2 = 1
      CHECK_CLOSE(e.sn * e.sn + e.cn * e.cn, 1.0, R, A);
      CHECK_CLOSE(e.dn * e.dn + g.m * e.sn * e.sn, 1.0, R, A);
    }
  }
}
