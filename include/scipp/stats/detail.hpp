#pragma once
// Internal special-function enablers for scipp::stats distributions:
// regularized incomplete gamma/beta and their inverses.

namespace scipp::stats::detail {

double gammainc(double a, double x);     // regularized lower P(a,x)
double gammaincc(double a, double x);    // regularized upper Q(a,x) = 1 - P
double betainc(double a, double b, double x);   // regularized I_x(a,b)
double gammaincinv(double a, double p);  // x such that gammainc(a,x) = p
double betaincinv(double a, double b, double p);  // x such that betainc(a,b,x) = p

}  // namespace scipp::stats::detail
