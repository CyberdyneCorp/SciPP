#pragma once
// scipp::stats — port of scipy.stats (Phase 7 subset): continuous distributions,
// summary statistics, correlation/regression, parametric tests, and gaussian_kde.

#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"

namespace scipp::stats {

using numpp::ndarray;

// ---- continuous distributions (static methods mirror scipy dist.method) ----
struct norm {
  static double pdf(double x, double loc = 0, double scale = 1);
  static double logpdf(double x, double loc = 0, double scale = 1);
  static double cdf(double x, double loc = 0, double scale = 1);
  static double sf(double x, double loc = 0, double scale = 1);
  static double ppf(double q, double loc = 0, double scale = 1);
  static double mean(double loc = 0, double scale = 1);
  static double var(double loc = 0, double scale = 1);
  static double std(double loc = 0, double scale = 1);
  static double median(double loc = 0, double scale = 1);
  static ndarray pdf(const ndarray& x, double loc = 0, double scale = 1);
  static ndarray cdf(const ndarray& x, double loc = 0, double scale = 1);
};
struct expon {
  static double pdf(double x, double loc = 0, double scale = 1);
  static double logpdf(double x, double loc = 0, double scale = 1);
  static double cdf(double x, double loc = 0, double scale = 1);
  static double sf(double x, double loc = 0, double scale = 1);
  static double ppf(double q, double loc = 0, double scale = 1);
  static double mean(double loc = 0, double scale = 1);
  static double var(double loc = 0, double scale = 1);
  static double median(double loc = 0, double scale = 1);
};
struct uniform {
  static double pdf(double x, double loc = 0, double scale = 1);
  static double cdf(double x, double loc = 0, double scale = 1);
  static double sf(double x, double loc = 0, double scale = 1);
  static double ppf(double q, double loc = 0, double scale = 1);
  static double mean(double loc = 0, double scale = 1);
  static double var(double loc = 0, double scale = 1);
};
struct gamma {
  static double pdf(double x, double a, double loc = 0, double scale = 1);
  static double logpdf(double x, double a, double loc = 0, double scale = 1);
  static double cdf(double x, double a, double loc = 0, double scale = 1);
  static double sf(double x, double a, double loc = 0, double scale = 1);
  static double ppf(double q, double a, double loc = 0, double scale = 1);
  static double mean(double a, double loc = 0, double scale = 1);
  static double var(double a, double loc = 0, double scale = 1);
};
struct chi2 {
  static double pdf(double x, double df, double loc = 0, double scale = 1);
  static double cdf(double x, double df, double loc = 0, double scale = 1);
  static double sf(double x, double df, double loc = 0, double scale = 1);
  static double ppf(double q, double df, double loc = 0, double scale = 1);
  static double mean(double df, double loc = 0, double scale = 1);
  static double var(double df, double loc = 0, double scale = 1);
};
struct beta {
  static double pdf(double x, double a, double b, double loc = 0, double scale = 1);
  static double cdf(double x, double a, double b, double loc = 0, double scale = 1);
  static double sf(double x, double a, double b, double loc = 0, double scale = 1);
  static double ppf(double q, double a, double b, double loc = 0, double scale = 1);
  static double mean(double a, double b, double loc = 0, double scale = 1);
  static double var(double a, double b, double loc = 0, double scale = 1);
};
struct t {
  static double pdf(double x, double df, double loc = 0, double scale = 1);
  static double cdf(double x, double df, double loc = 0, double scale = 1);
  static double sf(double x, double df, double loc = 0, double scale = 1);
  static double ppf(double q, double df, double loc = 0, double scale = 1);
  static double mean(double df, double loc = 0, double scale = 1);
  static double var(double df, double loc = 0, double scale = 1);
};
struct f {
  static double pdf(double x, double dfn, double dfd, double loc = 0, double scale = 1);
  static double cdf(double x, double dfn, double dfd, double loc = 0, double scale = 1);
  static double sf(double x, double dfn, double dfd, double loc = 0, double scale = 1);
  static double ppf(double q, double dfn, double dfd, double loc = 0, double scale = 1);
  static double mean(double dfn, double dfd, double loc = 0, double scale = 1);
};

// ---- discrete distributions ----
struct binom {
  static double pmf(int k, int n, double p);
  static double logpmf(int k, int n, double p);
  static double cdf(int k, int n, double p);
  static double sf(int k, int n, double p);
  static int ppf(double q, int n, double p);
  static double mean(int n, double p);
  static double var(int n, double p);
};
struct poisson {
  static double pmf(int k, double mu);
  static double logpmf(int k, double mu);
  static double cdf(int k, double mu);
  static double sf(int k, double mu);
  static int ppf(double q, double mu);
  static double mean(double mu);
  static double var(double mu);
};
struct geom {
  static double pmf(int k, double p);
  static double cdf(int k, double p);
  static double sf(int k, double p);
  static int ppf(double q, double p);
  static double mean(double p);
  static double var(double p);
};
struct bernoulli {
  static double pmf(int k, double p);
  static double cdf(int k, double p);
  static double mean(double p);
  static double var(double p);
};
struct nbinom {
  static double pmf(int k, double n, double p);
  static double logpmf(int k, double n, double p);
  static double cdf(int k, double n, double p);
  static double sf(int k, double n, double p);
  static double mean(double n, double p);
  static double var(double n, double p);
};
struct hypergeom {
  static double pmf(int k, int M, int n, int N);
  static double cdf(int k, int M, int n, int N);
  static double sf(int k, int M, int n, int N);
  static double mean(int M, int n, int N);
  static double var(int M, int n, int N);
};

// ---- summary statistics ----
struct DescribeResult { int nobs; double minv, maxv, mean, variance, skewness, kurtosis; };
double gmean(const ndarray& a);
double hmean(const ndarray& a);
double moment(const ndarray& a, int order);
double skew(const ndarray& a, bool bias = true);
double kurtosis(const ndarray& a, bool fisher = true, bool bias = true);
double sem(const ndarray& a);
double variation(const ndarray& a);
double iqr(const ndarray& a);
ndarray zscore(const ndarray& a);
ndarray rankdata(const ndarray& a);
double mode(const ndarray& a);
DescribeResult describe(const ndarray& a);

// ---- correlation & regression ----
struct CorrResult { double statistic; double pvalue; };
struct LinregressResult { double slope, intercept, rvalue, pvalue, stderr, intercept_stderr; };
CorrResult pearsonr(const ndarray& x, const ndarray& y);
CorrResult spearmanr(const ndarray& x, const ndarray& y);
LinregressResult linregress(const ndarray& x, const ndarray& y);

// ---- hypothesis tests ----
struct TestResult { double statistic; double pvalue; };
struct Chi2ContingencyResult { double statistic; double pvalue; int dof; ndarray expected; };
TestResult ttest_1samp(const ndarray& a, double popmean);
TestResult ttest_ind(const ndarray& a, const ndarray& b);
TestResult ttest_rel(const ndarray& a, const ndarray& b);
TestResult f_oneway(const std::vector<ndarray>& groups);
TestResult ks_2samp(const ndarray& a, const ndarray& b);
Chi2ContingencyResult chi2_contingency(const ndarray& table);
TestResult normaltest(const ndarray& a);

// ---- normality tests ----
struct ShapiroResult { double statistic; double pvalue; };
struct AndersonResult {
  double statistic;
  std::vector<double> critical_values;
  std::vector<double> significance_level;
};
ShapiroResult shapiro(const ndarray& x);
AndersonResult anderson(const ndarray& x);

// ---- nonparametric rank tests ----
TestResult mannwhitneyu(const ndarray& x, const ndarray& y);
TestResult wilcoxon(const ndarray& x, const ndarray& y);
TestResult kruskal(const std::vector<ndarray>& groups);
CorrResult kendalltau(const ndarray& x, const ndarray& y);

// ---- density estimation ----
class gaussian_kde {
 public:
  explicit gaussian_kde(const ndarray& dataset, const std::string& bw_method = "scott");
  ndarray operator()(const ndarray& points) const;  // alias for evaluate
  ndarray evaluate(const ndarray& points) const;
  double factor() const { return factor_; }

 private:
  std::vector<double> data_;  // d*n row-major (d dims, n points)
  int d_ = 1, n_ = 0;
  double factor_ = 1.0, norm_ = 1.0;
  std::vector<double> inv_cov_;  // d*d
};

}  // namespace scipp::stats
