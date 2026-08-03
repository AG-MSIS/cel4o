#include "cel4o/cel4o.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>

namespace cel4o {
namespace {

bool effectively_zero(double x, double scale) {
    return std::abs(x) <= 16.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, scale);
}

int real_roots_linear(double a, double b, double* roots) {
    const double scale = std::max(std::abs(a), std::abs(b));
    if (effectively_zero(a, scale)) return 0;
    roots[0] = -b / a;
    return 1;
}

int real_roots_quadratic(double a, double b, double c, double* roots) {
    const double scale = std::max({std::abs(a), std::abs(b), std::abs(c)});
    if (effectively_zero(a, scale)) return real_roots_linear(b, c, roots);

    double disc = b * b - 4.0 * a * c;
    const double disc_scale = b * b + 4.0 * std::abs(a * c);
    if (effectively_zero(disc, disc_scale)) disc = 0.0;
    if (disc < 0.0) return 0;

    const double s = std::sqrt(disc);
    if (s == 0.0) {
        roots[0] = -0.5 * b / a;
        return 1;
    }

    const double q = -0.5 * (b + std::copysign(s, b));
    roots[0] = q / a;
    roots[1] = c / q;
    return 2;
}

int real_roots_cubic(double a, double b, double c, double d, double* roots) {
    const double scale = std::max({std::abs(a), std::abs(b), std::abs(c), std::abs(d)});
    if (effectively_zero(a, scale)) return real_roots_quadratic(b, c, d, roots);

    const double A = b / a;
    const double B = c / a;
    const double C = d / a;

    const double p = B - A * A / 3.0;
    const double q = 2.0 * A * A * A / 27.0 - A * B / 3.0 + C;
    double disc = 0.25 * q * q + (p * p * p) / 27.0;

    const double disc_scale = 0.25 * q * q + std::abs((p * p * p) / 27.0);
    if (effectively_zero(disc, disc_scale)) disc = 0.0;

    const double shift = A / 3.0;

    if (disc > 0.0) {
        const double s = std::sqrt(disc);
        const double u = std::cbrt(-0.5 * q + s);
        const double v = std::cbrt(-0.5 * q - s);
        roots[0] = u + v - shift;
        return 1;
    }

    if (disc == 0.0) {
        if (effectively_zero(p, std::max(std::abs(p), std::abs(q)))) {
            roots[0] = -shift;
            return 1;
        }

        const double u = std::cbrt(-0.5 * q);
        roots[0] = 2.0 * u - shift;
        roots[1] = -u - shift;
        return 2;
    }

    const double r = 2.0 * std::sqrt(-p / 3.0);
    double arg = (3.0 * q / (2.0 * p)) * std::sqrt(-3.0 / p);
    arg = std::clamp(arg, -1.0, 1.0);

    const double phi = std::acos(arg) / 3.0;
    const double two_pi_over_three = 2.0 * std::numbers::pi / 3.0;

    roots[0] = r * std::cos(phi) - shift;
    roots[1] = r * std::cos(phi - two_pi_over_three) - shift;
    roots[2] = r * std::cos(phi - 2.0 * two_pi_over_three) - shift;
    return 3;
}

double polyval_quartic(const double* c, double x) {
    return ((((c[0] * x) + c[1]) * x + c[2]) * x + c[3]) * x + c[4];
}

} // namespace

std::pair<double, double> minimize_quartic_on_interval(
    std::span<const double> coefficients,
    std::pair<double, double> limits,
    bool only_limits)
{
    if (coefficients.empty())
        throw std::invalid_argument("coefficients must not be empty.");
    if (coefficients.size() > 5)
        throw std::invalid_argument("Polynomial degree must not exceed four.");

    for (double x : coefficients)
        if (!std::isfinite(x))
            throw std::invalid_argument("coefficients must contain only finite values.");

    double lo = limits.first;
    double hi = limits.second;
    if (!std::isfinite(lo) || !std::isfinite(hi))
        throw std::invalid_argument("limits must be finite.");
    if (lo > hi) std::swap(lo, hi);

    double c[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    const std::size_t offset = 5 - coefficients.size();
    for (std::size_t i = 0; i < coefficients.size(); ++i)
        c[offset + i] = coefficients[i];

    double x_min = lo;
    double value_min = polyval_quartic(c, lo);

    const double value_hi = polyval_quartic(c, hi);
    if (value_hi < value_min) {
        x_min = hi;
        value_min = value_hi;
    }

    if (only_limits || lo == hi) return {x_min, value_min};

    double roots[3];
    const int root_count = real_roots_cubic(
        4.0 * c[0], 3.0 * c[1], 2.0 * c[2], c[3], roots);

    for (int i = 0; i < root_count; ++i) {
        const double x = roots[i];
        if (!std::isfinite(x) || x < lo || x > hi) continue;

        const double value = polyval_quartic(c, x);
        if (value < value_min) {
            x_min = x;
            value_min = value;
        }
    }

    return {x_min, value_min};
}

void integrate_trailing_column_householder(ConstMatrixView L, std::span<const double> q, std::size_t j,
    MatrixView result, double atol) {
    const std::size_t n = L.rows;
    const std::size_t m = L.cols;

    if (j >= m) throw std::out_of_range("j must satisfy 0 <= j < m.");

    const std::size_t trailing_dimension = m - j;
    if (q.size() != trailing_dimension)
        throw std::invalid_argument("q has wrong size.");

    double q_squared_norm = 0.0;
    for (double x : q) {
        if (!std::isfinite(x))
            throw std::invalid_argument("q must contain only finite values.");
        q_squared_norm += x * x;
    }

    const double q_norm = std::sqrt(q_squared_norm);
    if (std::abs(q_norm - 1.0) > atol * std::max(1.0, q_norm))
        throw std::invalid_argument("q must be a unit vector.");

    if (result.rows != n || result.cols != m)
        throw std::invalid_argument("result must have the same shape as L.");

    // Copy L -> result.
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < m; ++k)
            result(i, k) = L(i, k);

    // Choose sign to avoid cancellation in v = q + s e_0.
    const double s = q[0] >= 0.0 ? 1.0 : -1.0;
    std::vector<double> v(q.begin(), q.end());
    v[0] += s;
    const double v_squared_norm = 2.0 * (1.0 + s * q[0]);
    const double factor = 2.0 / v_squared_norm;

    // trailing <- trailing * (I - 2 vv^T / v^T v)
    //
    // Compute one row at a time:
    //     dot = trailing_row * v
    //     trailing_row -= factor * dot * v
    for (std::size_t i = 0; i < n; ++i) {
        double dot = 0.0;
        for (std::size_t k = 0; k < trailing_dimension; ++k)
            dot += result(i, j + k) * v[k];

        const double row_factor = factor * dot;
        for (std::size_t k = 0; k < trailing_dimension; ++k)
            result(i, j + k) -= row_factor * v[k];
    }

    // H e_0 = -s q. Flip first trailing column to obtain q.
    for (std::size_t i = 0; i < n; ++i)
        result(i, j) *= -s;
}

std::array<double, 5> conditional_quartic_coefficients_Q (
    std::span<const double> phi, std::span<const double> alpha,
    std::span<const double> beta, std::span<const double> gamma,
    std::span<const double> sigma, double upsilon, std::size_t i,
    std::size_t j, std::size_t m, std::span<const double> row_weights)
{
    const std::size_t n = row_weights.size();

    if (phi.size() != n || alpha.size() != n || beta.size() != n ||
        gamma.size() != n || sigma.size() != n)
        throw std::invalid_argument("All row-dependent arrays must have equal shape.");

    const double lam = 3.0 / static_cast<double>(m - j + 1);
    const double Lambda = 1.0 + lam;
    const std::size_t tail_dimension = m - i - 1;

    double k0 = 0.0, k1 = 0.0, k2 = 0.0, k3 = 0.0, k4 = 0.0;

    for (std::size_t r = 0; r < n; ++r) {
        const double a = alpha[r];
        const double b = beta[r];
        const double g = gamma[r];
        const double s = sigma[r];
        const double w = row_weights[r];

        const double v = tail_dimension > 0 ? g / static_cast<double>(tail_dimension) : 0.0;
        const double h = tail_dimension > 0
            ? 3.0 * g * g / static_cast<double>(tail_dimension * (tail_dimension + 2))
            : 0.0;

        const double a2 = a * a;
        const double a4 = a2 * a2;
        const double b2 = b * b;
        const double b4 = b2 * b2;

        const double k0_r =
            Lambda * (a4 + 6.0 * v * upsilon * a2 + h * upsilon * upsilon)
            - 2.0 * lam * s * (a2 + v * upsilon)
            + lam * s * s + phi[r];

        const double k1_r =
            4.0 * a * b * (Lambda * (a2 + 3.0 * v * upsilon) - lam * s);

        const double k2_r =
            Lambda * (6.0 * a2 * b2
                    + 6.0 * v * (b2 * upsilon - a2)
                    - 2.0 * h * upsilon)
            - 2.0 * lam * s * (b2 - v);

        const double k3_r = 4.0 * Lambda * a * b * (b2 - 3.0 * v);
        const double k4_r = Lambda * (b4 - 6.0 * v * b2 + h);

        k0 += w * k0_r;
        k1 += w * k1_r;
        k2 += w * k2_r;
        k3 += w * k3_r;
        k4 += w * k4_r;
    }

    const double normalization =
        static_cast<double>(m) / static_cast<double>(n);
    return {normalization * k4, normalization * k3,
            normalization * k2, normalization * k1,
            normalization * k0};
}

std::pair<std::vector<double>, double> optimize_conditional_Q_column(
    ConstMatrixView L, std::size_t j, std::span<const double> row_weights)
{
    const std::size_t n = L.rows;
    const std::size_t m = L.cols;

    if (j >= m) throw std::out_of_range("j must satisfy 0 <= j < m.");
    if (row_weights.size() != n)
        throw std::invalid_argument("row_weights must have one entry per row.");

    std::vector<double> phi(n, 0.0);
    std::vector<double> alpha(n, 0.0);
    std::vector<double> beta(n);
    std::vector<double> gamma(n, 0.0);
    std::vector<double> sigma(n, 0.0);
    std::vector<double> q(m - j);

    // Quantities constant throughout this column.
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t k = 0; k < j; ++k) {
            const double x = L(r, k);
            const double x2 = x * x;
            phi[r] += x2 * x2;
        }

        for (std::size_t k = j; k < m; ++k) {
            const double x = L(r, k);
            sigma[r] += x * x;
        }

        beta[r] = L(r, j);

        for (std::size_t k = j + 1; k < m; ++k) {
            const double x = L(r, k);
            gamma[r] += x * x;
        }
    }

    double upsilon = 1.0;
    double expected_loss = std::numeric_limits<double>::quiet_NaN();
    const double tolerance = 1e-13 * static_cast<double>(std::max<std::size_t>(1, m - j));

    for (std::size_t i = j; i < m; ++i) {
        if (upsilon < 0.0) {
            if (upsilon < -tolerance)
                throw std::runtime_error("Remaining squared norm became negative.");
            upsilon = 0.0;
        }
        if (upsilon < tolerance) upsilon = 0.0;

        const double radius = std::sqrt(upsilon);
        double y = 0.0;

        if (radius != 0.0) {
            const auto coefficients = conditional_quartic_coefficients_Q(
                phi, alpha, beta, gamma, sigma, upsilon, i, j, m, row_weights);

            const auto minimum = minimize_quartic_on_interval(
                coefficients, {-radius, radius}, i == m - 1);

            y = minimum.first;
            expected_loss = minimum.second;
        }

        q[i - j] = y;

        // Incremental transition i -> i+1.
        for (std::size_t r = 0; r < n; ++r)
            alpha[r] += beta[r] * y;

        upsilon -= y * y;

        if (i + 1 < m) {
            for (std::size_t r = 0; r < n; ++r) {
                beta[r] = L(r, i + 1);
                gamma[r] -= beta[r] * beta[r];
                if (gamma[r] < 0.0) gamma[r] = 0.0;
            }
        }
    }

    double q_squared_norm = 0.0;
    for (double x : q) q_squared_norm += x * x;

    const double q_norm = std::sqrt(q_squared_norm);
    if (std::abs(q_norm - 1.0) > 1e-10)
        throw std::runtime_error("Optimized column is not normalized.");

    return {std::move(q), expected_loss};
}

std::vector<double> cel4o_Q(
    ConstMatrixView L, std::span<const double> row_weights)
{
    const std::size_t n = L.rows;
    const std::size_t m = L.cols;

    if (row_weights.size() != n)
        throw std::invalid_argument("row_weights must have one entry per row.");

    // Contiguous row-major result.
    std::vector<double> result_data(n * m);
    MatrixView result = row_major_view(result_data.data(), n, m);

    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t k = 0; k < m; ++k)
            result(r, k) = L(r, k);

    for (std::size_t j = 0; j < m; ++j) {
        const auto [q, expected_loss] =
            optimize_conditional_Q_column(result, j, row_weights);

        // Integrate q directly into result.
        integrate_trailing_column_householder(result, q, j, result);
    }

    return std::move(result_data);
}

} // namespace cel4o
