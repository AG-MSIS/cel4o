#include "cel4o/cel4o.h"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace cel4o {

void dct_matrix(MatrixView result, bool make_orthonormal) {
    if (result.rows < result.cols)
        throw std::invalid_argument("result must have at least as many rows as columns.");

    const std::size_t m = result.rows;
    const std::size_t n = result.cols;
    if (m == 0 || n == 0) return;

    const double norm = make_orthonormal ? 1.0 / std::sqrt(static_cast<double>(m)) : 1.0;
    const double scale = std::sqrt(2.0) * norm;
    const double pi_m = std::numbers::pi / static_cast<double>(m);

    // j > 0
    for (std::size_t i = 0; i < m; ++i) {
        result(i, 0) = norm;
        for (std::size_t j = 1; j < n; ++j)
            result(i, j) = scale * std::cos(pi_m * (static_cast<double>(i) + 0.5)
                                                 * static_cast<double>(j));
    }
}

void cel4o_D (ConstMatrixView L, ConstMatrixView Q, std::span<const double> row_weights,
              MatrixView result, std::span<double> workspace) {
    check_matrix_product_input(L, Q, result);

    const std::size_t n = L.rows;
    const std::size_t p = L.cols;
    const std::size_t m = Q.cols;
    if (row_weights.size() != n)
        throw std::invalid_argument("row_weights must contain one element per row of L.");
    if (workspace.size() < n * m)
        throw std::invalid_argument("workspace must contain at least L.rows*Q.cols doubles.");

    MatrixView gamma = row_major_view(workspace.data(), n, m);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            result(i, j) = 0.0;

    // gamma = square(L) @ square(Q)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) gamma(i, j) = 0.0;
        for (std::size_t k = 0; k < p; ++k) {
            const double lik = L(i, k);
            if (lik!=0.0) {
                const double l2 = lik * lik;
                for (std::size_t j = 0; j < m; ++j) {
                    const double qkj = Q(k, j);
                    gamma(i, j) += l2 * qkj * qkj;
                }
            }
        }
    }

    for (std::size_t k = 0; k < p; ++k) {
        double delta_beta_E = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            const double lik = L(i, k);
            const double wi = row_weights[i];

            if (lik!=0)
                for (std::size_t j = 0; j < m; ++j) {
                    const double beta = lik * Q(k, j);
                    const double beta2 = beta * beta;
                    gamma(i, j) -= beta2;

                    const double alpha = result(i, j);
                    const double alpha2 = alpha * alpha;
                    delta_beta_E += wi * alpha * beta
                                * (3.0 * gamma(i, j) + alpha2 + beta2);
                }
        }

        delta_beta_E *= 8.0;
        const double sign = (delta_beta_E > 0.0) ? -1.0 : 1.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double lik = L(i, k);
            if (lik!=0)
                for (std::size_t j = 0; j < m; ++j)
                    result(i, j) += sign * lik * Q(k, j);
        }
    }
}

void cel4o_D(ConstMatrixView L, ConstMatrixView Q, std::span<const double> row_weights,
             MatrixView result) {
    std::vector<double> workspace(L.rows * Q.cols);
    cel4o_D(L, Q, row_weights, result, workspace);
}

void cel4o_D(ConstMatrixView L, ConstMatrixView Q, MatrixView result) {
    const std::vector<double> row_weights = make_row_weights(L);
    std::vector<double> workspace(L.rows * Q.cols);
    cel4o_D(L, Q, row_weights, result, workspace);
}

} // namespace cel4o
