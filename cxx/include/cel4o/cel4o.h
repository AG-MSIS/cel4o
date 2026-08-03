#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>
#include <utility>


namespace cel4o {

struct ConstMatrixView {
    const double* data = nullptr;
    std::size_t rows = 0, cols = 0;
    std::ptrdiff_t row_stride = 0, col_stride = 0;

    double operator()(std::size_t i, std::size_t j) const {
        return data[static_cast<std::ptrdiff_t>(i) * row_stride
                  + static_cast<std::ptrdiff_t>(j) * col_stride];
    }
};

struct MatrixView {
    double* data = nullptr;
    std::size_t rows = 0, cols = 0;
    std::ptrdiff_t row_stride = 0, col_stride = 0;

    double& operator()(std::size_t i, std::size_t j) {
        return data[static_cast<std::ptrdiff_t>(i) * row_stride
                  + static_cast<std::ptrdiff_t>(j) * col_stride];
    }

    double operator()(std::size_t i, std::size_t j) const {
        return data[static_cast<std::ptrdiff_t>(i) * row_stride
                  + static_cast<std::ptrdiff_t>(j) * col_stride];
    }

    operator ConstMatrixView() const {
        return {data, rows, cols, row_stride, col_stride};
    }
};

inline ConstMatrixView row_major_view(const double* data, std::size_t rows, std::size_t cols) {
    return {data, rows, cols, static_cast<std::ptrdiff_t>(cols), 1};
}

inline MatrixView row_major_view(double* data, std::size_t rows, std::size_t cols) {
    return {data, rows, cols, static_cast<std::ptrdiff_t>(cols), 1};
}

inline ConstMatrixView column_major_view(const double* data, std::size_t rows, std::size_t cols) {
    return {data, rows, cols, 1, static_cast<std::ptrdiff_t>(rows)};
}

inline MatrixView column_major_view(double* data, std::size_t rows, std::size_t cols) {
    return {data, rows, cols, 1, static_cast<std::ptrdiff_t>(rows)};
}

void check_matrix_product_input(ConstMatrixView L, ConstMatrixView Q, MatrixView result);

double weighted_l4_loss(ConstMatrixView L, std::span<const double> row_weights);

double weighted_l4_loss(ConstMatrixView L);

std::vector<double> make_row_weights(ConstMatrixView L);

// Fills an m*n matrix (m >= n) with the first n columns of the m*m DCT matrix.
void dct_matrix(MatrixView result, bool make_orthonormal = false);

void cel4o_D(ConstMatrixView L, ConstMatrixView Q, std::span<const double> row_weights,
             MatrixView result, std::span<double> workspace);

void cel4o_D(ConstMatrixView L, ConstMatrixView Q, std::span<const double> row_weights,
             MatrixView result);

void cel4o_D(ConstMatrixView L, ConstMatrixView Q, MatrixView result);

std::pair<double, double> minimize_quartic_on_interval(
    std::span<const double> coefficients,
    std::pair<double, double> limits = {-1.0, 1.0},
    bool only_limits = false);

    // L and result may be the same
void integrate_trailing_column_householder(ConstMatrixView L, std::span<const double> q,
    std::size_t j, MatrixView result, double atol = 1e-12);

    std::array<double, 5> conditional_quartic_coefficients_Q(
    std::span<const double> phi, std::span<const double> alpha,
    std::span<const double> beta, std::span<const double> gamma,
    std::span<const double> sigma, double upsilon, std::size_t i,
    std::size_t j, std::size_t m, std::span<const double> row_weights);

std::pair<std::vector<double>, double> optimize_conditional_Q_column(
    ConstMatrixView L, std::size_t j, std::span<const double> row_weights);

std::vector<double> cel4o_Q(
    ConstMatrixView L, std::span<const double> row_weights);

std::vector<double> cel4o_Q(ConstMatrixView L);
} // namespace cel4o
