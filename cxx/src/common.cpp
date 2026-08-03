#include "cel4o/cel4o.h"
#include <stdexcept>

namespace cel4o {

void check_matrix_product_input(ConstMatrixView L, ConstMatrixView Q, MatrixView result) {
    if (L.data == nullptr || Q.data == nullptr || result.data == nullptr)
        throw std::invalid_argument("Matrix data pointer must not be null.");

    if (Q.rows != L.cols)
        throw std::invalid_argument("L and Q have incompatible shapes.");
    if (result.rows != L.rows || result.cols != Q.cols)
        throw std::invalid_argument("result must have shape (L.rows, Q.cols).");
}

std::vector<double> make_row_weights (ConstMatrixView L) {
    if (L.data == nullptr) throw std::invalid_argument("L data pointer must not be null.");

    const std::size_t n_rows = L.rows;
    const std::size_t n_cols = L.cols;

    std::vector<double> weights(n_rows);
    for (std::size_t i = 0; i < n_rows; ++i) {
        double variance = 0.0;
        for (std::size_t k = 0; k < n_cols; ++k) {
            const double x = L(i, k);
            variance += x * x;
        }
        if (variance > 0.0) weights[i] = 1.0 / (variance * variance);
        else weights[i] = 0;
    }
    return weights;
}

double weighted_l4_loss(ConstMatrixView L, std::span<const double> row_weights) {
    if (row_weights.size() != L.rows)
        throw std::invalid_argument(
            "row_weights must have one entry per row.");

    double loss = 0.0;
    for (std::size_t i = 0; i < L.rows; ++i) {
        double row_loss = 0.0;
        for (std::size_t j = 0; j < L.cols; ++j) {
            const double x = L(i, j);
            const double x2 = x * x;
            row_loss += x2 * x2;
        }
        loss += row_weights[i] * row_loss;
    }
    return static_cast<double>(L.cols) / static_cast<double>(L.rows) * loss;
}


double weighted_l4_loss(ConstMatrixView L) {
    const std::vector<double> row_weights = make_row_weights(L);
    return weighted_l4_loss(L, row_weights);
}

} // namespace cel4o
