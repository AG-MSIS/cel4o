#include "cel4o/cel4o.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::forcecast>;

cel4o::ConstMatrixView matrix_view(const DoubleArray& a, const char* name) {
    const py::buffer_info info = a.request();
    if (info.ndim != 2) throw std::invalid_argument(std::string(name) + " must be two-dimensional.");

    return {
        static_cast<const double*>(info.ptr),
        static_cast<std::size_t>(info.shape[0]),
        static_cast<std::size_t>(info.shape[1]),
        static_cast<std::ptrdiff_t>(info.strides[0] / static_cast<py::ssize_t>(sizeof(double))),
        static_cast<std::ptrdiff_t>(info.strides[1] / static_cast<py::ssize_t>(sizeof(double)))
    };
}

py::array_t<double> dct_matrix_py(const py::object& size, bool make_orthonormal) {
    std::size_t m;
    std::size_t n;
    if (py::isinstance<py::int_>(size) || PyIndex_Check(size.ptr())) {
        m = n = size.cast<std::size_t>();
    } else if (py::isinstance<py::tuple>(size)) {
        const py::tuple dimensions = size.cast<py::tuple>();
        if (dimensions.size() != 2)
            throw std::invalid_argument("size must be an int or a tuple (m, n).");
        m = dimensions[0].cast<std::size_t>();
        n = dimensions[1].cast<std::size_t>();
    } else {
        throw std::invalid_argument("size must be an int or a tuple (m, n).");
    }
    if (m < n)
        throw std::invalid_argument("landscape DCT matrix is not allowed.");

    py::array_t<double> result({
        static_cast<py::ssize_t>(m),
        static_cast<py::ssize_t>(n),
    });

    py::buffer_info info = result.request();

    cel4o::MatrixView result_view {
        static_cast<double*>(info.ptr), m, n,
        static_cast<std::ptrdiff_t>(
            info.strides[0] / static_cast<py::ssize_t>(sizeof(double))),
        static_cast<std::ptrdiff_t>(
            info.strides[1] / static_cast<py::ssize_t>(sizeof(double)))
    };

    {
        py::gil_scoped_release release;
        cel4o::dct_matrix(result_view, make_orthonormal);
    }

    return result;
}

py::array_t<double> cel4o_D_py(const DoubleArray& L_array, const DoubleArray& Q_array,
                               py::object row_weights_object) {
    const cel4o::ConstMatrixView L = matrix_view(L_array, "L");
    const cel4o::ConstMatrixView Q = matrix_view(Q_array, "Q");

    if (Q.rows != L.cols)
        throw std::invalid_argument("L and Q have incompatible shapes.");

    const std::size_t n = L.rows;
    const std::size_t m = Q.cols;
    py::array_t<double> result({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(m)});
    py::buffer_info result_info = result.request();
    cel4o::MatrixView result_view {
        static_cast<double*>(result_info.ptr), n, m,
        static_cast<std::ptrdiff_t>(result_info.strides[0] / static_cast<py::ssize_t>(sizeof(double))),
        static_cast<std::ptrdiff_t>(result_info.strides[1] / static_cast<py::ssize_t>(sizeof(double)))
    };

    std::vector<double> workspace(n * m);

    if (row_weights_object.is_none()) {
        const std::vector<double> row_weights = cel4o::make_row_weights(L);
        py::gil_scoped_release release;
        cel4o::cel4o_D(L, Q, row_weights, result_view, workspace);
    } else {
        py::array_t<double, py::array::c_style | py::array::forcecast> weights_array(
            row_weights_object);
        const py::buffer_info weights_info = weights_array.request();

        if (weights_info.ndim != 1 || static_cast<std::size_t>(weights_info.shape[0]) != n)
            throw std::invalid_argument("row_weights must have one element per row of L.");

        const py::buffer_info contiguous_info = weights_array.request();
        const auto* weights = static_cast<const double*>(contiguous_info.ptr);

        py::gil_scoped_release release;
        cel4o::cel4o_D(L, Q, std::span<const double>(weights, n), result_view, workspace);
    }

    return result;
}

py::tuple minimize_quartic_on_interval_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> coefficients,
    std::pair<double, double> limits,
    bool only_limits)
{
    const py::buffer_info info = coefficients.request();

    if (info.ndim != 1)
        throw std::invalid_argument("coefficients must be one-dimensional.");

    const auto* data = static_cast<const double*>(info.ptr);
    const std::size_t size = static_cast<std::size_t>(info.shape[0]);

    const auto [x_min, value_min] =
        cel4o::minimize_quartic_on_interval(
            std::span<const double>(data, size),
            limits,
            only_limits);

    return py::make_tuple(x_min, value_min);
}

py::array_t<double> integrate_trailing_column_householder_py(const DoubleArray& L_array, py::array_t<double, py::array::c_style | py::array::forcecast> q_array,
    std::size_t j, double atol)
{
    const cel4o::ConstMatrixView L = matrix_view(L_array, "L");

    const py::buffer_info q_info = q_array.request();
    if (q_info.ndim != 1)
        throw std::invalid_argument("q must be one-dimensional.");

    const std::size_t q_size = static_cast<std::size_t>(q_info.shape[0]);
    const auto* q = static_cast<const double*>(q_info.ptr);

    py::array_t<double> result({
        static_cast<py::ssize_t>(L.rows),
        static_cast<py::ssize_t>(L.cols)
    });

    py::buffer_info result_info = result.request();

    cel4o::MatrixView result_view {
        static_cast<double*>(result_info.ptr),
        L.rows,
        L.cols,
        static_cast<std::ptrdiff_t>(
            result_info.strides[0] / static_cast<py::ssize_t>(sizeof(double))),
        static_cast<std::ptrdiff_t>(
            result_info.strides[1] / static_cast<py::ssize_t>(sizeof(double)))
    };

    {
        py::gil_scoped_release release;
        cel4o::integrate_trailing_column_householder(
            L, std::span<const double>(q, q_size), j, result_view, atol);
    }

    return result;
}

py::array_t<double> conditional_quartic_coefficients_Q_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> phi,
    py::array_t<double, py::array::c_style | py::array::forcecast> alpha,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gamma,
    py::array_t<double, py::array::c_style | py::array::forcecast> sigma,
    double upsilon, std::size_t i, std::size_t j, std::size_t m,
    py::array_t<double, py::array::c_style | py::array::forcecast> row_weights)
{
    const py::buffer_info phi_info = phi.request();
    const py::buffer_info alpha_info = alpha.request();
    const py::buffer_info beta_info = beta.request();
    const py::buffer_info gamma_info = gamma.request();
    const py::buffer_info sigma_info = sigma.request();
    const py::buffer_info weights_info = row_weights.request();

    if (phi_info.ndim != 1 || alpha_info.ndim != 1 || beta_info.ndim != 1 ||
        gamma_info.ndim != 1 || sigma_info.ndim != 1 || weights_info.ndim != 1)
        throw std::invalid_argument("All row-dependent arrays must be one-dimensional.");

    const std::size_t n_rows = static_cast<std::size_t>(weights_info.shape[0]);

    if (static_cast<std::size_t>(phi_info.shape[0]) != n_rows ||
        static_cast<std::size_t>(alpha_info.shape[0]) != n_rows ||
        static_cast<std::size_t>(beta_info.shape[0]) != n_rows ||
        static_cast<std::size_t>(gamma_info.shape[0]) != n_rows ||
        static_cast<std::size_t>(sigma_info.shape[0]) != n_rows)
        throw std::invalid_argument("All row-dependent arrays must have equal shape.");

    const auto coefficients = cel4o::conditional_quartic_coefficients_Q(
        std::span<const double>(static_cast<const double*>(phi_info.ptr), n_rows),
        std::span<const double>(static_cast<const double*>(alpha_info.ptr), n_rows),
        std::span<const double>(static_cast<const double*>(beta_info.ptr), n_rows),
        std::span<const double>(static_cast<const double*>(gamma_info.ptr), n_rows),
        std::span<const double>(static_cast<const double*>(sigma_info.ptr), n_rows),
        upsilon, i, j, m,
        std::span<const double>(static_cast<const double*>(weights_info.ptr), n_rows));

    py::array_t<double> result(5);
    std::copy(coefficients.begin(), coefficients.end(), result.mutable_data());
    return result;
}

py::tuple optimize_conditional_Q_column_py(
    const DoubleArray& L_array, std::size_t j,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights_array)
{
    const cel4o::ConstMatrixView L = matrix_view(L_array, "L");

    const py::buffer_info weights_info = weights_array.request();
    if (weights_info.ndim != 1)
        throw std::invalid_argument("row_weights must be one-dimensional.");

    const auto* weights = static_cast<const double*>(weights_info.ptr);
    const std::size_t weights_size = static_cast<std::size_t>(weights_info.shape[0]);

    std::pair<std::vector<double>, double> result;
    {
        py::gil_scoped_release release;
        result = cel4o::optimize_conditional_Q_column(
            L, j, std::span<const double>(weights, weights_size));
    }

    py::array_t<double> q(result.first.size());
    std::memcpy(q.mutable_data(), result.first.data(), result.first.size() * sizeof(double));

    return py::make_tuple(q, result.second);
}

py::array_t<double> cel4o_Q_py(
    const DoubleArray& L_array,
    py::object row_weights_object)
{
    const cel4o::ConstMatrixView L = matrix_view(L_array, "L");

    std::vector<double> result;

    if (row_weights_object.is_none()) {
        const std::vector<double> row_weights = cel4o::make_row_weights(L);

        {
            py::gil_scoped_release release;
            result = cel4o::cel4o_Q(L, row_weights);
        }
    } else {
        py::array_t<double, py::array::c_style | py::array::forcecast>
            weights_array(row_weights_object);

        const py::buffer_info info = weights_array.request();

        if (info.ndim != 1 ||
            static_cast<std::size_t>(info.shape[0]) != L.rows)
            throw std::invalid_argument(
                "row_weights must be a one-dimensional array of length n.");

        const auto* weights = static_cast<const double*>(info.ptr);

        {
            py::gil_scoped_release release;
            result = cel4o::cel4o_Q(
                L, std::span<const double>(weights, L.rows));
        }
    }

    py::array_t<double> result_array({
        static_cast<py::ssize_t>(L.rows),
        static_cast<py::ssize_t>(L.cols)
    });

    std::memcpy(
        result_array.mutable_data(),
        result.data(),
        result.size() * sizeof(double));

    return result_array;
}

double weighted_l4_loss_py(const DoubleArray& L_array, py::object row_weights_object)
{
    const cel4o::ConstMatrixView L = matrix_view(L_array, "U");

    if (row_weights_object.is_none()) {
        py::gil_scoped_release release;
        return cel4o::weighted_l4_loss(L);
    }

    py::array_t<double, py::array::c_style | py::array::forcecast>
        weights_array(row_weights_object);

    const py::buffer_info info = weights_array.request();

    if (info.ndim != 1 ||
        static_cast<std::size_t>(info.shape[0]) != L.rows)
        throw std::invalid_argument(
            "row_weights must be a one-dimensional array of length n_rows.");

    const auto* weights = static_cast<const double*>(info.ptr);

    py::gil_scoped_release release;
    return cel4o::weighted_l4_loss(L, std::span<const double>(weights, L.rows));
}

} // namespace

PYBIND11_MODULE(_cel4o, m) {
    m.doc() = "C++ implementation of CEL4O algorithms.";
    m.def(
        "weighted_l4_loss",
        &weighted_l4_loss_py,
        py::arg("U"),
        py::arg("row_weights") = py::none(),
        "Compute m/n times the weighted L4 sum for an n-by-m matrix."
    );
    m.def("dct_matrix", &dct_matrix_py,
        py::arg("size"),
        py::arg("make_orthonormal") = false,
        "Return the first n columns of an m-by-m DCT matrix for size=(m, n).");
    m.def("cel4o_D", &cel4o_D_py, py::arg("L"), py::arg("Q"),
        py::arg("row_weights") = py::none(),
        "Conditional-expectation L4 optimization, D variant.");
    m.def(
        "minimize_quartic_on_interval",
        &minimize_quartic_on_interval_py,
        py::arg("coefficients"),
        py::arg("limits") = std::pair<double, double>{-1.0, 1.0},
        py::arg("only_limits") = false,
        "Minimize a polynomial of degree at most four on an interval.");
    m.def(
        "integrate_trailing_column_householder",
        &integrate_trailing_column_householder_py,
        py::arg("L"),
        py::arg("q"),
        py::arg("j"),
        py::arg("atol") = 1e-12
    );
    m.def(
        "conditional_quartic_coefficients_Q",
        &conditional_quartic_coefficients_Q_py,
        py::arg("phi"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("gamma"),
        py::arg("sigma"),
        py::arg("upsilon"),
        py::arg("i"),
        py::arg("j"),
        py::arg("m"),
        py::arg("row_weights")
    );
    m.def(
        "optimize_conditional_Q_column",
        &optimize_conditional_Q_column_py,
        py::arg("L"), py::arg("j"), py::arg("row_weights")
    );
    m.def(
        "cel4o_Q",
        &cel4o_Q_py,
        py::arg("L"),
        py::arg("row_weights") = py::none(),
        "Optimize L @ Q for L with shape (n, m) and an implicit orthogonal Q of shape (m, m)."
    );
}
