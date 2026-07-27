#include "lph/benchmark.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_matrix_arithmetic() {
    const lph::Matrix4 identity = lph::Matrix4::identity();
    lph::Matrix4 matrix;
    matrix(0, 1) = {2.0, 3.0};
    matrix(2, 3) = {-1.0, 0.5};
    const lph::Matrix4 difference = identity * matrix - matrix;
    require(difference.frobenius_squared() < 1.0e-28,
            "identity multiplication failed");
    const lph::Matrix4 hermitian = matrix + matrix.adjoint();
    require((hermitian - hermitian.adjoint()).frobenius_squared() <
                1.0e-28,
            "adjoint failed");
}

void test_chebyshev_interpolation_and_derivative() {
    constexpr int count = 8;
    std::vector<lph::Matrix4> values(count);
    for (int j = 0; j < count; ++j) {
        const double theta =
            std::acos(-1.0) * (static_cast<double>(j) + 0.5) / count;
        const double x = std::cos(theta);
        values[j](0, 0) = 2.0 - 0.5 * x + 0.75 * (2.0 * x * x - 1.0);
    }
    const auto coefficients = lph::chebyshev_coefficients(values);
    require(std::abs(coefficients[0](0, 0) -
                     lph::Complex{2.0, 0.0}) < 1.0e-13,
            "constant Chebyshev coefficient failed");
    require(std::abs(coefficients[1](0, 0) +
                     lph::Complex{0.5, 0.0}) < 1.0e-13,
            "linear Chebyshev coefficient failed");
    require(std::abs(coefficients[2](0, 0) -
                     lph::Complex{0.75, 0.0}) < 1.0e-13,
            "quadratic Chebyshev coefficient failed");

    const auto derivative = lph::differentiate_chebyshev(coefficients);
    for (const double x : {-0.8, -0.1, 0.4, 0.9}) {
        const double expected = -0.5 + 3.0 * x;
        const double actual =
            lph::evaluate_chebyshev(derivative, x)(0, 0).real();
        require(std::abs(actual - expected) < 1.0e-12,
                "Chebyshev differentiation failed");
    }
}

void test_noiseless_convergence() {
    lph::BenchmarkConfig config;
    config.sample_counts = {12, 24};
    config.sigma = 0.0;
    config.quadrature_order = 128;
    const auto results = lph::run_benchmark(config);
    require(results.size() == 2, "unexpected benchmark result count");
    require(results[1].relative_l2_error <
                results[0].relative_l2_error,
            "noiseless interpolation did not converge");
    require(results[1].relative_l2_error < 1.0e-7,
            "high-order noiseless error is unexpectedly large");
}

}  // namespace

int main() {
    try {
        test_matrix_arithmetic();
        test_chebyshev_interpolation_and_derivative();
        test_noiseless_convergence();
        std::cout << "All tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << '\n';
        return 1;
    }
}
