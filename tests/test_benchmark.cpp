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

void test_unitary_projection() {
    lph::Matrix4 positive_diagonal;
    positive_diagonal(0, 0) = {0.7, 0.0};
    positive_diagonal(1, 1) = {1.3, 0.0};
    positive_diagonal(2, 2) = {2.0, 0.0};
    positive_diagonal(3, 3) = {0.4, 0.0};
    const lph::Matrix4 projected =
        lph::project_to_unitary(positive_diagonal);
    require((projected - lph::Matrix4::identity()).
                frobenius_squared() < 1.0e-27,
            "positive diagonal polar factor is not identity");

    lph::Matrix4 perturbed = lph::Matrix4::identity();
    perturbed(0, 1) = {0.03, -0.02};
    perturbed(2, 3) = {-0.01, 0.04};
    perturbed(3, 2) = {0.02, 0.01};
    const lph::Matrix4 unitary = lph::project_to_unitary(perturbed);
    require((unitary.adjoint() * unitary -
             lph::Matrix4::identity()).frobenius_squared() < 1.0e-27,
            "polar projection is not unitary");
}

void test_noiseless_convergence() {
    lph::BenchmarkConfig config;
    config.sample_counts = {12, 24};
    config.sigma = 0.0;
    config.quadrature_order = 128;
    const auto results = lph::run_benchmark(config);
    require(results.size() == 4, "unexpected benchmark result count");
    require(results[2].algorithm == lph::Algorithm::unconstrained,
            "unexpected benchmark result order");
    require(results[2].relative_l2_error <
                results[0].relative_l2_error,
            "noiseless interpolation did not converge");
    require(results[2].relative_l2_error < 1.0e-7,
            "high-order noiseless error is unexpectedly large");
    require(std::abs(results[2].relative_l2_error -
                     results[3].relative_l2_error) < 1.0e-10,
            "polar projection changed the noiseless result");
}

}  // namespace

int main() {
    try {
        test_matrix_arithmetic();
        test_chebyshev_interpolation_and_derivative();
        test_unitary_projection();
        test_noiseless_convergence();
        std::cout << "All tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << '\n';
        return 1;
    }
}
