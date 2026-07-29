#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Complex = std::complex<double>;
constexpr Complex imaginary_unit{0.0, 1.0};
constexpr std::size_t qubits = 4;
constexpr std::size_t dimension = 1U << qubits;
constexpr std::size_t entries = dimension * dimension;
constexpr int timing_batches = 5;
constexpr int polar_max_iterations = 20;
constexpr double polar_relative_tolerance_squared = 1.0e-28;

class Matrix {
public:
    [[nodiscard]] static Matrix identity() {
        Matrix result;
        for (std::size_t index = 0; index < dimension; ++index) {
            result(index, index) = Complex{1.0, 0.0};
        }
        return result;
    }

    [[nodiscard]] Complex& operator()(std::size_t row,
                                      std::size_t column) {
        return data_[row * dimension + column];
    }

    [[nodiscard]] const Complex& operator()(std::size_t row,
                                            std::size_t column) const {
        return data_[row * dimension + column];
    }

    [[nodiscard]] Matrix adjoint() const {
        Matrix result;
        for (std::size_t row = 0; row < dimension; ++row) {
            for (std::size_t column = 0; column < dimension; ++column) {
                result(column, row) = std::conj((*this)(row, column));
            }
        }
        return result;
    }

    [[nodiscard]] Complex trace() const {
        Complex result{};
        for (std::size_t index = 0; index < dimension; ++index) {
            result += (*this)(index, index);
        }
        return result;
    }

    [[nodiscard]] double frobenius_squared() const {
        double result = 0.0;
        for (const Complex value : data_) {
            result += std::norm(value);
        }
        return result;
    }

    Matrix& operator+=(const Matrix& other) {
        for (std::size_t index = 0; index < entries; ++index) {
            data_[index] += other.data_[index];
        }
        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        for (std::size_t index = 0; index < entries; ++index) {
            data_[index] -= other.data_[index];
        }
        return *this;
    }

    Matrix& operator*=(Complex scalar) {
        for (Complex& value : data_) {
            value *= scalar;
        }
        return *this;
    }

private:
    std::array<Complex, entries> data_{};
};

[[nodiscard]] Matrix operator+(Matrix left, const Matrix& right) {
    left += right;
    return left;
}

[[nodiscard]] Matrix operator-(Matrix left, const Matrix& right) {
    left -= right;
    return left;
}

[[nodiscard]] Matrix operator*(Complex scalar, Matrix matrix) {
    matrix *= scalar;
    return matrix;
}

[[nodiscard]] Matrix operator*(double scalar, Matrix matrix) {
    matrix *= Complex{scalar, 0.0};
    return matrix;
}

[[nodiscard]] Matrix operator*(const Matrix& left, const Matrix& right) {
    Matrix result;
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t inner = 0; inner < dimension; ++inner) {
            const Complex value = left(row, inner);
            for (std::size_t column = 0; column < dimension; ++column) {
                result(row, column) += value * right(inner, column);
            }
        }
    }
    return result;
}

struct PauliTerm {
    double coefficient = 0.0;
    std::array<char, qubits> paulis{};
};

[[nodiscard]] double chebyshev_t(int degree, double x) {
    if (degree == 0) {
        return 1.0;
    }
    if (degree == 1) {
        return x;
    }
    double previous = 1.0;
    double current = x;
    for (int k = 2; k <= degree; ++k) {
        const double next = 2.0 * x * current - previous;
        previous = current;
        current = next;
    }
    return current;
}

[[nodiscard]] std::array<PauliTerm, 9>
hamiltonian_terms(double time, double final_time) {
    const double x = 2.0 * time / final_time - 1.0;
    const double a =
        0.60 + 0.25 * chebyshev_t(1, x) - 0.10 * chebyshev_t(3, x);
    const double b =
        -0.45 + 0.20 * chebyshev_t(2, x) + 0.12 * chebyshev_t(5, x);
    const double c =
        0.50 + 0.18 * chebyshev_t(1, x) - 0.14 * chebyshev_t(4, x);
    const double d =
        0.35 - 0.15 * chebyshev_t(2, x) + 0.10 * chebyshev_t(4, x);
    const double bridge = d / std::sqrt(3.0);
    return {{
        {a, {'Z', 'I', 'I', 'I'}},
        {a, {'I', 'I', 'Z', 'I'}},
        {b, {'I', 'X', 'I', 'I'}},
        {b, {'I', 'I', 'I', 'X'}},
        {c, {'Y', 'Z', 'I', 'I'}},
        {c, {'I', 'I', 'Y', 'Z'}},
        {bridge, {'I', 'X', 'X', 'I'}},
        {bridge, {'I', 'Y', 'Y', 'I'}},
        {bridge, {'I', 'Z', 'Z', 'I'}},
    }};
}

void add_pauli_left(Matrix& result,
                    const Matrix& input,
                    const PauliTerm& term) {
    for (std::size_t basis = 0; basis < dimension; ++basis) {
        std::size_t output_basis = basis;
        Complex phase{term.coefficient, 0.0};
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            const std::size_t bit_position = qubits - 1U - qubit;
            const bool bit = ((basis >> bit_position) & 1U) != 0U;
            switch (term.paulis[qubit]) {
                case 'I':
                    break;
                case 'X':
                    output_basis ^= (1U << bit_position);
                    break;
                case 'Y':
                    output_basis ^= (1U << bit_position);
                    phase *= bit ? Complex{0.0, -1.0}
                                 : Complex{0.0, 1.0};
                    break;
                case 'Z':
                    if (bit) {
                        phase = -phase;
                    }
                    break;
                default:
                    throw std::logic_error("invalid Pauli label");
            }
        }
        for (std::size_t column = 0; column < dimension; ++column) {
            result(output_basis, column) += phase * input(basis, column);
        }
    }
}

[[nodiscard]] Matrix hamiltonian_left_product(double time,
                                               const Matrix& input,
                                               double final_time) {
    Matrix result;
    for (const PauliTerm& term :
         hamiltonian_terms(time, final_time)) {
        add_pauli_left(result, input, term);
    }
    return result;
}

[[nodiscard]] Matrix target_hamiltonian(double time, double final_time) {
    return hamiltonian_left_product(
        time, Matrix::identity(), final_time);
}

[[nodiscard]] Matrix schrodinger_rhs(double time,
                                     const Matrix& unitary,
                                     double final_time) {
    return -imaginary_unit *
           hamiltonian_left_product(time, unitary, final_time);
}

[[nodiscard]] Matrix rk4_step(double time,
                              double step,
                              const Matrix& unitary,
                              double final_time) {
    const Matrix k1 = schrodinger_rhs(time, unitary, final_time);
    const Matrix k2 = schrodinger_rhs(
        time + 0.5 * step, unitary + (0.5 * step) * k1, final_time);
    const Matrix k3 = schrodinger_rhs(
        time + 0.5 * step, unitary + (0.5 * step) * k2, final_time);
    const Matrix k4 =
        schrodinger_rhs(time + step, unitary + step * k3, final_time);
    return unitary +
           (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

[[nodiscard]] std::vector<double>
chebyshev_root_times(int sample_count, double final_time) {
    if (sample_count < 2) {
        throw std::invalid_argument("sample count must be at least two");
    }
    std::vector<double> times(static_cast<std::size_t>(sample_count));
    for (int j = 0; j < sample_count; ++j) {
        const double theta =
            std::numbers::pi * (static_cast<double>(j) + 0.5) /
            static_cast<double>(sample_count);
        const double x = std::cos(theta);
        times[static_cast<std::size_t>(j)] =
            0.5 * final_time * (x + 1.0);
    }
    return times;
}

[[nodiscard]] std::vector<Matrix>
exact_unitaries(const std::vector<double>& times,
                double final_time,
                double max_step) {
    std::vector<std::pair<double, std::size_t>> ordered;
    ordered.reserve(times.size());
    for (std::size_t index = 0; index < times.size(); ++index) {
        ordered.emplace_back(times[index], index);
    }
    std::sort(ordered.begin(), ordered.end());

    std::vector<Matrix> result(times.size());
    Matrix unitary = Matrix::identity();
    double current_time = 0.0;
    for (const auto& [target_time, original_index] : ordered) {
        const double interval = target_time - current_time;
        const int steps =
            std::max(1, static_cast<int>(std::ceil(interval / max_step)));
        const double step = interval / static_cast<double>(steps);
        for (int iteration = 0; iteration < steps; ++iteration) {
            unitary = rk4_step(current_time, step, unitary, final_time);
            current_time += step;
        }
        current_time = target_time;
        result[original_index] = unitary;
    }
    return result;
}

[[nodiscard]] std::vector<Matrix>
add_noise(const std::vector<Matrix>& exact_values,
          double relative_rms_noise,
          std::uint64_t seed) {
    const double sigma =
        relative_rms_noise / std::sqrt(static_cast<double>(dimension));
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> normal(
        0.0, sigma / std::sqrt(2.0));
    std::vector<Matrix> noisy_values = exact_values;
    for (Matrix& matrix : noisy_values) {
        for (std::size_t row = 0; row < dimension; ++row) {
            for (std::size_t column = 0; column < dimension; ++column) {
                matrix(row, column) +=
                    Complex{normal(generator), normal(generator)};
            }
        }
    }
    return noisy_values;
}

[[nodiscard]] Matrix inverse(const Matrix& matrix) {
    Matrix left = matrix;
    Matrix right = Matrix::identity();
    double scale = 0.0;
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            scale = std::max(scale, std::abs(left(row, column)));
        }
    }
    for (std::size_t column = 0; column < dimension; ++column) {
        std::size_t pivot = column;
        double pivot_magnitude = std::abs(left(column, column));
        for (std::size_t row = column + 1; row < dimension; ++row) {
            const double candidate = std::abs(left(row, column));
            if (candidate > pivot_magnitude) {
                pivot = row;
                pivot_magnitude = candidate;
            }
        }
        if (pivot_magnitude <=
            64.0 * std::numeric_limits<double>::epsilon() * scale) {
            throw std::runtime_error(
                "polar projection received a numerically singular matrix");
        }
        if (pivot != column) {
            for (std::size_t entry = 0; entry < dimension; ++entry) {
                std::swap(left(column, entry), left(pivot, entry));
                std::swap(right(column, entry), right(pivot, entry));
            }
        }
        const Complex diagonal = left(column, column);
        for (std::size_t entry = 0; entry < dimension; ++entry) {
            left(column, entry) /= diagonal;
            right(column, entry) /= diagonal;
        }
        for (std::size_t row = 0; row < dimension; ++row) {
            if (row == column) {
                continue;
            }
            const Complex factor = left(row, column);
            for (std::size_t entry = 0; entry < dimension; ++entry) {
                left(row, entry) -= factor * left(column, entry);
                right(row, entry) -= factor * right(column, entry);
            }
        }
    }
    return right;
}

[[nodiscard]] Matrix project_to_unitary(const Matrix& matrix) {
    Matrix iterate = matrix;
    for (int iteration = 0; iteration < polar_max_iterations; ++iteration) {
        const Matrix next =
            0.5 * (iterate + inverse(iterate).adjoint());
        const double difference_squared =
            (next - iterate).frobenius_squared();
        const double iterate_scale_squared =
            std::max(1.0, next.frobenius_squared());
        iterate = next;
        if (difference_squared <=
            polar_relative_tolerance_squared * iterate_scale_squared) {
            return iterate;
        }
    }
    throw std::runtime_error("unitary polar iteration did not converge");
}

[[nodiscard]] std::pair<Matrix, Matrix>
project_to_unitary_with_derivative(const Matrix& matrix,
                                   const Matrix& derivative) {
    Matrix iterate = matrix;
    Matrix derivative_iterate = derivative;
    for (int iteration = 0; iteration < polar_max_iterations; ++iteration) {
        const Matrix inverse_adjoint = inverse(iterate).adjoint();
        const Matrix next = 0.5 * (iterate + inverse_adjoint);
        const Matrix derivative_next =
            0.5 * (derivative_iterate -
                   inverse_adjoint * derivative_iterate.adjoint() *
                       inverse_adjoint);
        const double difference_squared =
            (next - iterate).frobenius_squared();
        const double derivative_difference_squared =
            (derivative_next - derivative_iterate).frobenius_squared();
        const double iterate_scale_squared =
            std::max(1.0, next.frobenius_squared());
        const double derivative_scale_squared =
            std::max(1.0, derivative_next.frobenius_squared());
        iterate = next;
        derivative_iterate = derivative_next;
        if (difference_squared <=
                polar_relative_tolerance_squared * iterate_scale_squared &&
            derivative_difference_squared <=
                polar_relative_tolerance_squared *
                    derivative_scale_squared) {
            return {iterate, derivative_iterate};
        }
    }
    throw std::runtime_error(
        "differentiated unitary polar iteration did not converge");
}

[[nodiscard]] std::vector<Matrix>
chebyshev_coefficients(const std::vector<Matrix>& values) {
    const int sample_count = static_cast<int>(values.size());
    std::vector<Matrix> coefficients(values.size());
    for (int degree = 0; degree < sample_count; ++degree) {
        Matrix sum;
        for (int j = 0; j < sample_count; ++j) {
            const double angle =
                std::numbers::pi *
                (static_cast<double>(j) + 0.5) *
                static_cast<double>(degree) /
                static_cast<double>(sample_count);
            sum += std::cos(angle) * values[static_cast<std::size_t>(j)];
        }
        const double scale =
            degree == 0 ? 1.0 / sample_count : 2.0 / sample_count;
        coefficients[static_cast<std::size_t>(degree)] = scale * sum;
    }
    return coefficients;
}

[[nodiscard]] std::vector<Matrix>
differentiate_chebyshev(const std::vector<Matrix>& coefficients) {
    const int highest_degree = static_cast<int>(coefficients.size()) - 1;
    std::vector<Matrix> derivative(
        static_cast<std::size_t>(highest_degree));
    derivative[static_cast<std::size_t>(highest_degree - 1)] =
        (2.0 * highest_degree) *
        coefficients[static_cast<std::size_t>(highest_degree)];
    if (highest_degree > 1) {
        derivative[static_cast<std::size_t>(highest_degree - 2)] =
            (2.0 * (highest_degree - 1)) *
            coefficients[static_cast<std::size_t>(highest_degree - 1)];
    }
    for (int degree = highest_degree - 3; degree >= 0; --degree) {
        derivative[static_cast<std::size_t>(degree)] =
            derivative[static_cast<std::size_t>(degree + 2)] +
            (2.0 * (degree + 1)) *
                coefficients[static_cast<std::size_t>(degree + 1)];
    }
    derivative[0] *= Complex{0.5, 0.0};
    return derivative;
}

[[nodiscard]] Matrix
evaluate_chebyshev(const std::vector<Matrix>& coefficients, double x) {
    Matrix result = coefficients[0];
    if (coefficients.size() == 1) {
        return result;
    }
    double previous = 1.0;
    double current = x;
    result += current * coefficients[1];
    for (std::size_t degree = 2; degree < coefficients.size(); ++degree) {
        const double next = 2.0 * x * current - previous;
        result += next * coefficients[degree];
        previous = current;
        current = next;
    }
    return result;
}

[[nodiscard]] Matrix hermitian_traceless(const Matrix& matrix) {
    Matrix result = 0.5 * (matrix + matrix.adjoint());
    const Complex mean_trace =
        result.trace() / static_cast<double>(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        result(index, index) -= mean_trace;
    }
    return result;
}

enum class Algorithm {
    unconstrained,
    polar_projected,
    unitary_curve,
};

[[nodiscard]] std::string_view algorithm_name(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::unconstrained:
            return "unconstrained";
        case Algorithm::polar_projected:
            return "polar_projected";
        case Algorithm::unitary_curve:
            return "unitary_curve";
    }
    throw std::invalid_argument("unknown algorithm");
}

struct LearnedModel {
    Algorithm algorithm = Algorithm::unconstrained;
    std::vector<Matrix> coefficients;
    std::vector<Matrix> derivative_coefficients;
};

[[nodiscard]] LearnedModel learn(const std::vector<Matrix>& noisy_samples,
                                 Algorithm algorithm) {
    std::vector<Matrix> projected_samples;
    const std::vector<Matrix>* samples = &noisy_samples;
    if (algorithm == Algorithm::polar_projected ||
        algorithm == Algorithm::unitary_curve) {
        projected_samples.reserve(noisy_samples.size());
        for (const Matrix& sample : noisy_samples) {
            projected_samples.push_back(project_to_unitary(sample));
        }
        samples = &projected_samples;
    }
    LearnedModel model;
    model.algorithm = algorithm;
    model.coefficients = chebyshev_coefficients(*samples);
    model.derivative_coefficients =
        differentiate_chebyshev(model.coefficients);
    return model;
}

[[nodiscard]] Matrix reconstruct_hamiltonian(const LearnedModel& model,
                                             double x,
                                             double final_time) {
    const Matrix estimated_unitary =
        evaluate_chebyshev(model.coefficients, x);
    const Matrix derivative_x =
        evaluate_chebyshev(model.derivative_coefficients, x);
    const Matrix derivative_t = (2.0 / final_time) * derivative_x;
    if (model.algorithm == Algorithm::unitary_curve) {
        const auto [unitary, derivative_unitary] =
            project_to_unitary_with_derivative(estimated_unitary,
                                               derivative_t);
        return hermitian_traceless(
            imaginary_unit * derivative_unitary * unitary.adjoint());
    }
    return hermitian_traceless(
        imaginary_unit * derivative_t * estimated_unitary.adjoint());
}

[[nodiscard]] double reconstruct_grid_checksum(
    const LearnedModel& model,
    const std::vector<double>& nodes,
    double final_time) {
    double checksum = 0.0;
    for (const double x : nodes) {
        checksum +=
            reconstruct_hamiltonian(model, x, final_time).
                frobenius_squared();
    }
    return checksum;
}

void validate_unitary_curve(const LearnedModel& model,
                            double final_time) {
    constexpr int validation_intervals = 1024;
    for (int index = 0; index <= validation_intervals; ++index) {
        const double x =
            -1.0 + 2.0 * static_cast<double>(index) /
                       static_cast<double>(validation_intervals);
        const Matrix polynomial =
            evaluate_chebyshev(model.coefficients, x);
        const Matrix derivative_x =
            evaluate_chebyshev(model.derivative_coefficients, x);
        const auto [unitary, derivative_unitary] =
            project_to_unitary_with_derivative(
                polynomial, (2.0 / final_time) * derivative_x);
        if ((unitary.adjoint() * unitary -
             Matrix::identity()).frobenius_squared() > 1.0e-24) {
            throw std::runtime_error(
                "unitary curve failed dense-grid unitarity validation");
        }
        const Matrix tangent_residual =
            unitary.adjoint() * derivative_unitary +
            derivative_unitary.adjoint() * unitary;
        if (tangent_residual.frobenius_squared() > 1.0e-22) {
            throw std::runtime_error(
                "unitary curve derivative failed dense-grid tangent validation");
        }
    }

    constexpr double validation_x = 0.137;
    constexpr double difference_step = 1.0e-5;
    const Matrix polynomial =
        evaluate_chebyshev(model.coefficients, validation_x);
    const Matrix derivative_x =
        evaluate_chebyshev(model.derivative_coefficients, validation_x);
    const auto [unitary, derivative_unitary] =
        project_to_unitary_with_derivative(
            polynomial, (2.0 / final_time) * derivative_x);
    (void)unitary;
    const Matrix plus = project_to_unitary(
        evaluate_chebyshev(model.coefficients,
                           validation_x + difference_step));
    const Matrix minus = project_to_unitary(
        evaluate_chebyshev(model.coefficients,
                           validation_x - difference_step));
    const Matrix finite_difference =
        (1.0 / (difference_step * final_time)) * (plus - minus);
    const double relative_difference_squared =
        (finite_difference - derivative_unitary).frobenius_squared() /
        std::max(1.0, derivative_unitary.frobenius_squared());
    if (relative_difference_squared > 1.0e-16) {
        throw std::runtime_error(
            "differentiated polar iteration failed finite-difference validation");
    }
}

[[nodiscard]] std::pair<std::vector<double>, std::vector<double>>
gauss_legendre_rule(int order) {
    std::vector<double> nodes(static_cast<std::size_t>(order));
    std::vector<double> weights(static_cast<std::size_t>(order));
    const int half = (order + 1) / 2;
    constexpr double tolerance = 4.0 * std::numeric_limits<double>::epsilon();
    for (int i = 0; i < half; ++i) {
        double root = std::cos(
            std::numbers::pi * (static_cast<double>(i) + 0.75) /
            (static_cast<double>(order) + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 100; ++iteration) {
            double p_previous = 1.0;
            double p_current = root;
            for (int degree = 2; degree <= order; ++degree) {
                const double p_next =
                    ((2.0 * degree - 1.0) * root * p_current -
                     (degree - 1.0) * p_previous) /
                    static_cast<double>(degree);
                p_previous = p_current;
                p_current = p_next;
            }
            derivative =
                static_cast<double>(order) *
                (root * p_current - p_previous) /
                (root * root - 1.0);
            const double update = p_current / derivative;
            root -= update;
            if (std::abs(update) <= tolerance) {
                break;
            }
        }
        const double weight =
            2.0 / ((1.0 - root * root) * derivative * derivative);
        nodes[static_cast<std::size_t>(i)] = -root;
        nodes[static_cast<std::size_t>(order - 1 - i)] = root;
        weights[static_cast<std::size_t>(i)] = weight;
        weights[static_cast<std::size_t>(order - 1 - i)] = weight;
    }
    return {std::move(nodes), std::move(weights)};
}

template <typename Work>
[[nodiscard]] double best_average_milliseconds(int repetitions,
                                               Work&& work) {
    double best = std::numeric_limits<double>::infinity();
    for (int batch = 0; batch < timing_batches; ++batch) {
        const auto start = std::chrono::steady_clock::now();
        double checksum = 0.0;
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            checksum += work();
        }
        const auto end = std::chrono::steady_clock::now();
        volatile double timing_sink = checksum;
        (void)timing_sink;
        best = std::min(
            best,
            std::chrono::duration<double, std::milli>(end - start).count() /
                repetitions);
    }
    return best;
}

struct Result {
    double relative_noise = 0.0;
    int sample_count = 0;
    Algorithm algorithm = Algorithm::unconstrained;
    double relative_l2_error = 0.0;
    double fit_milliseconds = 0.0;
    double evaluation_256_milliseconds = 0.0;
    double data_generation_milliseconds = 0.0;
};

[[nodiscard]] std::vector<Result> run_benchmark(double ode_step) {
    constexpr double final_time = 2.0;
    constexpr int quadrature_order = 256;
    constexpr std::uint64_t seed = 20260727ULL;
    constexpr std::array sample_counts{3, 5, 8, 12, 16};
    constexpr std::array relative_noise_rates{0.005, 0.01, 0.02, 0.03};
    constexpr std::array algorithms{
        Algorithm::unconstrained,
        Algorithm::polar_projected,
        Algorithm::unitary_curve,
    };
    const auto [nodes, weights] = gauss_legendre_rule(quadrature_order);
    const Matrix check_hamiltonian =
        target_hamiltonian(0.37 * final_time, final_time);
    if ((check_hamiltonian - check_hamiltonian.adjoint()).
            frobenius_squared() > 1.0e-27 ||
        std::norm(check_hamiltonian.trace()) > 1.0e-27) {
        throw std::runtime_error(
            "four-qubit Hamiltonian failed Hermitian/traceless validation");
    }
    std::vector<Result> results;
    results.reserve(sample_counts.size() *
                    relative_noise_rates.size() *
                    algorithms.size());

    for (const int sample_count : sample_counts) {
        const std::vector<double> times =
            chebyshev_root_times(sample_count, final_time);
        const std::vector<Matrix> exact_samples =
            exact_unitaries(times, final_time, ode_step);

        const double dynamics_generation_ms =
            best_average_milliseconds(3, [&]() {
                const std::vector<Matrix> samples =
                    exact_unitaries(times, final_time, ode_step);
                return samples.back().frobenius_squared();
            });
        const double noise_generation_ms =
            best_average_milliseconds(100, [&]() {
                const std::vector<Matrix> samples =
                    add_noise(exact_samples, 0.01, seed);
                return samples.back().frobenius_squared();
            });
        const double data_generation_ms =
            dynamics_generation_ms + noise_generation_ms;

        for (const double relative_noise : relative_noise_rates) {
            const std::vector<Matrix> noisy_samples =
                add_noise(exact_samples, relative_noise, seed);
            const Matrix projected_check =
                project_to_unitary(noisy_samples.front());
            if ((projected_check.adjoint() * projected_check -
                 Matrix::identity()).frobenius_squared() > 1.0e-24) {
                throw std::runtime_error(
                    "polar projection failed unitarity validation");
            }
            for (const Algorithm algorithm : algorithms) {
                const int fit_repetitions =
                    algorithm == Algorithm::unconstrained ? 100 : 10;
                const double fit_ms =
                    best_average_milliseconds(fit_repetitions, [&]() {
                        const LearnedModel model =
                            learn(noisy_samples, algorithm);
                        return model.coefficients.front().frobenius_squared() +
                               model.derivative_coefficients.front().
                                   frobenius_squared();
                    });
                const LearnedModel model = learn(noisy_samples, algorithm);
                const int evaluation_repetitions =
                    algorithm == Algorithm::unitary_curve ? 3 : 10;
                const double evaluation_256_ms =
                    best_average_milliseconds(
                        evaluation_repetitions,
                        [&]() {
                            return reconstruct_grid_checksum(
                                model, nodes, final_time);
                        });
                if (algorithm == Algorithm::unitary_curve) {
                    validate_unitary_curve(model, final_time);
                }
                double numerator = 0.0;
                double denominator = 0.0;
                for (std::size_t index = 0; index < nodes.size(); ++index) {
                    const double x = nodes[index];
                    const double time = 0.5 * final_time * (x + 1.0);
                    const Matrix estimate =
                        reconstruct_hamiltonian(model, x, final_time);
                    const Matrix exact =
                        target_hamiltonian(time, final_time);
                    numerator +=
                        weights[index] *
                        (estimate - exact).frobenius_squared();
                    denominator +=
                        weights[index] * exact.frobenius_squared();
                }
                results.push_back(Result{
                    relative_noise,
                    sample_count,
                    algorithm,
                    std::sqrt(numerator / denominator),
                    fit_ms,
                    evaluation_256_ms,
                    data_generation_ms,
                });
            }
        }
    }
    return results;
}

void write_results(const std::vector<Result>& results,
                   const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("could not open result CSV");
    }
    output << "seed,qubits,relative_rms_noise,sigma,algorithm,M,"
              "relative_l2_error,fit_ms,evaluation_256_ms,"
              "total_postprocessing_256_ms,data_generation_ms\n";
    output << std::setprecision(17);
    for (const Result& result : results) {
        output << 20260727 << ','
               << qubits << ','
               << result.relative_noise << ','
               << result.relative_noise /
                      std::sqrt(static_cast<double>(dimension)) << ','
               << algorithm_name(result.algorithm) << ','
               << result.sample_count << ','
               << result.relative_l2_error << ','
               << result.fit_milliseconds << ','
               << result.evaluation_256_milliseconds << ','
               << result.fit_milliseconds +
                      result.evaluation_256_milliseconds << ','
               << result.data_generation_milliseconds << '\n';
    }
}

void print_results(const std::vector<Result>& results) {
    std::cout << std::left
              << std::setw(9) << "noise"
              << std::setw(6) << "M"
              << std::setw(19) << "algorithm"
              << std::setw(18) << "relative error"
              << std::setw(15) << "fit [ms]"
              << std::setw(18) << "eval 256 [ms]"
              << "data gen [ms]\n"
              << std::setprecision(8);
    for (const Result& result : results) {
        std::cout << std::left
                  << std::setw(9) << result.relative_noise
                  << std::setw(6) << result.sample_count
                  << std::setw(19) << algorithm_name(result.algorithm)
                  << std::setw(18) << result.relative_l2_error
                  << std::setw(15) << result.fit_milliseconds
                  << std::setw(18) << result.evaluation_256_milliseconds
                  << result.data_generation_milliseconds << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        double ode_step = 2.0e-4;
        if (argc == 3 && std::string_view(argv[1]) == "--ode-step") {
            ode_step = std::stod(argv[2]);
        } else if (argc != 1) {
            std::cerr << "Usage: " << argv[0]
                      << " [--ode-step STEP]\n";
            return 1;
        }
        const std::vector<Result> results = run_benchmark(ode_step);
        write_results(
            results, "results/four_qubit_noise_sweep.csv");
        print_results(results);
        std::cout << "Wrote results/four_qubit_noise_sweep.csv\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
