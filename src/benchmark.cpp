#include "lph/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <utility>

namespace lph {
namespace {

constexpr Complex imaginary_unit{0.0, 1.0};
constexpr int timing_batches = 5;

template <typename Work>
double best_average_milliseconds(int repetitions, Work&& work) {
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
        const double average =
            std::chrono::duration<double, std::milli>(end - start).count() /
            repetitions;
        best = std::min(best, average);
    }
    return best;
}

Matrix4 kronecker(const std::array<Complex, 4>& left,
                  const std::array<Complex, 4>& right) {
    Matrix4 result;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            for (std::size_t k = 0; k < 2; ++k) {
                for (std::size_t l = 0; l < 2; ++l) {
                    result(2 * i + k, 2 * j + l) =
                        left[2 * i + j] * right[2 * k + l];
                }
            }
        }
    }
    return result;
}

const Matrix4& pauli_zi() {
    static const std::array<Complex, 4> identity{
        Complex{1.0, 0.0}, Complex{}, Complex{}, Complex{1.0, 0.0}};
    static const std::array<Complex, 4> z{
        Complex{1.0, 0.0}, Complex{}, Complex{}, Complex{-1.0, 0.0}};
    static const Matrix4 value = kronecker(z, identity);
    return value;
}

const Matrix4& pauli_ix() {
    static const std::array<Complex, 4> identity{
        Complex{1.0, 0.0}, Complex{}, Complex{}, Complex{1.0, 0.0}};
    static const std::array<Complex, 4> x{
        Complex{}, Complex{1.0, 0.0}, Complex{1.0, 0.0}, Complex{}};
    static const Matrix4 value = kronecker(identity, x);
    return value;
}

const Matrix4& pauli_yz() {
    static const std::array<Complex, 4> y{
        Complex{}, Complex{0.0, -1.0}, Complex{0.0, 1.0}, Complex{}};
    static const std::array<Complex, 4> z{
        Complex{1.0, 0.0}, Complex{}, Complex{}, Complex{-1.0, 0.0}};
    static const Matrix4 value = kronecker(y, z);
    return value;
}

double chebyshev_t(int degree, double x) {
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

Matrix4 schrodinger_rhs(double time,
                        const Matrix4& unitary,
                        double final_time) {
    return -imaginary_unit * (target_hamiltonian(time, final_time) * unitary);
}

Matrix4 rk4_step(double time,
                 double step,
                 const Matrix4& unitary,
                 double final_time) {
    const Matrix4 k1 = schrodinger_rhs(time, unitary, final_time);
    const Matrix4 k2 = schrodinger_rhs(
        time + 0.5 * step, unitary + (0.5 * step) * k1, final_time);
    const Matrix4 k3 = schrodinger_rhs(
        time + 0.5 * step, unitary + (0.5 * step) * k2, final_time);
    const Matrix4 k4 =
        schrodinger_rhs(time + step, unitary + step * k3, final_time);
    return unitary +
           (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

std::vector<Matrix4> exact_unitaries(const std::vector<double>& times,
                                     double final_time,
                                     double max_step) {
    if (!(final_time > 0.0) || !(max_step > 0.0)) {
        throw std::invalid_argument("final_time and ode_max_step must be positive");
    }

    std::vector<std::pair<double, std::size_t>> ordered;
    ordered.reserve(times.size());
    for (std::size_t index = 0; index < times.size(); ++index) {
        if (times[index] < 0.0 || times[index] > final_time) {
            throw std::invalid_argument("query time outside [0, final_time]");
        }
        ordered.emplace_back(times[index], index);
    }
    std::sort(ordered.begin(), ordered.end());

    std::vector<Matrix4> result(times.size());
    Matrix4 unitary = Matrix4::identity();
    double current_time = 0.0;
    for (const auto& [target_time, original_index] : ordered) {
        const double interval = target_time - current_time;
        const int number_of_steps =
            std::max(1, static_cast<int>(std::ceil(interval / max_step)));
        const double step = interval / static_cast<double>(number_of_steps);
        for (int iteration = 0; iteration < number_of_steps; ++iteration) {
            unitary = rk4_step(current_time, step, unitary, final_time);
            current_time += step;
        }
        current_time = target_time;
        result[original_index] = unitary;
    }
    return result;
}

std::vector<double> chebyshev_root_times(int sample_count,
                                         double final_time) {
    if (sample_count < 2) {
        throw std::invalid_argument("sample count must be at least two");
    }
    std::vector<double> times(static_cast<std::size_t>(sample_count));
    for (int j = 0; j < sample_count; ++j) {
        const double theta =
            std::numbers::pi * (static_cast<double>(j) + 0.5) /
            static_cast<double>(sample_count);
        const double x = std::cos(theta);
        times[static_cast<std::size_t>(j)] = 0.5 * final_time * (x + 1.0);
    }
    return times;
}

std::vector<Matrix4> add_noise(const std::vector<Matrix4>& exact_values,
                               double sigma,
                               std::uint64_t seed) {
    if (sigma < 0.0) {
        throw std::invalid_argument("sigma must be nonnegative");
    }
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> normal(
        0.0, sigma / std::sqrt(2.0));

    std::vector<Matrix4> noisy_values = exact_values;
    for (Matrix4& matrix : noisy_values) {
        for (std::size_t row = 0; row < Matrix4::dimension; ++row) {
            for (std::size_t column = 0;
                 column < Matrix4::dimension;
                 ++column) {
                matrix(row, column) += Complex{normal(generator),
                                               normal(generator)};
            }
        }
    }
    return noisy_values;
}

std::pair<std::vector<double>, std::vector<double>>
gauss_legendre_rule(int order) {
    if (order < 2) {
        throw std::invalid_argument("quadrature order must be at least two");
    }
    std::vector<double> nodes(static_cast<std::size_t>(order));
    std::vector<double> weights(static_cast<std::size_t>(order));
    const int half = (order + 1) / 2;
    constexpr double tolerance = 4.0 * std::numeric_limits<double>::epsilon();

    for (int i = 0; i < half; ++i) {
        double root = std::cos(
            std::numbers::pi *
            (static_cast<double>(i) + 0.75) /
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

Matrix4 hermitian_traceless(const Matrix4& matrix) {
    Matrix4 hermitian = 0.5 * (matrix + matrix.adjoint());
    const Complex mean_trace =
        hermitian.trace() / static_cast<double>(Matrix4::dimension);
    for (std::size_t index = 0; index < Matrix4::dimension; ++index) {
        hermitian(index, index) -= mean_trace;
    }
    return hermitian;
}

std::vector<Matrix4> reconstruct_hamiltonians(
    const std::vector<Matrix4>& coefficients,
    const std::vector<Matrix4>& derivative_coefficients,
    const std::vector<double>& evaluation_nodes,
    double final_time) {
    std::vector<Matrix4> estimates;
    estimates.reserve(evaluation_nodes.size());
    for (const double x : evaluation_nodes) {
        const Matrix4 estimated_unitary =
            evaluate_chebyshev(coefficients, x);
        const Matrix4 derivative_x =
            evaluate_chebyshev(derivative_coefficients, x);
        const Matrix4 derivative_t = (2.0 / final_time) * derivative_x;
        estimates.push_back(hermitian_traceless(
            imaginary_unit * derivative_t * estimated_unitary.adjoint()));
    }
    return estimates;
}

}  // namespace

Matrix4 Matrix4::identity() {
    Matrix4 result;
    for (std::size_t index = 0; index < dimension; ++index) {
        result(index, index) = Complex{1.0, 0.0};
    }
    return result;
}

Complex& Matrix4::operator()(std::size_t row, std::size_t column) {
    return data_[row * dimension + column];
}

const Complex& Matrix4::operator()(std::size_t row,
                                   std::size_t column) const {
    return data_[row * dimension + column];
}

Matrix4 Matrix4::adjoint() const {
    Matrix4 result;
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            result(column, row) = std::conj((*this)(row, column));
        }
    }
    return result;
}

Complex Matrix4::trace() const {
    Complex result{};
    for (std::size_t index = 0; index < dimension; ++index) {
        result += (*this)(index, index);
    }
    return result;
}

double Matrix4::frobenius_squared() const {
    double result = 0.0;
    for (const Complex value : data_) {
        result += std::norm(value);
    }
    return result;
}

Matrix4& Matrix4::operator+=(const Matrix4& other) {
    for (std::size_t index = 0; index < entries; ++index) {
        data_[index] += other.data_[index];
    }
    return *this;
}

Matrix4& Matrix4::operator-=(const Matrix4& other) {
    for (std::size_t index = 0; index < entries; ++index) {
        data_[index] -= other.data_[index];
    }
    return *this;
}

Matrix4& Matrix4::operator*=(Complex scalar) {
    for (Complex& value : data_) {
        value *= scalar;
    }
    return *this;
}

Matrix4 operator+(Matrix4 left, const Matrix4& right) {
    left += right;
    return left;
}

Matrix4 operator-(Matrix4 left, const Matrix4& right) {
    left -= right;
    return left;
}

Matrix4 operator*(const Matrix4& left, const Matrix4& right) {
    Matrix4 result;
    for (std::size_t row = 0; row < Matrix4::dimension; ++row) {
        for (std::size_t inner = 0; inner < Matrix4::dimension; ++inner) {
            const Complex value = left(row, inner);
            for (std::size_t column = 0;
                 column < Matrix4::dimension;
                 ++column) {
                result(row, column) += value * right(inner, column);
            }
        }
    }
    return result;
}

Matrix4 operator*(Matrix4 matrix, Complex scalar) {
    matrix *= scalar;
    return matrix;
}

Matrix4 operator*(Complex scalar, Matrix4 matrix) {
    matrix *= scalar;
    return matrix;
}

Matrix4 operator*(Matrix4 matrix, double scalar) {
    matrix *= Complex{scalar, 0.0};
    return matrix;
}

Matrix4 operator*(double scalar, Matrix4 matrix) {
    matrix *= Complex{scalar, 0.0};
    return matrix;
}

Matrix4 target_hamiltonian(double time, double final_time) {
    const double x = 2.0 * time / final_time - 1.0;
    const double a =
        0.60 + 0.25 * chebyshev_t(1, x) - 0.10 * chebyshev_t(3, x);
    const double b =
        -0.45 + 0.20 * chebyshev_t(2, x) + 0.12 * chebyshev_t(5, x);
    const double c =
        0.50 + 0.18 * chebyshev_t(1, x) - 0.14 * chebyshev_t(4, x);
    return a * pauli_zi() + b * pauli_ix() + c * pauli_yz();
}

std::vector<Matrix4>
chebyshev_coefficients(const std::vector<Matrix4>& values) {
    const int sample_count = static_cast<int>(values.size());
    if (sample_count < 2) {
        throw std::invalid_argument("at least two samples are required");
    }

    std::vector<Matrix4> coefficients(values.size());
    for (int degree = 0; degree < sample_count; ++degree) {
        Matrix4 sum;
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

std::vector<Matrix4>
differentiate_chebyshev(const std::vector<Matrix4>& coefficients) {
    if (coefficients.size() < 2) {
        return {Matrix4{}};
    }
    const int highest_degree = static_cast<int>(coefficients.size()) - 1;
    std::vector<Matrix4> derivative(
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

Matrix4 evaluate_chebyshev(const std::vector<Matrix4>& coefficients,
                           double x) {
    if (coefficients.empty()) {
        return Matrix4{};
    }
    Matrix4 result = coefficients[0];
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

std::vector<BenchmarkResult>
run_benchmark(const BenchmarkConfig& config) {
    if (config.sample_counts.empty()) {
        throw std::invalid_argument("at least one sample count is required");
    }
    if (config.fit_timing_repetitions < 1 ||
        config.reconstruction_timing_repetitions < 1) {
        throw std::invalid_argument("timing repetitions must be positive");
    }
    const auto [quadrature_nodes, quadrature_weights] =
        gauss_legendre_rule(config.quadrature_order);

    std::vector<BenchmarkResult> results;
    results.reserve(config.sample_counts.size());
    for (const int sample_count : config.sample_counts) {
        const std::vector<double> sample_times =
            chebyshev_root_times(sample_count, config.final_time);
        const std::vector<Matrix4> exact_samples =
            exact_unitaries(sample_times,
                            config.final_time,
                            config.ode_max_step);
        const std::vector<Matrix4> noisy_samples =
            add_noise(exact_samples, config.sigma, config.seed);

        const double fit_milliseconds = best_average_milliseconds(
            config.fit_timing_repetitions,
            [&noisy_samples]() {
                const std::vector<Matrix4> timed_coefficients =
                    chebyshev_coefficients(noisy_samples);
                const std::vector<Matrix4> timed_derivative_coefficients =
                    differentiate_chebyshev(timed_coefficients);
                return timed_coefficients.front().frobenius_squared() +
                       timed_derivative_coefficients.front().
                           frobenius_squared();
            });

        const std::vector<Matrix4> coefficients =
            chebyshev_coefficients(noisy_samples);
        const std::vector<Matrix4> derivative_coefficients =
            differentiate_chebyshev(coefficients);

        const double reconstruction_milliseconds = best_average_milliseconds(
            config.reconstruction_timing_repetitions,
            [&coefficients,
             &derivative_coefficients,
             &quadrature_nodes,
             &config]() {
                const std::vector<Matrix4> timed_estimates =
                    reconstruct_hamiltonians(coefficients,
                                             derivative_coefficients,
                                             quadrature_nodes,
                                             config.final_time);
                return timed_estimates.back().frobenius_squared();
            });

        const std::vector<Matrix4> estimated_hamiltonians =
            reconstruct_hamiltonians(coefficients,
                                     derivative_coefficients,
                                     quadrature_nodes,
                                     config.final_time);
        double numerator = 0.0;
        double denominator = 0.0;
        for (std::size_t index = 0;
             index < quadrature_nodes.size();
             ++index) {
            const double x = quadrature_nodes[index];
            const double time = 0.5 * config.final_time * (x + 1.0);
            const Matrix4 exact_hamiltonian =
                target_hamiltonian(time, config.final_time);
            numerator += quadrature_weights[index] *
                         (estimated_hamiltonians[index] -
                          exact_hamiltonian).frobenius_squared();
            denominator += quadrature_weights[index] *
                           exact_hamiltonian.frobenius_squared();
        }

        const double relative_error = std::sqrt(numerator / denominator);
        results.push_back(BenchmarkResult{
            sample_count,
            relative_error,
            fit_milliseconds,
            reconstruction_milliseconds,
            fit_milliseconds + reconstruction_milliseconds});
    }
    return results;
}

void write_results(const BenchmarkConfig& config,
                   const std::vector<BenchmarkResult>& results) {
    if (config.output.has_parent_path()) {
        std::filesystem::create_directories(config.output.parent_path());
    }
    std::ofstream output(config.output);
    if (!output) {
        throw std::runtime_error("could not open output CSV");
    }
    output << "seed,sigma,M,relative_l2_error,fit_ms,"
              "reconstruction_256_ms,postprocessing_256_ms\n";
    output << std::setprecision(17);
    for (const BenchmarkResult& result : results) {
        output << config.seed << ','
               << config.sigma << ','
               << result.sample_count << ','
               << result.relative_l2_error << ','
               << result.fit_milliseconds << ','
               << result.reconstruction_milliseconds << ','
               << result.postprocessing_milliseconds << '\n';
    }
}

}  // namespace lph
