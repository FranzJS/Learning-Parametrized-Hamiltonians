#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace lph {

using Complex = std::complex<double>;

class Matrix4 {
public:
    static constexpr std::size_t dimension = 4;
    static constexpr std::size_t entries = dimension * dimension;

    Matrix4() = default;

    [[nodiscard]] static Matrix4 identity();
    [[nodiscard]] Complex& operator()(std::size_t row, std::size_t column);
    [[nodiscard]] const Complex& operator()(std::size_t row,
                                            std::size_t column) const;
    [[nodiscard]] Matrix4 adjoint() const;
    [[nodiscard]] Complex trace() const;
    [[nodiscard]] double frobenius_squared() const;

    Matrix4& operator+=(const Matrix4& other);
    Matrix4& operator-=(const Matrix4& other);
    Matrix4& operator*=(Complex scalar);

private:
    std::array<Complex, entries> data_{};
};

[[nodiscard]] Matrix4 operator+(Matrix4 left, const Matrix4& right);
[[nodiscard]] Matrix4 operator-(Matrix4 left, const Matrix4& right);
[[nodiscard]] Matrix4 operator*(const Matrix4& left, const Matrix4& right);
[[nodiscard]] Matrix4 operator*(Matrix4 matrix, Complex scalar);
[[nodiscard]] Matrix4 operator*(Complex scalar, Matrix4 matrix);
[[nodiscard]] Matrix4 operator*(Matrix4 matrix, double scalar);
[[nodiscard]] Matrix4 operator*(double scalar, Matrix4 matrix);

struct BenchmarkConfig {
    std::vector<int> sample_counts{3, 5, 8, 12, 16};
    std::uint64_t seed = 20260727ULL;
    double sigma = 0.005;
    double final_time = 2.0;
    double ode_max_step = 1.0e-4;
    int quadrature_order = 256;
    int fit_timing_repetitions = 2000;
    int reconstruction_timing_repetitions = 200;
    std::filesystem::path output =
        "results/chebyshev_baseline.csv";
};

struct BenchmarkResult {
    int sample_count = 0;
    double relative_l2_error = 0.0;
    double fit_milliseconds = 0.0;
    double reconstruction_milliseconds = 0.0;
    double postprocessing_milliseconds = 0.0;
};

[[nodiscard]] Matrix4 target_hamiltonian(double time,
                                         double final_time = 2.0);

[[nodiscard]] std::vector<Matrix4>
chebyshev_coefficients(const std::vector<Matrix4>& values);

[[nodiscard]] std::vector<Matrix4>
differentiate_chebyshev(const std::vector<Matrix4>& coefficients);

[[nodiscard]] Matrix4
evaluate_chebyshev(const std::vector<Matrix4>& coefficients, double x);

[[nodiscard]] std::vector<BenchmarkResult>
run_benchmark(const BenchmarkConfig& config);

void write_results(const BenchmarkConfig& config,
                   const std::vector<BenchmarkResult>& results);

}  // namespace lph
