#include "lph/benchmark.hpp"

#include <charconv>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

template <typename Integer>
Integer parse_integer(std::string_view text, std::string_view option) {
    Integer value{};
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("invalid value for " +
                                    std::string(option));
    }
    return value;
}

double parse_double(std::string_view text, std::string_view option) {
    char* end = nullptr;
    const std::string owned(text);
    const double value = std::strtod(owned.c_str(), &end);
    if (end != owned.c_str() + owned.size()) {
        throw std::invalid_argument("invalid value for " +
                                    std::string(option));
    }
    return value;
}

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --seed N          Noise seed (default: 20260727)\n"
        << "  --sigma X         Complex-noise standard deviation "
           "(default: 0.005)\n"
        << "  --quadrature N    Gauss-Legendre order (default: 256)\n"
        << "  --ode-step X      Maximum RK4 step (default: 1e-4)\n"
        << "  --output PATH     CSV output path\n"
        << "  --help            Show this message\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        lph::BenchmarkConfig config;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--help") {
                print_usage(argv[0]);
                return 0;
            }
            if (index + 1 >= argc) {
                throw std::invalid_argument("missing value for " +
                                            std::string(argument));
            }
            const std::string_view value(argv[++index]);
            if (argument == "--seed") {
                config.seed =
                    parse_integer<std::uint64_t>(value, argument);
            } else if (argument == "--sigma") {
                config.sigma = parse_double(value, argument);
            } else if (argument == "--quadrature") {
                config.quadrature_order =
                    parse_integer<int>(value, argument);
            } else if (argument == "--ode-step") {
                config.ode_max_step = parse_double(value, argument);
            } else if (argument == "--output") {
                config.output = value;
            } else {
                throw std::invalid_argument("unknown option " +
                                            std::string(argument));
            }
        }

        const auto results = lph::run_benchmark(config);
        lph::write_results(config, results);

        std::cout << "Chebyshev comparison: seed=" << config.seed
                  << ", sigma=" << config.sigma << '\n';
        std::cout << std::left << std::setw(19) << "algorithm"
                  << std::setw(8) << "M"
                  << std::setw(22) << "relative L2 error"
                  << std::setw(18) << "learning [ms]"
                  << "reconstruct 256 [ms]\n";
        std::cout << std::setprecision(8);
        for (const lph::BenchmarkResult& result : results) {
            std::cout << std::left << std::setw(19)
                      << lph::algorithm_name(result.algorithm)
                      << std::setw(8)
                      << result.sample_count
                      << std::setw(22) << result.relative_l2_error
                      << std::setw(18) << result.learning_milliseconds
                      << result.reconstruction_milliseconds << '\n';
        }
        std::cout << "Wrote " << config.output << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
