cmake_minimum_required(VERSION 3.20)

project(learning_parametrized_hamiltonians
        VERSION 0.1.0
        LANGUAGES CXX)

option(LPH_ENABLE_NATIVE "Enable architecture-specific compiler optimizations" OFF)

add_library(lph
    src/benchmark.cpp
)
target_include_directories(lph PUBLIC include)
target_compile_features(lph PUBLIC cxx_std_20)
target_compile_options(lph PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)

if(LPH_ENABLE_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lph PRIVATE -march=native)
endif()

add_executable(lph_benchmark app/main.cpp)
target_link_libraries(lph_benchmark PRIVATE lph)

add_executable(lph_four_qubit_benchmark app/four_qubit.cpp)
target_compile_features(lph_four_qubit_benchmark PRIVATE cxx_std_20)
target_compile_options(lph_four_qubit_benchmark PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)
if(LPH_ENABLE_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lph_four_qubit_benchmark PRIVATE -march=native)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(lph_tests tests/test_benchmark.cpp)
    target_link_libraries(lph_tests PRIVATE lph)
    add_test(NAME lph_tests COMMAND lph_tests)
endif()
