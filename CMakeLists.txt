cmake_minimum_required(VERSION 3.16)
project(ecs-bench-lemonhash)

# -O2: GCC 15 miscompiles LeMonHash/SimpleRibbon at -O3.
add_compile_options(-march=native -Wno-template-body)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "" FORCE)

add_subdirectory(MMPHF-Experiments/cpp/extlib/LeMonHash)

add_executable(bench_lemonhash bench_lemonhash.cpp)
target_compile_features(bench_lemonhash PRIVATE cxx_std_20)
target_link_libraries(bench_lemonhash PRIVATE LeMonHash)
