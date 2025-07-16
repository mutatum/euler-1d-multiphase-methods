cmake_minimum_required(VERSION 3.10)

project(DGTests CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find parent directory for the main project
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

# Add the parent directory to include path
include_directories(${PARENT_DIR})

# Add Eigen
find_package(Eigen3 REQUIRED)

# Create a library from the main project headers
add_library(dg_lib INTERFACE)
target_include_directories(dg_lib INTERFACE ${PARENT_DIR}/src)
target_link_libraries(dg_lib INTERFACE Eigen3::Eigen)

# Function to create a test executable
function(create_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE dg_lib)
    
    # Add test to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Create all test executables
create_test(test_euler)
create_test(test_legendre)
create_test(test_gauss_legendre)
create_test(test_field)
create_test(test_rusanov)
create_test(test_dg_scheme)
create_test(test_runge_kutta)
create_test(test_lagrange)
create_test(test_mesh)

# Test runner
add_executable(run_all_tests run_all_tests.cpp)

# Enable testing
enable_testing()

# Create a custom target that runs all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_euler test_legendre test_gauss_legendre test_field test_rusanov test_dg_scheme test_runge_kutta test_lagrange test_mesh
    COMMENT "Running all tests"
)

# Print summary
message(STATUS "Test configuration complete")
message(STATUS "Available test targets:")
message(STATUS "  - test_euler: Tests for Euler equations and physics")
message(STATUS "  - test_legendre: Tests for Legendre polynomials")
message(STATUS "  - test_gauss_legendre: Tests for Gauss-Legendre quadrature")
message(STATUS "  - test_field: Tests for Field data structure")
message(STATUS "  - test_rusanov: Tests for Rusanov numerical flux")
message(STATUS "  - test_dg_scheme: Tests for Discontinuous Galerkin scheme")
message(STATUS "  - test_runge_kutta: Tests for Runge-Kutta time integration")
message(STATUS "  - test_lagrange: Tests for Lagrange basis functions")
message(STATUS "  - run_all_tests: Runs all tests in sequence")
message(STATUS "  - run_tests: Uses CTest to run all tests")
