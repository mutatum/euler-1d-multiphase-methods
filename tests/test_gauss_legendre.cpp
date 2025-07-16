#include "test_framework.hpp"
#include "../src/physics/Euler/euler.hpp"
#include "../src/utils/quadrature/gauss_legendre.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TestFramework global_test_framework;

// Test basic quadrature functionality
TEST(gauss_legendre_quadrature_basic) {
    using Quadrature = GLQuadrature<double, 2>;
    
    // Test integration of constant function f(x) = 1 over [-1, 1]
    auto constant_func = [](double x) { return 1.0; };
    double result = Quadrature::integrate(constant_func, -1.0, 1.0);
    ASSERT_NEAR(2.0, result, 1e-14); // ∫₋₁¹ 1 dx = 2
}

TEST(gauss_legendre_quadrature_polynomial) {
    using Quadrature = GLQuadrature<double, 3>;
    
    // Test integration of x^2 over [-1, 1]
    auto x_squared = [](double x) { return x * x; };
    double result = Quadrature::integrate(x_squared, -1.0, 1.0);
    ASSERT_NEAR(2.0/3.0, result, 1e-14); // ∫₋₁¹ x² dx = 2/3
    
    // Test integration of x^3 over [-1, 1] (should be 0)
    auto x_cubed = [](double x) { return x * x * x; };
    result = Quadrature::integrate(x_cubed, -1.0, 1.0);
    ASSERT_NEAR(0.0, result, 1e-14); // ∫₋₁¹ x³ dx = 0
}

TEST(gauss_legendre_quadrature_exactness) {
    using Quadrature = GLQuadrature<double, 3>;
    
    // GL quadrature with n points should integrate polynomials of degree 2n-1 exactly
    // For n=3, should be exact for polynomials up to degree 5
    
    // Test x^4 over [-1, 1]
    auto x_power_4 = [](double x) { return x * x * x * x; };
    double result = Quadrature::integrate(x_power_4, -1.0, 1.0);
    ASSERT_NEAR(2.0/5.0, result, 1e-14); // ∫₋₁¹ x⁴ dx = 2/5
    
    // Test x^5 over [-1, 1]
    auto x_power_5 = [](double x) { return x * x * x * x * x; };
    result = Quadrature::integrate(x_power_5, -1.0, 1.0);
    ASSERT_NEAR(0.0, result, 1e-14); // ∫₋₁¹ x⁵ dx = 0
}

TEST(gauss_legendre_quadrature_arbitrary_interval) {
    using Quadrature = GLQuadrature<double, 4>;
    
    // Test integration over [0, 2]
    auto linear_func = [](double x) { return x; };
    double result = Quadrature::integrate(linear_func, 0.0, 2.0);
    ASSERT_NEAR(2.0, result, 1e-14); // ∫₀² x dx = 2
    
    // Test integration over [-2, 3]
    auto quadratic_func = [](double x) { return x * x; };
    result = Quadrature::integrate(quadratic_func, -2.0, 3.0);
    double expected = (std::pow(3.0, 3) - std::pow(-2.0, 3)) / 3.0; // ∫₋₂³ x² dx
    ASSERT_NEAR(expected, result, 1e-12);
}

TEST(gauss_legendre_quadrature_different_orders) {
    // Test quadrature with different numbers of points
    
    // Simple linear function
    auto linear_func = [](double x) { return 2.0 * x + 1.0; };
    double expected = 2.0; // ∫₋₁¹ (2x + 1) dx = 2
    
    // All should give exact result for linear functions
    ASSERT_NEAR(expected, (GLQuadrature<double, 1>::integrate(linear_func, -1.0, 1.0)), 1e-14);
    ASSERT_NEAR(expected, (GLQuadrature<double, 2>::integrate(linear_func, -1.0, 1.0)), 1e-14);
    ASSERT_NEAR(expected, (GLQuadrature<double, 3>::integrate(linear_func, -1.0, 1.0)), 1e-14);
    ASSERT_NEAR(expected, (GLQuadrature<double, 4>::integrate(linear_func, -1.0, 1.0)), 1e-14);
}

TEST(gauss_legendre_quadrature_transcendental) {
    using Quadrature = GLQuadrature<double, 5>;
    
    // Test integration of e^x over [-1, 1]
    auto exp_func = [](double x) { return std::exp(x); };
    double result = Quadrature::integrate(exp_func, -1.0, 1.0);
    double expected = std::exp(1.0) - std::exp(-1.0); // e - e^(-1)
    ASSERT_NEAR(expected, result, 1e-6); // Less precise for transcendental functions
}

TEST(gauss_legendre_quadrature_sin_cos) {
    using Quadrature = GLQuadrature<double, 6>;
    
    // Test integration of sin(x) over [-π, π]
    auto sin_func = [](double x) { return std::sin(x); };
    double result = Quadrature::integrate(sin_func, -M_PI, M_PI);
    ASSERT_NEAR(0.0, result, 1e-10); // ∫₋π^π sin(x) dx = 0
    
    // Test integration of cos(x) over [0, π/2]
    auto cos_func = [](double x) { return std::cos(x); };
    result = Quadrature::integrate(cos_func, 0.0, M_PI / 2.0);
    ASSERT_NEAR(1.0, result, 1e-10); // ∫₀^(π/2) cos(x) dx = 1
}

TEST(gauss_legendre_nodes_and_weights) {
    using Quadrature = GLQuadrature<double, 2>;
    
    // For GL quadrature with 2 points, nodes should be ±1/√3
    double expected_node = 1.0 / std::sqrt(3.0);
    ASSERT_NEAR(-expected_node, Quadrature::nodes[0], 1e-14);
    ASSERT_NEAR(expected_node, Quadrature::nodes[1], 1e-14);
    
    // Weights should be 1 for 2-point GL quadrature
    ASSERT_NEAR(1.0, Quadrature::weights[0], 1e-14);
    ASSERT_NEAR(1.0, Quadrature::weights[1], 1e-14);
}

TEST(gauss_legendre_symmetry) {
    using Quadrature = GLQuadrature<double, 5>;
    
    // Test that nodes are symmetric about 0
    for (std::size_t i = 0; i < Quadrature::nodes.size() / 2; ++i) {
        std::size_t j = Quadrature::nodes.size() - 1 - i;
        ASSERT_NEAR(Quadrature::nodes[i], -Quadrature::nodes[j], 1e-14);
        ASSERT_NEAR(Quadrature::weights[i], Quadrature::weights[j], 1e-14);
    }
}

TEST(gauss_legendre_vector_integration) {
    using Quadrature = GLQuadrature<double, 3>;

    auto vector_func = [](double x) -> Eigen::Vector3d {
        return Eigen::Vector3d(1.0, x, x * x);
    };
    
    auto result = Quadrature::integrate(vector_func, -1.0, 1.0);

    ASSERT_NEAR(2.0, result[0], 1e-14);     // ∫₋₁¹ 1 dx = 2
    ASSERT_NEAR(0.0, result[1], 1e-14);     // ∫₋₁¹ x dx = 0
    ASSERT_NEAR(2.0/3.0, result[2], 1e-14); // ∫₋₁¹ x² dx = 2/3
}

int main() {
    std::cout << "Running Gauss-Legendre quadrature tests...\n";
    global_test_framework.run_all_tests();
    return 0;
}
