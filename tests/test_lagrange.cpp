#include "test_framework.hpp"
#include "../src/utils/basis/lagrange.hpp"
#include <array>

TestFramework global_test_framework;

// Global constexpr arrays for testing
static constexpr std::array<double, 2> nodes_2 = {-1.0, 1.0};
static constexpr std::array<double, 3> nodes_3 = {-1.0, 0.0, 1.0};
static constexpr std::array<double, 3> nodes_3_nonuniform = {-1.0, 0.5, 1.0};
static constexpr std::array<double, 4> nodes_4 = {-1.0, -0.5, 0.5, 1.0};

// Test Lagrange basis with 2 nodes
TEST(lagrange_basis_2_nodes) {
    using Basis = Lagrange<nodes_2>;
    
    ASSERT_EQ(2, Basis::num_basis_functions);
    ASSERT_EQ(1, Basis::order);
    
    // Test cardinal property: L_i(x_j) = δ_ij
    ASSERT_NEAR(1.0, Basis::evaluate(0, -1.0), 1e-14);
    ASSERT_NEAR(0.0, Basis::evaluate(0, 1.0), 1e-14);
    ASSERT_NEAR(0.0, Basis::evaluate(1, -1.0), 1e-14);
    ASSERT_NEAR(1.0, Basis::evaluate(1, 1.0), 1e-14);
    
    // Test at intermediate point
    ASSERT_NEAR(0.5, Basis::evaluate(0, 0.0), 1e-14);
    ASSERT_NEAR(0.5, Basis::evaluate(1, 0.0), 1e-14);
    
    // Test partition of unity: sum of basis functions = 1
    ASSERT_NEAR(1.0, Basis::evaluate(0, 0.5) + Basis::evaluate(1, 0.5), 1e-14);
    ASSERT_NEAR(1.0, Basis::evaluate(0, -0.5) + Basis::evaluate(1, -0.5), 1e-14);
}

// Test Lagrange basis with 3 nodes
TEST(lagrange_basis_3_nodes) {
    using Basis = Lagrange<nodes_3>;
    
    ASSERT_EQ(3, Basis::num_basis_functions);
    ASSERT_EQ(2, Basis::order);
    
    // Test cardinal property
    ASSERT_NEAR(1.0, Basis::evaluate(0, -1.0), 1e-14);
    ASSERT_NEAR(0.0, Basis::evaluate(0, 0.0), 1e-14);
    ASSERT_NEAR(0.0, Basis::evaluate(0, 1.0), 1e-14);
    
    ASSERT_NEAR(0.0, Basis::evaluate(1, -1.0), 1e-14);
    ASSERT_NEAR(1.0, Basis::evaluate(1, 0.0), 1e-14);
    ASSERT_NEAR(0.0, Basis::evaluate(1, 1.0), 1e-14);
    
    ASSERT_NEAR(0.0, Basis::evaluate(2, -1.0), 1e-14);
    ASSERT_NEAR(0.0, Basis::evaluate(2, 0.0), 1e-14);
    ASSERT_NEAR(1.0, Basis::evaluate(2, 1.0), 1e-14);
    
    // Test partition of unity
    for (double x : {-0.5, 0.5, 0.25, -0.75}) {
        double sum = Basis::evaluate(0, x) + Basis::evaluate(1, x) + Basis::evaluate(2, x);
        ASSERT_NEAR(1.0, sum, 1e-14);
    }
}

// Test Lagrange basis derivatives with 2 nodes
TEST(lagrange_basis_derivatives_2_nodes) {
    using Basis = Lagrange<nodes_2>;
    
    // For linear basis on [-1, 1]: L0(x) = (1-x)/2, L1(x) = (1+x)/2
    // Derivatives: L0'(x) = -1/2, L1'(x) = 1/2
    
    ASSERT_NEAR(-0.5, Basis::derivative(0, 0.0), 1e-14);
    ASSERT_NEAR(0.5, Basis::derivative(1, 0.0), 1e-14);
    
    ASSERT_NEAR(-0.5, Basis::derivative(0, 0.5), 1e-14);
    ASSERT_NEAR(0.5, Basis::derivative(1, 0.5), 1e-14);
    
    // Sum of derivatives should be 0 (constant function has zero derivative)
    ASSERT_NEAR(0.0, Basis::derivative(0, 0.3) + Basis::derivative(1, 0.3), 1e-14);
}

// Test Lagrange basis derivatives with 3 nodes
TEST(lagrange_basis_derivatives_3_nodes) {
    using Basis = Lagrange<nodes_3>;
    
    // Sum of derivatives should be 0 (constant function has zero derivative)
    for (double x : {-0.5, 0.0, 0.5, 0.25}) {
        double sum = Basis::derivative(0, x) + Basis::derivative(1, x) + Basis::derivative(2, x);
        ASSERT_NEAR(0.0, sum, 1e-14);
    }
    
    // Test specific values at x = 0
    // L0(x) = 0.5*x*(x-1), L0'(0) = -0.5
    // L1(x) = 1-x^2, L1'(0) = 0
    // L2(x) = 0.5*x*(x+1), L2'(0) = 0.5
    ASSERT_NEAR(-0.5, Basis::derivative(0, 0.0), 1e-14);
    ASSERT_NEAR(0.0, Basis::derivative(1, 0.0), 1e-14);
    ASSERT_NEAR(0.5, Basis::derivative(2, 0.0), 1e-14);
}

// Test Lagrange basis with non-uniform nodes
TEST(lagrange_basis_non_uniform) {
    using Basis = Lagrange<nodes_3_nonuniform>;
    
    // Test cardinal property
    ASSERT_NEAR(1.0, Basis::evaluate(0, -1.0), 1e-14);
    ASSERT_NEAR(0.0, Basis::evaluate(0, 0.5), 1e-14);
    ASSERT_NEAR(0.0, Basis::evaluate(0, 1.0), 1e-14);
    
    ASSERT_NEAR(0.0, Basis::evaluate(1, -1.0), 1e-14);
    ASSERT_NEAR(1.0, Basis::evaluate(1, 0.5), 1e-14);
    ASSERT_NEAR(0.0, Basis::evaluate(1, 1.0), 1e-14);
    
    ASSERT_NEAR(0.0, Basis::evaluate(2, -1.0), 1e-14);
    ASSERT_NEAR(0.0, Basis::evaluate(2, 0.5), 1e-14);
    ASSERT_NEAR(1.0, Basis::evaluate(2, 1.0), 1e-14);
    
    // Test partition of unity
    for (double x : {0.0, 0.25, 0.75}) {
        double sum = Basis::evaluate(0, x) + Basis::evaluate(1, x) + Basis::evaluate(2, x);
        ASSERT_NEAR(1.0, sum, 1e-14);
    }
}

// Test Lagrange basis with 4 nodes
TEST(lagrange_basis_4_nodes) {
    using Basis = Lagrange<nodes_4>;
    
    ASSERT_EQ(4, Basis::num_basis_functions);
    ASSERT_EQ(3, Basis::order);
    
    // Test cardinal property for all nodes
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(expected, Basis::evaluate(i, nodes_4[j]), 1e-14);
        }
    }
    
    // Test partition of unity
    for (double x : {-0.75, 0.0, 0.25, 0.75}) {
        double sum = 0.0;
        for (std::size_t i = 0; i < 4; ++i) {
            sum += Basis::evaluate(i, x);
        }
        ASSERT_NEAR(1.0, sum, 1e-14);
    }
}

// Test polynomial reproduction
TEST(lagrange_basis_polynomial_reproduction) {
    using Basis = Lagrange<nodes_3>;
    
    // Test reproduction of constant function f(x) = 1
    for (double x : {-0.5, 0.0, 0.5}) {
        double interpolated = Basis::evaluate(0, x) * 1.0 + 
                            Basis::evaluate(1, x) * 1.0 + 
                            Basis::evaluate(2, x) * 1.0;
        ASSERT_NEAR(1.0, interpolated, 1e-14);
    }
    
    // Test reproduction of linear function f(x) = x
    for (double x : {-0.5, 0.0, 0.5}) {
        double interpolated = Basis::evaluate(0, x) * (-1.0) + 
                            Basis::evaluate(1, x) * 0.0 + 
                            Basis::evaluate(2, x) * 1.0;
        ASSERT_NEAR(x, interpolated, 1e-14);
    }
    
    // Test reproduction of quadratic function f(x) = x^2
    for (double x : {-0.5, 0.0, 0.5}) {
        double interpolated = Basis::evaluate(0, x) * 1.0 + 
                            Basis::evaluate(1, x) * 0.0 + 
                            Basis::evaluate(2, x) * 1.0;
        ASSERT_NEAR(x * x, interpolated, 1e-14);
    }
}

// Test derivative accuracy
TEST(lagrange_basis_derivative_accuracy) {
    using Basis = Lagrange<nodes_3>;
    
    // Test derivative of linear function f(x) = x
    // Should give exact derivative = 1
    for (double x : {-0.5, 0.0, 0.5}) {
        double derivative = Basis::derivative(0, x) * (-1.0) + 
                          Basis::derivative(1, x) * 0.0 + 
                          Basis::derivative(2, x) * 1.0;
        ASSERT_NEAR(1.0, derivative, 1e-14);
    }
    
    // Test derivative of quadratic function f(x) = x^2
    // Should give exact derivative = 2x
    for (double x : {-0.5, 0.0, 0.5}) {
        double derivative = Basis::derivative(0, x) * 1.0 + 
                          Basis::derivative(1, x) * 0.0 + 
                          Basis::derivative(2, x) * 1.0;
        ASSERT_NEAR(2.0 * x, derivative, 1e-14);
    }
}

int main() {
    std::cout << "Running Lagrange basis tests...\n";
    global_test_framework.run_all_tests();
    return 0;
}
