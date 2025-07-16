#include "test_framework.hpp"
#include "../src/utils/basis/legendre.hpp"
#include <cmath>

TestFramework global_test_framework;

// Test Legendre polynomial evaluation
TEST(legendre_polynomial_basic_values) {
    using Legendre = Legendre<double, 10>;
    
    // Test P0(x) = 1
    ASSERT_EQ(1.0, Legendre::evaluate(0, 0.0));
    ASSERT_EQ(1.0, Legendre::evaluate(0, 0.5));
    ASSERT_EQ(1.0, Legendre::evaluate(0, -0.5));
    ASSERT_EQ(1.0, Legendre::evaluate(0, 1.0));
    ASSERT_EQ(1.0, Legendre::evaluate(0, -1.0));
    
    // Test P1(x) = x
    ASSERT_EQ(0.0, Legendre::evaluate(1, 0.0));
    ASSERT_EQ(0.5, Legendre::evaluate(1, 0.5));
    ASSERT_EQ(-0.5, Legendre::evaluate(1, -0.5));
    ASSERT_EQ(1.0, Legendre::evaluate(1, 1.0));
    ASSERT_EQ(-1.0, Legendre::evaluate(1, -1.0));
}

TEST(legendre_polynomial_orthogonality) {
    using Legendre = Legendre<double, 10>;
    
    // Test P2(x) = 1/2 * (3x^2 - 1)
    ASSERT_EQ(-0.5, Legendre::evaluate(2, 0.0));   // P2(0) = -1/2
    ASSERT_EQ(-0.125, Legendre::evaluate(2, 0.5)); // P2(0.5) = 1/2 * (3*0.25 - 1) = 1/2 * (-0.25) = -0.125
    ASSERT_EQ(1.0, Legendre::evaluate(2, 1.0));    // P2(1) = 1/2 * (3 - 1) = 1
    ASSERT_EQ(1.0, Legendre::evaluate(2, -1.0));   // P2(-1) = 1/2 * (3 - 1) = 1
}

TEST(legendre_polynomial_specific_values) {
    using Legendre = Legendre<double, 10>;
    
    // Test P3(x) = 1/2 * (5x^3 - 3x)
    ASSERT_EQ(0.0, Legendre::evaluate(3, 0.0));    // P3(0) = 0
    ASSERT_EQ(1.0, Legendre::evaluate(3, 1.0));    // P3(1) = 1/2 * (5 - 3) = 1
    ASSERT_EQ(-1.0, Legendre::evaluate(3, -1.0));  // P3(-1) = 1/2 * (-5 + 3) = -1
    
    // Test P4(x) = 1/8 * (35x^4 - 30x^2 + 3)
    ASSERT_EQ(3.0/8.0, Legendre::evaluate(4, 0.0)); // P4(0) = 3/8
    ASSERT_EQ(1.0, Legendre::evaluate(4, 1.0));     // P4(1) = 1/8 * (35 - 30 + 3) = 1
    ASSERT_EQ(1.0, Legendre::evaluate(4, -1.0));    // P4(-1) = 1/8 * (35 - 30 + 3) = 1
}

// Test Legendre derivative evaluation
TEST(legendre_derivative_basic) {
    using Legendre = Legendre<double, 10>;
    
    // Test P0'(x) = 0
    ASSERT_EQ(0.0, Legendre::derivative(0, 0.0));
    ASSERT_EQ(0.0, Legendre::derivative(0, 0.5));
    ASSERT_EQ(0.0, Legendre::derivative(0, -0.5));
    
    // Test P1'(x) = 1
    ASSERT_EQ(1.0, Legendre::derivative(1, 0.0));
    ASSERT_EQ(1.0, Legendre::derivative(1, 0.5));
    ASSERT_EQ(1.0, Legendre::derivative(1, -0.5));
}

TEST(legendre_derivative_boundary_conditions) {
    using Legendre = Legendre<double, 10>;
    
    // Test derivatives at boundaries x = ±1
    // P_n'(1) = n(n+1)/2
    // P_n'(-1) = (-1)^(n+1) * n(n+1)/2
    
    for (std::size_t n = 1; n <= 5; ++n) {
        double expected_at_1 = n * (n + 1.0) / 2.0;
        double expected_at_minus_1 = ((n % 2 == 0) ? -1.0 : 1.0) * expected_at_1;
        
        ASSERT_NEAR(expected_at_1, Legendre::derivative(n, 1.0), 1e-10);
        ASSERT_NEAR(expected_at_minus_1, Legendre::derivative(n, -1.0), 1e-10);
    }
}

TEST(legendre_derivative_specific_values) {
    using Legendre = Legendre<double, 10>;
    
    // Test P2'(x) = 3x
    ASSERT_EQ(0.0, Legendre::derivative(2, 0.0));
    ASSERT_EQ(1.5, Legendre::derivative(2, 0.5));
    ASSERT_EQ(-1.5, Legendre::derivative(2, -0.5));
    
    // Test P3'(x) = 1/2 * (15x^2 - 3)
    ASSERT_EQ(-1.5, Legendre::derivative(3, 0.0));   // P3'(0) = -3/2
    ASSERT_NEAR(6.0, Legendre::derivative(3, 1.0), 1e-10);    // P3'(1) = 1/2 * (15 - 3) = 6
    ASSERT_NEAR(6.0, Legendre::derivative(3, -1.0), 1e-10);   // P3'(-1) = 1/2 * (15 - 3) = 6
}

// Test edge cases
TEST(legendre_edge_cases) {
    using Legendre = Legendre<double, 10>;
    
    // Test very small values near boundaries
    const double eps = 1e-11;
    
    // Should not throw for values very close to ±1
    ASSERT_TRUE(std::isfinite(Legendre::derivative(3, 1.0 - eps)));
    ASSERT_TRUE(std::isfinite(Legendre::derivative(3, -1.0 + eps)));
    
    // Test intermediate values
    ASSERT_TRUE(std::isfinite(Legendre::evaluate(5, 0.123)));
    ASSERT_TRUE(std::isfinite(Legendre::derivative(5, 0.123)));
}

int main() {
    std::cout << "Running Legendre polynomial tests...\n";
    global_test_framework.run_all_tests();
    return 0;
}
