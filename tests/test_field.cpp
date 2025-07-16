#include "test_framework.hpp"
#include "../src/field/field.hpp"
#include "../src/scheme/classic_discontinuous_galerkin.hpp"
#include "../src/physics/Euler/euler.hpp"
#include "../src/physics/numerical_flux/rusanov.hpp"
#include "../src/utils/basis/legendre.hpp"
#include "../src/utils/quadrature/gauss_legendre.hpp"
#include "../src/boundary/boundary_conditions.hpp"

TestFramework global_test_framework;

// Mock scheme for testing
struct MockScheme {
    using Scalar = double;
    using State = physics::euler::EulerState<double>;
    struct Element {
        Eigen::Matrix<Scalar, 3, 3> coeffs;
        Element() { coeffs.setZero(); }
    };
};

template<class Scheme> using LeftBC = PeriodicBC<Side::Left>;
template<class Scheme> using RightBC = PeriodicBC<Side::Right>;

// Test Field construction
TEST(field_construction) {
    Field<MockScheme> field(5);
    ASSERT_EQ(5, field.size());
    
    // Test construction with data
    std::vector<MockScheme::Element> data(3);
    Field<MockScheme> field2(std::move(data));
    ASSERT_EQ(3, field2.size());
}

TEST(field_element_access) {
    Field<MockScheme> field(3);
    
    // Test bounds checking
    ASSERT_THROWS(field(3)); // Out of bounds
    ASSERT_THROWS(field(10)); // Out of bounds
    
    // Test valid access
    auto& element = field(0);
    element.coeffs(0, 0) = 1.0;
    ASSERT_EQ(1.0, field(0).coeffs(0, 0));
    
    // Test const access
    const auto& const_field = field;
    ASSERT_EQ(1.0, const_field(0).coeffs(0, 0));
}

TEST(field_arithmetic_operations) {
    Field<MockScheme> field1(2);
    Field<MockScheme> field2(2);
    
    // Initialize with some values
    field1(0).coeffs(0, 0) = 1.0;
    field1(1).coeffs(0, 0) = 2.0;
    field2(0).coeffs(0, 0) = 3.0;
    field2(1).coeffs(0, 0) = 4.0;
    
    // Test addition
    Field<MockScheme> sum = field1 + field2;
    ASSERT_EQ(4.0, sum(0).coeffs(0, 0));
    ASSERT_EQ(6.0, sum(1).coeffs(0, 0));
    
    // Test subtraction
    Field<MockScheme> diff = field2 - field1;
    ASSERT_EQ(2.0, diff(0).coeffs(0, 0));
    ASSERT_EQ(2.0, diff(1).coeffs(0, 0));
    
    // Test scalar multiplication
    Field<MockScheme> scaled = field1 * 2.0;
    ASSERT_EQ(2.0, scaled(0).coeffs(0, 0));
    ASSERT_EQ(4.0, scaled(1).coeffs(0, 0));
    
    // Test commutative scalar multiplication
    Field<MockScheme> scaled2 = 2.0 * field1;
    ASSERT_EQ(2.0, scaled2(0).coeffs(0, 0));
    ASSERT_EQ(4.0, scaled2(1).coeffs(0, 0));
}

TEST(field_compound_assignment) {
    Field<MockScheme> field1(2);
    Field<MockScheme> field2(2);
    
    // Initialize with some values
    field1(0).coeffs(0, 0) = 1.0;
    field1(1).coeffs(0, 0) = 2.0;
    field2(0).coeffs(0, 0) = 3.0;
    field2(1).coeffs(0, 0) = 4.0;
    
    // Test compound addition
    field1 += field2;
    ASSERT_EQ(4.0, field1(0).coeffs(0, 0));
    ASSERT_EQ(6.0, field1(1).coeffs(0, 0));
    
    // Test compound subtraction
    field1 -= field2;
    ASSERT_EQ(1.0, field1(0).coeffs(0, 0));
    ASSERT_EQ(2.0, field1(1).coeffs(0, 0));
    
    // Test compound scalar multiplication
    field1 *= 3.0;
    ASSERT_EQ(3.0, field1(0).coeffs(0, 0));
    ASSERT_EQ(6.0, field1(1).coeffs(0, 0));
}

TEST(field_size_mismatch) {
    Field<MockScheme> field1(2);
    Field<MockScheme> field2(3);
    
    // Test that operations with mismatched sizes throw
    ASSERT_THROWS(field1 + field2);
    ASSERT_THROWS(field1 - field2);
    ASSERT_THROWS(field1 += field2);
    ASSERT_THROWS(field1 -= field2);
}

TEST(field_with_dg_scheme) {
    // Test with actual DG scheme
    // gamma = 1.4
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 3>;
    using Quadrature = GLQuadrature<double, 4>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    Field<DGScheme> field(5);
    ASSERT_EQ(5, field.size());
    
    // Test that elements have correct size
    auto& element = field(0);
    ASSERT_EQ(4, element.coeffs.rows()); // PolynomialOrder + 1
    ASSERT_EQ(3, element.coeffs.cols()); // Variables
}

TEST(field_real_world_operations) {
    // Test with realistic DG field operations
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 2>;
    using Quadrature = GLQuadrature<double, 3>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    Field<DGScheme> U(3);
    Field<DGScheme> R(3);
    
    // Initialize with some realistic values
    for (std::size_t i = 0; i < U.size(); ++i) {
        U(i).coeffs(0, 0) = 1.0; // density
        U(i).coeffs(0, 1) = 0.5; // momentum
        U(i).coeffs(0, 2) = 2.5; // total energy
    }
    
    // Test Runge-Kutta-like operations
    double dt = 0.01;
    Field<DGScheme> k1 = R;
    Field<DGScheme> U_temp = U + (dt * 0.5) * k1;
    
    // Should not throw
    ASSERT_EQ(3, U_temp.size());
}

int main() {
    std::cout << "Running Field class tests...\n";
    global_test_framework.run_all_tests();
    return 0;
}
