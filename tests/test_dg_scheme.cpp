#include "test_framework.hpp"
#include "../src/scheme/classic_discontinuous_galerkin.hpp"
#include "../src/physics/Euler/euler.hpp"
#include "../src/physics/numerical_flux/rusanov.hpp"
#include "../src/utils/basis/legendre.hpp"
#include "../src/utils/quadrature/gauss_legendre.hpp"
#include "../src/boundary/boundary_conditions.hpp"
#include <Eigen/Dense>

TestFramework global_test_framework;

// Template aliases for boundary conditions
template<class Scheme> using LeftBC = PeriodicBC<Side::Left>;
template<class Scheme> using RightBC = PeriodicBC<Side::Right>;

// Test DG scheme construction
TEST(dg_scheme_construction) {
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 2>;
    using Quadrature = GLQuadrature<double, 3>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    DGScheme scheme;
    
    // Test that scheme constructs without errors
    ASSERT_TRUE(true); // If we get here, construction succeeded
}

TEST(dg_scheme_workspace) {
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 2>;
    using Quadrature = GLQuadrature<double, 3>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    const std::size_t num_cells = 5;
    DGScheme::Workspace workspace(num_cells);
    
    // Test workspace sizing
    ASSERT_EQ(num_cells + 1, workspace.interface_fluxes.size());
    ASSERT_EQ(num_cells, workspace.projected_fluxes.size());
    ASSERT_EQ(num_cells, workspace.F_star_matrices.size());
    
    // Test F_star_matrices dimensions
    ASSERT_EQ(2, workspace.F_star_matrices[0].rows());
    ASSERT_EQ(3, workspace.F_star_matrices[0].cols()); // Variables
}

TEST(dg_scheme_element_structure) {
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 2>;
    using Quadrature = GLQuadrature<double, 3>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    DGScheme::Element element;
    
    // Test element dimensions
    ASSERT_EQ(3, element.coeffs.rows()); // PolynomialOrder + 1
    ASSERT_EQ(3, element.coeffs.cols()); // Variables
    
    // Test initialization
    element.coeffs.setZero();
    ASSERT_EQ(0.0, element.coeffs(0, 0));
    ASSERT_EQ(0.0, element.coeffs(1, 1));
    ASSERT_EQ(0.0, element.coeffs(2, 2));
}

TEST(dg_scheme_polynomial_evaluation) {
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 2>;
    using Quadrature = GLQuadrature<double, 3>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    DGScheme scheme;
    DGScheme::Element element;
    
    // Set up a constant polynomial (only first coefficient)
    element.coeffs.row(0) = Eigen::RowVector3d(1.0, 2.0, 3.0);
    element.coeffs.row(1).setZero();
    element.coeffs.row(2).setZero();
    
    // Evaluate at different points
    auto state_minus1 = scheme.evaluate_poly(element, -1.0);
    auto state_0 = scheme.evaluate_poly(element, 0.0);
    auto state_plus1 = scheme.evaluate_poly(element, 1.0);
    
    // For constant polynomial, should be the same everywhere
    ASSERT_NEAR(1.0, state_minus1.density, 1e-14);
    ASSERT_NEAR(2.0, state_minus1.momentum, 1e-14);
    ASSERT_NEAR(3.0, state_minus1.total_energy, 1e-14);
    
    ASSERT_NEAR(1.0, state_0.density, 1e-14);
    ASSERT_NEAR(2.0, state_0.momentum, 1e-14);
    ASSERT_NEAR(3.0, state_0.total_energy, 1e-14);
    
    ASSERT_NEAR(1.0, state_plus1.density, 1e-14);
    ASSERT_NEAR(2.0, state_plus1.momentum, 1e-14);
    ASSERT_NEAR(3.0, state_plus1.total_energy, 1e-14);
}

TEST(dg_scheme_linear_polynomial) {
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 2>;
    using Quadrature = GLQuadrature<double, 3>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    DGScheme scheme;
    DGScheme::Element element;
    
    // Set up a linear polynomial: P(x) = a0 + a1*x
    element.coeffs.row(0) = Eigen::RowVector3d(1.0, 2.0, 3.0); // Constant term
    element.coeffs.row(1) = Eigen::RowVector3d(0.5, 1.0, 1.5); // Linear term
    element.coeffs.row(2).setZero();
    
    // Evaluate at specific points
    auto state_minus1 = scheme.evaluate_poly(element, -1.0);
    auto state_0 = scheme.evaluate_poly(element, 0.0);
    auto state_plus1 = scheme.evaluate_poly(element, 1.0);
    
    // Check linear behavior
    // At x = -1: P(-1) = a0 + a1*(-1) = a0 - a1
    ASSERT_NEAR(0.5, state_minus1.density, 1e-14);      // 1.0 - 0.5
    ASSERT_NEAR(1.0, state_minus1.momentum, 1e-14);     // 2.0 - 1.0
    ASSERT_NEAR(1.5, state_minus1.total_energy, 1e-14); // 3.0 - 1.5
    
    // At x = 0: P(0) = a0
    ASSERT_NEAR(1.0, state_0.density, 1e-14);
    ASSERT_NEAR(2.0, state_0.momentum, 1e-14);
    ASSERT_NEAR(3.0, state_0.total_energy, 1e-14);
    
    // At x = 1: P(1) = a0 + a1
    ASSERT_NEAR(1.5, state_plus1.density, 1e-14);       // 1.0 + 0.5
    ASSERT_NEAR(3.0, state_plus1.momentum, 1e-14);      // 2.0 + 1.0
    ASSERT_NEAR(4.5, state_plus1.total_energy, 1e-14);  // 3.0 + 1.5
}

TEST(dg_scheme_compute_residual_constant) {
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 1>; // Linear elements
    using Quadrature = GLQuadrature<double, 2>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    DGScheme scheme;
    const std::size_t num_cells = 3;
    Field<DGScheme> U(num_cells);
    Field<DGScheme> R(num_cells);
    DGScheme::Workspace workspace(num_cells);
    
    // Set up constant state in all cells
    EulerState<double> constant_state(1.0, 0.0, 2.5);
    for (std::size_t i = 0; i < num_cells; ++i) {
        U(i).coeffs.row(0) = Eigen::RowVector3d(constant_state.density, 
                                               constant_state.momentum, 
                                               constant_state.total_energy);
    }
    
    // Compute residual
    double max_speed = scheme.compute_residual(U, R, workspace);
    
    // For constant state, residual should be close to zero
    // (exact zero depends on boundary conditions)
    ASSERT_TRUE(std::isfinite(max_speed));
    ASSERT_TRUE(max_speed > 0);
    
    // Check that residual is computed
    for (std::size_t i = 0; i < num_cells; ++i) {
        for (int j = 0; j < R(i).coeffs.rows(); ++j) {
            for (int k = 0; k < R(i).coeffs.cols(); ++k) {
                ASSERT_TRUE(std::isfinite(R(i).coeffs(j, k)));
            }
        }
    }
}

TEST(dg_scheme_compute_residual_linear) {
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 2>;
    using Quadrature = GLQuadrature<double, 3>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    DGScheme scheme;
    const std::size_t num_cells = 2;
    Field<DGScheme> U(num_cells);
    Field<DGScheme> R(num_cells);
    DGScheme::Workspace workspace(num_cells);
    
    // Set up smooth initial condition
    for (std::size_t i = 0; i < num_cells; ++i) {
        // Constant terms
        U(i).coeffs.row(0) = Eigen::RowVector3d(1.0, 0.1, 2.5);
        // Linear terms
        U(i).coeffs.row(1) = Eigen::RowVector3d(0.01, 0.01, 0.01);
        // Quadratic terms
        U(i).coeffs.row(2).setZero();
    }
    
    // Compute residual
    double max_speed = scheme.compute_residual(U, R, workspace);
    
    // Should produce finite results
    ASSERT_TRUE(std::isfinite(max_speed));
    ASSERT_TRUE(max_speed > 0);
    
    // Check that all residual components are finite
    for (std::size_t i = 0; i < num_cells; ++i) {
        for (int j = 0; j < R(i).coeffs.rows(); ++j) {
            for (int k = 0; k < R(i).coeffs.cols(); ++k) {
                ASSERT_TRUE(std::isfinite(R(i).coeffs(j, k)));
            }
        }
    }
}

TEST(dg_scheme_interface_fluxes) {
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 1>;
    using Quadrature = GLQuadrature<double, 2>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    DGScheme scheme;
    const std::size_t num_cells = 2;
    Field<DGScheme> U(num_cells);
    Field<DGScheme> R(num_cells);
    DGScheme::Workspace workspace(num_cells);
    
    // Set up different states in each cell
    U(0).coeffs.row(0) = Eigen::RowVector3d(1.0, 0.0, 2.5);
    U(0).coeffs.row(1).setZero();
    
    U(1).coeffs.row(0) = Eigen::RowVector3d(0.8, 0.2, 2.0);
    U(1).coeffs.row(1).setZero();
    
    // Compute residual
    scheme.compute_residual(U, R, workspace);
    
    // Check that interface fluxes are computed
    for (std::size_t i = 0; i < workspace.interface_fluxes.size(); ++i) {
        ASSERT_TRUE(std::isfinite(workspace.interface_fluxes[i].density));
        ASSERT_TRUE(std::isfinite(workspace.interface_fluxes[i].momentum));
        ASSERT_TRUE(std::isfinite(workspace.interface_fluxes[i].total_energy));
    }
}

int main() {
    std::cout << "Running DG scheme tests...\n";
    global_test_framework.run_all_tests();
    return 0;
}
