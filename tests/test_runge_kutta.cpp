#include "test_framework.hpp"
#include "../src/time_integration/runge_kutta.hpp"
#include "../src/scheme/classic_discontinuous_galerkin.hpp"
#include "../src/physics/Euler/euler.hpp"
#include "../src/physics/numerical_flux/rusanov.hpp"
#include "../src/utils/basis/legendre.hpp"
#include "../src/utils/quadrature/gauss_legendre.hpp"
#include "../src/boundary/boundary_conditions.hpp"
#include <iostream>
#include <cmath>

TestFramework global_test_framework;

// Template aliases for boundary conditions
template<class Scheme> using LeftBC = PeriodicBC<Side::Left>;
template<class Scheme> using RightBC = PeriodicBC<Side::Right>;

// Test RungeKutta4 construction
TEST(runge_kutta_construction) {
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 1>;
    using Quadrature = GLQuadrature<double, 2>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    const std::size_t num_cells = 5;
    RungeKutta4<DGScheme> rk4(num_cells);
    
    // If we get here, construction succeeded
    ASSERT_TRUE(true);
}

// Mock scheme for testing ODE integration
struct MockODEScheme {
    using Scalar = double;
    using State = Scalar;
    
    struct Element {
        Eigen::Matrix<Scalar, 1, 1> coeffs;
        Element() { coeffs.setZero(); }
    };
    
    struct Workspace {
        explicit Workspace(std::size_t) {}
    };
    
    // Simple ODE: dy/dt = -y (exponential decay)
    Scalar compute_residual(const Field<MockODEScheme>& U, Field<MockODEScheme>& R, Workspace&) const {
        for (std::size_t i = 0; i < U.size(); ++i) {
            R(i).coeffs(0, 0) = -U(i).coeffs(0, 0);
        }
        return 1.0; // Mock max speed
    }
};

TEST(runge_kutta_exponential_decay) {
    const std::size_t num_cells = 1;
    MockODEScheme scheme;
    MockODEScheme::Workspace workspace(num_cells);
    RungeKutta4<MockODEScheme> rk4(num_cells);
    
    Field<MockODEScheme> U(num_cells);
    U(0).coeffs(0, 0) = 1.0; // Initial condition: y(0) = 1
    
    const double dt = 0.1;
    const int num_steps = 10;
    
    // Integrate for several time steps
    for (int step = 0; step < num_steps; ++step) {
        rk4.step(U, workspace, dt, scheme);
    }
    
    // Analytical solution: y(t) = exp(-t)
    double t_final = dt * num_steps;
    double expected = std::exp(-t_final);
    
    // RK4 should be quite accurate
    ASSERT_NEAR(expected, U(0).coeffs(0, 0), 1e-6);
}

// Mock scheme for testing oscillatory behavior
struct MockOscillatorScheme {
    using Scalar = double;
    using State = Eigen::Matrix<Scalar, 2, 1>;
    
    struct Element {
        Eigen::Matrix<Scalar, 2, 1> coeffs;
        Element() { coeffs.setZero(); }
    };
    
    struct Workspace {
        explicit Workspace(std::size_t) {}
    };
    
    // Harmonic oscillator: d²y/dt² = -y
    // Written as system: dx/dt = y, dy/dt = -x
    Scalar compute_residual(const Field<MockOscillatorScheme>& U, Field<MockOscillatorScheme>& R, Workspace&) const {
        for (std::size_t i = 0; i < U.size(); ++i) {
            double x = U(i).coeffs(0, 0);
            double y = U(i).coeffs(1, 0);
            R(i).coeffs(0, 0) = y;   // dx/dt = y
            R(i).coeffs(1, 0) = -x;  // dy/dt = -x
        }
        return 1.0; // Mock max speed
    }
};

TEST(runge_kutta_harmonic_oscillator) {
    const std::size_t num_cells = 1;
    MockOscillatorScheme scheme;
    MockOscillatorScheme::Workspace workspace(num_cells);
    RungeKutta4<MockOscillatorScheme> rk4(num_cells);
    
    Field<MockOscillatorScheme> U(num_cells);
    U(0).coeffs(0, 0) = 1.0; // Initial position: x(0) = 1
    U(0).coeffs(1, 0) = 0.0; // Initial velocity: y(0) = 0
    
    const double dt = 0.01;
    const int num_steps = 628; // Approximately 2π (one period)
    
    // Integrate for one period
    for (int step = 0; step < num_steps; ++step) {
        rk4.step(U, workspace, dt, scheme);
    }
    
    // After one period, should return to initial conditions
    ASSERT_NEAR(1.0, U(0).coeffs(0, 0), 1e-3); // Position
    ASSERT_NEAR(0.0, U(0).coeffs(1, 0), 1e-3); // Velocity
}

TEST(runge_kutta_conservation) {
    const std::size_t num_cells = 1;
    MockOscillatorScheme scheme;
    MockOscillatorScheme::Workspace workspace(num_cells);
    RungeKutta4<MockOscillatorScheme> rk4(num_cells);
    
    Field<MockOscillatorScheme> U(num_cells);
    U(0).coeffs(0, 0) = 1.0; // Initial position
    U(0).coeffs(1, 0) = 0.0; // Initial velocity
    
    // Calculate initial energy
    double initial_energy = 0.5 * (U(0).coeffs(0, 0) * U(0).coeffs(0, 0) + 
                                  U(0).coeffs(1, 0) * U(0).coeffs(1, 0));
    
    const double dt = 0.01;
    const int num_steps = 100;
    
    // Integrate
    for (int step = 0; step < num_steps; ++step) {
        rk4.step(U, workspace, dt, scheme);
    }
    
    // Calculate final energy
    double final_energy = 0.5 * (U(0).coeffs(0, 0) * U(0).coeffs(0, 0) + 
                                U(0).coeffs(1, 0) * U(0).coeffs(1, 0));
    
    // Energy should be conserved (RK4 has good conservation properties)
    ASSERT_NEAR(initial_energy, final_energy, 1e-3);
}

TEST(runge_kutta_with_dg_scheme) {
    using namespace physics::euler;
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    using PolynomialBasis = Legendre<double, 1>;
    using Quadrature = GLQuadrature<double, 2>;
    using DGScheme = DG<Physics, NumericalFlux, PolynomialBasis, Quadrature, LeftBC, RightBC>;
    
    const std::size_t num_cells = 3;
    DGScheme scheme;
    DGScheme::Workspace workspace(num_cells);
    RungeKutta4<DGScheme> rk4(num_cells);
    
    Field<DGScheme> U(num_cells);
    
    // Initialize with smooth constant state
    for (std::size_t i = 0; i < num_cells; ++i) {
        U(i).coeffs.row(0) = Eigen::Matrix<double, 1, 3>(1.0, 0.0, 2.5);
        U(i).coeffs.row(1).setZero();
    }
    
    // Store initial state for comparison
    Field<DGScheme> U_initial = U;
    
    const double dt = 0.0001;
    const int num_steps = 1;
    
    // Integrate for several time steps
    for (int step = 0; step < num_steps; ++step) {
        rk4.step(U, workspace, dt, scheme);
    }
    
    // Check that integration produces finite results
    for (std::size_t i = 0; i < num_cells; ++i) {
        for (int j = 0; j < U(i).coeffs.rows(); ++j) {
            for (int k = 0; k < U(i).coeffs.cols(); ++k) {
                ASSERT_TRUE(std::isfinite(U(i).coeffs(j, k)));
            }
        }
    }
    
    // For constant initial conditions, solution should remain nearly constant
    // (this is a stability test)
    for (std::size_t i = 0; i < num_cells; ++i) {
        ASSERT_NEAR(U_initial(i).coeffs(0, 0), U(i).coeffs(0, 0), 1e-2); // Density
        ASSERT_NEAR(U_initial(i).coeffs(0, 1), U(i).coeffs(0, 1), 1e-2); // Momentum
        ASSERT_NEAR(U_initial(i).coeffs(0, 2), U(i).coeffs(0, 2), 1e-2); // Energy
    }
}

TEST(runge_kutta_order_test) {
    const std::size_t num_cells = 1;
    MockODEScheme scheme;
    MockODEScheme::Workspace workspace(num_cells);
    RungeKutta4<MockODEScheme> rk4(num_cells);
    
    // Test order of accuracy with different step sizes
    std::vector<double> step_sizes = {0.1, 0.05, 0.025};
    std::vector<double> errors;
    
    const double t_final = 1.0;
    
    for (double dt : step_sizes) {
        Field<MockODEScheme> U(num_cells);
        U(0).coeffs(0, 0) = 1.0; // Initial condition
        
        int num_steps = static_cast<int>(t_final / dt);
        
        // Integrate
        for (int step = 0; step < num_steps; ++step) {
            rk4.step(U, workspace, dt, scheme);
        }
        
        // Calculate error
        double exact = std::exp(-t_final);
        double error = std::abs(U(0).coeffs(0, 0) - exact);
        errors.push_back(error);
    }
    
    // Check that error decreases with step size
    ASSERT_TRUE(errors[1] < errors[0]);
    ASSERT_TRUE(errors[2] < errors[1]);
    
    // Check approximate order of accuracy
    double order_1_2 = std::log(errors[0] / errors[1]) / std::log(2.0);
    double order_2_3 = std::log(errors[1] / errors[2]) / std::log(2.0);
    
    // Should be approximately 4th order
    ASSERT_TRUE(order_1_2 > 3.0);
    ASSERT_TRUE(order_2_3 > 3.0);
}

int main() {
    std::cout << "Running Runge-Kutta integration tests...\n";
    global_test_framework.run_all_tests();
    return 0;
}
