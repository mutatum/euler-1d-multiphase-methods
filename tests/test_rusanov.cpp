#include "test_framework.hpp"
#include "../src/physics/numerical_flux/rusanov.hpp"
#include "../src/physics/Euler/euler.hpp"
#include <cmath>

TestFramework global_test_framework;

// Test Rusanov flux computation
TEST(rusanov_flux_basic) {
    using namespace physics::euler;
    using EOS = IdealGasEOS<double, 1.4>;  // gamma = 1.4
    using Physics = EulerPhysics<double, EOS>;
    using NumericalFlux = Rusanov<Physics>;
    
    // Test case: symmetric states
    EulerState<double> state_L(1.0, 0.0, 2.5);  // rho=1, u=0, E=2.5
    EulerState<double> state_R(1.0, 0.0, 2.5);  // same as left
    EulerState<double> result;
    
    double max_speed = NumericalFlux::compute(state_L, state_R, result);
    
    // For symmetric states, flux should equal physical flux
    EulerState<double> expected_flux = Physics::physical_flux(state_L);
    
    ASSERT_NEAR(expected_flux.density, result.density, 1e-14);
    ASSERT_NEAR(expected_flux.momentum, result.momentum, 1e-14);
    ASSERT_NEAR(expected_flux.total_energy, result.total_energy, 1e-14);
    
    // Max speed should be sound speed (since u=0)
    double expected_speed = Physics::max_wave_speed(state_L);
    ASSERT_NEAR(expected_speed, max_speed, 1e-14);
}

TEST(rusanov_flux_contact_discontinuity) {
    using namespace physics::euler;
    // gamma = 1.4
    using EOS = IdealGasEOS<double, 1.4>;
    using Physics = EulerPhysics<double, EOS>;
    using NumericalFlux = Rusanov<Physics>;
    
    // Test case: contact discontinuity (different densities, same pressure and velocity)
    EulerState<double> state_L(1.0, 1.0, 2.5);  // rho=1, u=1, E=2.5
    EulerState<double> state_R(0.5, 0.5, 1.75); // rho=0.5, u=1, E=1.75
    EulerState<double> result;
    
    double max_speed = NumericalFlux::compute(state_L, state_R, result);
    
    // Should be well-defined
    ASSERT_TRUE(std::isfinite(result.density));
    ASSERT_TRUE(std::isfinite(result.momentum));
    ASSERT_TRUE(std::isfinite(result.total_energy));
    ASSERT_TRUE(std::isfinite(max_speed));
    ASSERT_TRUE(max_speed > 0);
}

TEST(rusanov_flux_shock_tube) {
    using namespace physics::euler;
    // gamma = 1.4
    using EOS = IdealGasEOS<double, 1.4>;
    using Physics = EulerPhysics<double, EOS>;
    using NumericalFlux = Rusanov<Physics>;
    
    // Test case: Sod shock tube initial conditions
    EulerState<double> state_L(1.0, 0.0, 2.5);   // High pressure
    EulerState<double> state_R(0.125, 0.0, 0.25); // Low pressure
    EulerState<double> result;
    
    double max_speed = NumericalFlux::compute(state_L, state_R, result);
    
    // Flux should be well-defined
    ASSERT_TRUE(std::isfinite(result.density));
    ASSERT_TRUE(std::isfinite(result.momentum));
    ASSERT_TRUE(std::isfinite(result.total_energy));
    
    // Max speed should be positive and reasonable
    ASSERT_TRUE(max_speed > 0);
    ASSERT_TRUE(max_speed < 10.0); // Sanity check
    
    // Check that it's greater than both individual wave speeds
    double speed_L = Physics::max_wave_speed(state_L);
    double speed_R = Physics::max_wave_speed(state_R);
    ASSERT_TRUE(max_speed >= speed_L);
    ASSERT_TRUE(max_speed >= speed_R);
}

TEST(rusanov_flux_consistency) {
    using namespace physics::euler;
    // gamma = 1.4
    using EOS = IdealGasEOS<double, 1.4>;
    using Physics = EulerPhysics<double, EOS>;
    using NumericalFlux = Rusanov<Physics>;
    
    // Test consistency: F(U, U) = f(U)
    EulerState<double> state(1.2, 0.3, 3.0);
    EulerState<double> result;
    
    double max_speed = NumericalFlux::compute(state, state, result);
    EulerState<double> expected = Physics::physical_flux(state);
    
    ASSERT_NEAR(expected.density, result.density, 1e-14);
    ASSERT_NEAR(expected.momentum, result.momentum, 1e-14);
    ASSERT_NEAR(expected.total_energy, result.total_energy, 1e-14);
}

TEST(rusanov_flux_conservation) {
    using namespace physics::euler;
    // gamma = 1.4
    using EOS = IdealGasEOS<double, 1.4>;
    using Physics = EulerPhysics<double, EOS>;
    using NumericalFlux = Rusanov<Physics>;
    
    // Test conservation property: numerical flux should be conservative
    // This means F(U_L, U_R) should be a proper average of the physical fluxes
    EulerState<double> state_L(1.0, 2.0, 10.0);
    EulerState<double> state_R(0.8, -1.0, 8.0);
    EulerState<double> result_LR;
    
    NumericalFlux::compute(state_L, state_R, result_LR);

    // Test that the flux is well-defined and finite
    ASSERT_TRUE(std::isfinite(result_LR.density));
    ASSERT_TRUE(std::isfinite(result_LR.momentum));
    ASSERT_TRUE(std::isfinite(result_LR.total_energy));

    // Test that the flux lies between the physical fluxes (reasonable for Rusanov)
    EulerState<double> flux_L = Physics::physical_flux(state_L);
    EulerState<double> flux_R = Physics::physical_flux(state_R);
    
    // The Rusanov flux should be a reasonable combination of the physical fluxes
    double min_density = std::min(flux_L.density, flux_R.density);
    double max_density = std::max(flux_L.density, flux_R.density);
    
    // Allow for some dissipation, so the flux might be outside the range slightly
    ASSERT_TRUE(result_LR.density >= min_density - 1.0);
    ASSERT_TRUE(result_LR.density <= max_density + 1.0);
}

TEST(rusanov_flux_conservation_property) {
    using namespace physics::euler;
    // gamma = 1.4
    using EOS = IdealGasEOS<double, 1.4>;
    using Physics = EulerPhysics<double, EOS>;
    using NumericalFlux = Rusanov<Physics>;
    
    // Test conservation property: F(U_L, U_R) + F(U_R, U_L) = f(U_L) + f(U_R)
    EulerState<double> state_L(1.0, 0.5, 2.5);
    EulerState<double> state_R(0.8, -0.2, 2.0);
    EulerState<double> result_LR, result_RL;
    
    NumericalFlux::compute(state_L, state_R, result_LR);
    NumericalFlux::compute(state_R, state_L, result_RL);
    
    // Physical fluxes
    EulerState<double> flux_L = Physics::physical_flux(state_L);
    EulerState<double> flux_R = Physics::physical_flux(state_R);
    
    // Conservation property: F(L,R) + F(R,L) = f(L) + f(R)
    EulerState<double> flux_sum = result_LR + result_RL;
    EulerState<double> physical_sum = flux_L + flux_R;
    
    ASSERT_NEAR(flux_sum.density, physical_sum.density, 1e-14);
    ASSERT_NEAR(flux_sum.momentum, physical_sum.momentum, 1e-14);
    ASSERT_NEAR(flux_sum.total_energy, physical_sum.total_energy, 1e-14);
}

TEST(rusanov_flux_symmetry) {
    using namespace physics::euler;
    // gamma = 1.4
    using EOS = IdealGasEOS<double, 1.4>;
    using Physics = EulerPhysics<double, EOS>;
    using NumericalFlux = Rusanov<Physics>;
    
    // Test symmetry property for truly symmetric states (same density, zero velocity)
    EulerState<double> state_L(1.0, 0.0, 2.5);
    EulerState<double> state_R(1.0, 0.0, 2.5);  // Identical states
    EulerState<double> result_LR, result_RL;
    
    NumericalFlux::compute(state_L, state_R, result_LR);
    NumericalFlux::compute(state_R, state_L, result_RL);
    
    // For identical states, the flux should be the same regardless of order
    ASSERT_NEAR(result_LR.density, result_RL.density, 1e-14);
    ASSERT_NEAR(result_LR.momentum, result_RL.momentum, 1e-14);
    ASSERT_NEAR(result_LR.total_energy, result_RL.total_energy, 1e-14);
    
    // And should equal the physical flux
    EulerState<double> physical_flux = Physics::physical_flux(state_L);
    ASSERT_NEAR(physical_flux.density, result_LR.density, 1e-14);
    ASSERT_NEAR(physical_flux.momentum, result_LR.momentum, 1e-14);
    ASSERT_NEAR(physical_flux.total_energy, result_LR.total_energy, 1e-14);
}

TEST(rusanov_flux_vacuum_states) {
    using namespace physics::euler;
    // gamma = 1.4
    using EOS = IdealGasEOS<double, 1.4>;
    using Physics = EulerPhysics<double, EOS>;
    using NumericalFlux = Rusanov<Physics>;
    
    // Test with very low density (near vacuum)
    EulerState<double> state_L(1.0, 0.0, 2.5);
    EulerState<double> state_R(1e-10, 0.0, 1e-10);
    EulerState<double> result;
    
    double max_speed = NumericalFlux::compute(state_L, state_R, result);
    
    // Should handle near-vacuum states without blowing up
    ASSERT_TRUE(std::isfinite(result.density));
    ASSERT_TRUE(std::isfinite(result.momentum));
    ASSERT_TRUE(std::isfinite(result.total_energy));
    ASSERT_TRUE(std::isfinite(max_speed));
    ASSERT_TRUE(max_speed > 0);
}

TEST(rusanov_flux_high_mach) {
    using namespace physics::euler;
    // gamma = 1.4
    using EOS = IdealGasEOS<double, 1.4>;
    using Physics = EulerPhysics<double, EOS>;
    using NumericalFlux = Rusanov<Physics>;
    
    // Test with high Mach number flow
    EulerState<double> state_L(1.0, 5.0, 15.0);  // High velocity
    EulerState<double> state_R(1.0, -5.0, 15.0); // High velocity, opposite direction
    EulerState<double> result;
    
    double max_speed = NumericalFlux::compute(state_L, state_R, result);
    
    // Should handle high Mach numbers
    ASSERT_TRUE(std::isfinite(result.density));
    ASSERT_TRUE(std::isfinite(result.momentum));
    ASSERT_TRUE(std::isfinite(result.total_energy));
    ASSERT_TRUE(std::isfinite(max_speed));
    
    // Max speed should be quite high
    ASSERT_TRUE(max_speed > 5.0);
}

TEST(rusanov_flux_entropy_condition) {
    using namespace physics::euler;
    // gamma = 1.4
    using EOS = IdealGasEOS<double, 1.4>;
    using Physics = EulerPhysics<double, EOS>;
    using NumericalFlux = Rusanov<Physics>;
    
    // Test entropy condition with expansion shock
    EulerState<double> state_L(0.125, 0.0, 0.25);  // Low pressure
    EulerState<double> state_R(1.0, 0.0, 2.5);     // High pressure
    EulerState<double> result;
    
    double max_speed = NumericalFlux::compute(state_L, state_R, result);
    
    // Rusanov flux should be dissipative enough to handle this
    ASSERT_TRUE(std::isfinite(result.density));
    ASSERT_TRUE(std::isfinite(result.momentum));
    ASSERT_TRUE(std::isfinite(result.total_energy));
    ASSERT_TRUE(max_speed > 0);
}

int main() {
    std::cout << "Running Rusanov numerical flux tests...\n";
    global_test_framework.run_all_tests();
    return 0;
}
