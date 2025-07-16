#include "test_framework.hpp"
#include "../src/physics/Euler/euler.hpp"
#include <cmath>

TestFramework global_test_framework;

// Test EulerState basic operations
TEST(euler_state_construction) {
    using namespace physics::euler;
    EulerState<double> state1;
    ASSERT_EQ(0.0, state1.density);
    ASSERT_EQ(0.0, state1.momentum);
    ASSERT_EQ(0.0, state1.total_energy);
    
    constexpr EulerState<double> state2(1.0, 2.0, 3.0);
    ASSERT_EQ(1.0, state2.density);
    ASSERT_EQ(2.0, state2.momentum);
    ASSERT_EQ(3.0, state2.total_energy);
}

TEST(euler_state_velocity) {
    using namespace physics::euler;
    constexpr EulerState<double> state(2.0, 4.0, 10.0);
    ASSERT_EQ(2.0, state.velocity()); // momentum / density = 4.0 / 2.0 = 2.0
}

TEST(euler_state_specific_internal_energy) {
    using namespace physics::euler;
    constexpr EulerState<double> state(1.0, 2.0, 5.0);
    // specific_internal_energy = total_energy/density - 0.5 * velocity^2
    // = 5.0/1.0 - 0.5 * 2.0^2 = 5.0 - 2.0 = 3.0
    ASSERT_EQ(3.0, state.specific_internal_energy());
}

TEST(euler_state_arithmetic_operations) {
    using namespace physics::euler;
    EulerState<double> state1(1.0, 2.0, 3.0);
    EulerState<double> state2(2.0, 3.0, 4.0);
    
    // Addition
    EulerState<double> sum = state1 + state2;
    ASSERT_EQ(3.0, sum.density);
    ASSERT_EQ(5.0, sum.momentum);
    ASSERT_EQ(7.0, sum.total_energy);
    
    // Subtraction
    EulerState<double> diff = state2 - state1;
    ASSERT_EQ(1.0, diff.density);
    ASSERT_EQ(1.0, diff.momentum);
    ASSERT_EQ(1.0, diff.total_energy);
    
    // Scalar multiplication
    EulerState<double> scaled = state1 * 2.0;
    ASSERT_EQ(2.0, scaled.density);
    ASSERT_EQ(4.0, scaled.momentum);
    ASSERT_EQ(6.0, scaled.total_energy);
    
    // Scalar multiplication (commutative)
    EulerState<double> scaled2 = 2.0 * state1;
    ASSERT_EQ(scaled.density, scaled2.density);
    ASSERT_EQ(scaled.momentum, scaled2.momentum);
    ASSERT_EQ(scaled.total_energy, scaled2.total_energy);
}

TEST(euler_state_compound_assignment) {
    using namespace physics::euler;
    EulerState<double> state1(1.0, 2.0, 3.0);
    EulerState<double> state2(2.0, 3.0, 4.0);
    
    state1 += state2;
    ASSERT_EQ(3.0, state1.density);
    ASSERT_EQ(5.0, state1.momentum);
    ASSERT_EQ(7.0, state1.total_energy);
    
    state1 -= state2;
    ASSERT_EQ(1.0, state1.density);
    ASSERT_EQ(2.0, state1.momentum);
    ASSERT_EQ(3.0, state1.total_energy);
    
    state1 *= 2.0;
    ASSERT_EQ(2.0, state1.density);
    ASSERT_EQ(4.0, state1.momentum);
    ASSERT_EQ(6.0, state1.total_energy);
}

// Test IdealGasEOS
TEST(ideal_gas_eos_pressure) {
    using namespace physics::euler;
    using EOS = IdealGasEOS<double, 1.4>;  // gamma = 1.4
    
    EulerState<double> state(1.0, 2.0, 5.0);
    // pressure = (gamma - 1) * (E - 0.5 * mom^2 / rho)
    // = 0.4 * (5.0 - 0.5 * 4.0 / 1.0) = 0.4 * (5.0 - 2.0) = 1.2
    ASSERT_NEAR(1.2, EOS::pressure(state), 1e-10);
}

TEST(ideal_gas_eos_sound_speed) {
    using namespace physics::euler;
    using EOS = IdealGasEOS<double, 1.4>;  // gamma = 1.4
    
    EulerState<double> state(1.0, 2.0, 5.0);
    double pressure = EOS::pressure(state); // 1.2
    double expected_sound_speed = std::sqrt(EOS::gamma * pressure / state.density);
    // = sqrt(1.4 * 1.2 / 1.0) = sqrt(1.68)
    ASSERT_NEAR(expected_sound_speed, EOS::sound_speed(state), 1e-10);
    
    // Test with separate pressure and density
    ASSERT_NEAR(expected_sound_speed, EOS::sound_speed(pressure, state.density), 1e-10);
}

// Test EulerPhysics
TEST(euler_physics_basic) {
    using namespace physics::euler;
    using EOS = IdealGasEOS<double, 1.4>;  // gamma = 1.4
    using Physics = EulerPhysics<double, EOS>;
    
    EulerState<double> state(1.0, 2.0, 5.0);
    
    // Test pressure
    ASSERT_NEAR(1.2, Physics::pressure(state), 1e-10);  // Use ASSERT_NEAR instead
    
    // Test velocity
    ASSERT_EQ(2.0, Physics::velocity(state));  // This should be okay as it's exact division
    
    // Test variables count
    ASSERT_EQ(3, Physics::variables);
}

TEST(euler_physics_physical_flux) {
    using namespace physics::euler;
    using EOS = IdealGasEOS<double, 1.4>;  // gamma = 1.4
    using Physics = EulerPhysics<double, EOS>;
    
    EulerState<double> state(1.0, 2.0, 5.0);
    EulerState<double> flux = Physics::physical_flux(state);
    
    double u = state.velocity(); // 2.0
    double p = Physics::pressure(state); // 1.2
    
    // Expected flux components:
    // F1 = rho * u = 1.0 * 2.0 = 2.0
    // F2 = rho * u^2 + p = 1.0 * 4.0 + 1.2 = 5.2
    // F3 = (E + p) * u = (5.0 + 1.2) * 2.0 = 12.4
    
    ASSERT_NEAR(2.0, flux.density, 1e-10);       // Use ASSERT_NEAR instead
    ASSERT_NEAR(5.2, flux.momentum, 1e-10);      // Use ASSERT_NEAR instead
    ASSERT_NEAR(12.4, flux.total_energy, 1e-10); // Use ASSERT_NEAR instead
}

TEST(euler_physics_max_wave_speed) {
    using namespace physics::euler;
    using EOS = IdealGasEOS<double, 1.4>;  // gamma = 1.4
    using Physics = EulerPhysics<double, EOS>;
    
    EulerState<double> state(1.0, 2.0, 5.0);
    
    double u = state.velocity(); // 2.0
    double a = EOS::sound_speed(state); // sqrt(1.68)
    double expected_max_speed = std::abs(u) + a;
    
    ASSERT_NEAR(expected_max_speed, Physics::max_wave_speed(state), 1e-10);
    
    // Test with separate parameters
    double p = Physics::pressure(state);
    ASSERT_NEAR(expected_max_speed, Physics::max_wave_speed(u, p, state.density), 1e-10);
}

int main() {
    std::cout << "Running EulerState and EulerPhysics tests...\n";
    global_test_framework.run_all_tests();
    return 0;
}
