#pragma once
#include <cmath>

template <class PhysicsModel>
struct Rusanov {
    using Physics = PhysicsModel;
    using State = typename Physics::State;
    using Scalar = typename State::Scalar;
    
    static Scalar compute(const State& state_L,
                         const State& state_R,
                         State& result)
    {
        const State flux_L = Physics::physical_flux(state_L);
        const State flux_R = Physics::physical_flux(state_R);

        const Scalar speed_L = Physics::max_wave_speed(state_L);
        const Scalar speed_R = Physics::max_wave_speed(state_R);
        const Scalar max_speed = std::max(speed_L, speed_R);

        result = 0.5 * (flux_L + flux_R) - 0.5 * max_speed * (state_R - state_L);
        return max_speed;
    }
};