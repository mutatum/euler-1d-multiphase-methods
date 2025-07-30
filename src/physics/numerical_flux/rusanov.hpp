#pragma once
#include <cmath>
#include "../physics_concepts.hpp"

template <PhysicsModelConcept PhysicsModel>
struct Rusanov
{
    using Physics = PhysicsModel;
    using State = typename Physics::State;
    using Scalar = typename State::Scalar;

    static Scalar compute(const State &state_L,
                          const State &state_R,
                          State &result)
    {
        const State flux_L = Physics::physical_flux(state_L);
        const State flux_R = Physics::physical_flux(state_R);

        const Scalar speed_L = Physics::max_wave_speed(state_L);
        const Scalar speed_R = Physics::max_wave_speed(state_R);
        const Scalar max_speed = std::max(speed_L, speed_R);

        result.density = 0.5 * (flux_L.density + flux_R.density) - 0.5 * max_speed * (state_R.density - state_L.density);
        result.momentum = 0.5 * (flux_L.momentum + flux_R.momentum) - 0.5 * max_speed * (state_R.momentum - state_L.momentum);
        result.total_energy = 0.5 * (flux_L.total_energy + flux_R.total_energy) - 0.5 * max_speed * (state_R.total_energy - state_L.total_energy);

        return max_speed;
    }
};