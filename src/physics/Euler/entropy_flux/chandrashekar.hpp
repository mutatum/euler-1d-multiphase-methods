#pragma once
#include <cmath>
#include "../euler.hpp"

template <class PhysicsModel> // Euler physics
struct Chandrashekar {
    using Physics = PhysicsModel;
    using State = typename Physics::State;
    using Scalar = typename State::Scalar;

    static Scalar arithmetic_mean(Scalar a, Scalar b)
    {
        return static_cast<Scalar>(0.5) * (a + b);
    }

    static Scalar logarithmic_mean(Scalar a, Scalar b)
    {
        Scalar diff = b - a;
        if (std::abs(diff) < std::numeric_limits<Scalar>::epsilon() * std::max(std::abs(a), std::abs(b)))
        {
            return a;
        }
        return diff / (std::log(b) - std::log(a));
    }

    static Scalar Beta(const State &state)
    {
        return state.density / (static_cast<Scalar>(2.0) * Physics::pressure(state));
    }

    static void compute(const State &state_L,
                        const State &state_R,
                        State &result)
    {
        const Scalar BetaL = Beta(state_L);
        const Scalar BetaR = Beta(state_R);
        const Scalar aBeta = arithmetic_mean(BetaL, BetaR);
        const Scalar lBeta = logarithmic_mean(BetaL, BetaR);
        const Scalar densityL = state_L.density;
        const Scalar densityR = state_R.density;
        const Scalar aDensity= arithmetic_mean(densityL, densityR);
        const Scalar lDensity = logarithmic_mean(densityL, densityR);
        const Scalar velocityL = state_L.velocity();
        const Scalar velocityR = state_R.velocity();
        const Scalar aVelocity = arithmetic_mean(velocityL, velocityR);
        const Scalar asqVelocity = arithmetic_mean(velocityL*velocityL, velocityR*velocityR);

        // f_S^i calculation:
        result.density = lDensity * aVelocity;
        result.momentum = aDensity / (static_cast<Scalar>(2.0) * aBeta) + aVelocity * result.density;
        result.total_energy = ( (static_cast<Scalar>(0.5) / ((Physics::EOS::gamma-static_cast<Scalar>(1.0))*lBeta) - static_cast<Scalar>(0.5)*asqVelocity) )*result.density + aVelocity * result.momentum;
    }
};