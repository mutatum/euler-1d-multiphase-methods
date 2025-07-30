#pragma once
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include "../euler.hpp"

namespace physics::euler
{
    template <class PhysicsModel> // Euler physics
    struct Chandrashekar
    {
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
            if (std::abs(diff) < std::numeric_limits<Scalar>::epsilon() * 1e2)
            {
                return a;
            }
            return diff / (std::log(b) - std::log(a));
        }

        static Scalar Beta(const State &state)
        {
            return state.density / (static_cast<Scalar>(2.0) * Physics::pressure(state));
        }

        static void compute(const State &state_L, const State &state_R, State &result)
        {
            const Scalar BetaL = Beta(state_L);
            const Scalar BetaR = Beta(state_R);
            if (!std::isfinite(BetaL) || !std::isfinite(BetaR)) {
                std::cout << "Non-finite Beta: BetaL=" << BetaL << " BetaR=" << BetaR << std::endl;
                std::cout << "state_L: " << state_L << std::endl;
                std::cout << "state_R: " << state_R << std::endl;
            }
            const Scalar aBeta = arithmetic_mean(BetaL, BetaR);
            const Scalar lBeta = logarithmic_mean(BetaL, BetaR);
            if (!std::isfinite(lBeta) || lBeta <= 0) {
                std::cout << "Non-finite or non-positive lBeta: " << lBeta << std::endl;
                std::cout << "BetaL: " << BetaL << " BetaR: " << BetaR << std::endl;
            }
            const Scalar densityL = state_L.density;
            const Scalar densityR = state_R.density;
            const Scalar aDensity = arithmetic_mean(densityL, densityR);
            const Scalar lDensity = logarithmic_mean(densityL, densityR);
            if (!std::isfinite(lDensity) || lDensity <= 0) {
                // higher print precision
                std::cout << std::setprecision(17);
                std::cout << "state_L: " << state_L << std::endl;
                std::cout << "state_R: " << state_R << std::endl;
                std::cout << "Non-finite or non-positive lDensity: " << lDensity << std::endl;
                std::cout << "densityL: " << densityL << " densityR: " << densityR << std::endl;
            }
            const Scalar velocityL = state_L.velocity();
            const Scalar velocityR = state_R.velocity();
            const Scalar aVelocity = arithmetic_mean(velocityL, velocityR);
            const Scalar asqVelocity = arithmetic_mean(velocityL * velocityL, velocityR * velocityR);

            // f_S^i calculation:
            result.density = lDensity * aVelocity;
            result.momentum = aDensity / (static_cast<Scalar>(2.0) * aBeta) + aVelocity * result.density;
            result.total_energy = ((static_cast<Scalar>(0.5) / ((Physics::EOS::gamma - static_cast<Scalar>(1.0)) * lBeta) - static_cast<Scalar>(0.5) * asqVelocity)) * result.density + aVelocity * result.momentum;

            // Print result if non-finite
            if (!std::isfinite(result.density) || !std::isfinite(result.momentum) || !std::isfinite(result.total_energy)) {
                std::cout << "Non-finite flux result: " << result << std::endl;
            }
        }
    };
} // namespace physics::euler