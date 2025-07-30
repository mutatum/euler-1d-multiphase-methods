#pragma once
#include "../euler.hpp"

namespace physics::euler
{
    template <class PhysicsModel>
    struct IsmailRoe // entropy conservative flux (only for Ideal gas law)
    {
        using Physics = PhysicsModel;
        using State = typename Physics::State;
        using Scalar = typename State::Scalar;

        struct Z
        {
            Scalar z1, z2, z3;
        };

        static Z to_Z(const State &state)
        {
            const Scalar pressure = Physics::pressure(state);
            const Scalar factor = std::sqrt(Physics::density(state) / pressure);
            Z z;
            z.z1 = factor;
            z.z2 = factor * Physics::velocity(state);
            z.z3 = factor * pressure;
            return z;
        }

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

        static Z arithmetic_mean(const Z &z_L, const Z &z_R)
        {
            Z z_am;
            z_am.z1 = arithmetic_mean(z_L.z1, z_R.z1);
            z_am.z2 = arithmetic_mean(z_L.z2, z_R.z2);
            z_am.z3 = arithmetic_mean(z_L.z3, z_R.z3);
            return z_am;
        }

        static Z logarithmic_mean(const Z &z_L, const Z &z_R)
        {
            Z z_lm;
            z_lm.z1 = logarithmic_mean(z_L.z1, z_R.z1);
            // skipping cause velocity can be negative and isn't used
            // z_lm.z2 = logarithmic_mean(z_L.z2, z_R.z2);
            z_lm.z2 = 0.0;
            z_lm.z3 = logarithmic_mean(z_L.z3, z_R.z3);
            return z_lm;
        }

        static void compute(const State &state_L,
                            const State &state_R,
                            State &result)
        {
            const Z z_L = to_Z(state_L);
            const Z z_R = to_Z(state_R);

            const Z z_am = arithmetic_mean(z_L, z_R);
            const Z z_lm = logarithmic_mean(z_L, z_R);

            result.density = z_am.z2 * z_lm.z3;
            result.momentum = (z_am.z3 + result.density * z_am.z2) / z_am.z1;
            result.total_energy = static_cast<Scalar>(0.5) * (z_am.z2 / z_am.z1) * (((Physics::EOS::gamma + 1) / (Physics::EOS::gamma - 1)) * (z_lm.z3 / z_lm.z1) + result.momentum);
        }
    };
} // namespace physics::euler