#pragma once
#include <functional>
#include "flux_model.hpp"

template <typename State>
class Euler_Base : public FluxModel<State>
{

public:
    /* U = { rho,
             rho * v,
             rho * E} */

    Euler_Base() override {}

    State physical_flux(const State &U) const override
    {
        /* F(U) = { rho * v, 
                    rho * v tensor v + p * I,
                    rho * E * v + p * v } */
        State F{U};

        F[0] = U[1];
        F[1]
        return F;
    }
};

public:
virtual ~FluxModel() = default;

virtual State physical_flux(const State &U) const = 0;

virtual State numerical_flux(const State &UL, const State &UR) const = 0;

virtual void prepare_step(const Field<State> &solution_field) const {};

virtual double max_wave_speed(const State &U) const = 0;

virtual double max_wave_speed(const Field<State> &solution_field) const = 0;
}
;