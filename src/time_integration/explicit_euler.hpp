#pragma once
#include "time_integrator.hpp"

template <typename State>
class ExplicitEuler : public TimeIntegrator<State>
{
public:
    void step(Field<State> &solution_field,
              double dt,
              const FluxModel<State> &flux_model,
              const NumericalScheme<State> &numerical_scheme) override
    {
        Field<State> R(solution_field.get_mesh());
        numerical_scheme.compute_residual(solution_field, flux_model, R);

        for (std::size_t i = 0; i < solution_field.size(); ++i)
        {
            solution_field(i) += dt * R(i);
        } // needs to be vectorized later //!\\

    }
};