#pragma once
#include "time_integrator.hpp"

template <typename State>
class RungeKutta4: public TimeIntegrator<State>
{
public:
    void step(Field<State> &Un,
              double dt,
              const FluxModel<State> &flux_model,
              const NumericalScheme<State> &numerical_scheme) override
    {
        Field<State> k_1;
        Field<State> k_2;
        Field<State> k_3;
        Field<State> k_4;
        Field<State> R;

        k_1 = numerical_scheme.compute_residual(Un, flux_model, R);
        k_2 = numerical_scheme.compute_residual(Un+.5*dt*k_1, flux_model, R);
        k_3 = numerical_scheme.compute_residual(Un+.5*dt*k_2, flux_model, R);
        k_4 = numerical_scheme.compute_residual(Un+dt*k_3, flux_model, R);

        Un += (dt/6)*(k_1+2*k_2+2*k_3+k_4);
        //! This updates ghosts cells which is unnecessary
        //! however the cost for now is marginal.. (relative to complexity of changing it)
        //! as we are not in 2d or 3d
    }
};