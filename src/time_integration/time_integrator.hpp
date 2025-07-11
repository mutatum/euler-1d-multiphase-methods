#pragma once
#include "../field/field.hpp"
#include "../physics/flux_model.hpp"
#include "../scheme/numerical_scheme.hpp"

template <typename State>
class TimeIntegrator
{
public:
    virtual ~TimeIntegrator() = default;

    virtual void step(Field<State> &solution_field, // this will be updated in place
                      double dt,
                      const FluxModel<State> &flux_model,
                      const NumericalScheme<State> &numerical_scheme) = 0;
};