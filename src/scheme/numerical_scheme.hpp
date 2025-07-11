#pragma once

#include "../field/field.hpp"
#include "../physics/flux_model.hpp"
#include "../mesh/mesh_1d.hpp" // 1D uniform mesh for now

template <typename State>
class NumericalScheme
{
public:
    virtual ~NumericalScheme() = default;

    virtual void compute_residual(const Field<State> &solution_field,
                                  const FluxModel<State> &flux_model,
                                  Field<State> &R) const = 0;
};