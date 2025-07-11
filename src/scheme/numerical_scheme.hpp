#pragma once

#include "../field/field.hpp"
#include "../physics/flux_model.hpp"
#include "../mesh/mesh_1d.hpp" // 1D uniform mesh for now

// Resolving Circular Dependency
template <class SchemePolicy>
class Field;

template <typename StateType>
class NumericalScheme
{
public:
    using State = StateType;
    virtual ~NumericalScheme() = default;

    virtual void compute_residual(const Field<NumericalScheme<State>> &solution_field,
                                  Field<NumericalScheme<State>> &Residual) const = 0;
};