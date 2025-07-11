#pragma once
#include "../field/field.hpp"

template <typename State>
class FluxModel
{
public:
    virtual ~FluxModel() = default;

    virtual State physical_flux(const State &U) const = 0;

    virtual State numerical_flux(const State &UL, const State &UR) const = 0;

    virtual void prepare_step(const Field<State> &solution_field) const {};

    virtual double max_wave_speed(const State &U) const = 0;

    virtual double max_wave_speed(const Field<State> &solution_field) const = 0;
};