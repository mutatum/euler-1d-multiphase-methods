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

template <typename State>
class FirstOrderFV: public NumericalScheme<State>
{
public:
    void compute_residual(const Field<State> &solution_field,
                          const FluxModel<State> &flux_model,
                          Field<State> &R) const override
    {
        // 1D structured uniform mesh for now
        const mesh_structured_uniform_1d *mesh_ptr = dynamic_cast<const mesh_structured_uniform_1d *>(&solution_field.get_mesh());
        if (!mesh_ptr)
        {
            throw std::runtime_error("FirstOrderFV: Invalid mesh type. Expected mesh_structured_uniform_1d.");
        }
        const mesh_structured_uniform_1d &mesh = *mesh_ptr;

        const std::size_t n_ghost = mesh.num_ghost_cells();
        const std::size_t n_total = mesh.total_cells();

        for (std::size_t i = n_ghost; i < n_total - n_ghost; i++)
        {
            R(i) = mesh.dx() * (flux_model.numerical_flux(solution_field(i - 1), solution_field(i)) - flux_model.numerical_flux(solution_field(i), solution_field(i + 1)));
        }
    }
};