#pragma once
#include "flux_model.hpp"
#include "../mesh/mesh_1d.hpp"
#include <cmath>
#include <algorithm>

class BurgersFluxBase : public FluxModel<double>
{
public:
    inline double physical_flux(const double &U) const override { return U * U / 2.0; }

    inline double max_wave_speed(const double &U) const override { return std::abs(U); }

    virtual double max_wave_speed(const Field<double> &solution_field) const override
    {
        double max_speed{0.0};
        const mesh_structured_uniform_1d *mesh_ptr = dynamic_cast<const mesh_structured_uniform_1d *>(&solution_field.get_mesh());
        if (!mesh_ptr)
        {
            throw std::runtime_error("BurgersFluxBase: Invalid mesh type. Expected mesh_structured_uniform_1d.");
        }
        const mesh_structured_uniform_1d &mesh = *mesh_ptr;

        const std::size_t n_ghost = mesh.num_ghost_cells();
        const std::size_t n_total = mesh.total_cells();
        for (std::size_t i = n_ghost; i < n_total - n_ghost; i++)
        {
            max_speed = std::max(max_speed, std::abs(solution_field(i)));
        }
        return max_speed;
    }
};

class BurgersFluxLaxFriedrichs : public BurgersFluxBase
{
    mutable double cached_global_wave_speed_{0.0};

public:
    inline void prepare_step(const Field<double> &solution_field) const override { cached_global_wave_speed_ = this->max_wave_speed(solution_field); }

    double numerical_flux(const double &UL, const double &UR) const override
    {
        return 0.5 * (UL + UR - cached_global_wave_speed_ * (UR - UL));
    }
};