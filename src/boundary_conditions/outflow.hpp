#pragma once
#include "bc_base.hpp"

template <typename State>
class OutflowBC : public BoundaryCondition<State>
{

public:
    void apply(Field<State> &field, BoundaryLocation location) const
    {
        const auto n_ghost = field.get_mesh().num_ghost_cells();
        const auto n_real = field.get_mesh().num_real_cells();

        switch (location)
        {
        case BoundaryLocation::X_MIN:
        {
            const auto &source_cell = field(n_ghost);
            for (std::size_t i = 0; i < n_ghost; ++i)
            {
                field.cell(i) = source_cell;
            }
        }
        break;
        case BoundaryLocation::X_MAX:
        {
            const auto &source_cell = field(n_real + n_ghost - 1);
            for (std::size_t i = 0; i < n_ghost; ++i)
            {
                field.cell(n_real + n_ghost + i) = source_cell;
            }
        }
        break;
        }
    }
};