#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include "mesh.hpp"

class mesh_structured_uniform_1d : public Mesh
{
private:
    double x_min_;
    double x_max_;
    double dx_;
    std::size_t n_ghost_;
    std::size_t n_cells_real_;
    std::vector<double> cell_centers_;

public:
    mesh_structured_uniform_1d(double x_min, double x_max, std::size_t n_cells_real, std::size_t n_ghost) : Mesh(2 * n_ghost + n_cells_real),
                                                                                                            x_min_(x_min),
                                                                                                            x_max_(x_max),
                                                                                                            n_cells_real_(n_cells_real),
                                                                                                            dx_((x_max - x_min) / n_cells_real),
                                                                                                            n_ghost_(n_ghost),
                                                                                                            cell_centers_(2 * n_ghost + n_cells_real)
    {
        if (n_cells_real <= 0)
        {
            throw std::invalid_argument("Number of real cells must be positive.");
        }
        if (n_ghost <= 0)
        {
            throw std::invalid_argument("Number of ghost cells must be positive.");
        }
        if (x_min >= x_max)
        {
            throw std::invalid_argument("Invalid domain: x_min must be less than x_max.");
        }
        for (std::size_t i = 0; i < n_cells_real_ + n_ghost ; ++i)
        {
            cell_centers_.at(i) = x_min_ + (i - n_ghost + 0.5) * dx_;
        }
    }

    inline std::size_t num_cells() const { return n_cells_real_; };
    inline std::size_t num_ghost_cells() const { return n_ghost_; };
    inline std::size_t total_cells() const { return n_cells_total_; }
    inline double domain_start() const { return x_min_; }
    inline double domain_end() const { return x_max_; }
    inline double dx() const { return dx_; };
    inline double cell_center(std::size_t i) const { return cell_centers_.at(i); };
};