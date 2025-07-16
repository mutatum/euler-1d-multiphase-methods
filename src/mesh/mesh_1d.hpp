#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <type_traits>
#include "mesh.hpp"

class mesh_structured_uniform_1d : public Mesh
{
private:
    const double x_min_;
    const double x_max_;
    const double dx_;
    const std::size_t n_ghost_;
    const std::size_t n_cells_real_;
    const std::vector<double> cell_centers_;
    
    static std::vector<double> compute_cell_centers(double x_min, double dx, 
                                                   std::size_t n_ghost, std::size_t n_total)
    {
        std::vector<double> centers(n_total);
        for (std::size_t i = 0; i < n_total; ++i)
        {
            centers[i] = x_min + (static_cast<double>(i) - static_cast<double>(n_ghost) + 0.5) * dx;
        }
        return centers;
    }

public:
    mesh_structured_uniform_1d(double x_min, double x_max, std::size_t n_cells_real, std::size_t n_ghost) 
        : Mesh(2 * n_ghost + n_cells_real),
          x_min_(x_min),
          x_max_(x_max),
          n_cells_real_(n_cells_real),
          dx_((x_max - x_min) / static_cast<double>(n_cells_real)),
          n_ghost_(n_ghost),
          cell_centers_(compute_cell_centers(x_min, dx_, n_ghost, n_cells_total_))
    {
        if (n_cells_real == 0)
        {
            throw std::invalid_argument("Number of real cells must be positive.");
        }
        if (n_ghost == 0)
        {
            throw std::invalid_argument("Number of ghost cells must be positive.");
        }
        if (x_min >= x_max)
        {
            throw std::invalid_argument("Invalid domain: x_min must be less than x_max.");
        }
    }

    [[nodiscard]] constexpr std::size_t num_cells() const noexcept { return n_cells_real_; }
    [[nodiscard]] constexpr std::size_t num_ghost_cells() const noexcept { return n_ghost_; }
    [[nodiscard]] constexpr std::size_t total_cells() const noexcept { return n_cells_total_; }
    [[nodiscard]] constexpr double domain_start() const noexcept { return x_min_; }
    [[nodiscard]] constexpr double domain_end() const noexcept { return x_max_; }
    [[nodiscard]] constexpr double dx() const noexcept { return dx_; }
    [[nodiscard]] double cell_center(std::size_t i) const { return cell_centers_.at(i); }
};