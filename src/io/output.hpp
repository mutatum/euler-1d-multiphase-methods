#pragma once
#include <fstream>
#include <string>
#include <limits>
#include <iomanip>
#include <iostream>
#include "../field/field.hpp"
#include "../physics/Euler/isentropic_solution.hpp"

/**
 * @brief Write metadata as comments to the CSV output file.
 */
template <class Scheme>
void write_csv_metadata(std::ofstream &outfile, const std::string &scheme_name, int poly_order, int quad_order, std::size_t num_cells, const std::string &time_integrator, double cfl, std::size_t step_count, double final_time)
{
    outfile << std::setprecision(4) << std::scientific;
    outfile << "# scheme: " << scheme_name << "\n";
    outfile << "# polynomial_order: " << poly_order << "\n";
    outfile << "# quadrature_order: " << quad_order << "\n";
    outfile << "# num_cells: " << num_cells << "\n";
    outfile << "# time_stepper: " << time_integrator << "\n";
    outfile << "# cfl: " << cfl << "\n";
    outfile << "# step_count: " << step_count << "\n";
    outfile << "# final_time: " << final_time << "\n";
}
 
/**
 * @brief Write the exact solution to a CSV file for comparison.
 */
template <class Scheme>
void write_exact_solution_to_file(
    double domain_start,
    double dx,
    std::size_t num_cells,
    double t,
    const std::string &filename,
    int points_per_cell = 40,
    std::size_t step_count = 0)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    outfile << std::setprecision(17) << std::scientific;

    write_csv_metadata<Scheme>(
        outfile,
        "Exact",
        static_cast<int>(Scheme::PolynomialOrder + 1),
        static_cast<int>(Scheme::Quadrature::order + 1),
        num_cells,
        "none",
        0.0,
        step_count,
        t);

    outfile << std::numeric_limits<typename Scheme::Scalar>::digits10 + 2;
    outfile << "cell_index,cell_left,cell_right,x,rho,momentum,total_energy\n";
    for (std::size_t i = 0; i < num_cells; ++i)
    {
        const double x_left = domain_start + i * dx / num_cells;
        const double x_right = x_left + dx;
        for (int j = 0; j < points_per_cell; ++j)
        {
            const double x = x_left + i * dx;
            const auto exact_state = physics::euler::isentropic_solution<double>(x, t, 0.1);
            outfile << i << "," << x_left << "," << x_right << "," << x << ","
                    << exact_state.density << "," << exact_state.momentum << "," << exact_state.total_energy << "\n";
        }
    }
}

/**
 * @brief Write the numerical solution to a CSV file.
 */
template <class Scheme>
void write_solution_to_file(
    const Field<Scheme> &U,
    double domain_start,
    double dx,
    const std::string &filename,
    std::size_t step_count = 0,
    double final_time = 0.0,
    double cfl = 0.0,
    const std::string &time_integrator = "unknown",
    const std::string &scheme_name = "DG")
{
    std::ofstream outfile(filename);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    write_csv_metadata<Scheme>(
        outfile,
        scheme_name,
        Scheme::PolynomialOrder,
        Scheme::Quadrature::order + 1,
        U.size(),
        time_integrator,
        cfl,
        step_count,
        final_time);

    // using Quadrature = typename Scheme::Quadrature;
    // using Quadrature = GLLQuadrature<typename Scheme::Scalar, Scheme::PolynomialOrder + 0>;
    using Quadrature = GLLQuadrature<typename Scheme::Scalar, 15>;
    const std::size_t points_per_cell = Quadrature::nodes.size();
    outfile << std::setprecision(std::numeric_limits<typename Scheme::Scalar>::digits10 + 2);
    outfile << "cell_index,cell_left,cell_right,x,rho,momentum,total_energy\n";
    // Output the left edge of first cell
    for (std::size_t i = 0; i < U.size(); ++i)
    {
        const double x_left = domain_start + i * dx;
        const double x_right = x_left + dx;
        for (int j = 0; j < points_per_cell; ++j)
        {
            const double xi = Quadrature::nodes[j];
            const double x = x_left + (1.0+xi) * dx / 2.0;
            typename Scheme::State state = Scheme::evaluate_element(U(i), xi);
            outfile << i << "," << x_left << "," << x_right << "," << x << "," << state.density << "," << state.momentum << "," << state.total_energy << "\n";
        }
    }
}
