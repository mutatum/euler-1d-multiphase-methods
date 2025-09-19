#include "../src/physics/Euler/isentropic_solution.hpp"
#include "../src/scheme/DG.hpp"
#include "../src/physics/Euler/euler.hpp"
#include "../src/physics/numerical_flux/rusanov.hpp"
#include "../src/utils/functional.hpp"
#include "../src/utils/basis/legendre.hpp"
#include "../src/utils/basis/lagrange.hpp"
#include "../src/utils/quadrature/gauss_legendre.hpp"
#include "../src/utils/quadrature/gauss_lobatto_legendre.hpp"
#include "../src/boundary/boundary_conditions.hpp"
#include "../src/time_integration/runge_kutta.hpp"
#include "../src/io/output.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip> // Include iomanip for std::setprecision

template <class Scheme>
void initialize_field(Field<Scheme> &field, double domain_start, double domain_end, double factor = 0.1)
{
    const std::size_t num_cells = field.size();
    const double dx = field.dx;

    for (std::size_t i = 0; i < num_cells; ++i)
    {
        const double x_left = domain_start + i * dx; // Left edge of cell
        auto f = [x_left, factor, dx](double xi)
        {
            Eigen::Vector3d coeffs;
            // Correct mapping: xi ∈ [-1,1] -> physical_x ∈ [x_left, x_left + dx]
            const double physical_x = x_left + (xi + 1.0) * dx / 2.0;
            const auto result = physics::euler::isentropic_solution<double>(physical_x, 0.0, factor);
            coeffs(0) = result.density;
            coeffs(1) = result.momentum;
            coeffs(2) = result.total_energy;
            return coeffs;
        };
        field(i).coeffs = compute_L2_projection_onto_basis<GLQuadrature<double, 14>, typename Scheme::PolynomialBasis>(f);
    }
}
using namespace physics::euler;
struct DGConfig {
    using Physics = EulerPhysics<double, IdealGasEOS<double, 3.0>>;
    using NumericalFlux = Rusanov<Physics>;
    static constexpr std::size_t Order = 9;
    using Quadrature = GLLQuadrature<double, Order+1>;
    // using PolynomialBasis = Lagrange<GLQuadrature<double, Order>::nodes>;
    using PolynomialBasis = Lagrange<Quadrature::nodes>;
    // using PolynomialBasis = Legendre<double, Order>;
    template <class Scheme> using LeftBC = BoundaryConditions::LeftPeriodicBC<Scheme>;
    template <class Scheme> using RightBC = BoundaryConditions::RightPeriodicBC<Scheme>;
};

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <number of starting cells>" << " <final time>" << std::endl;
        return 1;
    }
    std::size_t starting_n_cells = std::stoi(argv[1]);
    const double final_time = std::stod(argv[2]);

    using DGScheme = DG<DGConfig>;

    DGScheme scheme;
    const double domain_start = -1.0;
    const double domain_end = 1.0;
    double t = 0.0;
    double dt = 0.0;

    // ------------------- Convergence Analysis -------------------
    std::cout << "Starting convergence analysis..." << std::endl;
    std::size_t num_cells = starting_n_cells;
    std::size_t prev_num_cells = num_cells;
    const double rate = 1.5;
    double last_error = 0.0;

    std::cout << "Final time: " << final_time << std::endl;
    for (std::size_t i = 0; i < 10; ++i)
    {
        const double dx_conv = (domain_end - domain_start) / num_cells;
        Field<DGScheme> U_convergence(num_cells, domain_start, domain_end);
        initialize_field(U_convergence, domain_start, domain_end, 0.1);
        DGScheme::Workspace workspace_convergence(num_cells);
        const double cfl = 0.3;
        RKSSP<DGScheme, 4> rk(num_cells, cfl);
        t = 0.0;

        std::size_t steps = 0;
        while (t < final_time)
        {
            dt = rk.step(U_convergence, workspace_convergence, scheme, dx_conv, final_time - t);

            t += dt;
            ++steps;
        }
        // std::cout << "Finished simulation for " << num_cells << " cells at t = " << t << " after " << steps << " steps." << std::endl;
        write_solution_to_file(U_convergence, domain_start, dx_conv, "solution_final" + std::to_string(i) + "_order" + std::to_string(DGScheme::PolynomialOrder + 1) + ".csv",
                               /*step_count=*/steps, /*final_time=*/t, /*cfl=*/cfl, /*time_integrator=*/"RungeKutta4");

        auto exact_solution = [&final_time](double x)
        {
            Eigen::Vector<double, 3> result;
            result.setZero();
            auto state = physics::euler::isentropic_solution<double>(x, final_time, 0.1);
            result << state.density, state.momentum, state.total_energy;
            return result;
        };

        // std::cout << "Error for " << num_cells << " cells: " << current_error << std::endl;
        double current_error = compute_L2_error(U_convergence, exact_solution);

        if (i > 0)
        {
            double rate = std::log(last_error / current_error) / std::log(static_cast<double>(num_cells) / prev_num_cells);
            std::cout << "Convergence rate for " << prev_num_cells << "->" << num_cells << " cells: " << rate << "\tStep count: " << steps << std::endl;
        }

        last_error = current_error;
        prev_num_cells = num_cells;
        num_cells = static_cast<double>(num_cells) * 1.5;
        if (current_error < 1e-14)
        {
            std::cout << "Error is sufficiently small to stop analysis: " << current_error << std::endl;
            break;
        }
    }
    double dx_convergence = (domain_end - domain_start) / prev_num_cells;
    write_exact_solution_to_file<DGScheme>(
        domain_start, dx_convergence, prev_num_cells, t,
        "solution_exact_order" + std::to_string(DGScheme::PolynomialOrder + 1) + ".csv",
        10 * (DGScheme::PolynomialOrder + 2));
    std::cout << "Order of polynomial basis: " << DGScheme::PolynomialOrder << std::endl;

    std::cout << "Convergence analysis completed." << std::endl;
    return 0;
}
