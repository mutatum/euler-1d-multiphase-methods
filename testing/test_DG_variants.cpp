#include "../src/physics/Euler/isentropic_solution.hpp"
#include "../src/scheme/DGSEM.hpp"
#include "../src/physics/Euler/euler.hpp"
#include "../src/physics/numerical_flux/rusanov.hpp"
#include "../src/physics/Euler/entropy_flux/chandrashekar.hpp"
#include "../src/physics/Euler/entropy_flux/ismail_roe.hpp"
#include "../src/utils/functional.hpp"
#include "../src/utils/basis/legendre.hpp"
#include "../src/utils/basis/lagrange.hpp"
#include "../src/utils/quadrature/gauss_legendre.hpp"
#include "../src/utils/quadrature/gauss_lobatto_legendre.hpp"
#include "../src/boundary/boundary_conditions.hpp"
#include "../src/time_integration/runge_kutta.hpp"
#include "../src/io/output.hpp"
#include "../src/scheme/DGSEM.hpp"
#include "../src/scheme/DGSEM_entropy_stable.hpp"
#include "../src/scheme/DG.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>

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
        field(i).coeffs = compute_L2_projection_onto_basis<typename Scheme::Quadrature, typename Scheme::PolynomialBasis>(f);
    }
}

using namespace physics::euler;
static constexpr std::size_t order = 7;

// Scheme configs
struct DGConfig {
    using Physics = EulerPhysics<double, IdealGasEOS<double, 3.0>>;
    using NumericalFlux = Rusanov<Physics>;
    static constexpr std::size_t order = ::order;
    using Quadrature = GLQuadrature<double, order+1>;
    using PolynomialBasis = Legendre<double, order>;
    template <class Scheme> using LeftBC = BoundaryConditions::LeftPeriodicBC<Scheme>;
    template <class Scheme> using RightBC = BoundaryConditions::RightPeriodicBC<Scheme>;
    static std::string name() { return "DG"; }
};

struct DGSEM_ESConfig {
    using Physics = EulerPhysics<double, IdealGasEOS<double, 3.0>>;
    using EntropyFlux = Chandrashekar<Physics>;
    using NumericalFlux = Rusanov<Physics>;
    static constexpr std::size_t order = ::order;
    template <class Scheme> using LeftBC = BoundaryConditions::LeftPeriodicBC<Scheme>;
    template <class Scheme> using RightBC = BoundaryConditions::RightPeriodicBC<Scheme>;
    static std::string name() { return "DGSEM_ES"; }
};

struct DGSEMConfig {
    using Physics = EulerPhysics<double, IdealGasEOS<double, 3.0>>;
    using NumericalFlux = Rusanov<Physics>;
    static constexpr std::size_t order = ::order;
    template <class Scheme> using LeftBC = BoundaryConditions::LeftPeriodicBC<Scheme>;
    template <class Scheme> using RightBC = BoundaryConditions::RightPeriodicBC<Scheme>;
    static std::string name() { return "DGSEM"; }
};

// Test runner for a given scheme
template <typename Scheme, typename Config>
void run_convergence(Scheme& scheme, const std::string& scheme_name, std::size_t starting_n_cells, double final_time) {
    using DGScheme = Scheme;
    const double domain_start = -1.0;
    const double domain_end = 1.0;
    double t = 0.0;
    double dt = 0.0;

    std::cout << "\n=== Testing scheme: " << scheme_name << " ===" << std::endl;
    std::size_t num_cells = starting_n_cells;
    std::size_t prev_num_cells = num_cells;
    double last_error = 0.0;

    auto get_poly_order = []() { return Config::order; };

    for (std::size_t i = 0; i < 10; ++i) {
        const double dx_conv = (domain_end - domain_start) / num_cells;
        Field<DGScheme> U_convergence(num_cells, domain_start, domain_end);
        initialize_field(U_convergence, domain_start, domain_end, 0.1);
        typename DGScheme::Workspace workspace_convergence(num_cells);
        const double cfl = 0.1;
        RKSSP<DGScheme, 4> rk(num_cells, cfl);
        t = 0.0;
        std::size_t steps = 0;
        while (t < final_time) {
            dt = rk.step(U_convergence, workspace_convergence, scheme, dx_conv, final_time - t);
            t += dt;
            ++steps;
        }
        // Output file with scheme name, order, cells, and final time
        std::string outname = "sol_" + scheme_name +
                              "_P" + std::to_string(get_poly_order()) +
                              "_N" + std::to_string(num_cells) +
                              "_T" + std::to_string(final_time) +
                              ".csv";
        write_solution_to_file(U_convergence, domain_start, dx_conv, outname,
                               steps, t, cfl, "RungeKutta4");

        auto exact_solution = [&final_time](double x) {
            Eigen::Vector<double, 3> result;
            result.setZero();
            auto state = physics::euler::isentropic_solution<double>(x, final_time, 0.1);
            result << state.density, state.momentum, state.total_energy;
            return result;
        };

        double current_error = compute_L2_error(U_convergence, exact_solution);

        if (i > 0) {
            double rate = std::log(last_error / current_error) / std::log(static_cast<double>(num_cells) / prev_num_cells);
            std::cout << "Convergence rate for " << prev_num_cells << "->" << num_cells << " cells: " << rate << "\tStep count: " << steps << std::endl;
        }

        last_error = current_error;
        prev_num_cells = num_cells;
        num_cells = static_cast<double>(num_cells) * 1.5;
        if (current_error < 1e-14) {
            std::cout << "Error is sufficiently small to stop analysis: " << current_error << std::endl;
            break;
        }
    }
    double dx_convergence = (domain_end - domain_start) / prev_num_cells;
    std::string exact_outname = "sol_exact_" + scheme_name +
                                "_P" + std::to_string(get_poly_order()) +
                                "_N" + std::to_string(prev_num_cells) +
                                "_T" + std::to_string(t) +
                                ".csv";
    write_exact_solution_to_file<DGScheme>(
        domain_start, dx_convergence, prev_num_cells, t,
        exact_outname,
        10 * (get_poly_order())
    );
    std::cout << "Order of polynomial basis: " << get_poly_order() << std::endl;
    std::cout << "=== Finished scheme: " << scheme_name << " ===\n" << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <number of starting cells> <final time>" << std::endl;
        return 1;
    }
    std::size_t starting_n_cells = std::stoi(argv[1]);
    const double final_time = std::stod(argv[2]);

    // Instantiate schemes
    DG<DGConfig> dg_scheme;
    DGSEM<DGSEMConfig> dgsem_scheme;
    DGSEM_ES<DGSEM_ESConfig> dgsem_es_scheme;

    // Run all schemes
    run_convergence<DG<DGConfig>, DGConfig>(dg_scheme, "DG", starting_n_cells, final_time);
    run_convergence<DGSEM<DGSEMConfig>, DGSEMConfig>(dgsem_scheme, "DGSEM", starting_n_cells, final_time);
    run_convergence<DGSEM_ES<DGSEM_ESConfig>, DGSEM_ESConfig>(dgsem_es_scheme, "DGSEM_ES", starting_n_cells, final_time);

    std::cout << "All convergence analyses completed." << std::endl;
    return 0;
}