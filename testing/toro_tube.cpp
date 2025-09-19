#include "../src/scheme/DG.hpp"
#include "../src/scheme/DGSEM.hpp"
#include "../src/scheme/DGSEM_entropy_stable.hpp"
#include "../src/physics/Euler/euler.hpp"
#include "../src/physics/numerical_flux/rusanov.hpp"
#include "../src/physics/Euler/entropy_flux/ismail_roe.hpp"
#include "../src/physics/Euler/entropy_flux/chandrashekar.hpp"
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
        field(i).coeffs.setZero();
        // project initial condition onto polynomial basis
        // using the quadrature nodes of the polynomial basis
        // Sod's shock tube initial condition
        auto f = [&](double xi) -> Eigen::Vector<double, Scheme::Variables>
        {
            Eigen::Vector<double, Scheme::Variables> result;
            auto physical_x = [&](double xi) -> double
            {
                return x_left + (xi + 1.0) * dx / 2.0; // Map xi ∈ [-1, 1] to physical_x ∈ [x_left, x_left + dx]
            };
            if (physical_x(xi) < 0.0)
            {
                result << 1.0, 0.0, 1.0 / (Scheme::Physics::EOS::gamma - 1);
            }
            else
            {
                result << 0.125, 0.0, 0.1 / (Scheme::Physics::EOS::gamma - 1);
            }
            return result;
        };
        // for (std::size_t j = 0; j < Scheme::PolynomialOrder + 1; ++j)
        // {
        //     const double xi = Scheme::Quadrature::nodes[j]; // Local coordinate in the cell
        //     // Correct mapping: xi ∈ [-1,1] -> physical_x ∈ [x_left, x_left + dx]
        //     const double physical_x = x_left + (xi + 1.0) * dx / 2.0;
        //     std::cout << "Physical x: " << physical_x << std::endl;
        //     // const auto result = physics::euler::isentropic_solution<double>(physical_x, 0.0, factor);
        //     // Sod's shock tube
        //     if (physical_x < 0.0)
        //     {
        //         field(i).coeffs.row(j) << 1.0, 0.0, 1.0 / (Scheme::Physics::EOS::gamma - 1);
        //     } else {
        //         field(i).coeffs.row(j) << 0.125, 0.0, 0.1 / (Scheme::Physics::EOS::gamma - 1);
        //     }
        // };
        field(i).coeffs = compute_L2_projection_onto_basis<typename Scheme::Quadrature, typename Scheme::PolynomialBasis>(f);
        std::cout << "Cell " << i << " initialized with coefficients: " << field(i).coeffs << std::endl;
    }
}

using namespace physics::euler;
struct DGSEM_ESConfig
{
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using EntropyFlux = Chandrashekar<Physics>;
    using NumericalFlux = Rusanov<Physics>;
    static constexpr std::size_t order = 5;
    template <class Scheme>
    using LeftBC = BoundaryConditions::LeftSolidWallBC<Scheme>;
    template <class Scheme>
    using RightBC = BoundaryConditions::RightSolidWallBC<Scheme>;
};

using namespace physics::euler;
struct DGSEMConfig
{
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    static constexpr std::size_t order = 5;
    template <class Scheme>
    using LeftBC = BoundaryConditions::LeftSolidWallBC<Scheme>;
    template <class Scheme>
    using RightBC = BoundaryConditions::RightSolidWallBC<Scheme>;
};

struct DGConfig
{
    using Physics = EulerPhysics<double, IdealGasEOS<double, 1.4>>;
    using NumericalFlux = Rusanov<Physics>;
    static constexpr std::size_t order = 1;
    using Quadrature = GLQuadrature<double, order + 9>;
    using PolynomialBasis = Lagrange<GLLQuadrature<double, order+1>::nodes>;
    // using PolynomialBasis = Legendre<double, order>;
    template <class Scheme>
    using LeftBC = BoundaryConditions::LeftCopyBC<Scheme>;
    template <class Scheme>
    using RightBC = BoundaryConditions::RightCopyBC<Scheme>;
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

    // using DGScheme = DGSEM_ES<DGSEM_ESConfig>;
    using DGScheme = DG<DGConfig>;

    DGScheme scheme;
    const double domain_start = -1.0;
    const double domain_end = 1.0;
    double t = 0.0;
    double dt = 0.0;

    // ------------------- Convergence Analysis -------------------
    std::cout << "Starting convergence analysis..." << std::endl;
    std::size_t num_cells = starting_n_cells;
    const double rate = 1.5;
    double last_error = 0.0;

    std::cout << "Final time: " << final_time << std::endl;
    for (std::size_t i = 0; i < 1; ++i)
    {
        const double dx_conv = (domain_end - domain_start) / num_cells;
        Field<DGScheme> U_convergence(num_cells, domain_start, domain_end);
        initialize_field(U_convergence, domain_start, domain_end, 0.1);
        DGScheme::Workspace workspace_convergence(num_cells);
        const double cfl = 0.1;
        RKSSP<DGScheme, 4> rkssp(num_cells, cfl);
        t = 0.0;

        std::size_t steps = 0;
        while (t < final_time)
        {
            std::cout << "Time: " << t << ", Cells: " << num_cells << ", Steps: " << steps << std::endl;
            dt = rkssp.step(U_convergence, workspace_convergence, scheme, dx_conv, final_time - t);
            if (steps%10==0)
            {
                std::cout << "Time: " << t << ", Cells: " << num_cells << ", Steps: " << steps << ", dt: " << dt << std::endl;
                write_solution_to_file(U_convergence, domain_start, dx_conv, "solution_final" + std::to_string(i) + "_order" + std::to_string(DGScheme::PolynomialOrder) + "_t=" + std::to_string(t) + ".csv", steps, t, cfl, "RungeKutta4");
                // write_solution_to_file(U_convergence, domain_start, dx_conv, "solution_final" + std::to_string(i) + "_order" + std::to_string(DGScheme::PolynomialOrder) + ".csv", steps, t, cfl, "RungeKutta4");
            }
            t += dt;
            ++steps;
        }
        write_solution_to_file(U_convergence, domain_start, dx_conv, "solution_final" + std::to_string(i) + "_order" + std::to_string(DGScheme::PolynomialOrder) + "_t=" + std::to_string(t) + ".csv", steps, t, cfl, "RungeKutta4");
        // Without time stamp in filename
        // write_solution_to_file(U_convergence, domain_start, dx_conv, "solution_final" + std::to_string(i) + "_order" + std::to_string(DGScheme::PolynomialOrder) + ".csv", steps, t, cfl, "RungeKutta4");
        std::cout << "Finished simulation for " << num_cells << " cells at t = " << t << " after " << steps << " steps." << std::endl;
        num_cells = static_cast<double>(num_cells) * 1.5;
    }

    std::cout << "Order of polynomial basis: " << DGScheme::PolynomialOrder << std::endl;

    std::cout << "Convergence analysis completed." << std::endl;


    return 0;
}