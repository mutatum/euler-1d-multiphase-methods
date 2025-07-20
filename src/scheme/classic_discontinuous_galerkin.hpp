#pragma once
#include <Eigen/Dense>
#include "../field/field.hpp"
#include <stdexcept>
#include <algorithm>
#include <type_traits>
#include <iostream>
#include <omp.h>
#define EIGEN_USE_THREADS

template <class SchemePolicy>
class Field;

template <class PhysicsModel,
          class NumericalFlux,
          class PolynomialBasisClass,
          class QuadratureClass,
          template <class> class LeftBC,
          template <class> class RightBC>
class DG
{
public:
    using Physics = PhysicsModel;
    using State = typename PhysicsModel::State;
    using Scalar = typename State::Scalar;
    using PolynomialBasis = PolynomialBasisClass;
    using Quadrature = QuadratureClass;
    static constexpr std::size_t Variables = PhysicsModel::variables;
    static constexpr std::size_t PolynomialOrder = PolynomialBasis::order;

    struct Element
    {
        Eigen::Matrix<Scalar, PolynomialOrder + 1, Variables> coeffs;
        
        // Eigen::Vector<Scalar, Variables> evaluate(const Scalar xi) {
        //     assert(xi>=-1.0 && xi<=1.0);
        //     Eigen::Vector<Scalar, Variables> result;
        //     result.setZero();
        //     for (std::size_t j = 0; j <= PolynomialOrder; ++j)
        //     {
        //         result += element.coeffs.row(j) * PolynomialBasis::evaluate(j, xi);
        //     }
        //     return result;
        // }
    };

    struct Workspace
    {
        std::vector<State> interface_fluxes;
        std::vector<Eigen::Matrix<Scalar, 2, Variables>> F_star_matrices;
        Field<DG> projected_fluxes;
        std::vector<State> ULs, URs;

        explicit Workspace(std::size_t num_cells) : interface_fluxes(num_cells + 1),
                                                    projected_fluxes(num_cells),
                                                    F_star_matrices(num_cells),
                                                    ULs(num_cells + 1),
                                                    URs(num_cells + 1) {}
    };

    static constexpr Scalar CFL_multiplier = static_cast<Scalar>(1.0/(2.0 * PolynomialOrder + 1.0));

    DG() : S_(compute_S()), M_(compute_M()), B_(compute_B()), M_inv_(M_.inverse()) {}

private:
    const Eigen::Matrix<Scalar, PolynomialOrder + 1, PolynomialOrder + 1> S_, M_, M_inv_;
    const Eigen::Matrix<Scalar, PolynomialOrder + 1, 2> B_;

    // Stiffness Matrix
    static Eigen::Matrix<Scalar, PolynomialOrder + 1, PolynomialOrder + 1> compute_S()
    {
        Eigen::Matrix<Scalar, PolynomialOrder + 1, PolynomialOrder + 1> S;
        for (std::size_t j = 0; j < PolynomialOrder + 1; ++j)
        {
            for (std::size_t k = 0; k < PolynomialOrder + 1; ++k)
            {
                auto integrand = [&](Scalar x) -> Scalar
                { return PolynomialBasis::evaluate(j, x) * PolynomialBasis::derivative(k, x); };
                S(j, k) = Quadrature::integrate(integrand, static_cast<Scalar>(-1.0), static_cast<Scalar>(1.0));
            }
        }
        return S;
    }

    // Mass matrix
    static Eigen::Matrix<Scalar, PolynomialOrder + 1, PolynomialOrder + 1> compute_M()
    {
        Eigen::Matrix<Scalar, PolynomialOrder + 1, PolynomialOrder + 1> M;
        for (std::size_t j = 0; j < PolynomialOrder + 1; ++j)
        {
            for (std::size_t k = 0; k < PolynomialOrder + 1; ++k)
            {
                auto integrand = [&](Scalar x)
                {
                    return PolynomialBasis::evaluate(j, x) * PolynomialBasis::evaluate(k, x);
                };
                M(j, k) = Quadrature::integrate(integrand, static_cast<Scalar>(-1.0), static_cast<Scalar>(1.0));
            }
        }
        // Check if M is invertible
        if (M.determinant() == 0)
        {
            throw std::runtime_error("Mass matrix M is singular and cannot be inverted.");
        }
        return M;
    }

    // Flux extraction Matrix
    static Eigen::Matrix<Scalar, PolynomialOrder + 1, 2> compute_B()
    {
        Eigen::Matrix<Scalar, PolynomialOrder + 1, 2> B;
        for (std::size_t k = 0; k < PolynomialOrder + 1; ++k)
        {
            B(k, 0) = PolynomialBasis::evaluate(k, static_cast<Scalar>(-1.0));
            B(k, 1) = PolynomialBasis::evaluate(k, static_cast<Scalar>(1.0));
        }
        return B;
    }

public:
    [[nodiscard]] static State evaluate_element(const Element &element, Scalar xi) noexcept
    {
        State result{};
        for (std::size_t j = 0; j <= PolynomialOrder; ++j)
        {
            result += element.coeffs.row(j) * PolynomialBasis::evaluate(j, xi);
        }
        return result;
    }

    Scalar compute_residual(const Field<DG> &U, Field<DG> &R, Workspace &workspace) const
    {
        const std::size_t num_cells = U.size();
        if (num_cells == 0) {
            throw std::invalid_argument("Field cannot be empty");
        }

        // Compute interface fluxes and get max cell speed
        Scalar max_cell_speed = static_cast<Scalar>(0.0);
        

        // Left boundary
        workspace.ULs[0] = LeftBC<DG>::evaluate(U);
        workspace.URs[0] = evaluate_element(U(0), static_cast<Scalar>(-1.0));
        max_cell_speed = std::max(max_cell_speed, NumericalFlux::compute(workspace.ULs[0], workspace.URs[0], workspace.interface_fluxes[0]));
        
        // Interior interfaces
        #pragma omp parallel for reduction(max: max_cell_speed)
        for (std::size_t i = 1; i < num_cells; ++i)
        {
            workspace.ULs[i] = evaluate_element(U(i - 1), static_cast<Scalar>(1.0));
            workspace.URs[i] = evaluate_element(U(i), static_cast<Scalar>(-1.0));

            // Computing numerical flux
            max_cell_speed = std::max(max_cell_speed, 
                                    NumericalFlux::compute(workspace.ULs[i], workspace.URs[i], workspace.interface_fluxes[i]));
        }
        
        // Right boundary
        workspace.ULs[num_cells] = evaluate_element(U(num_cells - 1), static_cast<Scalar>(1.0));
        workspace.URs[num_cells] = RightBC<DG>::evaluate(U);
        max_cell_speed = std::max(max_cell_speed, NumericalFlux::compute(workspace.ULs[num_cells], workspace.URs[num_cells], workspace.interface_fluxes[num_cells]));

        #pragma omp parallel for
        // Project flux onto polynomial basis
        for (std::size_t i = 0; i < num_cells; ++i)
        {
            for (std::size_t j = 0; j <= PolynomialOrder; ++j)
            {
                auto integrand = [&](Scalar xi) -> State
                {
                    return Physics::physical_flux(evaluate_element(U(i), xi)) * PolynomialBasis::evaluate(j, xi);
                };
                const auto integrated_flux = Quadrature::integrate(integrand, static_cast<Scalar>(-1.0), static_cast<Scalar>(1.0));
                workspace.projected_fluxes(i).coeffs.row(j) << integrated_flux.density,
                    integrated_flux.momentum,
                    integrated_flux.total_energy;
            }
        }
        #pragma omp parallel for
        // Compute residuals
        for (std::size_t i = 0; i < num_cells; ++i)
        {
            const auto &flux_L = workspace.interface_fluxes[i];
            workspace.F_star_matrices[i].row(0) << flux_L.density,
                flux_L.momentum,
                flux_L.total_energy;
            
            const auto &flux_R = workspace.interface_fluxes[i + 1];
            workspace.F_star_matrices[i].row(1) << -flux_R.density,
                -flux_R.momentum,
                -flux_R.total_energy;

            R(i).coeffs = (2.0 / U.dx) * (M_inv_ * (S_.transpose() * M_inv_ * workspace.projected_fluxes(i).coeffs + B_ * workspace.F_star_matrices[i]));
        }

        return max_cell_speed;
    }
};