#pragma once
#include <Eigen/Dense>
#include "DG_concepts.hpp"
#include "../field/field.hpp"
#include "../utils/basis/lagrange.hpp"
#include "../utils/quadrature/gauss_lobatto_legendre.hpp"
#include <stdexcept>
#include <algorithm>
#include <type_traits>
#include <iostream>
#include <omp.h>
// #define EIGEN_USE_THREADS

template <class SchemePolicy>
class Field;

// template <class PhysicsModel,
//           class EntropyStableFlux,
//           class NumericalFlux,
//           std::size_t Order,
//           template <class> class LeftBC,
//           template <class> class RightBC>
template <DGSEM_ESConfigConcept Config>
class DGSEM_ES
{
public:
    using Physics = Config::Physics;
    using State = typename Physics::State;
    using Scalar = typename State::Scalar;
    static constexpr std::size_t PolynomialOrder = Config::order;
    using Quadrature = GLLQuadrature<Scalar, PolynomialOrder + 1>;
    using PolynomialBasis = Lagrange<Quadrature::nodes>;
    static constexpr std::size_t Variables = Physics::variables;
    using EntropyStableFlux = Config::EntropyFlux;
    using NumericalFlux = Config::NumericalFlux;
    using LeftBC = typename Config::template LeftBC<DGSEM_ES>;
    using RightBC = typename Config::template RightBC<DGSEM_ES>;

    struct Element
    {
        Eigen::Matrix<Scalar, PolynomialOrder + 1, Variables> coeffs;
    };

    struct Workspace
    {
        std::vector<State> interface_fluxes;
        std::vector<Eigen::Matrix<Scalar, 2, Variables>> F;
        std::vector<State> ULs, URs;

        explicit Workspace(std::size_t num_cells) : interface_fluxes(num_cells + 1),
                                                    F(num_cells),
                                                    ULs(num_cells + 1),
                                                    URs(num_cells + 1) {}
    };

    static constexpr Scalar CFL_multiplier = static_cast<Scalar>(1.0/(2.0 * PolynomialOrder + 1.0));

    DGSEM_ES() : D_(compute_D()) {}

private:
    const Eigen::Matrix<Scalar, PolynomialOrder + 1, PolynomialOrder + 1> D_;

    // Differentiation Matrix
    static Eigen::Matrix<Scalar, PolynomialOrder + 1, PolynomialOrder + 1> compute_D()
    {
        Eigen::Matrix<Scalar, PolynomialOrder + 1, PolynomialOrder + 1> D;
        for (std::size_t j = 0; j < PolynomialOrder + 1; ++j)
        {
            for (std::size_t k = 0; k < PolynomialOrder + 1; ++k)
            {
                D(j,k) = PolynomialBasis::derivative(k, Quadrature::nodes[j]);
            }
        }
        return D;
    }

public:
    [[nodiscard]] static State evaluate_element(const Element &element, Scalar xi) noexcept
    {
        State result{};
        for (std::size_t j = 0; j < PolynomialOrder + 1; ++j)
        {
            result += element.coeffs.row(j) * PolynomialBasis::evaluate(j, xi);
        }
        return result;
    }

    Scalar compute_residual(const Field<DGSEM_ES> &U, Field<DGSEM_ES> &R, Workspace &workspace) const
    {
        const std::size_t num_cells = U.size();

        // Compute interface fluxes and get max cell speed
        Scalar max_cell_speed = static_cast<Scalar>(0.0);
        

        // Left boundary
        workspace.ULs[0] = LeftBC::evaluate(U);
        workspace.URs[0] = U(0).coeffs.row(0);
        max_cell_speed = std::max(max_cell_speed, NumericalFlux::compute(workspace.ULs[0], workspace.URs[0], workspace.interface_fluxes[0]));
        
        // Interior interfaces
        #pragma omp parallel for reduction(max: max_cell_speed)
        for (std::size_t i = 1; i < num_cells; ++i)
        {
            workspace.ULs[i] = U(i - 1).coeffs.row(PolynomialOrder);
            workspace.URs[i] = U(i).coeffs.row(0);

            // Computing numerical flux
            max_cell_speed = std::max(max_cell_speed, 
                                    NumericalFlux::compute(workspace.ULs[i], workspace.URs[i], workspace.interface_fluxes[i]));
        }
        
        // Right boundary
        workspace.ULs[num_cells] = U(num_cells - 1).coeffs.row(PolynomialOrder);
        workspace.URs[num_cells] = RightBC::evaluate(U);
        max_cell_speed = std::max(max_cell_speed, NumericalFlux::compute(workspace.ULs[num_cells], workspace.URs[num_cells], workspace.interface_fluxes[num_cells]));

        #pragma omp parallel for
        for (std::size_t i = 0; i < num_cells; ++i)
        {
            R(i).coeffs.setZero();
            for (std::size_t j = 0; j < PolynomialOrder + 1; ++j)
            {
                for (std::size_t k = 0; k < PolynomialOrder + 1; ++k) {
                    State state_j{U(i).coeffs.row(j)},
                          state_k{U(i).coeffs.row(k)},
                          flux;
                    EntropyStableFlux::compute(
                        state_j, state_k,flux
                    );
                    R(i).coeffs.row(j) -= static_cast<Scalar>(2.0) * D_(j, k) * flux.to_vector();
                }
            }
            workspace.F[i].row(0) = Physics::physical_flux(State{U(i).coeffs.row(0)}).to_vector();
            workspace.F[i].row(1) = Physics::physical_flux(State{U(i).coeffs.row(PolynomialOrder)}).to_vector();
            R(i).coeffs.row(0) -= (workspace.F[i].row(0) - workspace.interface_fluxes[i].to_vector()) / Quadrature::weights[0];
            R(i).coeffs.row(PolynomialOrder) += (workspace.F[i].row(1) - workspace.interface_fluxes[i+1].to_vector()) / Quadrature::weights[PolynomialOrder];
            R(i).coeffs *= (2.0 / U.dx);
        }

        return max_cell_speed;
    }
};