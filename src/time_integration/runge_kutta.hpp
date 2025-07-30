#pragma once
#include "../field/field.hpp"
#include <stdexcept>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include "../scheme/DG_concepts.hpp"

template <class SchemePolicy>
class Field;

template <SchemeConcept Scheme, std::size_t Order>
class RKSSP {
public:
    using State = typename Scheme::State;
    using Scalar = typename Scheme::Scalar;
    using Workspace = typename Scheme::Workspace;

    Scalar CFL;

    // Storage for intermediate stages
    Field<Scheme> k1_, k2_, k3_, k4_, R_, U_temp_;

    explicit RKSSP(std::size_t num_cells, Scalar CFL = 0.2)
        : k1_(num_cells), k2_(num_cells), k3_(num_cells), k4_(num_cells),
          R_(num_cells), U_temp_(num_cells), CFL(CFL)
    {
        if (num_cells == 0)
            throw std::invalid_argument("Number of cells must be positive");
    }

    Scalar step(Field<Scheme> &Un, Workspace &workspace, Scheme &scheme, Scalar dx, Scalar dt_max)
    {
        if constexpr (Order == 1) {
            Scalar max_speed = scheme.compute_residual(Un, R_, workspace);
            Scalar dt_cfl = CFL * Scheme::CFL_multiplier * dx / max_speed;
            Scalar dt = std::min(dt_cfl, dt_max);
            Un += dt * R_;
            if (max_speed == Scalar(0.0)) return dt_max;
            return dt;
        }
        else if constexpr (Order == 3) {
            Scalar max_speed = scheme.compute_residual(Un, k1_, workspace);
            Scalar dt_cfl = CFL * Scheme::CFL_multiplier * dx / max_speed;
            Scalar dt = std::min(dt_cfl, dt_max);
            if (max_speed == Scalar(0.0)) return dt_max;
            U_temp_ = Un + dt * k1_;
            scheme.compute_residual(U_temp_, k2_, workspace);
            U_temp_ = 0.75 * Un + 0.25 * U_temp_ + 0.25 * dt * k2_;
            scheme.compute_residual(U_temp_, k3_, workspace);
            Un = (1.0/3.0) * Un + (2.0/3.0) * U_temp_ + (2.0/3.0) * dt * k3_;
            return dt;
        }
        else if constexpr (Order == 4) {
            Scalar max_speed = scheme.compute_residual(Un, k1_, workspace);
            Scalar dt_cfl = CFL * Scheme::CFL_multiplier * dx / max_speed;
            Scalar dt = std::min(dt_cfl, dt_max);
            if (max_speed == Scalar(0.0)) return dt_max;
            U_temp_ = Un + (dt * Scalar(0.5)) * k1_;
            scheme.compute_residual(U_temp_, k2_, workspace);
            U_temp_ = Un + (dt * Scalar(0.5)) * k2_;
            scheme.compute_residual(U_temp_, k3_, workspace);
            U_temp_ = Un + dt * k3_;
            scheme.compute_residual(U_temp_, k4_, workspace);
            Un += (dt / Scalar(6.0)) * (k1_ + Scalar(2.0) * k2_ + Scalar(2.0) * k3_ + k4_);
            return dt;
        }
        else {
            static_assert(Order == 1 || Order == 3 || Order == 4, "Unsupported RKSSP order");
        }
    }
};