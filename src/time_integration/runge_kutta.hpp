#pragma once
#include "../field/field.hpp"
#include <stdexcept>
#include <type_traits>
#include <cmath>

template <class SchemePolicy>
class Field;

template <typename Scheme>
class RungeKutta1
{
public:
    using State = typename Scheme::State;
    using Scalar = typename Scheme::Scalar;
    using Workspace = typename Scheme::Workspace;

    static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating-point type");

private:
    Field<Scheme> R_;
    Field<Scheme> U_temp_;

public:
    Scalar CFL;
    
    explicit RungeKutta1(std::size_t num_cells, const Scalar CFL = 0.2)
        : R_(num_cells), U_temp_(num_cells), CFL(CFL)
    {
        if (num_cells == 0)
        {
            throw std::invalid_argument("Number of cells must be positive");
        }
    }

    // Returns time step
    Scalar step(Field<Scheme> &Un, Workspace &workspace, Scheme &scheme, Scalar dx, Scalar dt_max)
    {
        Scalar max_speed = scheme.compute_residual(Un, R_, workspace);

        const Scalar dt_cfl = CFL * Scheme::CFL_multiplier * dx / max_speed;
        const Scalar dt = std::min(dt_cfl, dt_max);


        Un += dt * R_;

        if (max_speed == static_cast<Scalar>(0.0))
        {
            return dt_max;
        }

        return dt;
    }
};
 
template <typename Scheme>
class RungeKutta3
{
public:
    using State = typename Scheme::State;
    using Scalar = typename Scheme::Scalar;
    using Workspace = typename Scheme::Workspace;

    static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating-point type");

private:
    Field<Scheme> k1_, k2_, k3_;
    Field<Scheme> R_;
    Field<Scheme> U_temp_;

public:
    Scalar CFL;

    explicit RungeKutta3(std::size_t num_cells, const Scalar CFL = 0.2)
        : k1_(num_cells), k2_(num_cells), k3_(num_cells),
          R_(num_cells), U_temp_(num_cells), CFL(CFL)
    {
        if (num_cells == 0)
        {
            throw std::invalid_argument("Number of cells must be positive");
        }
    }

    // Returns time step
    Scalar step(Field<Scheme> &Un, Workspace &workspace, Scheme &scheme, Scalar dx, Scalar dt_max)
    {
        Scalar max_speed = scheme.compute_residual(Un, k1_, workspace);

        const Scalar dt_cfl = CFL * Scheme::CFL_multiplier * dx / max_speed;
        // const Scalar dt = std::min(dt_cfl, dt_max);
        const Scalar dt = std::min({std::pow(dx,4.0/3.0),dt_cfl, dt_max});

        if (max_speed == static_cast<Scalar>(0.0))
        {
            return dt_max;
        }

        U_temp_ = Un + dt * k1_;
        scheme.compute_residual(U_temp_, k2_, workspace);

        U_temp_ = 0.75 * Un + 0.25 * U_temp_ + dt * k2_;
        scheme.compute_residual(U_temp_, k3_, workspace);

        Un = (1.0/3.0) * Un + (2.0/3.0) * (U_temp_ + (dt / 3.0) * k3_);

        return dt;
    }
};

template <typename Scheme>
class RungeKutta4
{
public:
    using State = typename Scheme::State;
    using Scalar = typename Scheme::Scalar;
    using Workspace = typename Scheme::Workspace;

    static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating-point type");

private:
    Field<Scheme> k1_, k2_, k3_, k4_;
    Field<Scheme> R_;
    Field<Scheme> U_temp_;

public:
    Scalar CFL;
    
    explicit RungeKutta4(std::size_t num_cells, const Scalar CFL = 0.2)
        : k1_(num_cells), k2_(num_cells), k3_(num_cells), k4_(num_cells),
          R_(num_cells), U_temp_(num_cells), CFL(CFL)
    {
        if (num_cells == 0)
        {
            throw std::invalid_argument("Number of cells must be positive");
        }
    }

    // Returns time step
    Scalar step(Field<Scheme> &Un, Workspace &workspace, Scheme &scheme, Scalar dx, Scalar dt_max, bool testing=false, std::size_t order=4)
    {
        Scalar max_speed = scheme.compute_residual(Un, k1_, workspace);

        const Scalar dt_cfl = CFL * Scheme::CFL_multiplier * dx / max_speed;
        const Scalar dt = (testing) ? std::min({std::pow(dx,static_cast<Scalar>(order)/4.0),dt_cfl, dt_max}) : std::min(dt_cfl, dt_max);

        if (max_speed == static_cast<Scalar>(0.0))
        {
            return dt_max;
        }

        U_temp_ = Un + (dt * static_cast<Scalar>(0.5)) * k1_;
        scheme.compute_residual(U_temp_, k2_, workspace);

        U_temp_ = Un + (dt * static_cast<Scalar>(0.5)) * k2_;
        scheme.compute_residual(U_temp_, k3_, workspace);

        U_temp_ = Un + dt * k3_;
        scheme.compute_residual(U_temp_, k4_, workspace);

        Un += (dt / static_cast<Scalar>(6.0)) * (k1_ + static_cast<Scalar>(2.0) * k2_ + static_cast<Scalar>(2.0) * k3_ + k4_);

        return dt;
    }
};