#pragma once
#include "../field/field.hpp"
#include <stdexcept>
#include <type_traits>

template <class SchemePolicy>
class Field;

template <typename Scheme>
class RungeKutta4
{
private:
    Field<Scheme> k1_, k2_, k3_, k4_;
    Field<Scheme> R_;
    Field<Scheme> U_temp_;

public:
    using State = typename Scheme::State;
    using Scalar = typename Scheme::Scalar;
    using Workspace = typename Scheme::Workspace;
    
    static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating-point type");
    
    explicit RungeKutta4(std::size_t num_cells) 
        : k1_(num_cells), k2_(num_cells), k3_(num_cells), k4_(num_cells),
          R_(num_cells), U_temp_(num_cells) 
    {
        if (num_cells == 0) {
            throw std::invalid_argument("Number of cells must be positive");
        }
    }
    
    void step(Field<Scheme>& Un, Workspace& workspace, Scalar dt, const Scheme& scheme)
    {
        if (dt <= 0) {
            throw std::invalid_argument("Time step must be positive");
        }
        
        // k1 = f(Un)
        scheme.compute_residual(Un, k1_, workspace);

        // k2 = f(Un + dt/2 * k1)
        U_temp_ = Un + (dt * static_cast<Scalar>(0.5)) * k1_;
        scheme.compute_residual(U_temp_, k2_, workspace);

        // k3 = f(Un + dt/2 * k2)
        U_temp_ = Un + (dt * static_cast<Scalar>(0.5)) * k2_;
        scheme.compute_residual(U_temp_, k3_, workspace);
        
        // k4 = f(Un + dt * k3)
        U_temp_ = Un + dt * k3_;
        scheme.compute_residual(U_temp_, k4_, workspace);
        
        // Un+1 = Un + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        Un += (dt / static_cast<Scalar>(6.0)) * (k1_ + static_cast<Scalar>(2.0) * k2_ + static_cast<Scalar>(2.0) * k3_ + k4_);
    }
};