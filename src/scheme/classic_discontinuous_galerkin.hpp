#pragma once
#include <Eigen/Dense>
#include "numerical_scheme.hpp"

template <class PhysicsModel, class PolynomialBasis>
class DG : public NumericalScheme<typename PhysicsModel::State>
{
private:
    using State = typename PhysicsModel::State;
    using PolynomialOrder = PolynomialBasis::order;
    Eigen::Matrix<typename State::scalar_type, PolynomialOrder+1, PolynomialOrder+1> S_, M_, M_inv_;

    // Stiffness Matrix
    void compute_S() {

    }

    void compute_M() {
        
    }

public:
    void compute_residual(const Field<State> &solution_field,
                          Field<State> &R) const override
    {
        ;
    }
};