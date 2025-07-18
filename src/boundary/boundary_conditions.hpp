#pragma once
#include "../field/field.hpp"
#include <type_traits>

enum class Side { Left, Right };

template<Side side>
struct PeriodicBC
{
    template <class Scheme>
    static typename Scheme::State evaluate(const Field<Scheme>& U)
    {
        if constexpr (side == Side::Left)
            return Scheme::evaluate_poly(U(U.size() - 1), static_cast<typename Scheme::Scalar>(1.0));
        else
            return Scheme::evaluate_poly(U(0), static_cast<typename Scheme::Scalar>(-1.0));
    }
};

template<Side side>
struct SolidWallBC
{
    template <class Scheme>
    static typename Scheme::State evaluate(const Field<Scheme>& U)
    {
        using State = typename Scheme::State;
        using Scalar = typename Scheme::Scalar;
        
        State adjacent_state;
        if constexpr (side == Side::Left)
            adjacent_state = Scheme::evaluate_poly(U(0), static_cast<Scalar>(1.0));
        else
            adjacent_state = Scheme::evaluate_poly(U(U.size() - 1), static_cast<Scalar>(1.0));

        return State{
            adjacent_state.density,
            -adjacent_state.momentum,
            adjacent_state.total_energy};
    }
};