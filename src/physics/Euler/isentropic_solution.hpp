#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <functional>
#include <stdexcept>
#include <limits>
#include "euler.hpp"

namespace physics {
namespace euler {

/**
 * @brief A generic Newton-Raphson solver for finding roots of a function.
 * 
 * @tparam Scalar The floating-point type (e.g., double, float).
 * @tparam Func The type of the function f(x).
 * @tparam DFunc The type of the derivative function df(x)/dx.
 * @param x0 The initial guess for the root.
 * @param f The function for which to find the root.
 * @param df The derivative of the function f.
 * @param eps The desired tolerance for convergence.
 * @param max_it The maximum number of iterations.
 * @return The root of the function.
 * @throws std::runtime_error if the derivative is close to zero or if the method fails to converge.
 */
template <typename Scalar, typename Func, typename DFunc>
Scalar newton_raphson(Scalar x0, Func&& f, DFunc&& df, 
                      Scalar eps = std::numeric_limits<Scalar>::epsilon(), int max_it = 100000) {
    Scalar x = x0;
    for (int i = 0; i < max_it; ++i) {
        Scalar fx = f(x);
        if (std::abs(fx) < eps) {
            return x;
        }
        Scalar dfx = df(x);
        if (std::abs(dfx) < std::numeric_limits<Scalar>::epsilon()) {
            throw std::runtime_error("Newton-Raphson derivative is zero.");
        }
        x = x - fx / dfx;
    }
    return 0.0;
    // throw std::runtime_error("Newton-Raphson failed to converge.");
}

/**
 * @brief Computes the exact isentropic Euler solution at a given point (x, t).
 * 
 * This solution corresponds to the interaction of two Riemann invariants and is
 * periodic on the domain [-1, 1].
 * 
 * @tparam Scalar The floating-point type (e.g., double, float).
 * @param x The spatial coordinate.
 * @param t The time.
 * @param factor A factor controlling the amplitude of the initial sine wave.
 * @return An EulerState object containing the density, momentum, and total energy.
 */
template <typename Scalar>
EulerState<Scalar> isentropic_solution(Scalar x, Scalar t, Scalar factor = 0.1) {
    const Scalar PI = static_cast<Scalar>(M_PI);
    const Scalar SQRT3 = std::sqrt(static_cast<Scalar>(3.0));

    // Define the functions and their derivatives for the Newton-Raphson solver
    auto fp = [&](Scalar X) { return X + SQRT3 * t * (1 + factor * std::sin(PI * X)) - x; };
    auto dfp = [&](Scalar X) { return 1 + SQRT3 * t * PI * factor * std::cos(PI * X); };
    auto fm = [&](Scalar X) { return X - SQRT3 * t * (1 + factor * std::sin(PI * X)) - x; };
    auto dfm = [&](Scalar X) { return 1 - SQRT3 * t * PI * factor * std::cos(PI * X); };

    // Find the characteristic variables Xp and Xm using Newton-Raphson
    Scalar Xp = newton_raphson(x, fp, dfp);
    Xp = Xp - 2 * std::floor((Xp + 1) / 2); // Map to [-1, 1)

    Scalar Xm = newton_raphson(x, fm, dfm);
    Xm = Xm - 2 * std::floor((Xm + 1) / 2); // Map to [-1, 1)

    // Calculate Riemann invariants
    const Scalar wp = SQRT3 * (1 + factor * std::sin(PI * Xp));
    const Scalar wm = -SQRT3 * (1 + factor * std::sin(PI * Xm));

    // Reconstruct primitive variables
    const Scalar u = (wp + wm) * static_cast<Scalar>(0.5);
    const Scalar rho = (wp - wm) / (static_cast<Scalar>(2.0) * SQRT3);

    const Scalar E = rho * (static_cast<Scalar>(0.5) * u * u + static_cast<Scalar>(0.5) * rho * rho);

    return EulerState<Scalar>{rho, rho * u, E};
}

} // namespace euler
} // namespace physics