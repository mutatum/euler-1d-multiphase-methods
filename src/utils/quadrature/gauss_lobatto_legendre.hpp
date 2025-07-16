#pragma once
#include <stdexcept>
#include <cstddef>
#include <array>
#include "gll_quadrature_data.hpp" // AI filled by coefsandweights.py script

template<typename Scalar, std::size_t N>
struct GLLQuadrature {
    static constexpr std::size_t order = N;
    static_assert(N >= 1 && N <= 15, "GLQuadrature is implemented for N between 1 and 15.");
    static constexpr std::array<Scalar, N> nodes = GLL::GLLData<Scalar, N>::nodes;
    static constexpr std::array<Scalar, N> weights = GLL::GLLData<Scalar, N>::weights;
    
    template<typename F, typename Output = decltype(std::declval<F>()(std::declval<Scalar>()))>
    static constexpr Output integrate(F&& f, Scalar a, Scalar b) {
        
        const Scalar jacobian = (b-a)/2.0;
        const Scalar offset   = (b+a)/2.0;

        Output integral_sum = weights[0] * f(offset + jacobian * nodes[0]);
        for (std::size_t i = 1; i< N; ++i) {
            const Scalar x = offset + jacobian * nodes[i]; // nodes[i] in [-1,1]
            integral_sum += weights[i] * f(x);
        }

        return jacobian * integral_sum;
    }

    constexpr GLLQuadrature() = default;
};