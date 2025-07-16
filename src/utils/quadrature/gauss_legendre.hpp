#pragma once
#include <stdexcept>
#include <cstddef>
#include <array>
#include <functional>
#include <type_traits>
#include "gl_quadrature_data.hpp" // Include the generated data

template<typename Scalar, std::size_t N>
struct GLQuadrature {
    static constexpr std::size_t order = N;
    static_assert(N >= 1 && N <= 15, "GLQuadrature is implemented for N between 1 and 15.");
    static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating-point type");
    
    static constexpr std::array<Scalar, N> nodes = GL::GLData<Scalar, N>::nodes;
    static constexpr std::array<Scalar, N> weights = GL::GLData<Scalar, N>::weights;

    template<typename F, typename Output = std::invoke_result_t<F, Scalar>>
    [[nodiscard]] static constexpr Output integrate(F&& f, Scalar a, Scalar b) {
        if (a >= b) {
            throw std::invalid_argument("Invalid integration bounds: a must be less than b");
        }
        
        const Scalar jacobian = (b - a) / static_cast<Scalar>(2.0);
        const Scalar offset = (b + a) / static_cast<Scalar>(2.0);

        Output integral_sum = weights[0] * f(offset + jacobian * nodes[0]);

        for (std::size_t i = 1; i < N; ++i) {
            const Scalar x = offset + jacobian * nodes[i];
            integral_sum += weights[i] * f(x);
        }

        return jacobian * integral_sum;
    }

    constexpr GLQuadrature() = default;
};
