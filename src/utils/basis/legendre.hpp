#pragma once
#include <cstddef>
#include <type_traits>
#include <cmath>

template <typename ScalarT, std::size_t MaxOrder>
struct Legendre
{
    static constexpr std::size_t order = MaxOrder;
    using Scalar = ScalarT;
    static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating-point type");

    template<std::size_t N>
    [[nodiscard]] static constexpr Scalar evaluate(Scalar x) noexcept
    {
        if constexpr (N == 0) return static_cast<Scalar>(1.0);
        if constexpr (N == 1) return x;
        return evaluate<N-1>(x) * x - static_cast<Scalar>(N-1) * evaluate<N-2>(x) / static_cast<Scalar>(N);
    }

    [[nodiscard]] static Scalar evaluate(std::size_t n, Scalar x) noexcept
    {
        if (n == 0) return static_cast<Scalar>(1.0);
        if (n == 1) return x;

        Scalar pm2 = static_cast<Scalar>(1.0);
        Scalar pm1 = x;
        Scalar p = static_cast<Scalar>(0.0);

        for (std::size_t k = 1; k < n; ++k)
        {
            const Scalar k_scalar = static_cast<Scalar>(k);
            p = ((static_cast<Scalar>(2.0) * k_scalar + static_cast<Scalar>(1.0)) * x * pm1 - k_scalar * pm2) / (k_scalar + static_cast<Scalar>(1.0));
            pm2 = pm1;
            pm1 = p;
        }

        return p;
    }

    [[nodiscard]] static Scalar orthonormal_evaluate(std::size_t n, Scalar x) noexcept
    {
        const Scalar norm_factor = std::sqrt((static_cast<Scalar>(2.0) * n + static_cast<Scalar>(1.0)) / static_cast<Scalar>(2.0));
        return evaluate(n, x) * norm_factor;
    }

    [[nodiscard]] static constexpr Scalar derivative(std::size_t n, Scalar x) noexcept
    {
        if (n == 0) return static_cast<Scalar>(0.0);

        constexpr Scalar tolerance = static_cast<Scalar>(1e-12);
        const Scalar n_scalar = static_cast<Scalar>(n);
        
        if (x >= static_cast<Scalar>(1.0) - tolerance)
            return n_scalar * (n_scalar + static_cast<Scalar>(1.0)) / static_cast<Scalar>(2.0);
        if (x <= static_cast<Scalar>(-1.0) + tolerance)
        {
            const Scalar sign = (n % 2 != 0) ? static_cast<Scalar>(1.0) : static_cast<Scalar>(-1.0);
            return sign * n_scalar * (n_scalar + static_cast<Scalar>(1.0)) / static_cast<Scalar>(2.0);
        }

        return n_scalar * (x * evaluate(n, x) - evaluate(n - 1, x)) / (x * x - static_cast<Scalar>(1.0));
    }

    [[nodiscard]] static constexpr Scalar orthonormal_derivative(std::size_t n, Scalar x) noexcept
    {
        const Scalar norm_factor = std::sqrt((static_cast<Scalar>(2.0) * n + static_cast<Scalar>(1.0)) / static_cast<Scalar>(2.0));
        return derivative(n, x) * norm_factor;
    }
};