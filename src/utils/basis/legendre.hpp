#pragma once
#include <cstddef>

template <typename Scalar, std::size_t MaxOrder>
struct Legendre
{
    static constexpr std::size_t max_order = MaxOrder;

    static constexpr Scalar evaluate(std::size_t n, Scalar x)
    {

        if (n == 0)
            return 1.0;
        if (n == 1)
            return x;

        Scalar pm2 = 1.0;
        Scalar pm1 = x;
        Scalar p = 0.0;

        for (std::size_t k = 1; k < n; ++k)
        {
            p = ((2.0 * k + 1.0) * x * pm1 - k * pm2) / (k + 1.0);
            pm2 = pm1;
            pm1 = p;
        }

        return p;
    }

    static constexpr Scalar derivative(std::size_t n, Scalar x)
    {
        if (n == 0)
            return 0.0;

        constexpr Scalar tolerance = 1e-12;
        if (x >= 1.0 - tolerance)
            return n * (n + 1.0) / 2.0;
        if (x <= -1.0 + tolerance)
        {
            const Scalar sign = (n % 2 == 0) ? -1.0 : 1.0;
            return sign * n * (n + 1.0) / 2.0;
        }

        return n * (x * evaluate(n, x) - evaluate(n - 1, x)) / (x * x - 1.0);
    }
};