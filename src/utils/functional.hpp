#pragma once
#include <cstddef>
#include "quadrature/gauss_legendre.hpp"
#include <Eigen/Dense>

template <class Quadrature, class PolynomialBasis, class Callable>
auto compute_L2_projection_onto_basis(const Callable &f)
{
    using Scalar = typename PolynomialBasis::Scalar;
    constexpr std::size_t N = PolynomialBasis::order + 1;

    using Output = std::invoke_result_t<Callable, Scalar>;
    constexpr int variables = Output::RowsAtCompileTime;

    Eigen::Matrix<Scalar, N, N> M;
    M.setZero();
    Eigen::Matrix<Scalar, N, variables> b;
    b.setZero();
    for (std::size_t j = 0; j < N; ++j)
    {
        for (std::size_t k = 0; k < N; ++k)
        {
            auto integrand = [&](Scalar xi)
            {
                return PolynomialBasis::evaluate(j, xi) * PolynomialBasis::evaluate(k, xi);
            };
            M(j, k) = Quadrature::integrate(integrand, static_cast<Scalar>(-1.0), static_cast<Scalar>(1.0));
        }
    }
    // Check if M is invertible
    if (M.determinant() == 0)
    {
        throw std::runtime_error("Mass matrix M is singular and cannot be inverted.");
    }

    for (std::size_t i = 0; i < N; ++i)
    {
        b.row(i) = Quadrature::integrate([&](Scalar xi) -> Output
                                                    { return f(xi) * PolynomialBasis::evaluate(i, xi); }, Scalar(-1.0), Scalar(1.0))
                                  .transpose();
    }
    Eigen::Matrix<Scalar, N, variables> coefficients = M.colPivHouseholderQr().solve(b);
    return coefficients;
}

template <class Quadrature, class Callable> // Assuming Callable is a f(Scalar)->Eigen::Vector even if vectorsize=1
auto compute_L2_norm(const Callable &f, typename Quadrature::Scalar a, typename Quadrature::Scalar b)
{
    assert(a < b);
    using Scalar = typename Quadrature::Scalar;
    using Output = std::invoke_result_t<Callable, Scalar>;
    constexpr int variables = Output::RowsAtCompileTime;
    Eigen::Matrix<Scalar, 1, variables> norm_values;

    return std::sqrt(Quadrature::integrate([&](Scalar x) -> Scalar
                                           { return f(x).array().square().sum(); }, a, b));
}

template <class Quadrature, class Callable1, class Callable2>
auto compute_L2_error(const Callable1 &f, const Callable2 &g, typename Quadrature::Scalar a, typename Quadrature::Scalar b)
{
    assert(a < b);
    using Scalar = typename Quadrature::Scalar;
    using Output = std::invoke_result_t<Callable1, Scalar>;
    auto difference = [&](Scalar x) -> Output
    { return f(x) - g(x); };
    return std::pow(compute_L2_norm<Quadrature>(difference, a, b), 2);
}

template <class Field, class Callable>
auto compute_L2_error(const Field &field, const Callable &exact_solution)
{
    using Scalar = typename Field::Scalar;
    using Scheme = typename Field::Scheme;
    using ErrorQuadrature = GLQuadrature<Scalar, 14>;

    Scalar total_error_sq = Scalar(0.0);
    Scalar a = field.domain_start;

    for (const auto &element : field)
    {
        const Scalar b = a + field.dx;
        auto field_evaluator = [&](Scalar x)
        {
            return field.evaluate(x);
        };
        total_error_sq += compute_L2_error<ErrorQuadrature>(field_evaluator, exact_solution, a, b);
        a = b;
    }

    return std::sqrt(total_error_sq);
}