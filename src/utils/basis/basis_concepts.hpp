#pragma once
#include <concepts>
 
template <typename T>
concept PolynomialBasisConcept = requires(std::size_t i, typename T::Scalar xi) {
    typename T::Scalar;
    { T::order } -> std::convertible_to<std::size_t>;
    { T::evaluate(i, xi) } -> std::convertible_to<typename T::Scalar>;
    { T::derivative(i, xi) } -> std::convertible_to<typename T::Scalar>;
};