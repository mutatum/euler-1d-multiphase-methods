#pragma once
#include <concepts>
#include <vector>
#include <ranges>

template <typename T>
concept QuadratureConcept = requires(std::size_t n, typename T::Scalar a, typename T::Scalar b) {
    typename T::Scalar;
    { T::nodes } -> std::ranges::range;
    { T::weights} -> std::ranges::range;
    { T::integrate([](typename T::Scalar x) { return x; }, a, b) };
};