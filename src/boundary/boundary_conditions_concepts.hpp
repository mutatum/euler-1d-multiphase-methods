#pragma once
#include <concepts>

template<class Scheme>
class Field;

template<typename BC, typename Scheme>
concept BoundaryConditionConcept = requires(const Field<Scheme>& U) {
    { BC::evaluate(U) } -> std::same_as<typename Scheme::State>;
};