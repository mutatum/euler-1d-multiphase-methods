#pragma once
#include <concepts>
#include "../physics_concepts.hpp"

template <typename T>
concept EulerStateConcept = requires(T a) {
    requires StateModelConcept<T>;
    // Specific to Euler
    { a.density } -> std::convertible_to<typename T::Scalar>;
    { a.momentum } -> std::convertible_to<typename T::Scalar>;
    { a.total_energy } -> std::convertible_to<typename T::Scalar>;
};

template <typename T>
concept EOSConcept = requires(const T::State &s) {
    typename T::Scalar;
    typename T::State;
    requires EulerStateConcept<typename T::State>;
    requires std::is_same_v<typename T::Scalar, typename T::State::Scalar>;
    requires std::same_as<typename T::Scalar, typename T::State::Scalar>;
    { T::pressure(s) } -> std::convertible_to<typename T::State::Scalar>;
    { T::sound_speed(s) } -> std::convertible_to<typename T::State::Scalar>;
};