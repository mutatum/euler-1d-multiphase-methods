#pragma once
#include <concepts>

template <typename T>
concept StateModelConcept = requires (T a, T b, typename T::Scalar s) {
    typename T::Scalar;
    { a += b } -> std::same_as<T&>;
    { a -= b } -> std::same_as<T&>;
    { a *= s } -> std::same_as<T&>;
    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
    { a * s } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
};

template <typename T>
concept PhysicsModelConcept = requires(const typename T::State& s) {
    typename T::State;
    requires StateModelConcept<typename T::State>;
    { T::variables } -> std::convertible_to<typename std::size_t>;
    { T::physical_flux(s) } -> std::same_as<typename T::State>;
    { T::max_wave_speed(s) } -> std::convertible_to<typename T::State::Scalar>;
};