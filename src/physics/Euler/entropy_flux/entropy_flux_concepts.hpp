#pragma once
#include <concepts>
#include "../../physics_concepts.hpp"

template <typename T>
concept EntropyFluxConcept = requires(const typename T::Physics::State &sL, const typename T::Physics::State &sR, typename T::Physics::State &result) {
    requires PhysicsModelConcept<typename T::Physics>;
    { T::compute(sL, sR, result) };
};