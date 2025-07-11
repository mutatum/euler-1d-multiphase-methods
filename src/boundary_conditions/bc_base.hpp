#pragma once
#include "../field/field.hpp"

enum class BoundaryLocation {
    X_MIN,
    X_MAX
};

template <typename State>
class BoundaryCondition {
    virtual ~BoundaryCondition() = default;

    virtual void apply(Field<State>& field, BoundaryLocation location) const = 0;
};