#pragma once
#include <concepts>
#include <vector>
#include <functional>
#include <cstddef>
#include <type_traits>
#include "../physics/physics_concepts.hpp"
#include "../utils/basis/basis_concepts.hpp"
#include "../utils/quadrature/quadrature_concepts.hpp"
#include "../physics/numerical_flux/numerical_flux_concepts.hpp"
#include "../physics/Euler/entropy_flux/entropy_flux_concepts.hpp"
#include "../boundary/boundary_conditions_concepts.hpp"

template <class SchemePolicy>
class Field;

template <typename T>
concept SchemeConcept = requires(T scheme, const Field<T> &U) {
    typename T::State;
    
    { T::PolynomialOrder } -> std::convertible_to<std::size_t>;
    { T::CFL_multiplier } -> std::convertible_to<typename T::Scalar>;

    { scheme.compute_residual(U, std::declval<Field<T>&>(), std::declval<typename T::Workspace&>()) } -> std::convertible_to<typename T::Scalar>;
};

template <typename Config>
concept DGConfigConcept =  
    PhysicsModelConcept<typename Config::Physics> &&
    NumericalFluxConcept<typename Config::NumericalFlux>;
    // PolynomialBasisConcept<typename Config::PolynomialBasis> ;
    // QuadratureConcept<typename Config::Quadrature>;
    // BoundaryConditionConcept<typename Config::LeftBC, typename Config::Scheme> &&
    // BoundaryConditionConcept<typename Config::RightBC, typename Config::Scheme>;

template <typename Config>
concept DGSEMConfigConcept =  
    PhysicsModelConcept<typename Config::Physics> &&
    NumericalFluxConcept<typename Config::NumericalFlux>;

// Specific to Euler DGSEM schemes
template <typename Config>
concept DGSEM_ESConfigConcept =  
    PhysicsModelConcept<typename Config::Physics> &&
    NumericalFluxConcept<typename Config::NumericalFlux> &&
    EntropyFluxConcept<typename Config::EntropyFlux>;