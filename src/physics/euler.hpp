#pragma once
#include <functional>
#include "flux_model.hpp"
#include <cmath>

template<typename Scalar>
struct EulerState {
    using scalar_type = Scalar;
    Scalar density;
    Scalar momentum;
    Scalar total_energy;

    inline Scalar velocity() const {return momentum / density; }

    inline Scalar specific_internal_energy() const { 
        Scalar _velocity = velocity();
        return total_energy/density - 0.5 * _velocity * _velocity;
    }
};

template<typename Scalar>
class IdealGasEOS {
private:
    Scalar gamma_;
public:
    explicit IdealGasEOS(Scalar gamma = 1.4): gamma_(gamma) {}

    inline Scalar pressure(EulerState<Scalar>& U) const {
        return (gamma_ - 1.0) * (U.total_energy - 0.5 * U.momentum * U.momentum / U.density);
    }

    inline Scalar sound_speed(EulerState<Scalar>& U) const {
        return std::sqrt(gamma_ * pressure(U) / U.density);
    }

    inline Scalar sound_speed(Scalar _pressure, Scalar density) const {
        return std::sqrt(gamma_ * _pressure / density);
    }
    
    inline void set_gamma(Scalar gamma) { gamma_ = gamma; }
    inline Scalar get_gamma() const { return gamma_; }
};


template <typename Scalar, typename EOS>
class EulerPhysics: public FluxModel<EulerState<Scalar>>
{
private:
    EOS eos_;
public:
    using State = EulerState<Scalar>;

    template<typename... Args>
    explicit EulerPhysics(Args&&... args) : eos_(std::forward<Args>(args)...) {}

    inline Scalar pressure(const State& U) const {
        return eos_.pressure(U);
    }
    
    inline Scalar velocity(const State& U) const {
        return U.velocity();
    }

    State physical_flux(const State& U) const override {
        const Scalar u = U.velocity();
        const Scalar p = pressure(U);
        return State{
            U.momentum,                 // rho * u
            U.momentum * u + p,         // rho * u^2 + p
            (U.total_energy + p) * u    // (E + p) * u
        };
    }


};