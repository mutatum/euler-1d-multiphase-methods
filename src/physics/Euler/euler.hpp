#pragma once
#include <functional>
#include <cmath>
#include <array>
#include <Eigen/Dense>
#include <type_traits>
#include <cassert>
#include <ostream>

namespace physics {
namespace euler {

template <typename ScalarType>
struct EulerState
{
    using Scalar = ScalarType;
    constexpr static std::size_t variables = 3;
    static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating-point type");
    
    Scalar density;
    Scalar momentum;
    Scalar total_energy;

    [[nodiscard]] constexpr Scalar velocity() const noexcept { 
        assert(density > Scalar(0));
        return momentum / density; 
    }

    [[nodiscard]] constexpr Scalar specific_internal_energy() const noexcept
    {
        const Scalar u = velocity();
        return total_energy / density - static_cast<Scalar>(0.5) * u * u;
    }

    [[nodiscard]] constexpr bool is_physically_valid() const noexcept
    {
        return density > Scalar(0) && total_energy > Scalar(0);
    }

    constexpr EulerState() noexcept : density(0), momentum(0), total_energy(0) {}

    constexpr EulerState(Scalar rho, Scalar mom, Scalar E) noexcept 
        : density(rho), momentum(mom), total_energy(E) {
        // assert(rho >= Scalar(0) && "Density must be non-negative");
        // assert(E >= Scalar(0) && "Total energy must be non-negative");
    }

    constexpr EulerState(const std::array<Scalar, 3>& arr) noexcept 
        : density(arr[0]), momentum(arr[1]), total_energy(arr[2]) {
        // assert(arr[0] >= Scalar(0) && "Density must be non-negative");
        // assert(arr[2] >= Scalar(0) && "Total energy must be non-negative");
    }

    constexpr EulerState(const Eigen::Vector3<Scalar>& vec) noexcept 
        : density(vec[0]), momentum(vec[1]), total_energy(vec[2]) {
        // assert(vec[0] >= Scalar(0) && "Density must be non-negative");
        // assert(vec[2] >= Scalar(0) && "Total energy must be non-negative");
    }

    // =================== Arithmetic operations
    EulerState &operator+=(const EulerState &other)
    {
        density += other.density;
        momentum += other.momentum;
        total_energy += other.total_energy;
        return *this;
    }

    Eigen::Matrix<Scalar, 1, 3> to_vector() const
    {
        Eigen::Matrix<Scalar, 1, 3> vec;
        vec << density, momentum, total_energy;
        return vec;
    }

    // This is to account for Eigen's expression template in DG
    template <typename Derived>
    EulerState &operator+=(const Eigen::MatrixBase<Derived> &other)
    {
        density += other(0);
        momentum += other(1);
        total_energy += other(2);
        return *this;
    }

    // This is to account for Eigen's expression template in DG
    template <typename Derived>
    EulerState &operator=(const Eigen::MatrixBase<Derived> &other)
    {
        density = other(0);
        momentum = other(1);
        total_energy = other(2);
        return *this;
    }

    EulerState &operator-=(const EulerState &other)
    {
        density -= other.density;
        momentum -= other.momentum;
        total_energy -= other.total_energy;
        return *this;
    }

    EulerState &operator*=(Scalar scalar)
    {
        density *= scalar;
        momentum *= scalar;
        total_energy *= scalar;
        return *this;
    }

    EulerState operator+(const EulerState &other) const
    {
        return EulerState{density + other.density, momentum + other.momentum, total_energy + other.total_energy};
    }

    EulerState operator-(const EulerState &other) const
    {
        return EulerState{density - other.density, momentum - other.momentum, total_energy - other.total_energy};
    }

    EulerState operator*(Scalar scalar) const
    {
        return EulerState{density * scalar, momentum * scalar, total_energy * scalar};
    }
    
    friend std::ostream& operator<<(std::ostream& os, const EulerState& state) {
        os << "EulerState{density=" << state.density << ", momentum=" 
           << state.momentum << ", total_energy=" << state.total_energy << "}";
        return os;
    }
};

template <typename ScalarType>
EulerState<ScalarType> operator*(ScalarType scalar, const EulerState<ScalarType> &state)
{
    return state * scalar;
}

template <typename Scalar, auto gamma_value>
struct IdealGasEOS
{
    using ScalarType = Scalar;
    static constexpr Scalar gamma = static_cast<Scalar>(gamma_value);
    
    static_assert(gamma_value > Scalar(1), "Gamma must be greater than 1 for ideal gas");
    
    static constexpr Scalar pressure(const EulerState<Scalar> &U)
    {
        assert(U.density > Scalar(0) && "Density must be positive for pressure calculation");
        const Scalar kinetic_energy = static_cast<Scalar>(0.5) * U.momentum * U.momentum / U.density;
        const Scalar internal_energy = U.total_energy - kinetic_energy;
        return (gamma - static_cast<Scalar>(1.0)) * internal_energy;
    }

    static constexpr Scalar sound_speed(const EulerState<Scalar> &U)
    {
        assert(U.density > Scalar(0) && "Density must be positive for sound speed calculation");
        const Scalar p = pressure(U);
        assert(p > Scalar(0) && "Pressure must be positive for sound speed calculation");
        return std::sqrt(gamma * p / U.density);
    }

    static constexpr Scalar sound_speed(const Scalar _pressure, const Scalar density)
    {
        assert(density > Scalar(0) && "Density must be positive for sound speed calculation");
        assert(_pressure > Scalar(0) && "Pressure must be positive for sound speed calculation");
        return std::sqrt(gamma * _pressure / density);
    }
};

namespace constants {
    template<typename T>
    constexpr T gamma_air = T(1.4);

    template<typename T>
    constexpr T gamma_monatomic = T(5.0/3.0);

    template<typename T>
    constexpr T gamma_diatomic = T(7.0/5.0);
}

template <typename Scalar, typename EquationOfState>
struct EulerPhysics
{
public:
    using State = EulerState<Scalar>;
    using EOS = EquationOfState;
    static constexpr std::size_t variables = 3; // density, momentum, total_energy

    static Scalar density(const State &U)
    {
        return U.density;
    }
    
    static Scalar pressure(const State &U)
    {
        return EOS::pressure(U);
    }

    static Scalar velocity(const State &U)
    {
        return U.velocity();
    }

    static State physical_flux(const State &U)
    {
        assert(U.density > Scalar(0) && "Density must be positive for flux calculation");
        const Scalar u = U.velocity();
        const Scalar p = pressure(U);
        assert(p > Scalar(0) && "Pressure must be positive for flux calculation");
        
        return State{
            U.momentum,              // rho * u
            U.momentum * u + p,      // rho * u^2 + p
            (U.total_energy + p) * u // (E + p) * u
        };
    }

    static Scalar max_wave_speed(const State &U)
    {
        return std::abs(U.velocity()) + EOS::sound_speed(U);
    }

    static Scalar max_wave_speed(const Scalar _velocity, const Scalar _pressure, const Scalar density)
    {
        return std::abs(_velocity) + EOS::sound_speed(_pressure, density);
    }
};

// Type aliases for common configurations
template<typename Scalar>
using AirPhysics = EulerPhysics<Scalar, IdealGasEOS<Scalar, constants::gamma_air<Scalar>>>;

template<typename Scalar>
using MonatomicGasPhysics = EulerPhysics<Scalar, IdealGasEOS<Scalar, constants::gamma_monatomic<Scalar>>>;

template<typename Scalar>
using DiatomicGasPhysics = EulerPhysics<Scalar, IdealGasEOS<Scalar, constants::gamma_diatomic<Scalar>>>;

} // namespace euler
} // namespace physics