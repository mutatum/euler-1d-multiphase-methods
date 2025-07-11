#pragma once
#include <cstddef>

template<const auto& Nodes>
struct Lagrange 
{
    using array_type = std::remove_cvref_t<decltype(Nodes)>;
    using Scalar = typename array_type::value_type;
    static constexpr std::size_t num_basis_functions = Nodes.size();
    static constexpr std::size_t order = num_basis_functions - 1;

    static constexpr Scalar evaluate(std::size_t i, Scalar x) {
        Scalar result = 1.0;
        for (std::size_t j=0; j< num_basis_functions; ++j) {
            if (i == j) continue;
            result *= (x-Nodes[j]) / (Nodes[i] - Nodes[j]);
        }
        return result;
    }

    static constexpr Scalar derivative(std::size_t i, Scalar x) {
        Scalar result = 0.0;
        for (std::size_t m=0; m < num_basis_functions; ++m) {
            if (m == i) continue;
            Scalar product = (1.0/(Nodes[i]-Nodes[m]));
            for (std::size_t k=0; k < num_basis_functions; ++k) {
                if (k == i or k == m) continue;
                product *= (x - Nodes[k]) / (Nodes[i] - Nodes[k]);
            }
            result += product;
        }
        return result;
    }



};