#pragma once
#include <vector>
#include <cstddef>

class Mesh
{
protected:
    std::size_t n_cells_total_;

public:
    constexpr Mesh() noexcept : n_cells_total_(0) {}
    constexpr explicit Mesh(std::size_t n_cells_total) noexcept : n_cells_total_(n_cells_total) {}
    virtual ~Mesh() = default;
    
    // Rule of five
    Mesh(const Mesh&) = default;
    Mesh& operator=(const Mesh&) = default;
    Mesh(Mesh&&) = default;
    Mesh& operator=(Mesh&&) = default;
    
    [[nodiscard]] constexpr std::size_t total_cells() const noexcept { return n_cells_total_; }
};