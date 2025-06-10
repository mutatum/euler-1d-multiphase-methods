#pragma once
#include <vector>
#include <cstddef>

class Mesh
{
protected:
    std::size_t n_cells_total_;

public:
    Mesh() : n_cells_total_(0) {}
    Mesh(std::size_t n_cells_total) : n_cells_total_(n_cells_total) {}
    virtual ~Mesh() = default;
    std::size_t total_cells() const { return n_cells_total_; }
};