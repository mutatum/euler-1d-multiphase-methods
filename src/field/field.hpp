#pragma once
#include "../mesh/mesh.hpp"
#include <stdexcept>
#include <vector>

template <typename SchemePolicy>
class Field
{
public:
    using Element = typename SchemePolicy::Element;
    using State = typename SchemePolicy::State;
    using Scalar = typename State::scalar_type;

private:
    std::vector<Element> data_;
    const Mesh &mesh_;

public:
    Field(const Mesh &mesh) : mesh_(mesh), data_(mesh.total_cells()) {};
    // Field(const Mesh &mesh, const State &initial_value) : mesh_(mesh), data_(mesh.total_cells(), initial_value) {};
    // Field(const Mesh &mesh, const std::vector<State> &data) : mesh_(mesh), data_(data) {
    //     if (data.size() != mesh.total_cells()) {
    //         throw std::invalid_argument("Data size does not match the number of cells in the mesh.");
    //     }
    // }
    std::size_t size() const { return data_.size(); }
    const Mesh &get_mesh() const { return this->mesh_; }
    const Element& operator()(std::size_t i) const {return data_.at(i); }
    Element& operator()(std::size_t i) {return data_.at(i); }
};