#pragma once
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <cassert>

template <typename SchemePolicy>
class Field
{
public:
    using Scheme = SchemePolicy;
    using Element = typename SchemePolicy::Element;
    using Scalar = typename SchemePolicy::Scalar;

private:
    std::vector<Element> data_;

public:
    Scalar dx;
    Scalar domain_start;
    Scalar domain_end;
    
    explicit Field(std::size_t num_elements, Scalar start=-1.0, Scalar end=1.0) : data_(num_elements), domain_start(start), domain_end(end), dx((end-start)/num_elements) {}

    explicit Field(std::vector<Element> data, Scalar start=-1.0, Scalar end=1.0) : data_(std::move(data)), domain_start(start), domain_end(end), dx((end-start)/data.size()) {}

    std::size_t size() const { return data_.size(); }
    const Element& operator()(std::size_t i) const { return data_[i]; }
    Element& operator()(std::size_t i) { return data_[i]; }

    // Arithmetic operations
    Field& operator+=(const Field& other) {
        if (data_.size() != other.data_.size()) {
            throw std::invalid_argument("Field sizes must match for addition");
        }
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i].coeffs += other.data_[i].coeffs;
        }
        return *this;
    }
    
    Field& operator-=(const Field& other) {
        if (data_.size() != other.data_.size()) {
            throw std::invalid_argument("Field sizes must match for subtraction");
        }
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i].coeffs -= other.data_[i].coeffs;
        }
        return *this;
    }
    
    Field& operator*=(Scalar scalar) {
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i].coeffs *= scalar;
        }
        return *this;
    }
    
    Field operator+(const Field& other) const {
        Field result = *this;
        result += other;
        return result;
    }
    
    Field operator-(const Field& other) const {
        Field result = *this;
        result -= other;
        return result;
    }
    
    Field operator*(Scalar scalar) const {
        Field result = *this;
        result *= scalar;
        return result;
    }

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
    
    const Element& at(std::size_t i) const { return data_.at(i); }
    Element& at(std::size_t i) { return data_.at(i); }

    Eigen::Vector<Scalar, Scheme::Variables> evaluate(Scalar x) const {
        // assert(x<=domain_end && x>=domain_start);
        const Scalar cell_pos = (x - domain_start)/dx;
        const std::size_t cell_index = cell_pos;
        const Scalar xi = (cell_pos-static_cast<Scalar>(cell_index))*static_cast<Scalar>(2.0) - static_cast<Scalar>(1.0);
        const auto state = Scheme::evaluate_element(data_[cell_index], xi);
        Eigen::Vector<Scalar, Scheme::Variables> result;
        result << state.density, state.momentum, state.total_energy;
        return result;
    }

};

template <typename SchemePolicy>
Field<SchemePolicy> operator*(typename SchemePolicy::Scalar scalar, const Field<SchemePolicy>& field) {
    return field * scalar;
}