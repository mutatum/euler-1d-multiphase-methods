#pragma once
#include "../mesh/mesh.hpp"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <type_traits>

template <typename SchemePolicy>
class Field
{
public:
    using Element = typename SchemePolicy::Element;
    using Scalar = typename SchemePolicy::Scalar;

private:
    std::vector<Element> data_;

public:
    explicit Field(std::size_t num_elements) : data_(num_elements) {}
    
    explicit Field(std::vector<Element> data) : data_(std::move(data)) {}

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

    // Range-based iteration support
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
    
    // Const element access with bounds checking
    const Element& at(std::size_t i) const { return data_.at(i); }
    Element& at(std::size_t i) { return data_.at(i); }
};

template <typename SchemePolicy>
Field<SchemePolicy> operator*(typename SchemePolicy::Scalar scalar, const Field<SchemePolicy>& field) {
    return field * scalar;
}