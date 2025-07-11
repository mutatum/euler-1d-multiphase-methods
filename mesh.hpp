#pragma once
#include <cstddef>
#include <stdexcept>
#include <array>
#include <vector>

template<typename T> 
class Cell {
    T x_center;
    T x_start;
    T x_end;
    T dx;
    bool is_border;
    Cell(T x_start, T x_end): x_start(x_start), x_end(x_end), x_center((x_start+x_end)/2), dx(x_end-x_start) {
        if (x_end <= x_start) {
            throw std::invalid_argument("Cell end must be greater than start");
        }
    }
};

template<typename Scalar>
struct ConservedState {
    Scalar density;
    Scalar momentum;
    Scalar total_energy;
};

template<typename Scalar, std::size_t PolynomialOrder>
struct Element {
    std::array<ConservedState<Scalar>, PolynomialOrder+1> coeffs;
};

template<typename Scalar, std::size_t PolynomialOrder, std::size_t NumCells>
struct SolutionField {
    std::vector<Element<Scalar, PolynomialOrder>> data;
    explicit SolutionField(): data(NumCells) {}
};

// 1D Structured Uniform Mesh
template<typename Scalar>
class Mesh {
    private:
    std::size_t n_cells; 
    Scalar dt; 
    std::vector<Scalar> cells;

    public:

};