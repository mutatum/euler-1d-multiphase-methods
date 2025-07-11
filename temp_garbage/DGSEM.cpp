#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <fstream>

#include <Eigen/Dense>

// Using directives for Eigen
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Vector3d;

// State variables indices
const int RHO = 0;
const int RHO_U = 1;
const int ENERGY = 2;

// Global constant for gamma
const double GAMMA = 1.4;

// GLL quadrature data structure
struct GLLQuadrature {
    VectorXd xi;
    VectorXd w;
};

// Function to get GLL data for a given polynomial degree p
GLLQuadrature get_gll_quadrature(int p) {
    switch (p) {
        case 0: {
            VectorXd xi(1); xi << -1.0;
            VectorXd w(1); w << 2.0;
            return {xi, w};
        }
        case 1: {
            VectorXd xi(2); xi << -1.0, 1.0;
            VectorXd w(2); w << 1.0, 1.0;
            return {xi, w};
        }
        case 2: {
            VectorXd xi(3); xi << -1.0, 0.0, 1.0;
            VectorXd w(3); w << 1.0/3.0, 4.0/3.0, 1.0/3.0;
            return {xi, w};
        }
        case 3: {
            VectorXd xi(4); xi << -1.0, -std::sqrt(1.0/5.0), std::sqrt(1.0/5.0), 1.0;
            VectorXd w(4); w << 1.0/6.0, 5.0/6.0, 5.0/6.0, 1.0/6.0;
            return {xi, w};
        }
        case 4: {
            VectorXd xi(5); xi << -1.0, -std::sqrt(3.0/7.0), 0.0, std::sqrt(3.0/7.0), 1.0;
            VectorXd w(5); w << 1.0/10.0, 49.0/90.0, 32.0/45.0, 49.0/90.0, 1.0/10.0;
            return {xi, w};
        }
        case 5: {
            VectorXd xi(6); 
            xi << -1.0, -std::sqrt(1.0/3.0 + 2.0*std::sqrt(7.0)/21.0), -std::sqrt(1.0/3.0 - 2.0*std::sqrt(7.0)/21.0),
                   std::sqrt(1.0/3.0 - 2.0*std::sqrt(7.0)/21.0),  std::sqrt(1.0/3.0 + 2.0*std::sqrt(7.0)/21.0), 1.0;
            VectorXd w(6);
            w << 1.0/15.0, (14.0 - std::sqrt(7.0))/30.0, (14.0 + std::sqrt(7.0))/30.0,
                 (14.0 + std::sqrt(7.0))/30.0, (14.0 - std::sqrt(7.0))/30.0, 1.0/15.0;
            return {xi, w};
        }
        case 6: {
            VectorXd xi(7);
            xi << -1.0, -std::sqrt(5.0/11.0 + 2.0*std::sqrt(5.0/3.0)/11.0), -std::sqrt(5.0/11.0 - 2.0*std::sqrt(5.0/3.0)/11.0),
                  0.0, std::sqrt(5.0/11.0 - 2.0*std::sqrt(5.0/3.0)/11.0), std::sqrt(5.0/11.0 + 2.0*std::sqrt(5.0/3.0)/11.0), 1.0;
            VectorXd w(7);
            w << 1.0/21.0, (124.0 - 7.0*std::sqrt(15.0))/350.0, (124.0 + 7.0*std::sqrt(15.0))/350.0,
                 256.0/525.0, (124.0 + 7.0*std::sqrt(15.0))/350.0, (124.0 - 7.0*std::sqrt(15.0))/350.0, 1.0/21.0;
            return {xi, w};
        }
        default:
            throw std::invalid_argument("Unsupported polynomial degree p.");
    }
}

double equation_of_state(double rho, double e) {
    return (GAMMA - 1.0) * rho * e;
}

Vector3d physical_flux(const Vector3d& U) {
    double rho = U(RHO);
    double u = U(RHO_U) / rho;
    double E = U(ENERGY);
    double e = E / rho - 0.5 * u * u;
    double p = equation_of_state(rho, e);
    
    Vector3d F;
    F(RHO) = U(RHO_U);
    F(RHO_U) = U(RHO_U) * u + p;
    F(ENERGY) = (E + p) * u;
    return F;
}

Vector3d rusanov_flux(const Vector3d& Ul, const Vector3d& Ur, double lambda_max) {
    return 0.5 * (physical_flux(Ul) + physical_flux(Ur)) - 0.5 * lambda_max * (Ur - Ul);
}

double compute_max_speed(const std::vector<MatrixXd>& U_cells) {
    int n_cells = U_cells.size();
    double lambda_max = 1e-6;

    for (int i = 0; i <= n_cells; ++i) {
        Vector3d UL, UR;
        if (i == 0) { // Reflecting boundaries
            UL = UR = U_cells[i].col(0);
        } else if (i == n_cells) {
            UR = UL = U_cells[i-1].col(U_cells[i-1].cols()-1);
        } else {
            UL = U_cells[i-1].col(U_cells[i-1].cols()-1);
            UR = U_cells[i].col(0);
        }

        double rhoR = std::max(1e-9, UR(RHO));
        double rhoL = std::max(1e-9, UL(RHO));
        double uR = UR(RHO_U) / rhoR;
        double uL = UL(RHO_U) / rhoL;
        double ER = UR(ENERGY);
        double EL = UL(ENERGY);
        double eR = ER / rhoR - 0.5 * uR * uR;
        double eL = EL / rhoL - 0.5 * uL * uL;
        double pR = std::max(1e-9, equation_of_state(rhoR, eR));
        double pL = std::max(1e-9, equation_of_state(rhoL, eL));
        
        double cR = std::sqrt(GAMMA * pR / rhoR);
        double cL = std::sqrt(GAMMA * pL / rhoL);
        
        lambda_max = std::max({lambda_max, std::abs(uL) + cL, std::abs(uR) + cR});
    }
    return lambda_max;
}

std::vector<Vector3d> compute_fluxes(const std::vector<MatrixXd>& U_cells, double lambda_max) {
    int n_cells = U_cells.size();
    int n_fluxes = n_cells + 1;
    std::vector<Vector3d> fluxes(n_fluxes);

    for (int i = 0; i < n_fluxes; ++i) {
        Vector3d UL, UR;
        if (i == 0) { // Reflecting boundaries
            UL = UR = U_cells[i].col(0);
        } else if (i == n_fluxes - 1) {
            UR = UL = U_cells[i-1].col(U_cells[i-1].cols()-1);
        } else {
            UL = U_cells[i-1].col(U_cells[i-1].cols()-1);
            UR = U_cells[i].col(0);
        }
        fluxes[i] = rusanov_flux(UL, UR, lambda_max);
    }
    return fluxes;
}

MatrixXd compute_differentiation_matrix(int p) {
    if (p == 0) {
        return MatrixXd::Zero(1, 1);
    }
    
    GLLQuadrature gll = get_gll_quadrature(p);
    const auto& xi = gll.xi;
    MatrixXd D = MatrixXd::Zero(p + 1, p + 1);
    
    VectorXd c = VectorXd::Ones(p + 1);
    c(0) = 2.0;
    c(p) = 2.0;

    for (int i = 0; i <= p; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j <= p; ++j) {
            if (i != j) {
                D(i, j) = (c(i) / c(j)) / (xi(i) - xi(j));
                row_sum += D(i, j);
            }
        }
        D(i, i) = -row_sum;
    }
    
    return D;
}

void dgsem_residual(std::vector<MatrixXd>& residual, const std::vector<MatrixXd>& U_cells, const MatrixXd& D, double dx, double max_speed, const VectorXd& w) {
    int n_cells = U_cells.size();
    int p = U_cells[0].cols() - 1;
    
    auto fluxes = compute_fluxes(U_cells, max_speed);
    
    for (int c = 0; c < n_cells; ++c) {
        // Volume integral term
        MatrixXd F_nodes(3, p + 1);
        for (int k = 0; k <= p; ++k) {
            F_nodes.col(k) = physical_flux(U_cells[c].col(k));
        }
        residual[c] = -F_nodes * D.transpose();

        // Surface integral term (flux corrections at boundaries)
        Vector3d F_left = physical_flux(U_cells[c].col(0));
        Vector3d F_right = physical_flux(U_cells[c].col(p));
        
        residual[c].col(0) -= (2.0 / (dx * w(0))) * (F_left - fluxes[c]);
        residual[c].col(p) += (2.0 / (dx * w(p))) * (F_right - fluxes[c+1]);
    }
}

void rk3_ssp(std::vector<MatrixXd>& U, std::vector<MatrixXd>& U_1, std::vector<MatrixXd>& U_2, std::vector<MatrixXd>& residual, const MatrixXd& D, double dx, double dt, double lambda_max, const VectorXd& w) {

    // Stage 1
    dgsem_residual(residual, U, D, dx, lambda_max, w);
    for (size_t i = 0; i < U.size(); ++i) {
        U_1[i] = U[i] + dt * residual[i];
    }

    // Stage 2
    dgsem_residual(residual, U_1, D, dx, lambda_max, w);
    for (size_t i = 0; i < U.size(); ++i) {
        U_2[i] = 0.75 * U[i] + 0.25 * (U_1[i] + dt * residual[i]);
    }

    // Stage 3
    dgsem_residual(residual, U_2, D, dx, lambda_max, w);
    for (size_t i = 0; i < U.size(); ++i) {
        U[i] = (1.0 / 3.0) * U[i] + (2.0 / 3.0) * (U_2[i] + dt * residual[i]);
    }
}

void init_sod(std::vector<MatrixXd>& U_cells, const VectorXd& cell_centers, double dx) {
    int n_cells = U_cells.size();
    int p = U_cells[0].cols() - 1;
    auto gll = get_gll_quadrature(p);
    const auto& xi = gll.xi;

    for (int i = 0; i < n_cells; ++i) {
        for (int j = 0; j <= p; ++j) {
            double x = cell_centers(i) + 0.5 * dx * xi(j);
            double rho, u, press;
            if (x < 0.5) {
                rho = 1.0;
                u = 0.0;
                press = 1.0;
            } else {
                rho = 0.125;
                u = 0.0;
                press = 0.1;
            }
            double E = press / (GAMMA - 1.0) + 0.5 * rho * u * u;
            
            U_cells[i](RHO, j) = rho;
            U_cells[i](RHO_U, j) = rho * u;
            U_cells[i](ENERGY, j) = E;
        }
    }
}

void write_solution_to_file(const std::vector<MatrixXd>& U_cells, const VectorXd& cell_centers, double dx, int p, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    auto gll = get_gll_quadrature(p);
    const auto& xi = gll.xi;

    outfile << "x,rho,u,pressure,e" << std::endl;

    for (size_t i = 0; i < U_cells.size(); ++i) {
        for (int j = 0; j <= p; ++j) {
            double x = cell_centers(i) + 0.5 * dx * xi(j);
            
            const auto& U = U_cells[i].col(j);
            double rho = U(RHO);
            double u = U(RHO_U) / rho;
            double E = U(ENERGY);
            double e = E / rho - 0.5 * u * u;
            double pressure = equation_of_state(rho, e);

            outfile << x << "," << rho << "," << u << "," << pressure << "," << e << std::endl;
        }
    }
    outfile.close();
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <N> <p> <T> <cfl>" << std::endl;
        return 1;
    }
    int N = std::stoi(argv[1]);
    int p = std::stoi(argv[2]);
    double T = std::stod(argv[3]);
    double cfl = std::stod(argv[4]);
    if (N <= 0 || p < 0 || T <= 0.0 || cfl <= 0.0) {
        std::cerr << "Invalid input parameters." << std::endl;
        return 1;
    }

    // Domain
    double x_min = 0.0, x_max = 1.0;
    VectorXd X = VectorXd::LinSpaced(N + 1, x_min, x_max);
    VectorXd cell_centers(N);
    for(int i=0; i<N; ++i) {
        cell_centers(i) = (X(i+1) + X(i)) / 2.0;
    }
    double dx = (x_max - x_min) / N;

    // Solution data structures
    std::vector<MatrixXd> U_cells(N, MatrixXd(3, p + 1));
    
    init_sod(U_cells, cell_centers, dx);

    MatrixXd D = compute_differentiation_matrix(p);
    auto gll = get_gll_quadrature(p);
    const auto& w = gll.w;

    // For rk3ssp and residual computation. Avoids reallocating
    std::vector<MatrixXd> U_1(N, MatrixXd(3, p + 1));
    std::vector<MatrixXd> U_2(N, MatrixXd(3, p + 1));
    std::vector<MatrixXd> residual(N, MatrixXd(3, p + 1));

    double t = 0.0;
    while (t < T) {
        double lambda_max = compute_max_speed(U_cells);
        double dt = cfl * dx / ((2 * p + 1) * lambda_max);
        if (t + dt > T) {
            dt = T - t;
        }

        rk3_ssp(U_cells, U_1, U_2, residual, D, dx, dt, lambda_max, w);

        t += dt;
        std::cout << "t = " << std::fixed << std::setprecision(4) << t << "/" << T 
                  << ", dt = " << std::scientific << std::setprecision(4) << dt << std::endl;
    }
    
    write_solution_to_file(U_cells, cell_centers, dx, p, "solution.txt");

    return 0;
}
