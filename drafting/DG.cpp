#include <iostream>
#include <Eigen/Dense>
#include <cmath>

using Eigen::Vector3d;

/* U = { rho,
         rho * v,
         rho * E} */

/* F(U) = { rho * v,
            rho * v tensor v + p * I,
            rho * E * v + p * v } */

double legendre(std::size_t n, double x)
{
    switch (n)
    {
    case 0:
        return 1.0;
    case 1:
        return x;
    default:
        return ((2. * n - 1.) * x * legendre(n - 1, x) - (n - 1.) * legendre(n - 2, x)) / n;
    }
}

inline double eq_of_state(double gamma, double rho, double e)
{
    if (rho <= 0.)
        throw std::runtime_error("Non-positive density");
    return (gamma - 1) * rho * e;
}

/* U = { rho,
         rho * u,
         rho * E} */

/* F(U) = { rho * u,
            rho * u * u + p * I,
            u(rho * E + p) } */

Vector3d physical_flux(const Vector3d &U)
{
    Vector3d F;
    double u = U[1] / U[0];
    double E = U[2] / U[0];
    double e = E - .5 * u * u;
    double p = eq_of_state(1.4, U[0], e);
    F[0] = U[1]; // rho * v
    F[1] = U[1] * u + p;
    F[2] = (E + p) * u;
    return F;
}

// Quarteroni Sacco Saleri Méthodes Numériques p.348
// Noeuds et poids des formules de Gauss-Legendre
// returns [x,w] où w est un vector ligne de poids
template <std::size_t n_nodes>
Eigen::Matrix<double, 2, n_nodes> zplege()
{
    Eigen::Matrix<double, 2, n_nodes> M; // will be returned. M is of size (2,n) with first row being nodes and second being weights
    Eigen::Matrix<double, n_nodes, n_nodes> JacM;
    M.setZero();
    JacM.setZero();

    if (n_nodes <= 1)
    {
        std::cerr << "n_nodes is too small" << std::endl;
        abort();
    } // error
    double b;
    for (std::size_t i = 0; i < n_nodes - 1; i++)
    {
        b = 1. / (4 - 1. / std::pow(i + 1, 2));
        JacM(i, i + 1) = JacM(i + 1, i) = std::sqrt(b); // sur et sous diag
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, n_nodes, n_nodes>> eigensolver(JacM);
    if (eigensolver.info() != Eigen::Success)
        abort();
    auto values = eigensolver.eigenvalues();
    auto vectors = eigensolver.eigenvectors();
    for (std::size_t i = 0; i < n_nodes; i++)
    {
        M(0, i) = values(i); // nodes
        M(1, i) = vectors(0, i) * vectors(0, i) * 2.; // weights
    }
    return M;
}

// Exact for polynomials of degree 2q-1. So for instance, if integrating P of degree n = 3, n_nodes should be at least 2.
// if integrating P of degree n=4, n_nodes should be at least 3
// if integrating P of degree n=5, n_nodes should be at least 3
// So if integrating P of degree n, n_nodes should be at least ceil((n+1) / 2)
template <std::size_t n_nodes, typename State>
State GL_integration(const std::function<State(double)> &f)
{
    auto M = zplege<n_nodes>(); // first row are nodes, second are weights
    State ret;
    for (std::size_t i = 0; i < n_nodes; i++)
    {
        ret += M(1, i) * f(M(0, i)); // M(1,i) is the weight, M(0,i) is the node
    }
    return ret * 0.5; 
}

template <size_t ndof>
Eigen::Matrix<double, ndof + 1, ndof + 1> build_mass_matrix() // + 1 because starting with deg(P)=0
{
    Eigen::Matrix<double, ndof + 1, ndof + 1> M;
    M.setZero();
    for (std::size_t i = 0; i < ndof + 1; i++)
    {
        M(i, i) = 2. / (2. * i + 1.); // <Lk, Lm> = delta{k,m} / (k+1/2)
    }
    return M;
}

template <size_t ndof>
Eigen::Matrix<double, ndof + 1, ndof + 1> build_D_matrix()
{
    Eigen::Matrix<double, ndof + 1, ndof + 1> D;
    D.setZero();
    for (std::size_t i = 0; i < ndof + 1; i++)
    {
        for (std::size_t j = 0; j < ndof + 1; j++)
        {
            // D(j,i) = <Pi, dPj> L2 scalar product
            D(j, i) = GL_integration<ndof + 1, double>([i, j](double x) -> double {
                return legendre(i, x) * (-j * x * legendre(j, x) + j * legendre(j - 1, x)) / (1. - x * x); // P_i * P_j' , with derivative given by formula
            });
        }
    }
    return D;
}

// Takes Uh's coordinates in Legendre Polynomials base and computes Uh at value x
template <size_t ndof>
Vector3d compute_uh_from_base_coordinate(const Eigen::Matrix<double, ndof + 1, 3> &base_coordinates, double x)
{
    Vector3d Uh;
    Uh.setZero();
    for (std::size_t i = 0; i < ndof + 1; i++)
    {
        Uh[0] += base_coordinates(i,0) * legendre(i, x); // rho
        Uh[1] += base_coordinates(i,1) * legendre(i, x); // rho u
        Uh[2] += base_coordinates(i,2) * legendre(i, x); // energy
    }

    return Uh;
}

// Compute projection of f onto f_j
// Integral(C_i) of { f(u_h)e_j } and multiply by Minv (see Vilar 2010, p.30)
template <size_t ndof>
Eigen::Matrix<double, ndof + 1, 3> build_F_matrix(const Eigen::Matrix<double, ndof + 1, 3> &uh_base_coordinates, const Eigen::Matrix<double, ndof + 1, ndof + 1> &Minv)
{
    Eigen::Matrix<double, ndof + 1, 3> F;
    F.setZero();
    for (std::size_t i = 0; i < ndof + 1; i++)
    {
        F.row(i) = GL_integration<ndof + 1, Vector3d>([i, uh_base_coordinates](double x) -> Vector3d {
            return physical_flux(compute_uh_from_base_coordinate<ndof>(uh_base_coordinates, x)) * legendre(i, x); });
    }

    return Minv * F;
}

// B is the matrix of basis functions evaluated at x
// B[i] = L_i(x) where L_i is the i-th Legendre polynomial
// B is of size (ndof+1,3) because we have 3 components in U
template <size_t ndof>
Vector3d build_B_matrix(double x)
{
    Vector3d B;
    B.setZero();
    for (std::size_t i = 0; i < ndof + 1; i++)
    {
        B[i] = legendre(i, x);
    }
    return B;
}


Vector3d numerical_flux(const Vector3d &ul, const Vector3d &ur) // Rusanov numerical flux
{
    Vector3d fl = physical_flux(ul);
    Vector3d fr = physical_flux(ur);

    double rhoL = ul[0], uL = ul[1] / ul[0], EL = ul[2] / ul[0];
    double eL = EL - 0.5 * uL * uL;
    double pL = (1.4 - 1) * rhoL * eL;
    double cL = std::sqrt(1.4 * pL / rhoL);

    double rhoR = ur[0], uR = ur[1] / ur[0], ER = ur[2] / ur[0];
    double eR = ER - 0.5 * uR * uR;
    double pR = (1.4 - 1) * rhoR * eR;
    double cR = std::sqrt(1.4 * pR / rhoR);

    double lambda = std::max(std::abs(uL) + cL, std::abs(uR) + cR);

    return 0.5 * (fl + fr) - 0.5 * lambda * (ur - ul);
}

template <std::size_t ndof>
Eigen::Matrix<double, ndof + 1, 3> residual(const Eigen::Matrix<double, ndof + 1, 3> &uh_base_coordinates,
                                            const Eigen::Matrix<double, ndof + 1, ndof + 1> Minv,
                                            const Vector3d &u_right_previous,
                                            const Vector3d &u_left_next)
{
    Eigen::Matrix<double, ndof + 1, 3> R;
    R.setZero();

    // R = -D * F
    R = -build_D_matrix<ndof>() * build_F_matrix<ndof>(uh_base_coordinates, Minv);
    Vector3d u_left_internal = compute_uh_from_base_coordinate<ndof>(uh_base_coordinates, -1);
    Vector3d u_right_internal = compute_uh_from_base_coordinate<ndof>(uh_base_coordinates, 1);

    Vector3d flux_left = numerical_flux(u_right_previous, u_left_internal);
    Vector3d flux_right = numerical_flux(u_right_internal, u_left_next);

    // Add numerical fluxes
    for (std::size_t i = 0; i < ndof + 1; i++)
    {
        R.row(i) += flux_right * legendre(i, 1) - flux_left * legendre(i, -1);
    }

    return Minv * R;
}

template<std::size_t ndof>
double compute_time_step(const std::vector<Eigen::Matrix<double, ndof+1, 3>>& cells, double cfl)
{
    double dt_min = std::numeric_limits<double>::max();
    double dx = 1.0 / cells.size(); // Assuming domain [0,1]
    
    for (const auto& cell : cells) {
        Vector3d U_avg = cell.row(0);
        
        double rho = U_avg[0];
        if (rho <= 0.0) {
            throw std::runtime_error("Non-positive density encountered in compute_time_step.");
        }
        double u = U_avg[1] / rho;
        double E = U_avg[2] / rho;
        double e = E - 0.5 * u * u;
        double p = eq_of_state(1.4, rho, e); // Assuming gamma = 1.4
        if (rho <= 0.0) {
            throw std::runtime_error("Non-positive density encountered in compute_time_step.");
        }
        double c = std::sqrt(1.4 * p / rho);
        
        double lambda_max = std::abs(u) + c;
        double dt_cell = cfl * dx / lambda_max;
        dt_min = std::min(dt_min, dt_cell);
    }
    
    return dt_min;
}

// Runge-Kutta 3rd order time integration Vilar 2010, p.32
template <std::size_t ndof>
Eigen::Matrix<double, ndof+1, 3> RK3_step(
    const Eigen::Matrix<double, ndof + 1, 3> &uh,
    const Eigen::Matrix<double, ndof + 1, ndof + 1> &Minv,
    const Vector3d &u_right_previous,
    const Vector3d &u_left_next,
    double dt,
    double dx)
{
    using Eigen::Matrix;
    Matrix<double, ndof + 1, 3> u1 = uh + dt * residual<ndof>(uh, Minv, u_right_previous, u_left_next);
    Matrix<double, ndof + 1, 3> u2 = .25* (3*uh+u1) + 0.25 * dt * residual<ndof>(u1, Minv,u_right_previous, u_left_next);
    Matrix<double, ndof + 1, 3> u_next = (uh+2*u2)/3 + (2/3) * dt * residual<ndof>(u2, Minv, u_right_previous, u_left_next);
    return u_next;
}



/*  A note about expression templates

This is an advanced topic that we explain on this page, but it is useful to just mention it now.
In Eigen, arithmetic operators such as operator+ don't perform any computation by themselves,
they just return an "expression object" describing the computation to be performed.
The actual computation happens later, when the whole expression is evaluated, typically in operator=.
While this might sound heavy, any modern optimizing compiler is able to optimize away that
abstraction and the result is perfectly optimized code. [https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html]
*/
int main()
{
    constexpr std::size_t ndof = 2; // Degree of the polynomial basis, spacial order is ndof+1 (here 3)
    constexpr std::size_t N_cells= 10;

    std::vector<Eigen::Matrix<double, ndof + 1, 3>> cells(N_cells);

    // Legendre projection. Only P0 is used because u0 is piecewise constant
    for (std::size_t i = 0; i < N_cells; i++) {
        cells[i].setZero();
        if (i < N_cells/2) {
            // Left state: high pressure
            cells[i](0, 0) = 1.0;    // rho
            cells[i](0, 1) = 0.0;    // rho*u
            cells[i](0, 2) = 2.5;    // rho*E
        } else {
            // Right state: low pressure  
            cells[i](0, 0) = 0.125;  // rho
            cells[i](0, 1) = 0.0;    // rho*u
            cells[i](0, 2) = 0.25;   // rho*E
        }
    }

    //! Remember that GD integration isn't yet adjusted to [a,b] integration, need to mult by dx

    auto M = build_mass_matrix<ndof>();        // M is n+1 square
    std::cout << "Mass Matrix:\n"
              << M << std::endl;
    std::cout << M.inverse() << std::endl;
    // std::cout << M*M.inverse() << std::endl; // this is indeed the inverse
    auto Minv = M.inverse();

    double T = 0.5;
    double t = 0.0;
    double cfl = 1/(2*ndof + 1); // CFL condition for RK3, see Vilar 2010, p.32;
    while (t < T)
    {
        double dt = compute_time_step<ndof>(cells, cfl);
        // drond_t Ui = Minv * (quadrature F drond_x phi_i - les Flux)
        // drond_t Ui = R le résidu
        // We will use RK3 for time integration since spacial order is 3
        auto cells_next = cells; // Copy cells
        Vector3d u_right_previous, u_left_next;
        for (std::size_t i = 0; i < N_cells; i++)
        {
            if (i == 0)
                u_right_previous = Vector3d(1.0, 0.0, 2.5); // Left
            else if (i == N_cells - 1)
                u_left_next = Vector3d(0.125, 0.0, 0.25); // Right
            else {
                u_right_previous = compute_uh_from_base_coordinate<ndof>(cells.at(i - 1), 1);
                u_left_next = compute_uh_from_base_coordinate<ndof>(cells.at(i + 1), -1);
            }

            cells_next[i] = RK3_step<ndof>(cells[i], Minv, u_right_previous, u_left_next, dt, dx);
        }
        
        cells = std::move(cells_next);
        t += dt;
        
        // Simple progress output
        if (static_cast<int>(t * 100) % 10 == 0) {
            std::cout << "t = " << t << ", dt = " << dt << std::endl;
        }
    }

    // Output final state for verification
    std::cout << "\nFinal state at t = " << t << std::endl;
    for (std::size_t i = 0; i < N_cells; i++) {
        Vector3d U = cells[i].row(0); // P0 component
        double rho = U[0];
        double u = U[1] / rho;
        double E = U[2] / U[0];
        double e = E - .5 * u * u;
        double p = eq_of_state(1.4, rho, e);
        std::cout << "Cell " << i << ": rho=" << rho << ", u=" << u << ", p=" << p << std::endl;
    }

    return 0;
}
