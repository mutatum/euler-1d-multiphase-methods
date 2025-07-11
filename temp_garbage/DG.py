# %%

import numpy as np


def burgers(u):
    return 0.5 * u**2

def lagrange_basis_generator(i, nodes):
    def lagrange_basis(x):
        product = 1.0
        for j, node in enumerate(nodes):
            if j != i:
                product *= (x - node) / (nodes[i] - node)
        return product
    return lagrange_basis

# using taylor polynomial base
def build_mass_matrix(N, h):
    M = np.zeros((N, N))
    for i in range(N):


def DiscontinuousGalerkin()

# %%
import numpy as np
from scipy.special import roots_legendre

# Taylor polynomial basis (at x=0, scaled to [-1,1])
def taylor_basis(k, x):
    return (x**k) / np.math.factorial(k)

# Derivative of Taylor basis
def taylor_basis_deriv(k, x):
    if k == 0:
        return 0
    return x**(k-1) / np.math.factorial(k-1)

# Gauss-Lobatto quadrature nodes and weights (scaled to [-1,1])
def gauss_lobatto(n_nodes):
    # Using Legendre roots for Gauss-Lobatto (approximation for simplicity)
    nodes, weights = roots_legendre(n_nodes)
    # Adjust to include endpoints for Gauss-Lobatto
    nodes = np.concatenate([[-1], nodes, [1]])
    weights = np.concatenate([[2/(n_nodes*(n_nodes-1))], weights, [2/(n_nodes*(n_nodes-1))]])
    return nodes, weights

# Numerical flux for Burgers' equation (Rusanov/local Lax-Friedrichs)
def numerical_flux(ul, ur):
    fl = ul**2
    fr = ur**2
    lambda_max = max(abs(2*ul), abs(2*ur))  # |f'(u)| = |2u|
    return 0.5 * (fl + fr) - 0.5 * lambda_max * (ur - ul)

# Compute u_h at point x from Taylor basis coefficients
def compute_uh(coeffs, x, ndof):
    uh = 0
    for k in range(ndof + 1):
        uh += coeffs[k] * taylor_basis(k, x)
    return uh

# Build mass matrix (diagonal for Taylor basis, L2 inner product)
def build_mass_matrix(ndof):
    M = np.zeros((ndof + 1, ndof + 1))
    for i in range(ndof + 1):
        M[i, i] = 2 / (2 * i + 1)  # Analytical for normalized Taylor basis
    return M

# Build D matrix (stiffness matrix, <phi_i, phi_j'>)
def build_D_matrix(ndof, n_nodes=5):
    nodes, weights = gauss_lobatto(n_nodes)
    D = np.zeros((ndof + 1, ndof + 1))
    for i in range(ndof + 1):
        for j in range(ndof + 1):
            integral = sum(weights[k] * taylor_basis(i, nodes[k]) * taylor_basis_deriv(j, nodes[k])
                         for k in range(n_nodes))
            D[i, j] = integral * 0.5  # Scale for [-1,1]
    return D

# Build F matrix (projection of f(u_h) onto basis)
def build_F_matrix(uh_coeffs, Minv, ndof, n_nodes=5):
    nodes, weights = gauss_lobatto(n_nodes)
    F = np.zeros(ndof + 1)
    for i in range(ndof + 1):
        integral = sum(weights[k] * (compute_uh(uh_coeffs, nodes[k], ndof)**2) * taylor_basis(i, nodes[k])
                     for k in range(n_nodes))
        F[i] = integral * 0.5  # Scale for [-1,1]
    return Minv @ F

# Compute residual
def residual(uh_coeffs, Minv, u_right_prev, u_left_next, ndof, n_nodes=5):
    D = build_D_matrix(ndof, n_nodes)
    F = build_F_matrix(uh_coeffs, Minv, ndof, n_nodes)
    R = -D @ F
    u_left = compute_uh(uh_coeffs, -1, ndof)
    u_right = compute_uh(uh_coeffs, 1, ndof)
    flux_right = numerical_flux(u_right, u_left_next)
    flux_left = numerical_flux(u_right_prev, u_left)
    for i in range(ndof + 1):
        R[i] += flux_right * taylor_basis(i, 1) - flux_left * taylor_basis(i, -1)
    return Minv @ R

# Compute time step (CFL condition)
def compute_time_step(cells, cfl, dx, ndof):
    dt_min = np.inf
    for cell in cells:
        u_avg = cell[0]  # P0 component
        lambda_max = abs(2 * u_avg)  # |f'(u)| = |2u|
        dt_cell = cfl * dx / lambda_max if lambda_max > 0 else np.inf
        dt_min = min(dt_min, dt_cell)
    return dt_min

# RK3 time stepping
def rk3_step(uh, Minv, u_right_prev, u_left_next, dt, dx, ndof, n_nodes=5):
    u1 = uh + dt * residual(uh, Minv, u_right_prev, u_left_next, ndof, n_nodes)
    u2 = 0.75 * uh + 0.25 * u1 + 0.25 * dt * residual(u1, Minv, u_right_prev, u_left_next, ndof, n_nodes)
    u_next = (uh + 2 * u2) / 3 + (2/3) * dt * residual(u2, Minv, u_right_prev, u_left_next, ndof, n_nodes)
    return u_next

# Main simulation
def main():
    ndof = 2  # Polynomial degree
    N_cells = 10  # Number of cells
    dx = 1.0 / N_cells  # Domain [0,1]
    cells = [np.zeros(ndof + 1) for _ in range(N_cells)]
    
    # Initial condition (shock-like)
    for i in range(N_cells):
        if i < N_cells // 2:
            cells[i][0] = 1.0  # Left state
        else:
            cells[i][0] = 0.1  # Right state
    
    M = build_mass_matrix(ndof)
    Minv = np.linalg.inv(M)
    T = 0.2
    t = 0.0
    cfl = 1 / (2 * ndof + 1)  # CFL condition
    
    while t < T:
        dt = compute_time_step(cells, cfl, dx, ndof)
        dt = min(dt, T - t)  # Ensure we don't overshoot T
        cells_next = [np.copy(cell) for cell in cells]
        for i in range(N_cells):
            u_right_prev = 1.0 if i == 0 else compute_uh(cells[i-1], 1, ndof)
            u_left_next = 0.1 if i == N_cells-1 else compute_uh(cells[i+1], -1, ndof)
            cells_next[i] = rk3_step(cells[i], Minv, u_right_prev, u_left_next, dt, dx, ndof)
        cells = cells_next
        t += dt
        print(f"t = {t:.4f}, dt = {dt:.4e}")
    
    # Output final state
    print("\nFinal state:")
    for i in range(N_cells):
        u = cells[i][0]  # P0 component
        print(f"Cell {i}: u = {u:.4f}")

if __name__ == "__main__":
    main()
# %%
