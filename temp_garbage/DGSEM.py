# %%
import matplotlib.pyplot as plt
do_plot = True
import numpy as np
from math import prod

""" p: {
    xi: p+1 x nodes,
    w: p+1 corresponding weights
}
"""
GLL_quadrature = [
    [
        np.array([-1.0]),
        np.array([2.0]),
    ],
    [
        np.array([-1.0,1.0]),
        np.array([1.0,1.0]),
    ],
    [
        np.array([-1.0, 0.0, 1.0]),
        np.array([1/3, 4/3, 1/3])
    ],
    [
        np.array([-1.0, -np.sqrt(1/5), np.sqrt(1/5), 1.0]),
        np.array([1/6, 5/6, 5/6, 1/6])
    ],
    [
        np.array([-1.0, -np.sqrt(3/7), 0.0, np.sqrt(3/7), 1.0]),
        np.array([1/10, 49/90, 32/45, 49/90, 1/10])
    ],
    [
        np.array([
            -1.0,
            -np.sqrt(1/3 + 2*np.sqrt(7)/21),
            -np.sqrt(1/3 - 2*np.sqrt(7)/21),
            np.sqrt(1/3 - 2*np.sqrt(7)/21),
            np.sqrt(1/3 + 2*np.sqrt(7)/21),
            1.0
        ]),
        np.array([
            1/15,
            (14 - np.sqrt(7))/30,
            (14 + np.sqrt(7))/30,
            (14 + np.sqrt(7))/30,
            (14 - np.sqrt(7))/30,
            1/15
        ])
    ],
    [
        np.array([
            -1.0,
            -np.sqrt(5/11 + 2*np.sqrt(5/3)/11),
            -np.sqrt(5/11 - 2*np.sqrt(5/3)/11),
            0.0,
            np.sqrt(5/11 - 2*np.sqrt(5/3)/11.),
            np.sqrt(5/11 + 2*np.sqrt(5/3)/11.),
            1.0
        ]),
        np.array([
            1/21,
            (124 - 7*np.sqrt(15))/350.,
            (124 + 7*np.sqrt(15))/350.,
            256/525,
            (124 + 7*np.sqrt(15))/350.,
            (124 - 7*np.sqrt(15))/350.,
            1/21
        ])
    ]
]

def lagrange_basis(xi_eval: float, xi, i: int):
    """
    Build Lagrange polynomials for the given order (p) using GLL quadrature points.
    Mapped on xi \in [-1,1]
    """
    if not (0 <= i < len(xi)):
        raise ValueError("Index (i) is out of range.")
    if not (-1. <= xi_eval <= 1.):
        raise ValueError("xi_eval must be in the range [-1, 1].")
    L = 1.0
    for j in range(len(xi)):
        if j!=i:
            L *= (xi_eval-xi[j])/(xi[i]-xi[j])
    return L

def lagrange_derivative(xi_eval: float, xi, i:int):
    return sum((1.0/(xi[i]-xi[m]))*prod( (xi_eval-xi[k])/(xi[i]-xi[k]) for k in range(len(xi)) if k != i and k!=m) for m in range(len(xi)) if m != i)

def eq_of_state(gamma, rho, e):
    return (gamma-1)*rho*e

def physical_flux(U,gamma):
    rho = U[0]
    u = U[1]/rho
    E = U[2]
    e = E/rho - .5*u**2.
    p = eq_of_state(gamma, rho, e)
    return np.array([
        U[1], # rho*u
        U[1] * u + p, #rho*u^2 + pressure
        (E + p) * u, # (rho*E + p) * u
    ])

def rusanov(Ul, Ur, gamma):
    """Rusanov flux that computes and returns local maximum speed"""
    # Compute local maximum speed
    rhoR = Ur[0]
    rhoL = Ul[0]
    uR = Ur[1]/rhoR
    uL = Ul[1]/rhoL
    ER = Ur[2]
    EL = Ul[2]
    eR = ER/rhoR - .5*uR**2.
    eL = EL/rhoL - .5*uL**2.
    pR = eq_of_state(gamma, rhoR, eR)
    pL = eq_of_state(gamma, rhoL, eL)
    cR = np.sqrt(gamma*pR / rhoR)
    cL = np.sqrt(gamma*pL / rhoL)
    lambda_local = max(abs(uL)+cL, abs(uR)+cR)
    
    # Compute Rusanov flux
    flux = 0.5 * (physical_flux(Ul, gamma) + physical_flux(Ur, gamma)) - 0.5 * lambda_local * (Ur - Ul)
    
    return flux, lambda_local

def compute_fluxes(U_cells, gamma, boundary):
    """Compute fluxes and return global maximum speed"""
    n_fluxes = U_cells.shape[1] + 1
    fluxes = np.zeros((3, n_fluxes))
    lambda_max = 1e-6
    
    # Left boundary
    UR_b = U_cells[:, 0, 0]
    if boundary[0] == 'reflecting':
        UL_b = U_cells[:, 0, 0].copy()
        UL_b[1] *= -1
    elif boundary[0] == 'periodic':
        UL_b = U_cells[:, -1, -1]
    else:
        raise ValueError(f"Unknown left boundary condition: {boundary[0]}")
    
    fluxes[:, 0], lambda_local = rusanov(UL_b, UR_b, gamma)
    lambda_max = max(lambda_max, lambda_local)

    # Internal fluxes
    for i in range(1, n_fluxes - 1):
        UL = U_cells[:, i-1, -1]
        UR = U_cells[:, i, 0]
        fluxes[:, i], lambda_local = rusanov(UL, UR, gamma)
        lambda_max = max(lambda_max, lambda_local)

    # Right boundary
    UL_b = U_cells[:, -1, -1]
    if boundary[1] == 'reflecting':
        UR_b = U_cells[:, -1, -1].copy()
        UR_b[1] *= -1
    elif boundary[1] == 'periodic':
        UR_b = U_cells[:, 0, 0]
    else:
        raise ValueError(f"Unknown right boundary condition: {boundary[1]}")
    
    fluxes[:, -1], lambda_local = rusanov(UL_b, UR_b, gamma)
    lambda_max = max(lambda_max, lambda_local)
    
    return fluxes, lambda_max

def compute_max_speed(U_cells, gamma, boundary):
    """Compute maximum speed by using flux computation"""
    _, lambda_max = compute_fluxes(U_cells, gamma, boundary)
    return lambda_max

def compute_differentiation_matrix(p): # Need to understand why c_i more
    if p == 0:
        return np.array([[0.0]])
    elif p == 1:
        return np.array([[-.5, .5],[-.5, .5]])
    xi = GLL_quadrature[p][0]
    D = np.zeros((p+1, p+1))
    for j in range(p+1):
        for l in range(p+1):
            D[j,l] = lagrange_derivative(xi[j], xi, l)
    return D

def DGSEM_residual(Residual, U_cells, D, dx, w,gamma, boundary): # Rhs in dU/dt = Rhs(U)
    p = U_cells.shape[2] - 1
    fluxes, max_speed = compute_fluxes(U_cells, gamma,boundary)
    for c in range(U_cells.shape[1]): # cells
        F_nodes = np.zeros((3,p+1))
        for node in range(p+1):
            F_nodes[:, node] = physical_flux(U_cells[:,c,node], gamma)
        vol_term = F_nodes @ D.T
        Residual[:, c, :] = -2.0 * vol_term/dx

        F_left = physical_flux(U_cells[:, c, 0],gamma)
        F_right = physical_flux(U_cells[:, c, -1], gamma)
        Residual[:, c, 0] -= (2.0 / (dx * w[0])) * (F_left - fluxes[:, c])
        Residual[:, c, -1] += (2.0 / (dx * w[-1])) * (F_right - fluxes[:, c+1])
    return max_speed

def rk_SSP(U, D, k, dx, max_dt, pre_dt, gamma, boundary):
    p = U.shape[2] - 1
    xi = np.array(GLL_quadrature[p][0], dtype=np.float64)
    w  = np.array(GLL_quadrature[p][1], dtype=np.float64)

    Residual = np.zeros_like(U)
    max_speed = DGSEM_residual(Residual, U, D, dx, w, gamma, boundary)
    dt = min(max_dt, pre_dt / max_speed)
    if k == 1:
        U_1 = U + Residual * dt
        return U_1, dt
    elif k == 2:
        U_1 = U + Residual * dt
        DGSEM_residual(Residual, U_1, D, dx, w, gamma, boundary)
        return (1.0 / 2.0) * U + (1.0 / 2.0) * (U_1 + dt * Residual), dt
    elif k == 3 or k==4 or k==5:
        U_1 = U + Residual * dt
        DGSEM_residual(Residual, U_1, D, dx, w, gamma, boundary)
        U_2 = 0.75 * U + 0.25 * (U_1 + dt * Residual)
        DGSEM_residual(Residual, U_2, D, dx, w, gamma, boundary)
        return (1.0 / 3.0) * U + (2.0 / 3.0) * (U_2 + dt * Residual), dt
    elif k == 4:
        U_1 = U + .25 * dt * Residual
        DGSEM_residual(Residual, U_1, D, dx, w, gamma, boundary)
        U_2 = U_1 + .25 * dt * Residual
        DGSEM_residual(Residual, U_2, D, dx, w, gamma, boundary)
        U_3 = U_2 + .25 * dt * Residual
        DGSEM_residual(Residual, U_3, D, dx, w, gamma, boundary)
        U_4 = (1.0/3.0) * U + (2.0/3.0) * (U_3 + 0.25 * dt * Residual)
        DGSEM_residual(Residual, U_4, D, dx, w, gamma, boundary)
        return 0.5 * U + 0.5 * (U_4 + .25 * dt * Residual), dt
    elif k == 5:
        U_1 = U + .2 * dt * Residual
        DGSEM_residual(Residual, U_1, D, dx, w, gamma, boundary)
        U_2 = U_1 + .2 * dt * Residual
        DGSEM_residual(Residual, U_2, D, dx, w, gamma, boundary)
        U_3 = U_2 + .2 * dt * Residual
        DGSEM_residual(Residual, U_3, D, dx, w, gamma, boundary)
        U_4 = U_3 + .2 * dt * Residual
        DGSEM_residual(Residual, U_4, D, dx, w, gamma, boundary)
        U_5 = .25 * U + .75 * (U_4 + 0.2 * dt * Residual)
        DGSEM_residual(Residual, U_5, D, dx, w, gamma, boundary)
        return 0.2 * U + 0.8 * (U_5 + .2 * dt * Residual), dt
    else:
        raise ValueError(f"Rk for precision {k} is not implemented")

# %%
def init_sod(U_cells, cell_centers, dx):
    _, N_cells, N_internal = U_cells.shape
    p = N_internal - 1
    xi = GLL_quadrature[p][0]
    for i in range(N_cells):
        for j in range(N_internal):
            x = cell_centers[i] + 0.5*dx*xi[j]
            if x < 0.0:
                rho = 1.0
                u = 0.0
                p = 1.0
            else:
                rho = 0.125
                u = 0.0
                p = 0.1
            e = p / ((1.4 - 1) * rho)
            E = p / (1.4-1) + 0.5*rho*u*u
            U_cells[0, i, j] = rho
            U_cells[1, i, j] = rho * u
            U_cells[2, i, j] = E

def init_isentropic(U_cells, cell_centers, dx, factor=0.1):
    _, N_cells, N_internal = U_cells.shape
    p = N_internal - 1
    xi = GLL_quadrature[p][0]
    for i in range(N_cells):
        for j in range(N_internal):
            x = cell_centers[i] + 0.5*dx*xi[j]
            u=0.0
            rho = 1.0+factor*np.sin(np.pi*x)
            p = rho**3.
            E = rho*(0.5*u**2+0.5*rho**2)
            U_cells[0, i, j] = rho
            U_cells[1, i, j] = rho * u
            U_cells[2, i, j] = E


def run_simulation(N=100, p=2, T=0.2, cfl=0.5, test_case='sod', verbose=False):
    """Run DGSEM simulation with the specified parameters"""
    # N cells
    # p polynomial degree of approx in cell
    # T final time sought

    if test_case=='sod':
        Omega = [-.5,.5] # x domain
    elif test_case=='isentropic':
        Omega = [-1., 1.] # isentropic case
    
    X = np.linspace(*Omega, N+1)
    cell_centers = (X[1:]+X[:-1])/2.0
    dx = (Omega[1]-Omega[0])/N
    U_cells = np.zeros((3, N, p+1))

    if test_case == 'sod':
        boundary = ['reflecting']*2
        gamma = 1.4
        init_sod(U_cells, cell_centers, dx)
    elif test_case == 'isentropic':
        boundary = ['periodic']*2
        gamma=3.
        init_isentropic(U_cells, cell_centers, dx)
    else:
        raise ValueError(f"Unknown test case: {test_case}")

    D = compute_differentiation_matrix(p)
    t=0.0

    while t < T:
        if p<5:
            time_precision = p+1
        else:
            time_precision = 5
        pre_dt = cfl * dx / (2*p+1)
        U_cells, dt = rk_SSP(U_cells, D, time_precision, dx, T-t, pre_dt, gamma, boundary)
        t += dt
        if verbose:
            print(f"t = {t:.4f}/{T}, dt = {dt:.4e}")

    # Create x coordinates for plotting
    x_plot = np.zeros((N, p+1))
    for i in range(N):
        x_plot[i,:] = cell_centers[i] + 0.5*dx*GLL_quadrature[p][0]
    
    return U_cells, x_plot, gamma, dx

def plot_results(U_cells, x_plot, gamma, test_case, T):
    """Plot simulation results"""
    rho = U_cells[0,:,:].flatten()
    u = U_cells[1,:,:].flatten() / rho
    E = U_cells[2,:,:].flatten()
    e = E/rho - 0.5 * u**2
    pressure = eq_of_state(gamma, rho, e)

    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.plot(x_plot.flatten(), rho, 'o-')
    plt.title("Density")
    plt.grid(True)

    plt.subplot(2,2,2)
    plt.plot(x_plot.flatten(), u, 'o-')
    plt.title("Velocity")
    plt.grid(True)

    plt.subplot(2,2,3)
    plt.plot(x_plot.flatten(), pressure, 'o-')
    plt.title("Pressure")
    plt.grid(True)
    
    plt.subplot(2,2,4)
    plt.plot(x_plot.flatten(), E/rho, 'o-')
    plt.title("Total Energy")
    plt.grid(True)

    plt.suptitle(f"DGSEM Results for {test_case} at T={T}")
    plt.tight_layout()
    plt.show()

def plot_comparison_with_exact(U_cells, x_plot, gamma, T, solution):
    """Plot comparison between simulation results and exact solution"""
    # Extract numerical solution data
    rho = U_cells[0,:,:].flatten()
    u = U_cells[1,:,:].flatten() / rho
    E = U_cells[2,:,:].flatten()
    e = E/rho - 0.5 * u**2
    pressure = eq_of_state(gamma, rho, e)

    # Compute exact solution
    X_exact = np.linspace(min(x_plot.flatten()), max(x_plot.flatten()), 200)
    Y_exact = np.array([solution(x, T) for x in X_exact])
    rho_exact = Y_exact[:, 0]
    u_exact = Y_exact[:, 1] / rho_exact
    E_exact = Y_exact[:, 2]
    e_exact = E_exact/rho_exact - 0.5 * u_exact**2
    pressure_exact = eq_of_state(gamma, rho_exact, e_exact)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(X_exact, rho_exact, '-', label='Exact')
    plt.plot(x_plot.flatten(), rho, 'o', label='DGSEM')
    plt.title("Density Comparison")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(X_exact, u_exact, '-', label='Exact')
    plt.plot(x_plot.flatten(), u, 'o', label='DGSEM')
    plt.title("Velocity Comparison")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(X_exact, pressure_exact, '-', label='Exact')
    plt.plot(x_plot.flatten(), pressure, 'o', label='DGSEM')
    plt.title("Pressure Comparison")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(X_exact, E_exact/rho_exact, '-', label='Exact')
    plt.plot(x_plot.flatten(), E/rho, 'o', label='DGSEM')
    plt.title("Total Energy Comparison")
    plt.legend()
    plt.grid(True)

    plt.suptitle(f"DGSEM vs Exact Solution at T={T}")
    plt.tight_layout()
    plt.show()

def compute_error(U_cells, x_plot, gamma, T, solution):
    """Compute L2 error between numerical and exact solutions"""
    # Extract numerical density
    x_flat = x_plot.flatten()
    rho_num = U_cells[0,:,:].flatten()
    
    # Compute exact density at the same points
    rho_exact = np.array([solution(x, T)[0] for x in x_flat])
    
    # Compute L2 error
    error = np.sqrt(np.sum((rho_num - rho_exact)**2) / len(rho_num))
    return error

def convergence_analysis(p, T, cfl, test_case, solution):
    """Perform convergence analysis by running simulations with different grid resolutions"""
    N_values = [20, 40, 80, 160]
    errors = []
    dx_values = []

    print(f"Starting convergence analysis for p={p}, T={T}, test_case={test_case}")
    for N in N_values:
        print(f"Running simulation with N={N} cells...")
        U_cells, x_plot, gamma, dx = run_simulation(N=N, p=p, T=T, cfl=cfl, test_case=test_case, verbose=False)
        error = compute_error(U_cells, x_plot, gamma, T, solution)
        errors.append(error)
        dx_values.append(dx)
        print(f"  N={N}, dx={dx:.6f}, L2 error={error:.6e}")

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.loglog(dx_values, errors, 'o-', linewidth=2, markersize=8)
    
    # Fit a line to determine convergence order
    log_dx = np.log(dx_values)
    log_err = np.log(errors)
    slope, intercept = np.polyfit(log_dx, log_err, 1)
    
    # Plot reference lines
    x_ref = np.array([min(dx_values), max(dx_values)])
    for order in range(1, p+3):
        factor = errors[0] / (dx_values[0]**order)
        y_ref = factor * x_ref**order
        plt.loglog(x_ref, y_ref, '--', label=f'Order {order}')
    
    plt.title(f'Convergence Analysis (p={p})\nObserved order: {slope:.2f}')
    plt.xlabel('Cell size (dx)')
    plt.ylabel('L2 Error')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()
    
    # Print convergence rates between consecutive refinements
    print("\nConvergence rates between consecutive refinements:")
    for i in range(1, len(N_values)):
        rate = np.log(errors[i-1]/errors[i]) / np.log(dx_values[i-1]/dx_values[i])
        print(f"  {N_values[i-1]} -> {N_values[i]} cells: order {rate:.2f}")
    
    print(f"Overall convergence rate (least squares fit): {slope:.2f}")
    
    return dx_values, errors, slope

def main(N=100, p=2, T=0.2, cfl=0.5, test_case='sod', solution=None, run_convergence=False, verbose=False):
    """Main function to run simulations and generate plots"""
    if run_convergence:
        if solution is None:
            print("Error: Convergence analysis requires an analytical solution.")
            return
        convergence_analysis(p, T, cfl, test_case, solution)
    else:
        # Run a single simulation
        U_cells, x_plot, gamma, _ = run_simulation(N=N, p=p, T=T, cfl=cfl, test_case=test_case, verbose=verbose)
        
        # Plot results
        if do_plot:
            plot_results(U_cells, x_plot, gamma, test_case, T)
            if solution:
                plot_comparison_with_exact(U_cells, x_plot, gamma, T, solution)

if __name__ == '__main__':
    from isentropic_solution import sol
    
    # Run a single simulation with comparison to exact solution
    # main(N=10, T=0.9, p=6, cfl=0.2, test_case='isentropic', solution=sol, verbose=False)
    
    # Run convergence analysis
    # main(p=2, T=0.7, cfl=0.2, test_case='isentropic', solution=sol, run_convergence=True)
    
    # Uncomment to run Sod shock tube test
    main(N=2000, T=0.13, p=0, cfl=0.15, test_case='sod', verbose = False)
# %%
