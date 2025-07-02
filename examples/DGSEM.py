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
        np.array([-1.0, -np.sqrt(1/5), 0.0, np.sqrt(1/5), 1.0]),
        np.array([1/6, 5/6, 5/6, 5/6, 1/6])
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
    return sum((1.0/(xi[i]-xi[j]))*prod( (xi_eval-xi[k])/(xi[j]-xi[k]) for k in range(len(xi)) if k != j and k!=i) for j in range(len(xi)) if j != i)

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

def rusanov(Ul,Ur,lambda_max,gamma):
    return 0.5 * (physical_flux(Ul,gamma)+physical_flux(Ur,gamma)) - 0.5 * lambda_max * (Ur-Ul)

def compute_max_speed(U, gamma, boundary):
    cells = U.shape[1]
    lambda_max = 1e-6

    # Internal interfaces
    for i in range(cells - 1):
        UL = U[:,i,-1]
        UR = U[:,i+1,0]
        rhoR = np.maximum(1e-9,UR[0])
        rhoL = np.maximum(1e-9,UL[0])
        uR = UR[1]/rhoR
        uL = UL[1]/rhoL
        ER = UR[2]
        EL = UL[2]
        eR = ER/rhoR - .5*uR**2.
        eL = EL/rhoL - .5*uL**2.
        pR = np.maximum(eq_of_state(gamma, rhoR, eR),1e-9)
        pL = np.maximum(eq_of_state(gamma, rhoL, eL),1e-9)
        # if pR < 0 or pL < 0 or rhoL <= 0 or rhoR <= 0:
        #     print(f"pR,pL,rhoR, rhoL: {pR,pL,rhoR, rhoL}")
        #     exit()
        cR = np.sqrt(gamma*pR / rhoR)
        cL = np.sqrt(gamma*pL / rhoL)
        lambda_max = max(lambda_max, abs(uL)+cL, abs(uR)+cR)

    # Boundary interfaces
    # Left boundary
    UL_b, UR_b = None, U[:, 0, 0]
    if boundary[0] == 'reflecting':
        UL_b = U[:, 0, 0].copy()
        UL_b[1] *= -1 
    elif boundary[0] == 'periodic':
        UL_b = U[:, -1, -1]
    
    # Right boundary
    UL_r, UR_r = U[:, -1, -1], None
    if boundary[1] == 'reflecting':
        UR_r = U[:, -1, -1].copy()
        UR_r[1] *= -1
    elif boundary[1] == 'periodic':
        UR_r = U[:, 0, 0]

    for U_pair in [(UL_b, UR_b), (UL_r, UR_r)]:
        if U_pair[0] is not None and U_pair[1] is not None:
            UL, UR = U_pair
            rhoR = np.maximum(1e-9,UR[0])
            rhoL = np.maximum(1e-9,UL[0])
            uR = UR[1]/rhoR
            uL = UL[1]/rhoL
            ER = UR[2]
            EL = UL[2]
            eR = ER/rhoR - .5*uR**2.
            eL = EL/rhoL - .5*uL**2.
            pR = np.maximum(eq_of_state(gamma, rhoR, eR),1e-9)
            pL = np.maximum(eq_of_state(gamma, rhoL, eL),1e-9)
            cR = np.sqrt(gamma*pR / rhoR)
            cL = np.sqrt(gamma*pL / rhoL)
            lambda_max = max(lambda_max, abs(uL)+cL, abs(uR)+cR)
            
    return lambda_max

def compute_fluxes(U_cells,lambda_max,gamma,boundary):
    #njit unusable:
    #U_borders = np.pad(U[:,:,[0,-1]].flatten(),pad_width=1,mode='wrap') # getting border value for each cell
    n_fluxes = U_cells.shape[1]+1
    fluxes = np.zeros((3,n_fluxes))
    n_cells = U_cells.shape[1]
    
    # Left boundary
    UR_b = U_cells[:, 0, 0]
    if boundary[0] == 'reflecting':
        UL_b = U_cells[:, 0, 0]
        UL_b[1] = 0.
        UL_b[2] = UL_b[0]/(gamma-1)
    elif boundary[0] == 'periodic':
        UL_b = U_cells[:, -1, -1]
    else:
        raise ValueError(f"Unknown left boundary condition: {boundary[0]}")
    fluxes[:, 0] = rusanov(UL_b, UR_b, lambda_max, gamma)

    # Internal fluxes
    for i in range(1, n_fluxes - 1):
        UL = U_cells[:,i-1,-1]
        UR = U_cells[:,i,0]
        fluxes[:, i] = rusanov(UL, UR, lambda_max,gamma)

    # Right boundary
    UL_b = U_cells[:, -1, -1]
    if boundary[1] == 'reflecting':
        UR_b = U_cells[:, -1, -1]
        UR_b[1] = 0
        UR_b[2] = UR_b[0]/(gamma-1)
    elif boundary[1] == 'periodic':
        UR_b = U_cells[:, 0, 0]
    else:
        raise ValueError(f"Unknown right boundary condition: {boundary[1]}")
    fluxes[:, -1] = rusanov(UL_b, UR_b, lambda_max, gamma)
    
    return fluxes

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

def DGSEM_residual(Residual, U_cells, D, dx, max_speed, w,gamma, boundary): # Rhs in dU/dt = Rhs(U)
    p = U_cells.shape[2] - 1
    fluxes = compute_fluxes(U_cells, max_speed, gamma,boundary)
    for c in range(U_cells.shape[1]): # cells
        for j in range(U_cells.shape[2]): # nodes (p+1)
            vol_term = D[j, :] @ physical_flux(U_cells[:, c, :], gamma).T
            Residual[:, c, j] = -vol_term/dx

        F_left = physical_flux(U_cells[:, c, 0],gamma)
        F_right = physical_flux(U_cells[:, c, -1], gamma)
        Residual[:, c, 0] -= (2.0 / (dx * w[0])) * (F_left - fluxes[:, c])
        Residual[:, c, -1] += (2.0 / (dx * w[-1])) * (F_right - fluxes[:, c+1])

def rk3_SSP(U, D, dx, dt, lambda_max, gamma, boundary):
    p = U.shape[2] - 1
    xi = np.array(GLL_quadrature[p][0], dtype=np.float64)
    w  = np.array(GLL_quadrature[p][1], dtype=np.float64)

    Residual = np.zeros_like(U)
    DGSEM_residual(Residual, U, D, dx, lambda_max, w, gamma, boundary)
    U_1 = U + dt * Residual
    
    DGSEM_residual(Residual, U_1, D, dx, lambda_max, w, gamma, boundary)
    U_2 = 0.75 * U + 0.25 * (U_1 + dt * Residual)
    
    DGSEM_residual(Residual, U_2, D, dx, lambda_max, w, gamma, boundary)
    return (1.0 / 3.0) * U + (2.0 / 3.0) * (U_2 + dt * Residual)

# %%
def init_sod(U_cells, cell_centers, dx):
    _, N_cells, N_internal = U_cells.shape
    p = N_internal - 1
    xi = GLL_quadrature[p][0]
    for i in range(N_cells):
        for j in range(N_internal):
            x = cell_centers[i] + 0.5*dx*xi[j]
            if x < 0.5:
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

def init_isentropic(U_cells, cell_centers, dx):
    _, N_cells, N_internal = U_cells.shape
    p = N_internal - 1
    xi = GLL_quadrature[p][0]
    for i in range(N_cells):
        for j in range(N_internal):
            x = cell_centers[i] + 0.5*dx*xi[j]
            u=0.0
            rho = 1.0+0.9999999*np.sin(np.pi*x)
            p = rho**3.
            E = rho*(0.5*u**2+0.5*rho**2)
            U_cells[0, i, j] = rho
            U_cells[1, i, j] = rho * u
            U_cells[2, i, j] = E


def main(N=100, p=2, T=0.2, cfl=0.5, test_case='sod', solution=None):
    # N cells
    # p polynomial degree of approx in cell
    # T final time sought

    # sod case
    if test_case=='sod':
        Omega = [0,1] # x domain
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

    D = compute_differentiation_matrix(p)
    t=0.0

    while t < T:
        lambda_max = compute_max_speed(U_cells, gamma, boundary)
        dt = cfl * dx / ((2*p+1)*lambda_max)
        if t + dt > T:
            dt = T - t
        
        U_cells = rk3_SSP(U_cells, D, dx, dt,lambda_max, gamma, boundary)
        t += dt
        print(f"t = {t:.4f}/{T}, dt = {dt:.4e}")

    if do_plot:
        x_plot = np.zeros((N, p+1))
        for i in range(N):
            x_plot[i,:] = cell_centers[i] + 0.5*dx*GLL_quadrature[p][0]
        
        rho = U_cells[0,:,:].flatten()
        u = U_cells[1,:,:].flatten() / rho
        E = U_cells[2,:,:].flatten() / rho
        e = E - 0.5 * u**2
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
        plt.plot(x_plot.flatten(), e, 'o-')
        plt.title("Internal Energy")
        plt.grid(True)

        plt.suptitle(f"DGSEM Results for {test_case} at T={T}")
        plt.tight_layout()
        plt.show()
        
        if solution:
            # Compare with the isentropic solution
            X_exact = np.linspace(-1, 1, 100)
            Y_exact = np.array([solution(x, T) for x in X_exact])
            rho_exact = Y_exact[:, 0]
            u_exact = Y_exact[:, 1] / rho_exact
            E_exact = Y_exact[:, 2] / rho_exact
            e_exact = E_exact - 0.5 * u_exact**2
            pressure_exact = eq_of_state(gamma, rho_exact, e_exact)

            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.plot(X_exact, rho_exact, label='Exact')
            plt.plot(x_plot.flatten(), rho, 'o-', label='DGSEM')
            plt.title("Density Comparison")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.plot(X_exact, u_exact, label='Exact')
            plt.plot(x_plot.flatten(), u, 'o-', label='DGSEM')
            plt.title("Velocity Comparison")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.plot(X_exact, pressure_exact, label='Exact')
            plt.plot(x_plot.flatten(), pressure, 'o-', label='DGSEM')
            plt.title("Pressure Comparison")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot(X_exact, e_exact, label='Exact')
            plt.plot(x_plot.flatten(), e, 'o-', label='DGSEM')
            plt.title("Internal Energy Comparison")
            plt.legend()
            plt.grid(True)

            plt.suptitle(f"DGSEM vs Exact Solution at T={T}")
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    from isentropic_solution import sol
    # main(N=30, T=0.1, p=2, cfl=.2, test_case='isentropic', solution=sol)
    main(N=100, T=0.15, p=0, cfl=.2, test_case='sod')
# %%
