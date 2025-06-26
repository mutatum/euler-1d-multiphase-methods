# %%
import matplotlib.pyplot as plt
do_plot = True
import numpy as np

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

def eq_of_state(gamma, rho, e):
    return (gamma-1)*rho*e

def physical_flux(U):
    rho = U[0]
    u = U[1]/rho
    E = U[2]
    e = E/rho - .5*u**2.
    p = eq_of_state(1.4, rho, e)
    return np.array([
        U[1], # rho*u
        U[1] * u + p, #rho*u^2 + pressure
        (E + p) * u, # (rho*E + p) * u
    ])

def rusanov(Ul,Ur,lambda_max):
    return 0.5 * (physical_flux(Ul)+physical_flux(Ur)) - 0.5 * lambda_max * (Ur-Ul)

def compute_max_speed(U):
    cells = U.shape[1]
    gamma = 1.4
    lambda_max = 1e-6
    for i in range(cells+1):
        if i == 0: # reflecting boundaries
            UL = UR = U[:,i,0]
        elif i == cells:
            UR = UL = U[:,i-1,-1]
        else:
            UL = U[:,i-1,-1]
            UR = U[:,i,0]
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
    return lambda_max

def compute_fluxes(U_cells,lambda_max):
    #njit unusable:
    #U_borders = np.pad(U[:,:,[0,-1]].flatten(),pad_width=1,mode='wrap') # getting border value for each cell
    n_fluxes = U_cells.shape[1]+1
    fluxes = np.zeros((3,n_fluxes))
    for i in range(n_fluxes):
        if i == 0: # reflecting boundaries
            UL = UR = U_cells[:,i,0]
        elif i == n_fluxes-1:
            UR = UL = U_cells[:,i-1,-1]
        else:
            UL = U_cells[:,i-1,-1]
            UR = U_cells[:,i,0]
        fluxes[:, i] = rusanov(UL, UR, lambda_max)
    return fluxes

def compute_differentiation_matrix(p): # Need to understand why c_i more
    if p == 0:
        return np.array([[0.0]])
    elif p == 1:
        return np.array([[-.5, .5],[-.5, .5]])
    xi = GLL_quadrature[p][0]
    D = np.zeros((p+1, p+1))
    c = np.ones(p+1)
    c[0] = c[-1] = 2
    for i in range(p+1):
        for j in range(p+1):
            if i!=j:
                D[i,j] = (c[i]/c[j])/(xi[i]-xi[j])
        D[i,i] = -np.sum(D[i,:])
    return D

def DGSEM_residual(Residual, U_cells, D, dx, max_speed, w): # Rhs in dU/dt = Rhs(U)
    p = U_cells.shape[2] - 1
    fluxes = compute_fluxes(U_cells, max_speed)
    for c in range(U_cells.shape[1]): # cells
        for j in range(U_cells.shape[2]): # nodes (p+1)
            vol_term = np.zeros(3)
            for k in range(p+1):
                vol_term += D[j,k] * physical_flux(U_cells[:, c, k])
            Residual[:, c, j] = -vol_term

        F_left = physical_flux(U_cells[:, c, 0])
        F_right = physical_flux(U_cells[:, c, -1])
        Residual[:, c, 0] -= (2.0 / (dx * w[0])) * (F_left - fluxes[:, c])
        Residual[:, c, -1] += (2.0 / (dx * w[-1])) * (F_right - fluxes[:, c+1])

def rk3_SSP(U, D, dx, dt, lambda_max):
    p = U.shape[2] - 1
    xi = np.array(GLL_quadrature[p][0], dtype=np.float64)
    w  = np.array(GLL_quadrature[p][1], dtype=np.float64)

    Residual = np.zeros_like(U)
    DGSEM_residual(Residual, U, D, dx, lambda_max, w)
    # U_1 = U + dt * Residual
    
    # DGSEM_residual(Residual, U_1, D, dx, lambda_max, w)
    # U_2 = 0.75 * U + 0.25 * (U_1 + dt * Residual)
    
    # DGSEM_residual(Residual, U_2, D, dx, lambda_max, w)
    # return (1.0 / 3.0) * U + (2.0 / 3.0) * (U_2 + dt * Residual)
    return U + dt*Residual

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


def main(N=100, p=2, T=0.2, cfl=0.5, test_case='sod'):
    # N cells
    # p polynomial degree of approx in cell
    # T final time sought

    Omega = [0,1] # x domain
    X = np.linspace(*Omega, N+1)
    cell_centers = (X[1:]+X[:-1])/2.0
    dx = (Omega[1]-Omega[0])/N
    U_cells = np.zeros((3, N, p+1))
    
    init_sod(U_cells, cell_centers, dx)

    D = compute_differentiation_matrix(p)
    t=0.0

    while t < T:
        lambda_max = compute_max_speed(U_cells)
        dt = cfl * dx / ((2*p+1)*lambda_max)
        if t + dt > T:
            dt = T - t
        
        U_cells = rk3_SSP(U_cells, D, dx, dt,lambda_max)
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
        pressure = eq_of_state(1.4, rho, e)

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

if __name__ == '__main__':
    main(N=30, T=0.45, p=3, cfl=.2, test_case='sod')

# %%
