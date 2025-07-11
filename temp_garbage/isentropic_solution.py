# %%
import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(x0, f, df, eps):
    
    x = x0
    max_it=800
    while abs(f(x)) > eps:
        x -= f(x)/df(x)
        max_it-=1
        if max_it<=0:
            print("Max iterations reached in Newton-Raphson method")
            break
    return x
    
def rho0(X, factor=0.1):
    return 1 + factor * np.sin(np.pi*X)

def sol(x, t, factor=0.1):
    fp = lambda X, x=x, t=t: X + np.sqrt(3) * t * (1+factor*np.sin(np.pi*X)) - x
    dfp = lambda X, x=x ,t=t: 1 + np.sqrt(3) * t * np.pi*factor*np.cos(np.pi*X)
    fm = lambda X, x=x, t=t: X - np.sqrt(3) * t * (1+factor*np.sin(np.pi*X)) - x
    dfm = lambda X, x=x ,t=t: 1 - np.sqrt(3) * t * np.pi*factor*np.cos(np.pi*X)
    Xp = newton_raphson(x, fp, dfp, np.finfo(float).eps)
    Xp = Xp - 2 * np.floor((Xp + 1) / 2)
    Xm = newton_raphson(x, fm, dfm, np.finfo(float).eps)
    Xm = Xm - 2 * np.floor((Xm + 1) / 2)

    # print("Xp: ", Xp, "Xm: ", Xm)
    wp = np.sqrt(3) * (1+factor*np.sin(np.pi*Xp))
    wm = -np.sqrt(3) * (1+factor*np.sin(np.pi*Xm))

    u = (wp+wm)*0.5
    rho = (wp-wm)/(2*np.sqrt(3))
    E = rho*(.5*u**2 + .5*rho**2)

    return np.array([rho,rho*u,E])

# # Time evolution parameters
# t_start = 0.0
# t_end = 1.0/(np.pi*np.sqrt(3)) - .01
# num_time_steps = 10  # Adjust for smoother or coarser time evolution

# # Spatial domain
# X = np.linspace(-1, 1, 100)  # Keep the spatial discretization

# # Initialize lists to store solutions at each time step
# rho_solutions = []
# u_solutions = []
# p_solutions = []
# c_solutions = []

# # Time evolution loop
# time_points = np.linspace(t_start, t_end, num_time_steps)
# for t in time_points:
#     Y = np.array([sol(x, t) for x in X])
#     rho_solutions.append(Y[:, 0])  # Collect rho values
#     u_solutions.append(Y[:, 1])    # Collect u values
#     p_solutions.append(Y[:, 2])    # Collect p values
#     c_solutions.append(np.sqrt(3)*(Y[:,0]))    # Collect c values

# # Plotting the time evolution of each state variable
# plt.figure(figsize=(12, 8))

# # Plot rho(x, t)
# plt.subplot(2, 2, 1)
# for i, t in enumerate(time_points):
#     plt.plot(X, rho_solutions[i], label=f't = {t:.2f}')
# plt.xlabel('x')
# plt.ylabel('Density (rho)')
# plt.title('Time Evolution of Density')
# plt.legend()

# # Plot u(x, t)
# plt.subplot(2, 2, 2)
# for i, t in enumerate(time_points):
#     plt.plot(X, u_solutions[i], label=f't = {t:.2f}')
# plt.xlabel('x')
# plt.ylabel('Velocity (u)')
# plt.title('Time Evolution of Velocity')
# plt.legend()

# # Plot p(x, t)
# plt.subplot(2, 2, 3)
# for i, t in enumerate(time_points):
#     plt.plot(X, p_solutions[i], label=f't = {t:.2f}')
# plt.xlabel('x')
# plt.ylabel('Pressure (p)')
# plt.title('Time Evolution of Pressure')
# plt.legend()

# # Plot c(x, t)
# plt.subplot(2, 2, 4)
# for i, t in enumerate(time_points):
#     plt.plot(X, c_solutions[i], label=f't = {t:.2f}')
# plt.xlabel('x')
# plt.ylabel('Wave Speed (c)')
# plt.title('Time Evolution of Wave Speed')
# plt.legend()

# plt.tight_layout()
# plt.show()
# %%
