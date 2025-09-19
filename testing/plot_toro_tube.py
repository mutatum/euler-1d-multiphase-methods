# %%
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import sys
import os
import matplotlib.animation as animation

sys.path.append(os.path.dirname(__file__))
from TORO_exact_riemann_solver import exact_riemann_solver

# Specify the polynomial order to plot
poly_order = 4  # Change this to match the order used in your simulation
num_displayed_solutions = 6  # Number of first solutions to display

def load_csv(filename):
    """
    Load solution data from a CSV file with metadata in comments.
    Args:
        filename: Path to CSV file
    Returns:
        Dictionary with solution arrays and metadata
    """
    try:
        metadata = {}
        header_lines = 0
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    header_lines += 1
                    if ':' in line:
                        key, value = line[1:].strip().split(':', 1)
                        metadata[key.strip()] = value.strip()
                else:
                    header_lines += 1
                    break
        data = np.loadtxt(filename, delimiter=',', skiprows=header_lines, dtype=np.longdouble)
        result = {
            'cell_index': data[:, 0].astype(int),
            'cell_left': data[:, 1],
            'cell_right': data[:, 2],
            'x': data[:, 3],
            'rho': data[:, 4],
            'momentum': data[:, 5],
            'total_energy': data[:, 6],
            'metadata': metadata
        }
        return result
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def get_toro_exact_arrays(x_array, t, gamma=1.4):
    """
    Use Toro's exact Riemann solver to get density, momentum, total energy arrays.
    """
    left_state = (1.0, 0.0, 1.0)
    right_state = (0.125, 0.0, 0.1)
    if t == 0:
        # Initial discontinuity at x=0.0
        x0 = 0.0
        rho = np.where(x_array < x0, left_state[0], right_state[0])
        u = np.where(x_array < x0, left_state[1], right_state[1])
        p = np.where(x_array < x0, left_state[2], right_state[2])
    else:
        rho, u, p = exact_riemann_solver(left_state, right_state, gamma, x_array, t)
    E = p / (gamma - 1) + 0.5 * rho * u**2
    return rho, rho * u, E

def plot_toro_tube():
    """
    Reads solution data from CSV files and plots the numerical and exact solutions for the Toro tube problem.
    Also computes and plots L2 error and convergence rates.
    """
    def extract_index(fname):
        import re
        match = re.search(r'_final(\d+)_order', fname)
        return int(match.group(1)) if match else -1

    solution_files = sorted(
        glob.glob(f"solution_final*_order{poly_order}.csv"),
        key=extract_index
    )
    num_files = len(solution_files)
    if num_files == 0:
        print(f"No solution files found for polynomial order {poly_order}.")
        return

    numerical_solutions = []
    for fname in solution_files:
        data = load_csv(fname)
        if data is not None:
            num_cells = len(np.unique(data['cell_index']))
            metadata = data.get('metadata', {})
            time_integrator = metadata.get('time_stepper', 'Unknown')
            cfl = metadata.get('cfl', 'Unknown')
            final_time = metadata.get('final_time', 'Unknown')
            poly_order_meta = metadata.get('polynomial_order', 'Unknown')
            label = f"P{poly_order_meta}, {num_cells} cells, CFL={cfl}"
            numerical_solutions.append((data, label, num_cells, metadata))
        else:
            print(f"Warning: Could not load {fname}")

    if not numerical_solutions:
        print("Error: Missing required data files.")
        return

    first_metadata = numerical_solutions[0][3] if numerical_solutions else {}
    sim_poly_order = first_metadata.get('polynomial_order', poly_order)
    final_time = float(first_metadata.get('final_time', '0.1'))

    variables_to_plot = [
        ('rho', 0, 'Density (ρ)'),
        ('momentum', 1, 'Momentum (ρu)'),
        ('total_energy', 2, 'Total Energy (E)')
    ]
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_displayed_solutions))

    # --- Main solution plots ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(f'Toro Shock Tube (Polynomial Order P{sim_poly_order})', fontsize=16, fontweight='bold')
    # Create a high-resolution x grid for the exact solution
    x_min = min([np.min(df_num['x']) for df_num, _, _, _ in numerical_solutions])
    x_max = max([np.max(df_num['x']) for df_num, _, _, _ in numerical_solutions])
    x_exact_highres = np.linspace(x_min, x_max, 2000)  # 2000 points for high resolution

    for var, i, ylabel in variables_to_plot:
        for j, (df_num, label, cell_count, metadata) in enumerate(numerical_solutions[:num_displayed_solutions]):
            axes[i].plot(df_num['x'], df_num[var], 'o-', color=colors[j], 
                        label=label, markersize=3, linewidth=1.5, alpha=0.8)
        # Plot high-resolution exact solution
        rho_exact, mom_exact, E_exact = get_toro_exact_arrays(x_exact_highres, final_time)
        axes[i].plot(x_exact_highres, [rho_exact, mom_exact, E_exact][i], 'r-', linewidth=2.5,
                    label=f'Exact Solution (Toro)', alpha=0.9)
        axes[i].set_ylabel(ylabel, fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=10, framealpha=0.9)
        axes[i].set_title(f'{ylabel} vs Position', fontsize=11, fontweight='bold')
    axes[-1].set_xlabel('Position (x)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def animate_toro_evolution(poly_order=2):
    """
    Animates the evolution of the Toro tube solution over time.
    """
    import re
    solution_files = sorted(
        glob.glob(f"solution_final*_order{poly_order}_t=*.csv"),
        key=lambda fname: float(re.search(r'_t=([0-9\.eE+-]+)\.csv', fname).group(1)) if re.search(r'_t=([0-9\.eE+-]+)\.csv', fname) else 0.0
    )
    if not solution_files:
        print(f"No time-evolution solution files found for polynomial order {poly_order}.")
        return

    # Load all solutions and times
    solutions = []
    times = []
    for fname in solution_files:
        data = load_csv(fname)
        if data is not None:
            solutions.append(data)
            match = re.search(r'_t=([0-9\.eE+-]+)\.csv', fname)
            t = float(match.group(1)) if match else 0.0
            times.append(t)

    def compute_pressure(data, gamma=1.4):
        # p = (gamma - 1) * (E - 0.5 * rho * u^2)
        rho = data['rho']
        mom = data['momentum']
        E = data['total_energy']
        u = mom / rho
        p = (gamma - 1) * (E - 0.5 * rho * u**2)
        return p

    variables_to_plot = [
        ('rho', 'Density (ρ)'),
        ('momentum', 'Momentum (ρu)'),
        ('total_energy', 'Total Energy (E)'),
        ('pressure', 'Pressure (p)')
    ]
    colors = ['b', 'g', 'm', 'orange']

    fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
    fig.suptitle(f'Toro Shock Tube Evolution (Polynomial Order P{poly_order})', fontsize=16, fontweight='bold')

    # Create empty lines for numerical and exact solutions
    lines = []
    dots = []
    exact_lines = []
    for ax, (var, ylabel), color in zip(axes, variables_to_plot, colors):
        line, = ax.plot([], [], '-', lw=2, label='Numerical', color=color)
        dot, = ax.plot([], [], 'o', markersize=6, alpha=0.8, label='Points', color=color)
        exact_line, = ax.plot([], [], 'r--', lw=2, alpha=0.7, label='Exact')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.set_title(f'{ylabel} vs Position', fontsize=11, fontweight='bold')
        lines.append(line)
        dots.append(dot)
        exact_lines.append(exact_line)
    axes[-1].set_xlabel('Position (x)', fontsize=12, fontweight='bold')

    def update(frame):
        data = solutions[frame]
        t = times[frame]
        x = data['x']
        pressure = compute_pressure(data)
        for i, (var, _) in enumerate(variables_to_plot):
            if var == 'pressure':
                y = pressure
            else:
                y = data[var]
            lines[i].set_data(x, y)
            dots[i].set_data(x, y)  # Add dots at data points
            axes[i].set_xlim(np.min(x), np.max(x))
            axes[i].set_ylim(np.min(y), np.max(y) * 1.1)
            # Update exact solution line
            x_exact = np.linspace(np.min(x), np.max(x), 2000)
            rho_exact, mom_exact, E_exact = get_toro_exact_arrays(x_exact, t)
            if var == 'pressure':
                u_exact = mom_exact / rho_exact
                p_exact = (1.4 - 1) * (E_exact - 0.5 * rho_exact * u_exact**2)
                exact_lines[i].set_data(x_exact, p_exact)
            else:
                exact_lines[i].set_data(x_exact, [rho_exact, mom_exact, E_exact][i])
        fig.suptitle(f'Toro Shock Tube Evolution (P{poly_order})   t = {t:.4f}', fontsize=16, fontweight='bold')
        return lines + dots + exact_lines

    ani = animation.FuncAnimation(fig, update, frames=len(solutions), blit=False, interval=200, repeat=True)
    ani.save("animation_toro_tube.gif", writer='pillow', fps=5)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

if __name__ == "__main__":
    # plot_toro_tube()
    # To show animation, uncomment the next line:
    animate_toro_evolution(poly_order=5)
