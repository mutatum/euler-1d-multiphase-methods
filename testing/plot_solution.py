# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Specify the polynomial order to plot
poly_order = 12 # Change this to match the order used in your simulation

# Number of first solutions to display on main and error plots
num_displayed_solutions = 5  # Change this to control how many are shown on main/error plots

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


def load_csv(filename):
    # Assumes CSV with header: cell_index,cell_left,cell_right,x,rho,momentum,total_energy
    try:
        # Extract metadata from header comments
        metadata = {}
        header_lines = 0
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    header_lines += 1
                    # Parse metadata from comments
                    if ':' in line:
                        key, value = line[1:].strip().split(':', 1)
                        metadata[key.strip()] = value.strip()
                else:
                    header_lines += 1  # Add one more for the column header line
                    break
        
        data = np.loadtxt(filename, delimiter=',', skiprows=header_lines, dtype=np.float64)
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

def plot_solutions():
    """
    Reads solution data from CSV files and plots the numerical, exact,
    and initial solutions for density, momentum, and total energy.
    """
    # Automatically find all solution_final*_order{poly_order}.csv files
    def extract_index(fname):
        # Extract the integer between '_final' and '_order'
        import re
        match = re.search(r'_final(\d+)_order', fname)
        return int(match.group(1)) if match else -1

    # Sort solution files by the extracted index in ascending order (coarse to fine)
    solution_files = sorted(
        glob.glob(f"solution_final*_order{poly_order}.csv"),
        key=extract_index
    )
    # Reverse the order so that files are from fine to coarse (largest to smallest index)
    # solution_files = solution_files[::-1]

    num_files = len(solution_files)
    if num_files == 0:
        print(f"No solution files found for polynomial order {poly_order}.")
        return

    # Load numerical solutions
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

    # Use analytical solution for plotting and error analysis
    # Extract simulation parameters from the first numerical solution
    first_metadata = numerical_solutions[0][3] if numerical_solutions else {}
    sim_poly_order = first_metadata.get('polynomial_order', poly_order)
    final_time = float(first_metadata.get('final_time', '0.1'))

    # Data to plot: (column_name, axis_index, y_label)
    variables_to_plot = [
        ('rho', 0, 'Density (ρ)'),
        ('momentum', 1, 'Momentum (ρu)'),
        ('total_energy', 2, 'Total Energy (E)')
    ]

    colors = plt.cm.viridis(np.linspace(0, 0.8, num_displayed_solutions))

    # --- Main solution plots ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(f'DG Scheme for Euler Equations (Polynomial Order P{sim_poly_order})', fontsize=16, fontweight='bold')
    for var, i, ylabel in variables_to_plot:
        for j, (df_num, label, cell_count, metadata) in enumerate(numerical_solutions[:num_displayed_solutions]):
            axes[i].plot(df_num['x'], df_num[var], 'o-', color=colors[j], 
                        label=label, markersize=3, linewidth=1.5, alpha=0.8)
        # Plot analytical solution
        x_ana = np.sort(numerical_solutions[0][0]['x'])
        ana_vals = np.array([sol(x, final_time, 0.1) for x in x_ana])
        axes[i].plot(x_ana, ana_vals[:,i], 'r-', linewidth=2.5, 
                    label=f'Analytical Solution (t={final_time})', alpha=0.9)
        axes[i].set_ylabel(ylabel, fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=10, framealpha=0.9)
        axes[i].set_title(f'{ylabel} vs Position', fontsize=11, fontweight='bold')
    axes[-1].set_xlabel('Position (x)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    # --- Difference plots ---
    fig_diff, axes_diff = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    fig_diff.suptitle(f'Numerical Error vs Analytical Solution (P{sim_poly_order})', fontsize=16, fontweight='bold')
    for var, i, ylabel in variables_to_plot:
        for j, (df_num, label, cell_count, metadata) in enumerate(numerical_solutions[:num_displayed_solutions]):
            x_num = df_num['x']
            y_num = df_num[var]
            # Interpolate analytical solution at numerical points
            ana_vals = np.array([sol(x, final_time, 0.1) for x in x_num])
            diff = y_num - ana_vals[:,i]
            axes_diff[i].plot(x_num, diff, color=colors[j], label=label, linewidth=1.5, alpha=0.8)
        axes_diff[i].set_ylabel(f'Error in {ylabel}', fontsize=12, fontweight='bold')
        axes_diff[i].grid(True, alpha=0.3)
        axes_diff[i].legend(fontsize=10, framealpha=0.9)
        axes_diff[i].set_title(f'Numerical - Analytical {ylabel}', fontsize=11, fontweight='bold')
        axes_diff[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes_diff[-1].set_xlabel('Position (x)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    # --- Log-Log L2 Error vs dx plot ---
    # For each solution, compute dx and L2 error for each variable
    dxs = []
    l2_errors = {var: [] for var, _, _ in variables_to_plot}
    cell_counts = []
    labels_for_loglog = []

    # Get final time from exact solution metadata
    final_time = float(first_metadata.get('final_time', '0.1'))

    for df_num, label, cell_count, metadata in numerical_solutions:
        # Compute dx as the mean cell width
        cell_lefts = df_num['cell_left']
        cell_rights = df_num['cell_right']
        cell_indices = df_num['cell_index']
        unique_cells, unique_indices = np.unique(cell_indices, return_index=True)
        dx = np.mean(cell_rights[unique_indices] - cell_lefts[unique_indices])
        dxs.append(dx)
        cell_counts.append(cell_count)
        time_integrator = metadata.get('time_stepper', 'Unknown')
        cfl = metadata.get('cfl', 'Unknown')
        labels_for_loglog.append(f"{cell_count} cells")

        # --- Corrected L2 Error Calculation using analytical solution ---
        x_num = df_num['x']
        
        # Ensure x coordinates are sorted for integration
        sort_indices = np.argsort(x_num)
        x_num_sorted = x_num[sort_indices]

        # Compute analytical solution at each grid point
        rho_exact = []
        momentum_exact = []
        energy_exact = []
        
        for x in x_num_sorted:
            rho, mom, E = sol(x, final_time, 0.1)
            rho_exact.append(rho)
            momentum_exact.append(mom)
            energy_exact.append(E)
        
        rho_exact = np.array(rho_exact)
        momentum_exact = np.array(momentum_exact)
        energy_exact = np.array(energy_exact)

        for var, _, _ in variables_to_plot:
            # Get the numerical solution values and sort them
            y_num_sorted = df_num[var][sort_indices]
            
            # Use the correct analytical solution
            if var == 'rho':
                y_exact = rho_exact
            elif var == 'momentum':
                y_exact = momentum_exact
            elif var == 'total_energy':
                y_exact = energy_exact
            
            # Calculate the difference and then the L2 norm using the trapezoidal rule
            diff = y_num_sorted - y_exact
            l2 = np.sqrt(np.trapz(diff**2, x_num_sorted))
            l2_errors[var].append(l2)

    # Sort all arrays by dx (from coarse to fine grid)
    sort_idx = np.argsort(dxs)[::-1]  # Largest dx first (coarsest grid)
    dxs_sorted = np.array(dxs)[sort_idx]
    cell_counts_sorted = np.array(cell_counts)[sort_idx]
    labels_for_loglog_sorted = np.array(labels_for_loglog)[sort_idx]
    l2_errors_sorted = {var: np.array(l2_errors[var])[sort_idx] for var in l2_errors}
    
    # Debug: Print sorted data to verify correct pairing
    # print(f"\nDebug: Sorted dx and errors for variable 'rho':")
    # for i, (dx, err) in enumerate(zip(dxs_sorted, l2_errors_sorted['rho'])):
    #     print(f"Index {i}: dx={dx:.5e}, L2 error={err:.5e}, cells={cell_counts_sorted[i]}")
    # print()

    fig_loglog, axes_loglog = plt.subplots(1, 3, figsize=(18, 6))
    time_integrator = numerical_solutions[0][3].get('time_stepper', 'Unknown') if numerical_solutions else 'Unknown'
    cfl = numerical_solutions[0][3].get('cfl', 'Unknown') if numerical_solutions else 'Unknown'
    fig_loglog.suptitle(f'Convergence Analysis: L2 Error vs Grid Spacing (P{sim_poly_order}, {time_integrator}, CFL={cfl})', 
                       fontsize=14, fontweight='bold')
    
    for idx, (var, _, ylabel) in enumerate(variables_to_plot):
        axes_loglog[idx].scatter(dxs_sorted, l2_errors_sorted[var], c='black', marker='o', s=60, alpha=0.8, edgecolors='black', linewidth=1)
        axes_loglog[idx].plot(dxs_sorted, l2_errors_sorted[var], 'k--', alpha=0.6, linewidth=1.5)
        axes_loglog[idx].set_xscale('log')
        axes_loglog[idx].set_yscale('log')
        axes_loglog[idx].set_xlabel('Grid Spacing (Δx)', fontsize=11, fontweight='bold')
        axes_loglog[idx].set_ylabel(f'L2 Error in {ylabel}', fontsize=11, fontweight='bold')
        axes_loglog[idx].set_title(f'{ylabel} Convergence', fontsize=12, fontweight='bold')
        
        # Annotate points with cell counts
        for i, (dx, err, label_ll) in enumerate(zip(dxs_sorted, l2_errors_sorted[var], labels_for_loglog_sorted)):
            if not np.isnan(err):
                axes_loglog[idx].annotate(label_ll, (dx, err), textcoords="offset points", 
                                        xytext=(0,8), ha='center', fontsize=9, fontweight='bold')
        axes_loglog[idx].grid(True, which="both", ls="--", alpha=0.4)

        # Add reference lines for convergence orders
        dx_ref = np.array(sorted(dxs_sorted))
        if len(dx_ref) > 0 and len(l2_errors_sorted[var]) > 0:
            # Align reference line with the coarsest grid (largest dx)
            x0 = dx_ref[-1]
            y0 = l2_errors_sorted[var][np.argmax(dxs_sorted)] if not np.isnan(l2_errors_sorted[var][np.argmax(dxs_sorted)]) else 1.0
            colors_ref = ['gray', 'orange', 'green', 'purple', 'brown', 'red', 'blue', 'pink', 'olive', 'cyan']
            for order in range(1, poly_order + 2):
                color_idx = (order - 1) % len(colors_ref)
                ref_line = y0 * (dx_ref / x0) ** order
                axes_loglog[idx].plot(dx_ref, ref_line, '--', color=colors_ref[color_idx], 
                                    label=f'O(h^{order})', linewidth=1.5, alpha=0.7)
        axes_loglog[idx].legend(fontsize=9, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    # Print L2 errors and convergence ratios
    # print("\nL2 Errors and Convergence Ratios:")
    # for var in l2_errors_sorted:
    #     print(f"\nVariable: {var}")
    #     errs = l2_errors_sorted[var]
    #     dxs_log = dxs_sorted
    #     for i, (dx, err) in enumerate(zip(dxs_log, errs)):
    #         print(f"dx={dx:.5e}, L2 error={err:.5e}")
    #         if i > 0:
    #             # Calculate convergence rate: log(err_prev/err_curr) / log(dx_prev/dx_curr)
    #             ratio = np.log(errs[i-1]/err) / np.log(dxs_log[i-1]/dx)
    #             print(f"  Convergence rate (order) from previous: {ratio:.3f}")

plot_solutions()