# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

import argparse


# Optionally allow CLI arguments for flexibility
def parse_args():
    parser = argparse.ArgumentParser(description="Plot solutions for multiple schemes.")
    parser.add_argument('--poly_order', type=int, default=2, help='Polynomial order to plot')
    parser.add_argument('--num_displayed_solutions', type=int, default=3, help='Number of solutions to display per scheme')
    parser.add_argument('--schemes', nargs='+', default=schemes, help='List of schemes to plot')
    parser.add_argument('--save_pdf', action='store_true', help='Save plots as PDF')
    # Accept unknown args to avoid errors in Jupyter/IPython
    args, unknown = parser.parse_known_args()
    return args


def newton_raphson(x0, f, df, eps):
    
    x = x0
    max_it=1500
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


def plot_solutions(poly_order=2, num_displayed_solutions=3, schemes=None, save_pdf=False):
    """
    Reads solution data from CSV files for all specified schemes and plots them together.
    """
    if schemes is None:
        schemes = ["DG", "DGSEM", "DGSEM_ES"]

    # Data to plot: (column_name, axis_index, y_label)
    variables_to_plot = [
        ('rho', 0, 'Density (ρ)'),
        ('momentum', 1, 'Momentum (ρu)'),
        ('total_energy', 2, 'Total Energy (E)')
    ]

    # Assign a color for each scheme
    scheme_colors = {
        "DG": "tab:blue",
        "DGSEM": "tab:green",
        "DGSEM_ES": "tab:purple"
    }
    # Fallback for unknown schemes
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(schemes)))

    # Load all numerical solutions for each scheme
    all_schemes_data = {}
    for idx, scheme in enumerate(schemes):
        def extract_index(fname):
            import re
            match = re.search(r'_N(\d+)_T', fname)
            return int(match.group(1)) if match else -1

        # Find all files for this scheme and poly_order
        pattern = f"sol_{scheme}_P{poly_order}_N*_T*.csv"
        files = sorted(glob.glob(pattern), key=extract_index)
        if not files:
            print(f"No files found for scheme {scheme} and poly_order {poly_order}.")
            continue
        scheme_data = []
        for fname in files:
            data = load_csv(fname)
            if data is not None:
                num_cells = len(np.unique(data['cell_index']))
                metadata = data.get('metadata', {})
                time_integrator = metadata.get('time_stepper', 'Unknown')
                cfl = metadata.get('cfl', 'Unknown')
                final_time = metadata.get('final_time', 'Unknown')
                poly_order_meta = metadata.get('polynomial_order', poly_order)
                label = f"{scheme} P{poly_order_meta}, {num_cells} cells, CFL={cfl}"
                scheme_data.append((data, label, num_cells, metadata))
            else:
                print(f"Warning: Could not load {fname}")
        if scheme_data:
            all_schemes_data[scheme] = scheme_data

    # Load exact solution from CSV for plotting (if available)
    exact_csv_file = f"sol_exact_DG_P{poly_order}_N*_T*.csv"
    exact_files = glob.glob(exact_csv_file)
    use_csv_exact = False
    exact_data = None
    if exact_files:
        # Use the one with the largest N (finest grid)
        def extract_n(fname):
            import re
            match = re.search(r'_N(\d+)_T', fname)
            return int(match.group(1)) if match else -1
        exact_files = sorted(exact_files, key=extract_n)
        exact_data = load_csv(exact_files[-1])
        if exact_data is not None:
            use_csv_exact = True
    if not use_csv_exact:
        print("Warning: Could not load exact solution file. Using internal analytical solution for all plots.")

    # --- Main solution plots ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(
        f'Comparison of Schemes for Euler Equations (P{poly_order}, {basis} basis, {quadrature} quadrature)',
        fontsize=16, fontweight='bold'
    )
    for var, i, ylabel in variables_to_plot:
        # Plot analytical solution first
        if use_csv_exact and exact_data is not None:
            x_ana = exact_data['x']
            y_ana = exact_data[var]
            axes[i].plot(x_ana, y_ana, color='red', linewidth=2.5, 
                         label=f'Exact Solution (CSV)', alpha=0.7, zorder=1)
            final_time = float(exact_data['metadata'].get('final_time', '0.1'))
        else:
            # Use the first available scheme's finest grid for x
            first_scheme = next(iter(all_schemes_data.values())) if all_schemes_data else []
            if first_scheme:
                x_ana = np.sort(first_scheme[0][0]['x'])
                final_time = float(first_scheme[0][3].get('final_time', '0.1'))
            else:
                x_ana = np.linspace(-1, 1, 200)
                final_time = 0.1
            ana_vals = np.array([sol(x, final_time, 0.1) for x in x_ana])
            axes[i].plot(x_ana, ana_vals[:,i], color='red', linewidth=2.5, 
                         label=f'Analytical Solution', alpha=0.7, zorder=1)
        # Plot numerical solutions for each scheme
        for sidx, scheme in enumerate(schemes):
            scheme_data = all_schemes_data.get(scheme, [])
            color = scheme_colors.get(scheme, default_colors[sidx % len(default_colors)])
            for j, (df_num, label, cell_count, metadata) in enumerate(scheme_data[:num_displayed_solutions]):
                axes[i].plot(df_num['x'], df_num[var], 'o-', color=color, 
                             label=f'{label}', markersize=3, linewidth=1.5, alpha=0.8, zorder=2)
        axes[i].set_ylabel(ylabel, fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=10, framealpha=0.9)
        axes[i].set_title(f'{ylabel} vs Position', fontsize=11, fontweight='bold')
    axes[-1].set_xlabel('Position (x)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_pdf:
        fig.savefig(f"main_solutions_all_schemes_P{poly_order}.pdf", bbox_inches='tight')
    plt.show()

    # --- Difference plots ---
    fig_diff, axes_diff = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    fig_diff.suptitle(
        f'Numerical Error vs Analytical Solution (All Schemes, P{poly_order}, {basis} basis, {quadrature} quadrature)',
        fontsize=16, fontweight='bold'
    )
    for var, i, ylabel in variables_to_plot:
        for sidx, scheme in enumerate(schemes):
            scheme_data = all_schemes_data.get(scheme, [])
            color = scheme_colors.get(scheme, default_colors[sidx % len(default_colors)])
            for j, (df_num, label, cell_count, metadata) in enumerate(scheme_data[:num_displayed_solutions]):
                x_num = df_num['x']
                y_num = df_num[var]
                # Interpolate analytical solution at numerical points
                ana_vals = np.array([sol(x, final_time, 0.1) for x in x_num])
                diff = y_num - ana_vals[:,i]
                axes_diff[i].plot(x_num, diff, color=color, label=f'{label}', linewidth=1.5, alpha=0.8)
        axes_diff[i].set_ylabel(f'Error in {ylabel}', fontsize=12, fontweight='bold')
        axes_diff[i].grid(True, alpha=0.3)
        axes_diff[i].legend(fontsize=10, framealpha=0.9)
        axes_diff[i].set_title(f'Numerical - Analytical {ylabel}', fontsize=11, fontweight='bold')
        axes_diff[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes_diff[-1].set_xlabel('Position (x)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_pdf:
        fig_diff.savefig(f"error_plots_all_schemes_P{poly_order}.pdf", bbox_inches='tight')
    plt.show()

    # --- Log-Log L2 Error vs dx plot ---
    fig_loglog, axes_loglog = plt.subplots(1, 3, figsize=(18, 6))
    fig_loglog.suptitle(
        f'Convergence Analysis: L2 Error vs Grid Spacing (All Schemes, P{poly_order}, {basis} basis, {quadrature} quadrature)',
        fontsize=14, fontweight='bold'
    )
    for idx, (var, _, ylabel) in enumerate(variables_to_plot):
        # Define linestyles and markers for schemes
        scheme_linestyles = ['-', '--', '-.', ':']
        scheme_markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h', '1']
        for sidx, scheme in enumerate(schemes[::-1]):
            scheme_data = all_schemes_data.get(scheme, [])
            color = scheme_colors.get(scheme, default_colors[sidx % len(default_colors)])
            linestyle = scheme_linestyles[sidx % len(scheme_linestyles)]
            marker = scheme_markers[sidx % len(scheme_markers)]
            dxs = []
            l2_errors = []
            cell_counts = []
            labels_for_loglog = []
            for df_num, label, cell_count, metadata in scheme_data:
                cell_lefts = df_num['cell_left']
                cell_rights = df_num['cell_right']
                cell_indices = df_num['cell_index']
                unique_cells, unique_indices = np.unique(cell_indices, return_index=True)
                dx = np.mean(cell_rights[unique_indices] - cell_lefts[unique_indices])
                dxs.append(dx)
                cell_counts.append(cell_count)
                labels_for_loglog.append(f"{scheme} {cell_count} cells")
                # L2 error
                x_num = df_num['x']
                sort_indices = np.argsort(x_num)
                x_num_sorted = x_num[sort_indices]
                ana_vals = np.array([sol(x, final_time, 0.1) for x in x_num_sorted])
                y_num_sorted = df_num[var][sort_indices]
                y_exact = ana_vals[:, idx]
                diff = y_num_sorted - y_exact
                l2 = np.sqrt(np.trapz(diff**2, x_num_sorted))
                l2_errors.append(l2)
            # Sort by dx
            sort_idx = np.argsort(dxs)[::-1]
            dxs_sorted = np.array(dxs)[sort_idx]
            l2_errors_sorted = np.array(l2_errors)[sort_idx]
            labels_sorted = np.array(labels_for_loglog)[sort_idx]
            axes_loglog[idx].scatter(
                dxs_sorted, l2_errors_sorted, 
                c=[color], marker=marker, s=60, alpha=0.6, edgecolors='black', linewidth=1, label=f'{scheme}'
            )
            axes_loglog[idx].plot(
                dxs_sorted, l2_errors_sorted, linestyle, color=color, alpha=0.6, linewidth=1.5
            )
            # Annotate points
            # for i, (dx, err, label_ll) in enumerate(zip(dxs_sorted, l2_errors_sorted, labels_sorted)):
            #     if not np.isnan(err):
            #         axes_loglog[idx].annotate(label_ll, (dx, err), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, fontweight='bold')
        axes_loglog[idx].set_xscale('log')
        axes_loglog[idx].set_yscale('log')
        axes_loglog[idx].set_xlabel('Grid Spacing (Δx)', fontsize=11, fontweight='bold')
        axes_loglog[idx].set_ylabel(f'L2 Error in {ylabel}', fontsize=11, fontweight='bold')
        axes_loglog[idx].set_title(f'{ylabel} Convergence', fontsize=12, fontweight='bold')
        axes_loglog[idx].grid(True, which="both", ls="--", alpha=0.4)
        # Add reference lines for convergence orders
        dx_ref = np.array(sorted(np.concatenate([np.array([dx for dx in np.array(dxs)]) for dxs in [dxs]])))
        if len(dx_ref) > 0 and len(dx_ref.shape) == 1:
            x0 = dx_ref[-1]
            y0 = l2_errors_sorted[0] if len(l2_errors_sorted) > 0 else 1.0
            colors_ref = ['gray', 'orange', 'green', 'purple', 'brown', 'red', 'blue', 'pink', 'olive', 'cyan']
            for order in range(1, poly_order + 3):
                color_idx = (order - 1) % len(colors_ref)
                ref_line = y0 * (dx_ref / x0) ** order
                axes_loglog[idx].plot(dx_ref, ref_line, linestyle='-', color=colors_ref[color_idx], label=f'O(h^{order})', linewidth=1.5, alpha=0.7)
        axes_loglog[idx].legend(fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_pdf:
        fig_loglog.savefig(f"convergence_loglog_all_schemes_P{poly_order}.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Specify the polynomial order to plot
    poly_order = 7  # Default, can be overridden by CLI

    # Number of first solutions to display on main and error plots
    num_displayed_solutions = 1  # Default, can be overridden by CLI

    quadrature = 'GLL' + str(poly_order)
    basis = 'Lagrange'

    # List of schemes to plot (must match output file naming convention)
    schemes = ["DG", "DGSEM", "DGSEM_ES"]
    plot_solutions(poly_order=poly_order, num_displayed_solutions=num_displayed_solutions, schemes=schemes, save_pdf=True)

