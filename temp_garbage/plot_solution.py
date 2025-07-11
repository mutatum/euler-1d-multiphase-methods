# %%
import pandas as pd
import matplotlib.pyplot as plt

def plot_solution(filename="solution.txt"):
    # Read the data from the file
    data = pd.read_csv(filename)

    # Sort the data by the x-coordinate
    data = data.sort_values(by='x')

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(data['x'], data['rho'], 'o-')
    plt.title("Density")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(data['x'], data['u'], 'o-')
    plt.title("Velocity")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(data['x'], data['pressure'], 'o-')
    plt.title("Pressure")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(data['x'], data['e'], 'o-')
    plt.title("Internal Energy")
    plt.grid(True)

    plt.suptitle(f"DGSEM Results from {filename}")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_solution()

# %%
