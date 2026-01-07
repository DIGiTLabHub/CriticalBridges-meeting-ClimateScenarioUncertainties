import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def profile_fit(d, f, dy_grid):
    best_J = np.inf
    best_params = (None, None, None)
    for dy in dy_grid:
        I1 = d < dy
        I2 = ~I1
        if np.sum(I1) == 0 or np.sum(I2) == 0:
            continue
        # k1 >= 0
        num1 = (d[I1] * f[I1]).sum()
        den1 = (d[I1]**2).sum()
        k1 = max(0, num1 / den1)
        # k2
        num2 = ((d[I2] - dy) * (f[I2] - k1 * dy)).sum()
        den2 = ((d[I2] - dy)**2).sum()
        k2 = num2 / den2
        # objective J
        fhat = np.where(d < dy, k1 * d, k2 * d + (k1 - k2) * dy)
        J = ((f - fhat)**2).sum()
        if J < best_J:
            best_J = J
            best_params = (k1, k2, dy)
    return best_params

def fit_bilinear_profile(d, f, num_grid=200):
    dy_min, dy_max = d.min(), d.max()
    dy_grid = np.linspace(dy_min + 1e-6, dy_max - 1e-6, num_grid)
    k1, k2, dy = profile_fit(d, f, dy_grid)
    fy = k1 * dy
    return k1, k2, dy, fy

if __name__ == "__main__":
    # Hardcoded input file and settings
    excel_file = "Pushover_Colorado.xlsx"
    sheet_name = "scour_1976"
    d_col = "Disp (m)"
    f_col = "Shear (kN)"

    # Read data
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    d = df[d_col].values
    f = df[f_col].values

    # Fit model
    k1_hat, k2_hat, dy_hat, fy_hat = fit_bilinear_profile(d, f)

    # Print results
    print(f"Estimated parameters:\n"
          f"  k1 = {k1_hat:.6f}\n"
          f"  k2 = {k2_hat:.6f}\n"
          f"  d_y = {dy_hat:.6f}\n"
          f"  f_y = {fy_hat:.6f}")

    # Plot data and fitted model
    d_range = np.linspace(d.min(), d.max(), 500)
    f_model = np.where(
        d_range < dy_hat,
        k1_hat * d_range,
        k2_hat * d_range + (k1_hat - k2_hat) * dy_hat
    )

    plt.figure(figsize=(8,6))
    plt.scatter(d, f, label="Data", alpha=0.7)
    plt.plot(
        d_range, f_model,
        color="#0000FF",        # sharp blue
        linewidth=2,
        label="Fitted Model"
    )
    plt.scatter(
        [dy_hat], [fy_hat],
        color="red",
        label="Yield Point",
        zorder=5
    )
    plt.xlabel("d (Disp [m])")
    plt.ylabel("f (Shear [kN])")
    plt.title(f"Bilinear Model Fit  —  $d_y$={dy_hat:.3f},  $f_y$={fy_hat:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
