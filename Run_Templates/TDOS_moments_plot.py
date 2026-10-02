# Import dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
from KPM.measure import calculate_LDOS_from_moments

# ==============================================================================
# PARAMETERS
# ==============================================================================
wts_list = [0.10, 0.20, 0.30, 0.40, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]
eval_list = [3.12, 3.12, 3.12, 3.12, 3.12, 3.50, 4.00, 5.00, 5.50, 5.75, 6.00, 6.25, 6.50, 7.00, 7.5]
dir_name = ['010', '020', '030', '040', '050', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']

# --- NEW: Target moments for extrapolation ---
moment_list = [512, 1024, 2048, 4096, 8192, 16384]

data_dir = './numericals/LDOS_p10_n7'
data_name_template = 'LDOS_mass_{}.npy'
pval = 10
total_num_dr = 60
num_chunks = 7

save_filename = 'compiled_TDOS_extrapolation.npz'
full_path = os.path.join(data_dir, save_filename)

# ==============================================================================
# COMPUTATION & CACHING
# ==============================================================================

if os.path.exists(full_path):
    print(f"Found existing TDOS extrapolation cache: '{save_filename}'. Loading directly...")
    data = np.load(full_path)
    whole_tdos_matrix = data['whole_tdos_matrix']

else:
    print("TDOS extrapolation cache not found. Commencing fast-slice computations...")

    # We will store the global TDOS in a 2D array: Rows = moments, Cols = W_M
    whole_tdos_matrix = np.zeros((len(moment_list), len(wts_list)))

    # Iterating over all weights
    for ind, wt in enumerate(wts_list):
        print(f"Processing W_M = {wt:.2f}...")

        # Dictionary to temporarily hold run data for each moment cut
        runs_per_moment = {Nm: [] for Nm in moment_list}

        for i in range(total_num_dr):
            try:
                data = np.load(data_dir + '/' + dir_name[ind] + '/' + data_name_template.format(i))
            except FileNotFoundError:
                continue # Skip if a file is missing

            if np.any(np.isnan(data)) or np.max(np.abs(data) > 1.0):
                continue  # Skip corrupted runs

            # Normalize LDOS for each site across all moments
            data = data / data[0, :].reshape(1, -1)

            # THE MAGIC TRICK: Slice the moments in memory!
            for N_m in moment_list:
                sliced_data = data[:N_m, :] # Grab the first N_m rows (moments)
                chunked_data = np.split(sliced_data, num_chunks, axis=1)

                current_run_chunks = []
                for cd in chunked_data:
                    ldos_values = calculate_LDOS_from_moments(cd, np.array([0]), -eval_list[ind], eval_list[ind])
                    safe_ldos = np.clip(ldos_values, a_min=1e-14, a_max=None)
                    current_run_chunks.append(np.log(safe_ldos).reshape(-1))

                runs_per_moment[N_m].append(current_run_chunks)

        # Now compute the final TDOS for this specific weight across all moment sizes
        for m_idx, N_m in enumerate(moment_list):
            runs_stack = np.array(runs_per_moment[N_m])
            whole_runs_log = np.mean(runs_stack, axis=(1, 2)) # Collapse plaquettes and sites
            whole_log_mean = np.mean(whole_runs_log) # Average over realizations

            # Store the physical Global TDOS
            whole_tdos_matrix[m_idx, ind] = np.exp(whole_log_mean)

    # Save to disk!
    np.savez(full_path, whole_tdos_matrix=whole_tdos_matrix)
    print(f"Computations finished! Data successfully saved to '{save_filename}'")


# ==============================================================================
# PLOT 1: Normal N_m Convergence Plot
# ==============================================================================
plt.figure(figsize=(12, 7))

# Colormap for the 15 different weights
colors_nm = cm.viridis(np.linspace(0, 1, len(wts_list)))

for w_idx, wm in enumerate(wts_list):
    y_vals = whole_tdos_matrix[:, w_idx]
    c = colors_nm[w_idx]

    # Only label a few key weights in the legend to avoid clutter
    label_str = f'W_M = {wm:.2f}' if wm in [0.1, 0.4, 0.5, 1.0, 5.0, 10.0] else None

    plt.plot(moment_list, y_vals, 's-', color=c, label=label_str, markersize=5, alpha=0.8)

plt.xlabel(r"Number of moments ($N_m$)")
plt.ylabel(r"Global Typical DOS, $\rho_t(E=0)$")
plt.title(r"TDOS Convergence vs Number of Moments")

plt.xticks(ticks=moment_list) # Force x-ticks exactly at your calculated moments
plt.xlim(400, 16500)
plt.ylim(0, np.max(whole_tdos_matrix) * 1.1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Disorder Strengths")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('TDOS_Nm_Convergence.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# PLOT: 1/N Extrapolation for the Typical DOS (Order Parameter)
# ==============================================================================
inv_N = 1.0 / np.array(moment_list)

plt.figure(figsize=(12, 7))

# Colormap for the 15 different weights
colors = cm.viridis(np.linspace(0, 1, len(wts_list)))

for w_idx, wm in enumerate(wts_list):
    y_vals = whole_tdos_matrix[:, w_idx]

    # Fit using the top 4 moments (2048, 4096, 8192, 16384)
    fit_idx = -4
    slope, intercept, r_value, p_value, std_err = linregress(inv_N[fit_idx:], y_vals[fit_idx:])

    x_fit = np.linspace(0, max(inv_N), 100)
    y_fit = slope * x_fit + intercept

    c = colors[w_idx]

    # Only label a few key weights in the legend to avoid clutter, but plot all lines
    label_str = f'W_M = {wm:.2f}' if wm in [0.1, 0.4, 0.5, 1.0, 5.0, 10.0] else None

    plt.plot(inv_N, y_vals, 'o', color=c, alpha=0.8)
    plt.plot(x_fit, y_fit, '--', color=c, alpha=0.5, label=label_str)

    # Plot the extrapolated intercept star
    plt.plot(0, intercept, '*', color=c, markersize=8, markeredgecolor='black', alpha=0.9)

# Formatting
plt.axvline(0, color='black', linewidth=1)
plt.axhline(0, color='black', linewidth=1, linestyle='--')

plt.xlabel(r"Inverse Moments ($1/N_m$)")
plt.ylabel(r"Global Typical DOS, $\rho_t(E=0)$")
plt.title(r"Typical DOS Extrapolation to Infinite Resolution ($N_m \to \infty$)")

# Adjust x-limit slightly into the negative so you can clearly see the negative intercepts
plt.xlim(-0.0002, 0.0025)
plt.ylim([None, max(whole_tdos_matrix[-1, :]) * 1.1]) # Auto-scale max Y based on 16k moment data

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Disorder Strengths")
plt.grid(alpha=0.3)
# plt.tight_layout()
plt.savefig('TDOS_Extrapolation.png', dpi=300, bbox_inches='tight')
plt.show()
