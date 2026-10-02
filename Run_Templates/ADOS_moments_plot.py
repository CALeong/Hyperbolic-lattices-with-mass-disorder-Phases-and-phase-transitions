# Import dependencies
import numpy as np
from KPM.measure import calculate_ADOS_from_moments
import matplotlib.pyplot as plt

# Import function to count exact number of sites in system
from Lattice.General_Hamiltonian import number_points_q3_general_from_repeating_pattern

p_val = 10
n_val = 7
R = 12 # number of random vectors for stochastic trace
_, D = number_points_q3_general_from_repeating_pattern(p_val, n_val) # total number of sites
print(f"Number of points in our lattice: {D}")
rel_stoch_err = 1.0/np.sqrt(R * D)
print(f"Relative stochastic error: {rel_stoch_err}")
# # Define run parameters
# data_directory = './numericals/ADOS_p10_n7'
# moment_dir = ['512', '1024', '2048', '4096', '8192', '16384']
# moment_list = [int(mom) for mom in moment_dir]
# data_subdirectory_list = ['025', '030','035','040','045', '050', '075', '100']
# wm_list = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75, 1.00]  # Need to enter numerical values with ith value corresponding to ith value in subdir list manually
# data_name_template = 'mass' + '_{}'
# data_indices = np.arange(10)
#
# energy_values = np.array([0.0])
# eigenspectrum_bounds_list = [3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.50, 3.50]  # list of eigval bounds, ith element corresponds to ith subdir in list above
#
# all_moments_ados = [] # storing ados for different wts + moments
# all_moments_ados_err = [] # storing ados for errs for differents wts + moments
#
# print(f"{'Moments':} | {'W_M:'} | {'DOS(0)':} | {'Std Error':} | {'Stoch Error':} | {'Combined Err'}")
# print("-" * 85)
#
# for ind, m_dir in enumerate(moment_dir):
#     # Load in moments and calculate ADOS from them
#     all_ados = np.array([])
#     all_ados_err = np.array([]) # error for current moment
#
#     for subdir, ev_bound in zip(data_subdirectory_list, eigenspectrum_bounds_list):
#         current_ados = np.array([])
#         for i in data_indices:
#             moments = np.load(data_directory + '/' + m_dir + '/' + subdir + '/' + data_name_template.format(i) + '.npy')
#             current_ados = np.append(current_ados, calculate_ADOS_from_moments(moments, energy_values, -ev_bound, ev_bound))
#
#         mean_ados = np.average(current_ados)
#         stoch_err = rel_stoch_err * mean_ados
#         std_err = np.std(current_ados) / np.sqrt(len(data_indices))
#         combined_err = np.sqrt(std_err**2 + stoch_err**2)
#
#         print(f"{m_dir:<8} | {subdir} | {mean_ados:} | {std_err:} | {stoch_err:} | {combined_err:}")
#
#         all_ados = np.append(all_ados, mean_ados)
#         all_ados_err = np.append(all_ados_err, combined_err)
#
#
#     all_moments_ados.append(all_ados)
#     all_moments_ados_err.append(all_ados_err)
#

# Define run parameters
data_directory = './numericals/ADOS_p10_n7'

# We ONLY load from the master 16384 directory now
master_m_dir = '16384'

# These are our TARGET moment truncations
moment_list = [512, 1024, 2048, 4096, 8192, 16384]

data_subdirectory_list = ['025', '030','035','040','045', '050', '075', '100']
wm_list = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75, 1.00]
data_name_template = 'mass' + '_{}'
data_indices = np.arange(10)

energy_values = np.array([0.0])
eigenspectrum_bounds_list = [3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.50, 3.50]

all_moments_ados = []
all_moments_ados_err = []

print(f"{'Moments':<8} | {'W_M':<5} | {'DOS(0)':<12} | {'Std Error':<12} | {'Stoch Error':<12} | {'Combined Err'}")
print("-" * 85)

for N_m in moment_list:
    all_ados = np.array([])
    all_ados_err = np.array([])

    for subdir, ev_bound in zip(data_subdirectory_list, eigenspectrum_bounds_list):
        current_ados = np.array([])
        for i in data_indices:
            # 1. ALWAYS load the pristine 16,384 moment array
            moments_16k = np.load(data_directory + '/' + master_m_dir + '/' + subdir + '/' + data_name_template.format(i) + '.npy')

            # 2. THE MAGIC TRICK: Slice it down to the target moment length!
            sliced_moments = moments_16k[:N_m]

            # 3. Calculate DOS. The function will see the length of 'sliced_moments'
            # and automatically apply the perfect Jackson kernel for that length.
            ados_val = calculate_ADOS_from_moments(sliced_moments, energy_values, -ev_bound, ev_bound)
            current_ados = np.append(current_ados, ados_val)

        mean_ados = np.average(current_ados)
        stoch_err = rel_stoch_err * mean_ados
        std_err = np.std(current_ados) / np.sqrt(len(data_indices))
        combined_err = np.sqrt(std_err**2 + stoch_err**2)

        print(f"{N_m:<8} | {subdir:<5} | {mean_ados:<12.6e} | {std_err:<12.6e} | {stoch_err:<12.6e} | {combined_err:<12.6e}")

        all_ados = np.append(all_ados, mean_ados)
        all_ados_err = np.append(all_ados_err, combined_err)

    all_moments_ados.append(all_ados)
    all_moments_ados_err.append(all_ados_err)

# ... Plotting blocks remain exactly the same! ...
x_tick_locs = np.array(moment_list)

for i in range(len(wm_list)):
    plt.errorbar(moment_list, [ados[i] for ados in all_moments_ados], yerr= [err[i] for err in all_moments_ados_err], fmt = 's-', capsize= 3, label = f"weight {wm_list[i]}")
plt.xlabel("Number of moments")
plt.ylabel("ADOS at zero energy")
plt.xticks(ticks = x_tick_locs)
plt.legend()
plt.xlim(400, 16500)
plt.ylim([0, None])
plt.show()

from scipy.stats import linregress

# ==============================================================================
# PLOT 2: 1/N Extrapolation
# ==============================================================================

ados_matrix = np.array(all_moments_ados)
inv_N = 1.0 / np.array(moment_list)

plt.figure(figsize=(10, 6))

# --- NEW: Safely get the default Matplotlib color cycle ---
color_cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])

for w_idx, wm in enumerate(wm_list):
    y_vals = ados_matrix[:, w_idx]

    fit_idx = -4
    slope, intercept, r_value, p_value, std_err = linregress(inv_N[fit_idx:], y_vals[fit_idx:])

    x_fit = np.linspace(0, max(inv_N), 100)
    y_fit = slope * x_fit + intercept

    # --- NEW: Grab the next color gracefully ---
    color = next(color_cycle)

    plt.plot(inv_N, y_vals, 'o', label=f'W_M = {wm}', color=color)
    plt.plot(x_fit, y_fit, '--', color=color, alpha=0.6)
    plt.plot(0, intercept, '*', color=color, markersize=10,
             markeredgecolor='black', label=f'  Intercept: {intercept:.2e}')

plt.axvline(0, color='black', linewidth=1)
plt.axhline(0, color='black', linewidth=1, linestyle='--')

plt.xlabel(r"Inverse Moments ($1/N_m$)")
plt.ylabel(r"DOS at $E=0$")
plt.title(r"Finite-Resolution Extrapolation to $N_m \to \infty$")
plt.xlim(-0.0002, 0.0025)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
# plt.tight_layout()
plt.show()
