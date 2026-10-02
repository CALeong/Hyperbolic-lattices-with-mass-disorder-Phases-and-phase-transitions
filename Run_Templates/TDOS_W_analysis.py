# Import dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from KPM.measure import calculate_LDOS_from_moments


# User inputs
wts_list = [0.10, 0.20, 0.30, 0.40, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00] # list of weights
eval_list = [3.12, 3.12, 3.12, 3.12, 3.12, 3.50, 4.00, 5.00, 5.50, 5.75, 6.00, 6.25, 6.50, 7.00, 7.5] # eigenvalue bounds, corresponding to weights
dir_name = ['010', '020', '030', '040', '050', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000'] # folder suffixes
data_dir = './numericals/LDOS_p10_n7'  # Path to the parent directory of the directory holding all the moments data, suffix to be added later
data_name_template = 'LDOS_mass_{}.npy'  # Name of data file with {} for save index to be inserted later
pval = 10  # number of sides of fundamental polygon
total_num_dr = 60  # total number of disorder realization
num_chunks = 7  # Number of plaquettes considered

save_filename = 'compiled_TDOS_results.npz'
full_path = os.path.join(data_dir, save_filename)
########################################################################################################################

if os.path.exists(full_path):
    print("TDOS data exists, loading directly.")
    data = np.load(full_path)

    all_tdos_list = data['all_tdos_list']
    all_tdos_lower_err = data['all_tdos_lower_err']
    all_tdos_upper_err = data['all_tdos_upper_err']
    whole_tdos_list = data['whole_tdos_list']
    whole_tdos_lower_err = data['whole_tdos_lower_err']
    whole_tdos_upper_err = data['whole_tdos_upper_err']

else:
    print("TDOS file doesn't exist, computing.")

    all_tdos_list = [] # empty list to store TDOS values (plaquette wise)
    # all_tdos_err_list = [] # plaquette wise error storage
    all_tdos_upper_err = [] # upper error for asymmetric error
    all_tdos_lower_err = [] # lower error for asymetric error
    whole_tdos_list = [] # empty list to store average TDOS of whole plaquette
    # whole_tdos_err_list = []
    whole_tdos_upper_err = []
    whole_tdos_lower_err = []
    # iterating over all weights
    for ind, wt in enumerate(wts_list):

        # Results holder (Rows = plaquette, Cols = sites on plaquette)
        all_tdos = np.zeros((num_chunks, pval), dtype=np.float64)
        all_valid_runs = []
        for i in range(total_num_dr):  # Iterate over all disorder realizations
            # Load in data
            data = np.load(data_dir + '/' + dir_name[ind] + '/' + data_name_template.format(i))

            # if/else check for diverging data due to bad eigenvalue bounds
            if np.any(np.isnan(data)) or np.max(np.abs(data) > 1.0):
                total_num_dr += -1  # bad dataset, do nothing and reduce total number of disorder realizations by one
            else:
                # Split data for each plaquette and compute log(LDOS) values for each site on each plaquette
                # print("Shape of data", data.shape)
                data = data / data[0, :].reshape(1, -1)  # Normalize LDOS for each site
                chunked_data = np.split(data, num_chunks, axis=1)

                current_run_chunks = [] # temporary list to hold chunks for current disorder realization
                for cdi, cd in enumerate(chunked_data):
                    ldos_values = calculate_LDOS_from_moments(cd, np.array([0]), -eval_list[ind], eval_list[ind]) #change eigenvalue bounds here
                    bad_values = ldos_values[ldos_values <= 0]
                    # print(f"Found {len(bad_values)} invalid values:")
                    # print(bad_values)
                    safe_ldos = np.clip(ldos_values, a_min=1e-14, a_max=None)
                    log_ldos_values = np.log(safe_ldos)
                    current_run_chunks.append(log_ldos_values.reshape(-1))
                    all_tdos[cdi, :] += log_ldos_values.reshape(-1)

                all_valid_runs.append(current_run_chunks)

        # std dev and mean in log space
        runs_stack = np.array(all_valid_runs)
        runs_plaquette_log = np.mean(runs_stack, axis = 2)
        log_mean = np.mean(runs_plaquette_log, axis = 0)
        log_std_dev = np.std(runs_plaquette_log, axis = 0)
        log_std_dev = log_std_dev/ np.sqrt(total_num_dr) # std = sigma/sqrt(N)

        print("Current weight being analyzed: ", wt)
        # Average over all sites on plaquette and take exp, thereby giving TDOS of plaquette
        plaquette_tdos = np.exp(log_mean)

        # plaquette_tdos_err = plaquette_tdos * log_std_dev # first order error
        bdd_up = np.exp(log_mean + log_std_dev) # absolute upper bounds
        bdd_down = np.exp(log_mean - log_std_dev) # absolute lower bounds

        upper_err = np.abs(bdd_up - plaquette_tdos)
        lower_err = np.abs(plaquette_tdos - bdd_down)

        print("TDOS each plaquette (center to edge): ", plaquette_tdos)
        # print("Upper error each plaquette: ", plaquette_tdos_err)
        print("Upper error each plaquette: ", upper_err)
        print("Lower  error each plaquette: ", lower_err)

        # all_tdos_list.append(all_tdos)
        all_tdos_list.append(plaquette_tdos)
        # all_tdos_err_list.append(plaquette_tdos_err)
        all_tdos_upper_err.append(upper_err)
        all_tdos_lower_err.append(lower_err)

        whole_runs_log = np.mean(runs_stack, axis=(1, 2)) # tdos for whole system per realization, collapse both plaquettes and sites to get one value per run

        whole_log_mean = np.mean(whole_runs_log) # global mean across the independent runs (axis 0)
        whole_log_sem = np.std(whole_runs_log) / np.sqrt(total_num_dr) # standard error of mean, std dev of data over all runs, divided by sqrt of number of runs

        whole_tdos = np.exp(whole_log_mean) # exponentiate to get physical tdos

        whole_bdd_up = np.exp(whole_log_mean + whole_log_sem)
        whole_bdd_down = np.exp(whole_log_mean - whole_log_sem)
        # whole_tdos_err = whole_tdos * whole_log_sem # propagate error

        whole_upper = np.abs(whole_bdd_up - whole_tdos)
        whole_lower = np.abs(whole_tdos - whole_bdd_down)
        print('TDOS all sampled sites: ', whole_tdos)
        print('TDOS lower error total: ', whole_lower)
        print('TDOS upper error total: ', whole_upper)
        # print('Error all sampled sites: ', whole_tdos_err)

        # Append to your master lists
        whole_tdos_list.append(whole_tdos)
        # whole_tdos_err_list.append(whole_tdos_err)
        whole_tdos_upper_err.append(whole_upper)
        whole_tdos_lower_err.append(whole_lower)

    compiled_data = {
        'all_tdos_list': all_tdos_list,
        'all_tdos_lower_err': all_tdos_lower_err,
        'all_tdos_upper_err': all_tdos_upper_err,
        'whole_tdos_list': whole_tdos_list,
        'whole_tdos_lower_err': whole_tdos_lower_err,
        'whole_tdos_upper_err': whole_tdos_upper_err
    }
    np.savez(full_path, **compiled_data)
    print(f"Computations finished! Data successfully saved to '{save_filename}'")


# plt.xlabel("W_M")
# plt.ylabel("TDOS for each plaquette")
# plt.ylim(0, 0.12)
# plt.xlim(0.0, 10.0)
# plt.title("TDOS v/s W_M for each plaquette")
#
# # for wt in range(len(wts_list)):
# for i in range(len(all_tdos_list[0])):
#     yerr_lower = [err[i] for err in all_tdos_lower_err]
#     yerr_upper = [err[i] for err in all_tdos_upper_err]
#     plt.errorbar(wts_list, [tdos[i] for tdos in all_tdos_list], yerr =[yerr_lower, yerr_upper], label = f"plaquette {i+1}", fmt = 's-', capsize = 3)
# plt.legend()
# plt.show()

# plt.xlabel("W_M")
# plt.ylabel("TDOS all sampled sites")
# plt.ylim(0, 0.03)
# plt.xlim(0.0, 10.0)
# plt.errorbar(wts_list, whole_tdos_list, yerr = [whole_tdos_lower_err, whole_tdos_upper_err], fmt='s-', capsize = 3)
# plt.show()


# ==============================================================================
# PLOT 1: The 2x4 Subplot Grid (7 Generations + 1 Global)
# ==============================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()  # Flatten the 2x4 array into a 1D list of 8 axes

# 1. Plot the 7 individual plaquettes (Generations)
for i in range(num_chunks):
    yerr_lower = [err[i] for err in all_tdos_lower_err]
    yerr_upper = [err[i] for err in all_tdos_upper_err]
    y_vals = [tdos[i] for tdos in all_tdos_list]

    axes[i].errorbar(wts_list, y_vals, yerr=[yerr_lower, yerr_upper])
    axes[i].set_title(f"Generation {i}")
    axes[i].set_xlabel(r"$W_M$")
    axes[i].set_ylabel(r"$\rho_t(E=0)$")
    axes[i].set_xlim(0, 10.5)
    axes[i].set_ylim(0,0.12) # Auto-scales y-axis nicely

# 2. Plot the Global Whole System TDOS on the 8th subplot
axes[7].errorbar(wts_list, whole_tdos_list,
                 yerr=[whole_tdos_lower_err, whole_tdos_upper_err])
axes[7].set_title("Global System (All Sites)")
axes[7].set_xlabel(r"$W_M$")
axes[7].set_ylabel(r"$\rho_t(E=0)$")
axes[7].set_xlim(0, 10.5)
axes[7].set_ylim(0, 0.12)

plt.tight_layout() # Prevents labels from overlapping
plt.savefig('TDOS_2x4_Subplots.png', dpi=300, bbox_inches='tight') # SAVES THE FIGURE
plt.show()


# ==============================================================================
# PLOT 2: The Condensed "Fig 2d" Style Overlay
# ==============================================================================
plt.figure(figsize=(10, 6))

# Use a colormap so the generations look like a smooth gradient (e.g., from bulk to boundary)
colors = cm.viridis(np.linspace(0, 1, num_chunks))

for i in range(num_chunks):
    yerr_lower = [err[i] for err in all_tdos_lower_err]
    yerr_upper = [err[i] for err in all_tdos_upper_err]
    y_vals = [tdos[i] for tdos in all_tdos_list]

    plt.errorbar(wts_list, y_vals, yerr=[yerr_lower, yerr_upper],
                 label=f"Generation {i}", fmt='s-', capsize=3,
                 color=colors[i], markersize=5, alpha=0.8)

plt.xlabel(r"Disorder Strength $(W_M)$")
plt.ylabel(r"Typical DOS, $\rho_t(E=0)$")
plt.title("Generation-Resolved TDOS at Zero Energy")
plt.xlim(0, 10.5)
plt.ylim(0, 0.12) # Hardcoded based on your original script, adjust if needed
plt.legend(loc='upper right', framealpha=0.9)
# plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('TDOS_Condensed_Overlay.png', dpi=300, bbox_inches='tight') # SAVES THE FIGURE
plt.show()
