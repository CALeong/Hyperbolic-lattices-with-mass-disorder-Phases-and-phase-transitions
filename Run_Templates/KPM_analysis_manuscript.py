import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from plot_utils import fixed_box_figure, fixed_box_row
from scipy.stats import linregress
from KPM.measure import calculate_ADOS_from_moments, calculate_LDOS_from_moments

"""
General configuration and parameters for our code. We will save directory paths here pointing toward where the existing data is stored. We also specify 'cache' folders, where we save computed TDOS and ADOS values so that we do not have to recompute these values over and over while making plotting tweaks.
"""
MOMENT_TARGET = 16384 # Default max moment for our analyses
# MOMENT_LIST_p8 = [512, 1024, 2048, 4096, 8192, 16384]
MOMENT_LIST_p8 = [512, 1024, 2048, 4096, 8192]
MOMENT_LIST_p10 = [512, 1024, 2048, 4096, 6144]

# Cache directory stores computed ADOS/TDOS values to access quickly while plotting
CACHE_BASE = './analysis_cache'
PLOT_BASE_ADOS = './numericals/plots/ADOS' # save directory for ADOS plots
PLOT_BASE_TDOS = './numericals/plots/TDOS' # save directory for TDOS plots

# Master Dictionary for all lattice geometries (contains directory paths for data, list of weights, energy bounds etc)
SYSTEMS = {
    'p10_n7': {
        'num_chunks': 7,
        'ados_dir': './numericals/ADOS_p10_n7',
        'tdos_dir': './numericals/LDOS_p10_n7',

        'wm_list_ados': [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00],
        'dir_names_ados': ['000', '005', '010', '015', '020', '025', '030', '035', '040', '045', '050', '055', '060', '070', '075', '080', '090', '095', '100', '105', '110', '115', '120', '125', '150', '160', '170', '180', '190', '200'],
        'eval_list_ados': [3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 4.00 , 4.00 , 4.00 , 4.00 , 4.00], # energy bounds in corresponding order of weights

        'wm_list_tdos': [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 0.90, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 8.00, 9.00, 10.00],
        'dir_names_tdos': ['000', '005', '010', '015', '020', '025', '030', '035', '040', '045', '050', '060', '075', '090', '100', '125', '150', '175',  '200', '250', '300', '350', '400', '450', '475', '500', '525', '550', '575', '600', '625', '650', '675', '700', '800', '900', '1000'],
        'eval_list_tdos': [3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.50, 3.50, 3.50, 3.50, 4.00, 4.00, 4.00, 4.00, 5.00, 5.00, 5.50, 5.50, 5.75, 5.75, 5.75, 6.00, 6.00, 6.00, 6.00,  6.25, 6.25, 6.25, 6.25, 6.50, 7.00, 7.50]
    },
    'p8_n9': {
        'num_chunks': 9,
        'ados_dir': None, # This is a Fermi liquid system, so skip ADOS
        'tdos_dir': './numericals/LDOS_p8_n9',

        'wm_list_tdos': [0.00, 0.50, 0.75, 1.00, 1.25, 2.00, 3.00, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.0],
        'dir_names_tdos': ['000', '050', '075', '100', '125', '200', '300', '350', '375',  '400', '425', '450', '475', '500', '525', '550', '575', '600', '625', '650', '700', '750', '800', '850', '900', '950', '1000'],
        'eval_list_tdos': [3.4, 3.4, 3.4, 3.4, 3.6, 3.6, 3.9, 4.3, 4.3, 4.3, 4.7, 4.7, 4.7, 4.7, 5.1, 5.1, 5.1, 5.1, 5.6, 5.6, 5.6, 6.0, 6.0, 6.5, 6.5, 7.0, 7.0]
    }
}

"""
Main computation functions. These functions exist to carry out the following analyses (compute and save data for these):
1. get_ados_data: ADOS(E=0) v/s W
2. get_tdos_data: TDOS(E=0) v/s W
3. get_energy_resolved_data: computes both ADOS v/s E and TDOS v/s E for all weights
4. get_convergence_data: computes DOS(0) v/s N_m and DOS(0) v/s (1/N_m) for a set of moments
"""

def get_ados_data(sys_name, cfg):
    cache_file = os.path.join(CACHE_BASE, sys_name, f"ados_vs_w_{sys_name}.npy") # cache file to save computed ADOS values
    if os.path.exists(cache_file): return np.load(cache_file) # if file exists, we simply load, otherwise, we compute ADOS and save cache

    print(f"[{sys_name}] Computing ADOS at E=0...") # computing if cache file doesn't exist'
    ados_results = []
    MOMENT_TARGET_p10 = 6144
    for ind, subdir in enumerate(cfg['dir_names_ados']):
        current_ados = []
        for i in range(10):
            file_path = f"{cfg['ados_dir']}/{subdir}/mass_{i}.npy"
            if os.path.exists(file_path):
                moments = np.load(file_path)[:MOMENT_TARGET_p10]
                val = calculate_ADOS_from_moments(moments, np.array([0.0]), -cfg['eval_list_ados'][ind], cfg['eval_list_ados'][ind]) # use existing functions to compute ADOS
                current_ados.append(val)
        if current_ados: ados_results.append(np.mean(current_ados))

    np.save(cache_file, ados_results) # save cache
    return np.array(ados_results)

def get_tdos_data(sys_name, cfg):
    cache_file = os.path.join(CACHE_BASE, sys_name, f"tdos_vs_w_{sys_name}.npz")
    if os.path.exists(cache_file): # same as ADOS case, load if cache file exists, otherwise compute
        data = np.load(cache_file)
        return data['global_tdos'], data['plaquette_tdos']

    if sys_name=='p10_n7':
        MOMENT_TARGET=6144
    elif sys_name == 'p8_n9':
        MOMENT_TARGET = 8192
    print(f"[{sys_name}] Computing TDOS at E=0...")
    global_tdos = []
    plaquette_tdos = np.zeros((cfg['num_chunks'], len(cfg['wm_list_tdos'])))
    # size of plaquette_tdos is (num_chunks, len(wm_list_tdos))
    for ind, subdir in enumerate(cfg['dir_names_tdos']):
        runs_log_ldos = []
        for i in range(60):
            file_path = f"{cfg['tdos_dir']}/{subdir}/LDOS_mass_{i}.npy"
            if os.path.exists(file_path):
                data = np.load(file_path)[:MOMENT_TARGET, :]
                data = data / data[0, :].reshape(1, -1)
                chunked = np.split(data, cfg['num_chunks'], axis=1)
                run_chunks = []
                for cd in chunked:
                    ldos_vals = calculate_LDOS_from_moments(cd, np.array([0.0]), -cfg['eval_list_tdos'][ind], cfg['eval_list_tdos'][ind]) # using existing code to compute LDOS
                    run_chunks.append(np.log(np.clip(ldos_vals, a_min=1e-14, a_max=None)).reshape(-1))
                runs_log_ldos.append(run_chunks)

        # if runs_log_ldos:
        #     runs_stack = np.array(runs_log_ldos)
        #     plaquette_tdos[:, ind] = np.exp(np.mean(runs_stack, axis=(0, 2)))
        #     global_tdos.append(np.exp(np.mean(runs_stack)))
        if runs_log_ldos:
            runs_stack = np.array(runs_log_ldos)
            plaq_tdos = np.exp(np.mean(runs_stack, axis=(0, 2)))
            plaquette_tdos[:, ind] = plaq_tdos
            global_tdos.append(np.mean(plaq_tdos))

    np.savez(cache_file, global_tdos=global_tdos, plaquette_tdos=plaquette_tdos) # save cache
    return np.array(global_tdos), plaquette_tdos

def get_energy_resolved_data(sys_name, cfg, mode="TDOS", num_points=101, zoom_window=None): #compute for 3001 energy points
    cache_dir = os.path.join(CACHE_BASE, sys_name, f"{mode}_vs_E")
    os.makedirs(cache_dir, exist_ok=True)
    results = {}

    # we select weights list and corresponding energy bounds list depending on the case we look at
    wm_list = cfg['wm_list_tdos'] if mode == "TDOS" else cfg['wm_list_ados']
    dir_names = cfg['dir_names_tdos'] if mode == "TDOS" else cfg['dir_names_ados']
    eval_list = cfg['eval_list_tdos'] if mode == "TDOS" else cfg['eval_list_ados']

    for ind, wm in enumerate(wm_list):
        if sys_name=='p10_n7':
            MOMENT_TARGET = 6144
        elif sys_name == 'p8_n9':
            MOMENT_TARGET = 8192

        if wm>2.0 and mode == "ADOS":
            continue
        # cache_file = os.path.join(cache_dir, f"{mode}_E_W{wm}_{sys_name}.npz")
        # if os.path.exists(cache_file):
        #     data = np.load(cache_file)
        #     results[wm] = {'energies': data['energies'], 'vals': data['vals']}
        #     continue
        a = eval_list[ind]

        # --- NEW LOGIC: Dynamic KPM Resolution Grid ---
        if zoom_window is not None and wm == 0.60:
            kpm_resolution = (np.pi * a) / MOMENT_TARGET
            # kpm_resolution = 1
            energies = np.arange(zoom_window[0], zoom_window[1] + kpm_resolution, kpm_resolution)
            # Separate cache file so it doesn't overwrite your full spectrum data
            cache_file = os.path.join(cache_dir, f"{mode}_E_W{wm}_{sys_name}_zoomed.npz")
        else:
            energies = np.linspace(-0.05, 0.05, num_points)
            cache_file = os.path.join(cache_dir, f"{mode}_E_W{wm}_{sys_name}.npz")

        if os.path.exists(cache_file):
            data = np.load(cache_file)
            results[wm] = {'energies': data['energies'], 'vals': data['vals']}
            continue

        print(f"[{sys_name}] Computing {mode} vs E for W_M = {wm}...")
        # print(f"[{sys_name}] Computing {mode} vs E for W_M = {wm}...")
        # energies = np.linspace(-eval_list[ind], eval_list[ind], num_points)
        # energies = np.linspace(-0.05, 0.05, num_points)


        if mode == "ADOS":
            runs = []
            for i in range(10): # number of disorder realization
                file_path = f"{cfg['ados_dir']}/{dir_names[ind]}/mass_{i}.npy"
                if os.path.exists(file_path):
                    moments = np.load(file_path)[:]
                    runs.append(calculate_ADOS_from_moments(moments, energies, -eval_list[ind], eval_list[ind])) # append ADOS v/s E data for a disorder realization
            if runs:
                mean_val = np.mean(runs, axis=0) # average the ADOS values
                np.savez(cache_file, energies=energies, vals=mean_val)
                results[wm] = {'energies': energies, 'vals': mean_val}

        elif mode == "TDOS":
            runs_log_ldos = []
            for i in range(60): # number of disorder realization
                file_path = f"{cfg['tdos_dir']}/{dir_names[ind]}/LDOS_mass_{i}.npy"
                if os.path.exists(file_path):
                    data = np.load(file_path)[:MOMENT_TARGET, :]
                    data = data / data[0, :].reshape(1, -1)
                    chunked = np.split(data, cfg['num_chunks'], axis=1)
                    run_chunks = []
                    for cd in chunked:
                        ldos_vals = calculate_LDOS_from_moments(cd, energies, -eval_list[ind], eval_list[ind])
                        run_chunks.append(np.log(np.clip(ldos_vals, a_min=1e-14, a_max=None)))
                    runs_log_ldos.append(np.array(run_chunks))
            # if runs_log_ldos:
            #     runs_stack = np.array(runs_log_ldos)
            #     mean_log = np.mean(runs_stack, axis=(0, 1, 2))
            #     phys_tdos = np.exp(mean_log)
            #     np.savez(cache_file, energies=energies, vals=phys_tdos)
            #     results[wm] = {'energies': energies, 'vals': phys_tdos}
            if runs_log_ldos:
                runs_stack = np.array(runs_log_ldos)
                mean_log_per_plaquette = np.mean(runs_stack, axis=(0, 2))
                tdos_per_plaquette = np.exp(mean_log_per_plaquette)
                phys_tdos = np.mean(tdos_per_plaquette, axis=0)

                np.savez(cache_file, energies=energies, vals=phys_tdos)
                results[wm] = {'energies': energies, 'vals': phys_tdos}
    return results

def get_convergence_data(sys_name, cfg, mode="TDOS"):
# we slice the matrix with 16384 moments to truncate the list to smaller moment sizes in order to compute moment convergence
    cache_file = os.path.join(CACHE_BASE, sys_name, f"{mode}_convergence_matrix_{sys_name}.npy")
    if os.path.exists(cache_file): return np.load(cache_file)

    print(f"[{sys_name}] Computing {mode} convergence extrapolation data...")
    wm_list = cfg['wm_list_tdos'] if mode == "TDOS" else cfg['wm_list_ados']
    dir_names = cfg['dir_names_tdos'] if mode == "TDOS" else cfg['dir_names_ados']
    eval_list = cfg['eval_list_tdos'] if mode == "TDOS" else cfg['eval_list_ados']
    base_dir = cfg['tdos_dir'] if mode == "TDOS" else cfg['ados_dir']

    if sys_name == "p10_n7":
        MOMENT_LIST = MOMENT_LIST_p10
    elif sys_name == "p8_n9":
        MOMENT_LIST = MOMENT_LIST_p8
    extrap_matrix = np.zeros((len(MOMENT_LIST), len(wm_list)))

    for ind, subdir in enumerate(dir_names):
        runs_per_moment = {Nm: [] for Nm in MOMENT_LIST}
        num_runs = 60 if mode == "TDOS" else 10
        file_template = "LDOS_mass_{}.npy" if mode == "TDOS" else "mass_{}.npy"

        for i in range(num_runs):
            file_path = f"{base_dir}/{subdir}/{file_template.format(i)}"
            if os.path.exists(file_path):
                full_data = np.load(file_path)
                if mode == "TDOS": full_data = full_data / full_data[0, :].reshape(1, -1)

                for Nm in MOMENT_LIST:
                    if mode == "TDOS":
                        sliced = full_data[:Nm, :] # slicing only first Nm moments from full data
                        # same process as before to extract LDOS/TDOS/ADOS as required
                        chunked = np.split(sliced, cfg['num_chunks'], axis=1)
                        run_chunks = []
                        for cd in chunked:
                            ldos_vals = calculate_LDOS_from_moments(cd, np.array([0.0]), -eval_list[ind], eval_list[ind])
                            run_chunks.append(np.log(np.clip(ldos_vals, a_min=1e-14, a_max=None)).reshape(-1))
                        runs_per_moment[Nm].append(np.array(run_chunks))
                    else:
                        sliced = full_data[:Nm]
                        val = calculate_ADOS_from_moments(sliced, np.array([0.0]), -eval_list[ind], eval_list[ind])
                        runs_per_moment[Nm].append(val)

        for m_idx, Nm in enumerate(MOMENT_LIST):
            if runs_per_moment[Nm]:
                if mode == "TDOS":
                    runs_stack = np.array(runs_per_moment[Nm])
                    plaq_tdos = np.exp(np.mean(runs_stack, axis=(0, 2)))
                    extrap_matrix[m_idx, ind] = np.mean(plaq_tdos)
                else:
                    extrap_matrix[m_idx, ind] = np.mean(runs_per_moment[Nm])

    np.save(cache_file, extrap_matrix)
    return extrap_matrix

def fit_power_law(ax, x_vals, y_vals, xlabel, ylabel, title, color, param_name, y_offset=0.0):
    """Core function to fit and plot a pure log-log power law with error bounds."""
    # First we subtract the finite-size background offset
    y_shifted = y_vals - y_offset

    # Remove  zeros or negatives to prevent issues with log
    valid = (x_vals > 1e-6) & (y_shifted > 1e-10) & (y_vals < 10.0)
    x_fit = x_vals[valid]
    y_fit = y_shifted[valid]

    ax.scatter(x_vals, y_vals, color='black', s=40, zorder=5)

    if len(x_fit) < 2:
        ax.set_title(f"{title}\n(Insufficient Data)")
        return

    fit, cov = curve_fit(lambda x, m, b: m * x + b, np.log(x_fit), np.log(y_fit))
    exponent, intercept = fit[0], fit[1]
    err = np.sqrt(cov[0, 0])

    plot_x = np.linspace(1e-4, max(x_fit)*1.1, 100)
    C = np.exp(intercept)
    plot_y = C * (plot_x ** exponent) + y_offset

    ax.plot(plot_x, plot_y, color=color, linewidth=2, marker="",
            label=rf"Fit: ${param_name} = {exponent:.4f} \pm {err:.4f}$")

    # standard erro bounds
    def error_func(x, covariance_mat):
        err_slope = np.sqrt(covariance_mat[0, 0])
        err_int = np.sqrt(covariance_mat[1, 1])
        err_cov = covariance_mat[0, 1]
        return np.sqrt((np.log(x) * err_slope)**2 + err_int**2 + 2 * np.log(x) * err_cov)

    plot_x_err = np.linspace(1e-4, max(x_fit)*1.1, 100)
    err_margin = error_func(plot_x_err, cov)

    ax.plot(plot_x_err, C * (plot_x_err ** exponent) * np.exp(err_margin) + y_offset,
            color=color, linestyle='--', alpha=0.6, marker="")
    ax.plot(plot_x_err, C * (plot_x_err ** exponent) * np.exp(-err_margin) + y_offset,
            color=color, linestyle='--', alpha=0.6, marker="")
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=0.0)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, rotation=0, labelpad=25, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.spines[:].set_linewidth(1.5)
    ax.tick_params(axis='both', direction='in', length=4, width=1.5, pad =5)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))


def plot_base_dos_vs_W(sys_name, cfg, mode="TDOS"):
    # Fig 1a/1b: DOS at E=0 vs W
    if mode == "ADOS":
        # --- ADOS STANDARD PLOT ---
        fig, ax = fixed_box_figure()
        vals = get_ados_data(sys_name, cfg)
        ax.plot(cfg['wm_list_ados'], vals, ls='', marker='o', color='tab:red', linewidth=2, markersize=6)

        print("Printing ados values")
        for i in range(len(vals)):
            print(f"(wm, ados) = ({cfg['wm_list_ados'][i]}, {vals[i]})")
        # ax.set_xticks([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
        my_ticks = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        ax.set_xticks(my_ticks, labels=[f"{val:.1f}" for val in my_ticks])
        ax.set_yticks([0.00, 0.02, 0.04])
        ax.set_xlim(0, cfg['wm_list_ados'][-1])
        ax.set_ylim(0, 0.04)

        ax.set_ylabel(r" $\rho_a(0)$", rotation=0, labelpad=-7, y=0.6)
        ax.set_xlabel(r"$W_M$")

        # Formatting
        ax.spines[:].set_linewidth(1.5)
        ax.tick_params(axis='both', direction='in', length=4, width=1.5, pad =5)

        if sys_name == 'p10_n7':
            ax.set_title(rf"{mode} at E=0 vs Disorder Strength $\{{ 10, 3 \}} $")
        elif sys_name == 'p8_n9':
            ax.set_title(rf"{mode} at E=0 vs Disorder Strength $\{{ 8, 3\}}$")

    elif mode =='TDOS' and sys_name=='p10_n7':
        # --- TDOS BROKEN AXIS PLOT ---
        # 1. Create two side-by-side subplots sharing the Y-axis
        # fig, (ax1, ax2) = ax.subplots(1, 2, sharey=True, figsize=(6, 6), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.01})
        fig, (ax1, ax2) = fixed_box_row(n=2, width_ratios=[1, 1], gap=0.06, sharey=False)
        # fig.subplots_adjust(wspace=0.05) # Jam them together

        vals, _ = get_tdos_data(sys_name, cfg)
        w_vals = cfg['wm_list_tdos']

        if 1.5 in w_vals:
            idx = w_vals.index(1.5)
            w_vals_filtered = np.delete(w_vals, idx)
            vals_filtered = np.delete(vals, idx)
        else:
            w_vals_filtered = np.copy(w_vals)
            vals_filtered = np.copy(vals)
        # print("printing tdos values for {10,3}")
        # for i in range(len(w_vals_filtered)):
        #     print(f"(w, tdos) = {w_vals_filtered[i], vals_filtered[i]}")

        # 2. Plot data on BOTH axes
        ax1.plot(w_vals_filtered, vals_filtered, ls='',marker='o', color='tab:red', linewidth=2, markersize=6)
        ax2.plot(w_vals_filtered, vals_filtered, ls='',marker='o', color='tab:red', linewidth=2, markersize=6)

        # 3. Limit the X-axes to create the "break"
        ax1.set_xlim(0.0, 1.07)
        ax2.set_xlim(1.60, 10.0)
        ax1.set_ylim(0, 0.05)
        ax2.set_ylim(0, 0.05)

        # Apply the specific requested ticks
        ax1.set_xticks([0.0, 0.5, 1.0])
        my_ticks = [2.0, 6.0, 10.0]
        ax2.set_xticks(my_ticks, labels=[f"{val:.1f}" for val in my_ticks])
        # ax2.set_xticks([2.0, 5.0, 10.0])
        ax2.set_yticks([])
        ax1.set_yticks([0.000, 0.025, 0.050])
        # 4. Hide the inner spines between them
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        # ax2.tick_params(left=False) # Hide duplicate y-ticks on the right

        # Apply thickness to remaining visible spines
        for spine in ['top', 'bottom', 'left']:
            ax1.spines[spine].set_linewidth(1.5)
        for spine in ['top', 'bottom', 'right']:
            ax2.spines[spine].set_linewidth(1.5)

        ax1.tick_params(axis='both', direction='in', length=4, width=1.5, pad =5)
        ax2.tick_params(axis='x', direction='in', length=4, width=1.5, pad=5)

        # 5. Draw the diagonal slash marks to show the cut
        d = 0.012
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.5)
        ax1.plot((1 - d, 1 + d), (-d, +d), marker='', **kwargs)        # Bottom-right diagonal
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), marker='', **kwargs)  # Top-right diagonal
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (-d, +d), marker='', **kwargs)              # Bottom-left diagonal
        ax2.plot((-d, +d), (1 - d, 1 + d), marker='', **kwargs)        # Top-left diagonal

        # Labels & Titles (Centering X-label across the whole figure)
        ax1.set_ylabel(r"$\rho_t(0)$", rotation=0, labelpad=-10, y=0.6)
        fig.supxlabel(r"$W_M$")

        if sys_name == 'p10_n7':
            fig.suptitle(rf"{mode} at E=0 vs Disorder Strength $\{{ 10, 3 \}} $")
        elif sys_name == 'p8_n9':
            fig.suptitle(rf"{mode} at E=0 vs Disorder Strength $\{{ 8, 3\}}$")

    else:

        fig, ax = fixed_box_figure()
        vals, _ = get_tdos_data(sys_name, cfg)
        vals_filtered=vals[3:]
        wm_tdos_filtered = cfg['wm_list_tdos'][3:]

        # print("printing  tdos for {8,3}")
        # for i in range(len(vals_filtered)):
        #     print(f"(w, tdos) = {wm_tdos_filtered[i], vals_filtered[i]}")
        ax.plot(wm_tdos_filtered, vals_filtered, ls='', marker='o', color='tab:red', linewidth=2, markersize=6)
        my_ticks = [0.0, 1.0, 2.0, 5.0, 10.0]
        ax.set_xticks(my_ticks, labels=[f"{val:.1f}" for val in my_ticks])
        # ax.set_xticks([0.0, 1.0, 2.0, 5.0, 10.0])
        ax.set_yticks([0.00, 0.05, 0.10])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 0.10)

        ax.set_ylabel(r" $\rho_t(0)$", rotation=0, labelpad=-10, y=0.6)
        ax.set_xlabel(r"$W_M$")
        ax.set_title(rf"{mode} at E=0 vs Disorder Strength $\{{ 8, 3\}}$")

        # Formatting
        ax.spines[:].set_linewidth(1.5)
        ax.tick_params(axis='both', direction='in', length=4, width=1.5, pad =5)

    # Save output
    out_dir = os.path.join(PLOT_BASE_ADOS if mode == "ADOS" else PLOT_BASE_TDOS, sys_name)
    save_path = os.path.join(out_dir, f"{mode}_vs_W_{sys_name}.svg")
    plt.savefig(save_path)
    print(f"[{sys_name}] Saved: {save_path}")
    plt.close()

def plot_tdos_plaquettes(sys_name, cfg):
    # Fig 2d: TDOS at E=0 for different plaquettes
    _, plaquettes = get_tdos_data(sys_name, cfg)
    fig, ax = fixed_box_figure(right=2.0)  # room for the plaquette legend
    colors = cm.viridis(np.linspace(0, 1, cfg['num_chunks']))

    for i in range(cfg['num_chunks']):
        ax.plot(cfg['wm_list_tdos'], plaquettes[i, :], 's-', label=rf"$n= {i}$", color=colors[i], alpha=0.8, markersize=6)
    ax.set_xlim(0,10)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$W_M$")
    ax.set_ylabel(r" $\rho_t^n(0)$", rotation=0, labelpad=30)
    ax.spines[:].set_linewidth(1.5)
    ax.tick_params(axis='both', direction='in', length=4, width=1.5, pad =5)
    if sys_name=='p10_n7':
        ax.set_title(rf"Generation-resolved TDOS, $\rho_t^n(E=0), \{{10, 3\}}$")
    elif sys_name=='p8_n9':
        ax.set_title(rf"Generation-resolved TDOS, $\rho_t^n(E=0)$, \{{8, 3\}}")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    out_dir = os.path.join(PLOT_BASE_TDOS, sys_name)
    save_path = os.path.join(out_dir, f"Plaquettes_{sys_name}.svg")
    fig.savefig(save_path)
    print(f"[{sys_name}] Saved: {save_path}")
    plt.close(fig)

def plot_energy_resolved(sys_name, cfg, mode="TDOS"):
    # Fig 1 Insets: DOS vs Energy
    data = get_energy_resolved_data(sys_name, cfg, mode)
    fig, ax = fixed_box_figure(right=3.0)  # extra right margin for the outside legend

    wm_list = cfg['wm_list_tdos'] if mode == "TDOS" else cfg['wm_list_ados']
    if mode=="ADOS" and sys_name == "p10_n7":
        allowed_weights = [0.25, 0.50, 0.55, 0.60, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    elif mode == "TDOS" and sys_name == "p10_n7":
        allowed_weights = [0.1, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.75, 1, 4, 5, 6, 7]
    elif mode == "TDOS" and sys_name == "p8_n9":
        allowed_weights = [0.0, 0.5, 1.0, 3.0, 4, 4.5, 5, 5.5, 6.0, 6.5, 7.0]

    plot_weights = [wm for wm in allowed_weights if wm in data]
    # plot_weights = list(data.keys())
    colors = cm.turbo(np.linspace(0,1, len(plot_weights)))

    for wm, color in zip(plot_weights, colors):
        plot_data = data[wm]
        label_str = f"${wm:.2f}$"
        ax.plot(plot_data['energies'], plot_data['vals'], label=label_str, color=color, linewidth=1.5, marker="")

    ax.set_xlabel(r"$E$")
    if mode=="TDOS" and sys_name=="p8_n9":
        ax.set_ylabel(rf"$\rho_t(E)$", rotation=0, labelpad=30)
        ax.set_ylim(0,0.16)
        ax.set_xlim(-0.03,0.03)
        ax.set_xticks([-0.03, 0.00, +0.03])
        ax.set_yticks([0.00, 0.08, 0.16])
    elif mode=="TDOS" and sys_name=="p10_n7":
        ax.set_ylabel(rf"$\rho_t(E)$", rotation=0, labelpad=30)
        ax.set_ylim(0,0.08)
        ax.set_xlim(-0.025,0.025)
        ax.set_xticks([-0.025, 0.000, +0.025])
        ax.set_yticks([0.00, 0.04, 0.08])
    elif mode=="ADOS":
        ax.set_ylabel(rf"$\rho_a(E)$", rotation=0, labelpad=30)
        ax.set_ylim(0,0.06)
        ax.set_xlim(-0.04,0.04)
        ax.set_xticks([-0.04, 0.00, +0.04])
        ax.set_yticks([0.00, 0.03, 0.06])

    if sys_name=='p10_n7':
        ax.set_title(rf"{mode} v/s Energy $\{{ 10, 3\}}$")
    elif sys_name=='p8_n9':
        ax.set_title(rf"{mode} v/s Energy $\{{ 8, 3\}}$")

    ax.legend(loc='upper left', bbox_to_anchor=(1.20, 1))
    ax.spines[:].set_linewidth(1.5)
    ax.tick_params(axis='both', direction='in', length=4, width=1.5, pad =7)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    out_dir = os.path.join(PLOT_BASE_ADOS if mode=="ADOS" else PLOT_BASE_TDOS, sys_name)
    save_path = os.path.join(out_dir, f"{mode}_vs_E_{sys_name}.svg")
    fig.savefig(save_path)
    print(f"[{sys_name}] Saved: {save_path}")
    plt.close(fig)

def plot_convergence(sys_name, cfg, mode="TDOS", type="invNm"):
    # Fig 2a-c: Convergence vs Nm or 1/Nm
    extrap_matrix = get_convergence_data(sys_name, cfg, mode)
    if sys_name == "p10_n7":
        MOMENT_LIST = MOMENT_LIST_p10
    elif sys_name == "p8_n9":
        MOMENT_LIST = MOMENT_LIST_p8
    x_vals = 1.0 / np.array(MOMENT_LIST) if type == "invNm" else MOMENT_LIST

    wm_list = cfg['wm_list_tdos'] if mode == "TDOS" else cfg['wm_list_ados']

    # Filter weights into two distinct groups
    groups = {
        "W_le_2": [(idx, wm) for idx, wm in enumerate(wm_list) if wm <= 2.0],
        "W_gt_2": [(idx, wm) for idx, wm in enumerate(wm_list) if wm > 2.0]
    }

    for group_name, group_data in groups.items():
        # Skip if a group is empty
        if not group_data:
            continue

        fig, ax = fixed_box_figure(right=2.0)  # room for the legend
        # we generate colors specific only to the number of lines in current plot
        colors = cm.viridis(np.linspace(0, 1, len(group_data)))

        for c_idx, (w_idx, wm) in enumerate(group_data):
            y_vals = extrap_matrix[:, w_idx]
            c = colors[c_idx]
            # if mode =='TDOS':
            #     label_str = f'$W_M = {wm:.2f}$'
            # else:
            #     label_str = f'$W_M = {wm:.2f}$' if w_idx % max(1, len(wm_list)//5) == 0 else None

            if type!="invNm":
                ax.plot(x_vals, y_vals, marker='o', color=c, alpha=0.8, label=wm, markersize=6)

            # print(y_vals)

        ax.set_xlabel(r"$N_m$")

        if mode == 'TDOS':
            yname = r'$\rho_t(0)$'
        else:
            yname = r'$\rho_a(0)$'
        ax.set_ylabel(yname, rotation=0, labelpad=30)

        # naming convention for titlr
        title_suffix = r" ($W_M \leq 2.0$)" if group_name == "W_le_2" else r" ($W_M > 2.0$)"
        if sys_name == 'p10_n7':
            base_title = rf"{mode} Convergence $\{{10,3\}}$"
            if type=='Nm' and mode =='TDOS':
                ax.set_yticks([0.000, 0.045, 0.090])
                ax.set_ylim(0, 0.09)
                ax.set_xlim(0, 6500)
                ax.set_xticks(MOMENT_LIST)

            if type=='Nm' and mode =='ADOS':
                ax.set_yticks([0.000, 0.025, 0.050])
                ax.set_ylim(0, 0.05)
                ax.set_xlim(0, 6500)
                ax.set_xticks(MOMENT_LIST)



        elif sys_name == 'p8_n9':
            base_title = rf"{mode} Extrapolation $\{{8,3\}}$" if type == "invNm" else rf"{mode} Convergence $\{{8,3\}}$"
            if group_name == 'W_gt_2':
                ax.set_yticks([0.00, 0.04, 0.08])
                ax.set_ylim(0, 0.08)
            else:
                ax.set_yticks([0.00, 0.07, 0.14])
                ax.set_ylim(0, 0.14)
            ax.set_xlim(0, 9000)
            ax.set_xticks(MOMENT_LIST)
            # ax.tick_params(axis='x', labelsize=8)

        ax.set_title(base_title + title_suffix)

        # if type == "invNm":
        #     ax.set_xlim(-0.0002, max(x_vals) * 1.1)
        # else:
        #     ax.set_xticks(ticks=MOMENT_LIST)
        #     ax.tick_params(axis='x', labelsize=10)
        #     ax.set_xlim(0, 6500)

        # scalilng y axis to current group
        group_y_max = np.max([extrap_matrix[:, idx] for idx, _ in group_data])
        # ax.set_ylim(0 if type == "Nm" else None, group_y_max * 1.1)
        ax.spines[:].set_linewidth(1.5)
        ax.tick_params(axis='both', direction='in', length=4, width=1.5, pad =5)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        # suffixes for proper naming of output files
        out_dir = os.path.join(PLOT_BASE_ADOS if mode == "ADOS" else PLOT_BASE_TDOS, sys_name)
        save_path = os.path.join(out_dir, f"{mode}_{type}_{sys_name}_{group_name}.svg")
        fig.savefig(save_path)
        print(f"[{sys_name}] Saved: {save_path}")
        plt.close(fig)


def plot_comprehensive_scaling(sys_name, cfg):
    """Generates individual independent PNG figures for each scaling panel."""
    if sys_name != 'p10_n7':
        return

    print(f"[{sys_name}] Generating Individual Scaling Figures...")

    ados_data = np.array(get_ados_data(sys_name, cfg))
    tdos_raw = get_tdos_data(sys_name, cfg)
    # tdos_data = np.array(tdos_raw[-1]) #trying something out
    tdos_data = np.array(tdos_raw[0]) #full tdos

    w_vals_a = np.array(cfg['wm_list_ados'])
    w_vals_t = np.array(cfg['wm_list_tdos'])

    # critical points
    W_c1 = 0.60       # Semimetal-Metal Transition (ADOS) was 0.60
    W_c1_tdos = 0.60  # Semimetal-Metal Transition (TDOS closest point) was 0.60
    # W_c2 = 6.50       # Metal-Anderson Insulator Transition

    out_dir = os.path.join(PLOT_BASE_TDOS, sys_name)
    os.makedirs(out_dir, exist_ok=True)
    #panel (a) from paper, energy scaling $\alpha_a$ at W_c,1
    # try:
    #     fig, ax = fixed_box_figure()
    #     er_data = get_energy_resolved_data(sys_name, cfg, mode="ADOS", num_points=101)
    #     w0_data = er_data[W_c1]
    #     e_full, ados_full = w0_data['energies'], w0_data['vals']
    #
    #     mask = (e_full >= 0.002) & (e_full <= 0.04)
    #     fit_power_law(
    #         ax, e_full[mask], ados_full[mask],
    #         xlabel=r"$|E|$", ylabel=r"$\rho_a(E)$",
    #         title=r"Average DOS Energy Scaling ($W=0.75$)",
    #         color='tab:purple', param_name=r"\alpha_a",
    #         y_offset=0.0
    #     )
    #     ax.set_xlim(0,0.04)
    #     ax.set_ylim(0,0.026)
    #     ax.set_xticks([0, 0.02, 0.04])
    #     ax.set_yticks([0, 0.013, 0.026])
    #     save_path = os.path.join(out_dir, f"Scaling_Panel_A_Energy_{sys_name}.svg")
    #     plt.savefig(save_path)
    #     plt.close()
    # except Exception as e:
    #     print(f"[{sys_name}] Panel A failed: {e}")

    # panel (a) from paper, energy scaling \alpha_a at W_c,1
    try:
        fig, ax = fixed_box_figure()

        # Trigger the pi a / N dynamic grid by passing the zoom window
        er_data = get_energy_resolved_data(sys_name, cfg, mode="ADOS", zoom_window=(-0.04, 0.04))

        w0_data = er_data[W_c1]
        e_full, ados_full = w0_data['energies'], w0_data['vals']
        # 1. Isolate the strictly positive energies for the x-axis
        pos_mask = (e_full >= 0.001) & (e_full <= 0.022)
        e_pos = e_full[pos_mask]

        # 2. Extract ADOS for both positive and negative matching energies
        # (Since your dynamic grid is perfectly symmetric around E=0)
        ados_pos = ados_full[pos_mask]

        neg_mask = (e_full <= -0.001) & (e_full >= -0.022)
        # Flip the negative array so its indices align with the ascending positive energies
        ados_neg = ados_full[neg_mask][::-1]

        # 3. Average the two branches to cancel out numerical asymmetry
        ados_symmetric = (ados_pos + ados_neg) / 2.0

        # 4. Pass the symmetrized data into the fitter
        fit_power_law(
            ax, e_pos, ados_symmetric,
            xlabel=r"$|E|$", ylabel=r"$\rho_a(E)$",
            title=rf"Average DOS Energy Scaling ($W$={W_c1})",
            color='tab:purple', param_name=r"\alpha_a",
            y_offset=0.0
        )

        # mask = (e_full >= 0.001) & (e_full <= 0.028)
        # fit_power_law(
        #     ax, e_full[mask], ados_full[mask],
        #     xlabel=r"$|E|$", ylabel=r"$\rho_a(E)$",
        #     title=r"Average DOS Energy Scaling ($W=0.60$)",
        #     color='tab:purple', param_name=r"\alpha_a",
        #     y_offset=0.0
        # )
        ax.set_xlim(0, 0.02)
        ax.set_ylim(0, 0.01)
        ax.set_xticks([0.000, 0.01, 0.02])
        ax.set_yticks([0, 0.005, 0.01])
        save_path = os.path.join(out_dir, f"Scaling_Panel_A_Energy_{sys_name}.svg")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"[{sys_name}] Panel A failed: {e}")

    #panel (b) ADOS \betha_a (semimetal-metal)
    try:
        fig, ax = fixed_box_figure()
        offset_b = ados_data[w_vals_a == W_c1][0] if W_c1 in w_vals_a else np.min(ados_data)
        delta_b = (w_vals_a - W_c1) / W_c1
        mask_b = (w_vals_a >= W_c1) & (w_vals_a <2.5 * W_c1)

        fit_power_law(
            ax, delta_b[mask_b], ados_data[mask_b],
            xlabel=r"$\delta$", ylabel=r"$\rho_a(0)$",
            title=rf"ADOS Order Parameter ($\beta_a$, $W_{{c,1}}={W_c1}$)",
            color='tab:red', param_name=r"\beta_a",
            y_offset=offset_b
        )
        ax.set_xlim(0,1.0)
        ax.set_ylim(0, 0.01)
        ax.set_xticks([0.0, 0.5, 1.0] )
        ax.set_yticks([0.00, 0.005, 0.010])
        save_path = os.path.join(out_dir, f"Scaling_Panel_B_ADOS_{sys_name}.svg")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"[{sys_name}] Panel B failed: {e}")


    W_c2 = 7.00     # Metal-Anderson Insulator Transition
    # panel (d), TDOS \beta_A (Anderson Localization)
    try:
        fig, ax = fixed_box_figure()
        offset_d = tdos_data[w_vals_t == W_c2][0] if W_c2 in w_vals_t else np.min(tdos_data)
        delta_d = (W_c2 - w_vals_t) / W_c2
        mask_d = (w_vals_t <= W_c2) & (w_vals_t >=4.00)

        fit_power_law(
            ax, delta_d[mask_d], tdos_data[mask_d],
            xlabel=r"$\delta_A$", ylabel=r"$\rho_t(0)$",
            title=rf"Anderson Order Parameter ($\beta_A$, $W_{{c,2}}={W_c2}$)",
            color='tab:orange', param_name=r"\beta_A",
            y_offset=0
        )
        ax.set_xlim(0.00, 0.40)
        ax.set_ylim(0.00, 0.02)
        ax.set_xticks([0.00, 0.20, 0.40])
        ax.set_yticks([0.00, 0.01, 0.02])
        save_path = os.path.join(out_dir, f"Scaling_Panel_D_Anderson_{sys_name}.svg")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"[{sys_name}] Panel D failed: {e}")

    print(f"[{sys_name}] All scaling panels successfully saved as individual files!")

def plot_anderson_scaling_p8(sys_name, cfg):
    """Generates Panel (d) only  for the {8,3} Fermi liquid."""
    if sys_name != 'p8_n9':
        return

    print(f"[{sys_name}] Generating Anderson Localization Scaling (Panel D style)...")

    tdos_raw = get_tdos_data(sys_name, cfg)
    tdos_data = np.array(tdos_raw[0])
    w_vals_t = np.array(cfg['wm_list_tdos'])

    # metal-insulator critical point for {8,3}
    W_c2 = 6.00

    out_dir = os.path.join(PLOT_BASE_TDOS, sys_name)
    os.makedirs(out_dir, exist_ok=True)

    try:
        fig, ax = fixed_box_figure()
        offset_d = tdos_data[w_vals_t == W_c2][0] if W_c2 in w_vals_t else np.min(tdos_data)

        # Approaching the transition from the metallic side: W < W_{c2}
        delta_d = (W_c2 - w_vals_t) / W_c2
        mask_d = (w_vals_t <= W_c2) & (w_vals_t >= 3.00)

        fit_power_law(
            ax, delta_d[mask_d], tdos_data[mask_d],
            xlabel=r"$\delta_A$", ylabel=r"$\rho_t(0)$",
            title=rf"Anderson Order Parameter ($\beta_A$, $W_{{c,2}}={W_c2}$)",
            color='tab:orange', param_name=r"\beta_A",
            y_offset=0
        )
        ax.set_xlim(0, 0.5)
        ax.set_xticks([0.00, 0.25, 0.50])
        ax.set_ylim(0.00, 0.04)
        ax.set_yticks([0.000, 0.02, 0.04])
        save_path = os.path.join(out_dir, f"Scaling_Panel_D_Anderson_{sys_name}.svg")
        plt.savefig(save_path)
        plt.close()
        print(f"[{sys_name}] Saved Anderson Scaling Panel D!")
    except Exception as e:
        print(f"[{sys_name}] Panel D failed: {e}")

"""
SUPPLEMENTARY FIGURE: single-site (generation-resolved) TDOS near E = 0
"""

SUPP_SITE_INDICES = {
    'p10_n7': [0, 12, 22, 32, 42, 52, 62, 64],
    'p8_n9':  [0, 10, 18, 26, 34, 42, 50, 58, 66, 68],
}

SUPP_SITE_LABELS = {
    'p10_n7': [r"$n=1, q=3$", r"$n=2, q=3$", r"$n=3, q=3$", r"$n=4, q=3$",
               r"$n=5, q=3$", r"$n=6, q=3$", r"$n=7, q=2$", r"$n=7, q=3$"],
    'p8_n9':  [r"$n=1, q=3$", r"$n=2, q=3$", r"$n=3, q=3$", r"$n=4, q=3$",
               r"$n=5, q=3$", r"$n=6, q=3$", r"$n=7, q=3$", r"$n=8, q=3$",
               r"$n=9, q=2$", r"$n=9, q=3$"],
}

# Disorder strengths superimposed inside each panel (all W <= 1.0 that exist
# in wm_list_tdos for the given system).
SUPP_WEIGHTS = {
    'p10_n7': [0.10, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.00, 4.00, 5.00, 6.00, 7.00],
    'p8_n9':  [0.00, 0.50, 0.75, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00],
}


def get_sitewise_tdos_data(sys_name, cfg, num_points=101):
    """
    TDOS vs E for a single representative site of each generation.

    Returns {wm: {'energies': (n_E,), 'vals': (n_sites, n_E),
                  'site_indices': (n_sites,)}}

    """
    cache_dir = os.path.join(CACHE_BASE, sys_name, "TDOS_sitewise_vs_E")
    os.makedirs(cache_dir, exist_ok=True)

    site_indices = SUPP_SITE_INDICES[sys_name]
    wm_list = cfg['wm_list_tdos']
    dir_names = cfg['dir_names_tdos']
    eval_list = cfg['eval_list_tdos']

    results = {}
    for ind, wm in enumerate(wm_list):
        if wm not in SUPP_WEIGHTS[sys_name]:
            continue

        if sys_name == 'p10_n7':
            MOMENT_TARGET = 6144
        elif sys_name == 'p8_n9':
            MOMENT_TARGET = 8192

        energies = np.linspace(-0.05, 0.05, num_points)
        cache_file = os.path.join(cache_dir, f"TDOS_sitewise_E_W{wm}_{sys_name}.npz")
        if os.path.exists(cache_file):
            data = np.load(cache_file)
            results[wm] = {'energies': data['energies'], 'vals': data['vals'],
                           'site_indices': data['site_indices']}
            continue

        print(f"[{sys_name}] Computing site-resolved TDOS vs E for W_M = {wm}...")
        a = eval_list[ind]

        runs_log_ldos = []
        for i in range(60):  # disorder realizations
            file_path = f"{cfg['tdos_dir']}/{dir_names[ind]}/LDOS_mass_{i}.npy"
            if not os.path.exists(file_path):
                continue
            data = np.load(file_path)[:MOMENT_TARGET, :]
            data = data / data[0, :].reshape(1, -1)

            # one single-site "chunk" per generation. The trailing singleton
            # axis is kept so the array stays (N_m, n_sites) shaped, exactly
            # as calculate_LDOS_from_moments expects.
            site_chunks = [data[:, j:j + 1] for j in site_indices]

            run_sites = []
            for cd in site_chunks:
                ldos_vals = calculate_LDOS_from_moments(cd, energies, -a, a)
                run_sites.append(np.log(np.clip(ldos_vals, a_min=1e-14, a_max=None)))
            runs_log_ldos.append(np.array(run_sites))  # (n_sites, 1, n_E)

        if runs_log_ldos:
            runs_stack = np.array(runs_log_ldos)          # (n_cfg, n_sites, 1, n_E)
            mean_log = np.mean(runs_stack, axis=(0, 2))   # geometric mean over disorder
            vals = np.exp(mean_log)                       # (n_sites, n_E)

            np.savez(cache_file, energies=energies, vals=vals,
                     site_indices=np.array(site_indices))
            results[wm] = {'energies': energies, 'vals': vals,
                           'site_indices': np.array(site_indices)}

    return results


def plot_sitewise_tdos(sys_name, cfg, log_y=False):

    data_all = get_sitewise_tdos_data(sys_name, cfg)
    if not data_all:
        print(f"[{sys_name}] No site-resolved data found; skipping supplementary figure.")
        return

    labels = SUPP_SITE_LABELS[sys_name]
    site_indices = SUPP_SITE_INDICES[sys_name]
    plot_weights = [wm for wm in SUPP_WEIGHTS[sys_name] if wm in data_all]
    colors = cm.turbo(np.linspace(0, 1.0, len(plot_weights)))

    out_dir = os.path.join(PLOT_BASE_TDOS, sys_name, "supplementary")
    os.makedirs(out_dir, exist_ok=True)

    for s, site in enumerate(site_indices):
        fig, ax = fixed_box_figure(right=2.0)  # room for the W legend

        for wm, color in zip(plot_weights, colors):
            entry = data_all[wm]
            ax.plot(entry['energies'], entry['vals'][s, :], color=color,
                    linewidth=1.5, marker="", label=f"${wm:.2f}$")

        ax.set_xlabel(r"$E$")
        ax.set_ylabel(r"$\rho_t^{n}(E)$", rotation=0, labelpad=30)
        if log_y:
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)

        if sys_name == 'p10_n7':
            ax.set_xlim(-0.015, 0.015)
            ax.set_xticks([-0.015, 0.000, +0.015])
            if s == 0:
                ax.set_ylim(0, 0.20)
                ax.set_yticks([0.00, 0.10, 0.20])
            elif s == 1:
                ax.set_ylim(0, 0.20)
                ax.set_yticks([0.00, 0.10, 0.20])
            elif s == 2:
                ax.set_ylim(0, 0.10)
                ax.set_yticks([0.00, 0.05, 0.10])
            elif s == 3:
                ax.set_ylim(0, 0.048)
                ax.set_yticks([0.00, 0.024, 0.048])
            elif s == 4:
                ax.set_ylim(0, 0.016)
                ax.set_yticks([0.00, 0.008, 0.016])
            elif s ==5:
                ax.set_ylim(0, 0.0012)
                ax.set_yticks([0.00, 0.0006, 0.0012])
            elif s == 6:
                ax.set_ylim(0, 0.0004)
                ax.set_yticks([0.0000, 0.0002, 0.0004])
            elif s ==7:
                ax.set_ylim(0, 0.0004)
                ax.set_yticks([0.0000, 0.0002, 0.0004])

            geom = r"\{10,3\}"
        elif sys_name == 'p8_n9':
            ax.set_xlim(-0.015, 0.015)
            ax.set_xticks([-0.015, 0.000, +0.015])

            if s == 0:
                ax.set_ylim(0, 0.24)
                ax.set_yticks([0.00, 0.12, 0.24])
            elif s == 1:
                ax.set_ylim(0, 0.28)
                ax.set_yticks([0.00, 0.14, 0.28])
            elif s == 2:
                ax.set_ylim(0, 0.32)
                ax.set_yticks([0.00, 0.16, 0.32])
            elif s == 3:
                ax.set_ylim(0, 0.20)
                ax.set_yticks([0.00, 0.10, 0.20])
            elif s == 4:
                ax.set_ylim(0, 0.30)
                ax.set_yticks([0.00, 0.15, 0.30])
            elif s ==5:
                ax.set_ylim(0, 0.10)
                ax.set_yticks([0.00, 0.05, 0.10])
            elif s == 6:
                ax.set_ylim(0, 0.06)
                ax.set_yticks([0.00, 0.03, 0.06])
            elif s ==7:
                ax.set_ylim(0, 0.02)
                ax.set_yticks([0.00, 0.01, 0.02])
            elif s == 8:
                ax.set_ylim(0, 0.016)
                ax.set_yticks([0.00, 0.008, 0.016])
            elif s == 9:
                ax.set_ylim(0, 0.016)
                ax.set_yticks([0.00, 0.008, 0.016])
            geom = r"\{8,3\}"

        ax.set_title(rf"Single-site TDOS ${geom}$, {labels[s]}")
        ax.spines[:].set_linewidth(1.5)
        ax.tick_params(axis='both', direction='in', length=4, width=1.5, pad=5)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title=r"$W_M$")

        suffix = "_log" if log_y else ""
        save_path = os.path.join(
            out_dir, f"Supp_TDOS_site{site}_panel{s}_{sys_name}{suffix}.svg")
        fig.savefig(save_path)
        print(f"[{sys_name}] Saved: {save_path}")
        plt.close(fig)


"""
    Trying out extra fits:
    (a) rho_a(0)~delta^beta -> rho_a(0) = A delta^beta + B -> A^{-1} (rho_a(0) -B) = delta^beta -> delta = A^{-1/beta} (rho_a(0) -B)^{1/beta}
    (b) for bethe lattice rho_t(0) = A exp[-c delta^{-1/2}]
"""
def testing_new_fits(sys_name, cfg):
    print(f"[{sys_name}] Running new exploratory fits")

    # Grab the raw data
    ados_data = np.array(get_ados_data(sys_name, cfg))
    tdos_raw = get_tdos_data(sys_name, cfg)
    tdos_data = np.array(tdos_raw[0])

    w_vals_a = np.array(cfg['wm_list_ados'])
    w_vals_t = np.array(cfg['wm_list_tdos'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # FIT A: Linearized ADOS Scaling (Semimetal-Metal)
    W_c1 = 0.60
    if W_c1 in w_vals_a:
        # 1. Isolate the critical region
        mask_a = (w_vals_a > W_c1) & (w_vals_a <= 1.5)
        delta_a = (w_vals_a[mask_a] - W_c1) / W_c1
        rho_a = ados_data[mask_a]

        # B is the ADOS value exactly at the critical point
        B = ados_data[w_vals_a == W_c1][0]

        # 2. Fit the standard power law to extract beta
        # ln(rho - B) = ln(A) + beta * ln(delta)
        def log_fit(x, m, c): return m * x + c
        valid = (rho_a - B) > 0
        popt, _ = curve_fit(log_fit, np.log(delta_a[valid]), np.log(rho_a[valid] - B))
        beta = popt[0]

        # 3. Plot the linearized relation: y = (rho - B)^(1/beta) vs x = delta
        y_linear = (rho_a[valid] - B) ** (1.0 / beta)
        x_linear = delta_a[valid]

        ax1.plot(x_linear, y_linear, 'o-', color='tab:purple')
        ax1.set_xlabel(r"$\delta$")
        ax1.set_ylabel(r"$(\rho_a(0) - B)^{1/\beta}$")
        ax1.set_title(rf"Linearized ADOS ($\beta \approx {beta:.2f}$)")
        ax1.grid(True, linestyle='--', alpha=0.6)

    # FIT B: Essential Singularity TDOS (Anderson Transition)
    W_c2 = 7.00 if sys_name == 'p10_n7' else 6.00

    mask_t = (w_vals_t < W_c2) & (w_vals_t >= 4.0)
    delta_t = (W_c2 - w_vals_t[mask_t]) / W_c2
    rho_t = tdos_data[mask_t]

    # rho = A * exp(-c * delta^{-1/2})  =>  ln(rho) = ln(A) - c * delta^{-1/2}
    x_ess = delta_t ** (-0.5)
    y_ess = np.log(rho_t)

    valid_t = (rho_t > 0)
    popt_t, _ = curve_fit(lambda x, m, c: m * x + c, x_ess[valid_t], y_ess[valid_t])
    slope_c = -popt_t[0]

    ax2.plot(x_ess[valid_t], y_ess[valid_t], 's-', color='tab:orange')
    # Plot the line of best fit to guide the eye
    ax2.plot(x_ess[valid_t], popt_t[0] * x_ess[valid_t] + popt_t[1], 'k--', alpha=0.5)

    ax2.set_xlabel(r"$\delta^{-1/2}$")
    ax2.set_ylabel(r"$\ln(\rho_t(0))$")
    ax2.set_title(rf"Essential Singularity Fit ($c \approx {slope_c:.2f}$)")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"exploratory_fits_{sys_name}.svg")
    plt.close()
    print(f"[{sys_name}] new fits saved")

"""
Now we execute our data collection and plotting functions
"""
if __name__ == "__main__":
    for sys_name, cfg in SYSTEMS.items():
        print(f"\n******Processing System: {sys_name}*******")

        # We setup system-specific directories here
        os.makedirs(os.path.join(CACHE_BASE, sys_name), exist_ok=True)
        os.makedirs(os.path.join(PLOT_BASE_TDOS, sys_name), exist_ok=True)
        if cfg['ados_dir'] is not None:
            os.makedirs(os.path.join(PLOT_BASE_ADOS, sys_name), exist_ok=True)

        # Execute ADOS functions (Skips {8,3} automatically)
        if cfg['ados_dir'] is not None:
            plot_base_dos_vs_W(sys_name, cfg, mode="ADOS")
            plot_energy_resolved(sys_name, cfg, mode="ADOS")
            plot_convergence(sys_name, cfg, mode="ADOS", type="Nm")
        #     # plot_convergence(sys_name, cfg, mode="ADOS", type="invNm")
        #     # new exploratory fits
        #     testing_new_fits(sys_name, cfg)
        # Execute TDOS functions
        # if cfg['tdos_dir'] is not None:
        #     plot_base_dos_vs_W(sys_name, cfg, mode="TDOS")
        #     plot_tdos_plaquettes(sys_name, cfg)
        #     plot_energy_resolved(sys_name, cfg, mode="TDOS")
        #     plot_convergence(sys_name, cfg, mode="TDOS", type="Nm")
            # plot_convergence(sys_name, cfg, mode="TDOS", type="invNm")
            # plot_comprehensive_scaling(sys_name, cfg)
            # plot_anderson_scaling_p8(sys_name, cfg)

            # Supplementary figure: single-site (generation-resolved) TDOS
            # plot_sitewise_tdos(sys_name, cfg)


    print("\nAll systems processed, cached, and plotted successfully!")
