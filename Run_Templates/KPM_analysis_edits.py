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
MOMENT_LIST = [512, 1024, 2048, 4096, 8192, 16384] # Moments to evaluate for the DOS(0) v/s N_m extrapolation plot

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

        'wm_list_ados': [0.00, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.50, 2.00],
        'dir_names_ados': ['000', '025', '030', '035', '040', '045', '050', '060', '070', '075', '080', '090', '095', '100', '105', '110', '115', '120', '125', '150', '200'],
        'eval_list_ados': [3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 4.00], # energy bounds in corresponding order of weights

        'wm_list_tdos': [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 8.00, 9.00, 10.00],
        'dir_names_tdos': ['000', '010', '020', '030', '040', '050', '060', '075', '090', '100', '125', '150', '175',  '200', '250', '300', '350', '400', '450', '475', '500', '525', '550', '575', '625', '650', '675', '600', '700', '800', '900', '1000'],
        'eval_list_tdos': [3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.50, 3.50, 3.50, 3.50, 4.00, 4.00, 4.00, 4.00, 5.00, 5.00, 5.50, 5.50, 5.75, 5.75, 5.75, 6.00, 6.00, 6.00, 6.00,  6.25, 6.25, 6.25, 6.25, 6.50, 7.00, 7.50]
    },
    'p8_n9': {
        'num_chunks': 9,
        'ados_dir': None, # This is a Fermi liquid system, so skip ADOS
        'tdos_dir': './numericals/LDOS_p8_n9',

        'wm_list_tdos': [0.00, 0.50, 0.75, 1.00, 1.25, 2.00, 3.00, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 7.00, 8.00, 9.00, 10.0],
        'dir_names_tdos': ['000', '050', '075', '100', '125', '200', '300', '350', '375',  '400', '425', '450', '475', '500', '525', '550', '575', '600', '625', '650', '700', '800', '900', '1000'],
        'eval_list_tdos': [3.4, 3.4, 3.4, 3.4, 3.6, 3.6, 3.9, 4.3, 4.3, 4.3, 4.7, 4.7, 4.7, 4.7, 5.1, 5.1, 5.1, 5.1, 5.6, 5.6, 5.6, 6.0, 6.5, 7.0]
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
    for ind, subdir in enumerate(cfg['dir_names_ados']):
        current_ados = []
        for i in range(10):
            file_path = f"{cfg['ados_dir']}/{subdir}/mass_{i}.npy"
            if os.path.exists(file_path):
                moments = np.load(file_path)[:MOMENT_TARGET]
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

def get_energy_resolved_data(sys_name, cfg, mode="TDOS", num_points=101): #compute for 3001 energy points
    cache_dir = os.path.join(CACHE_BASE, sys_name, f"{mode}_vs_E")
    os.makedirs(cache_dir, exist_ok=True)
    results = {}

    # we select weights list and corresponding energy bounds list depending on the case we look at
    wm_list = cfg['wm_list_tdos'] if mode == "TDOS" else cfg['wm_list_ados']
    dir_names = cfg['dir_names_tdos'] if mode == "TDOS" else cfg['dir_names_ados']
    eval_list = cfg['eval_list_tdos'] if mode == "TDOS" else cfg['eval_list_ados']

    for ind, wm in enumerate(wm_list):
        if wm>2.0:
            continue
        cache_file = os.path.join(cache_dir, f"{mode}_E_W{wm}_{sys_name}.npz")
        if os.path.exists(cache_file):
            data = np.load(cache_file)
            results[wm] = {'energies': data['energies'], 'vals': data['vals']}
            continue

        print(f"[{sys_name}] Computing {mode} vs E for W_M = {wm}...")
        # energies = np.linspace(-eval_list[ind], eval_list[ind], num_points)
        energies = np.linspace(-0.05, 0.05, num_points)

        if mode == "ADOS":
            runs = []
            for i in range(10): # number of disorder realization
                file_path = f"{cfg['ados_dir']}/{dir_names[ind]}/mass_{i}.npy"
                if os.path.exists(file_path):
                    moments = np.load(file_path)[:MOMENT_TARGET]
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

        # for m_idx, Nm in enumerate(MOMENT_LIST):
        #     if runs_per_moment[Nm]:
        #         if mode == "TDOS":
        #             extrap_matrix[m_idx, ind] = np.exp(np.mean(np.array(runs_per_moment[Nm])))
        #         else:
        #             extrap_matrix[m_idx, ind] = np.mean(runs_per_moment[Nm])

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
    ax.legend(loc="best")

"""
Plotting functions for all the analyses we carried out.
# """
# def plot_base_dos_vs_W(sys_name, cfg, mode="TDOS"):
#     # Fig 1a/1b: DOS at E=0 vs W
#     ax.figure(figsize=(6, 6))
#     if mode == "ADOS":
#         vals = get_ados_data(sys_name, cfg)
#         ax.plot(cfg['wm_list_ados'], vals, marker='o', color='tab:red', linewidth=2, markersize=6)
#         ax.xticks([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
#         ax.yticks([0.00, 0.02, 0.04])
#         ax.ylim(0,0.04)
#         ax.xlim(0, cfg['wm_list_ados'][-1])
#         ax.ylabel(r" $\rho_a(0)$", rotation=0, labelpad=30)
#     else:
#         vals, _ = get_tdos_data(sys_name, cfg)
#         ax.plot(cfg['wm_list_tdos'], vals, marker='o', color='tab:red', linewidth=2, markersize=6)
#         ax.xticks([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
#         ax.xlim(0,10)
#         ax.ylabel(r"$\rho_t(0)$", rotation=0, labelpad=30)
#
#     ax = ax.gca()
#     ax.spines[:].set_linewidth(1.5)
#     ax.tick_params(axis='x', direction='in', length=4, width=1.5)
#     ax.tick_params(axis='y', direction='in', length=4, width=1.5)
#
#
#     ax.ylim(bottom=0)
#     ax.xlabel(r"$W_M$")
#     if sys_name=='p10_n7':
#         ax.title(rf"{mode} at E=0 vs Disorder Strength $\{{ 10, 3 \}} $")
#     elif sys_name=='p8_n9':
#         ax.title(rf"{mode} at E=0 vs Disorder Strength $\{{ 8, 3\}}$")
#
#     out_dir = os.path.join(PLOT_BASE_ADOS if mode=="ADOS" else PLOT_BASE_TDOS, sys_name)
#     save_path = os.path.join(out_dir, f"{mode}_vs_W_{sys_name}.svg")
#     plt.savefig(save_path)
#     print(f"[{sys_name}] Saved: {save_path}")
#     plt.close()

def plot_base_dos_vs_W(sys_name, cfg, mode="TDOS"):
    # Fig 1a/1b: DOS at E=0 vs W
    if mode == "ADOS":
        # --- ADOS STANDARD PLOT ---
        fig, ax = fixed_box_figure()
        vals = get_ados_data(sys_name, cfg)
        ax.plot(cfg['wm_list_ados'], vals, ls='', marker='o', color='tab:red', linewidth=2, markersize=6)

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
        fig, (ax1, ax2) = fixed_box_row(n=2, width_ratios=[1, 1], gap=0.03, sharey=True)
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



        # 2. Plot data on BOTH axes
        ax1.plot(w_vals_filtered, vals_filtered, ls='',marker='o', color='tab:red', linewidth=2, markersize=6)
        ax2.plot(w_vals_filtered, vals_filtered, ls='',marker='o', color='tab:red', linewidth=2, markersize=6)

        # 3. Limit the X-axes to create the "break"
        ax1.set_xlim(0.0, 1.07)
        ax2.set_xlim(1.60, 10.0)
        ax1.set_ylim(bottom=0)

        # Apply the specific requested ticks
        ax1.set_xticks([0.0, 0.5, 1.0])
        my_ticks = [2.0, 5.0, 10.0]
        ax2.set_xticks(my_ticks, labels=[f"{val:.1f}" for val in my_ticks])
        # ax2.set_xticks([2.0, 5.0, 10.0])
        ax1.set_yticks([0.000, 0.025, 0.050])

        # 4. Hide the inner spines between them
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params(left=False) # Hide duplicate y-ticks on the right

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

        ax.plot(wm_tdos_filtered, vals_filtered, ls='', marker='o', color='tab:red', linewidth=2, markersize=6)
        my_ticks = [0.0, 1.0, 2.0, 5.0, 10.0]
        ax.set_xticks(my_ticks, labels=[f"{val:.1f}" for val in my_ticks])
        # ax.set_xticks([0.0, 1.0, 2.0, 5.0, 10.0])
        ax.set_yticks([0.000, 0.045, 0.090])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 0.09)

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
    # ax.figure(figsize=(10, 6))
    fixed_box_figure()
    colors = cm.viridis(np.linspace(0, 1, cfg['num_chunks']))

    for i in range(cfg['num_chunks']):
        ax.plot(cfg['wm_list_tdos'], plaquettes[i, :], 's-', label=rf"$n= {i}$", color=colors[i], alpha=0.8, markersize=6)
    ax.xlim(0,10)
    ax.ylim(bottom=0)
    ax.xlabel(r"$W_M$")
    ax.ylabel(r" $\rho_t^n(0)$", rotation=0, labelpad=30)
    if sys_name=='p10_n7':
        ax.title(rf"Generation-resolved TDOS, $\rho_t^n(E=0), \{{10, 3\}}$")
    elif sys_name=='p8_n9':
        ax.title(rf"Generation-resolved TDOS, $\rho_t^n(E=0)$, \{{8, 3\}}")
    ax.legend(loc="upper right")

    out_dir = os.path.join(PLOT_BASE_TDOS, sys_name)
    save_path = os.path.join(out_dir, f"Plaquettes_{sys_name}.svg")
    plt.savefig(save_path)
    print(f"[{sys_name}] Saved: {save_path}")
    plt.close()

# def plot_energy_resolved(sys_name, cfg, mode="TDOS"):
#     # Fig 1 Insets: DOS vs Energy
#     data = get_energy_resolved_data(sys_name, cfg, mode)
#     ax.figure(figsize=(6, 6))
#
#     wm_list = cfg['wm_list_tdos'] if mode == "TDOS" else cfg['wm_list_ados']
#     # if mode == "ADOS":
#     #             # cutting down list
#     #     allowed_weights = [0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
#     # elif mode=="TDOS" and sys_name=="p10_n7":
#     #     allowed_weights = [0.00, 0.20, 0.30, 0.50, 1.00, 2.00]
#     # elif mode=="TDOS" and sys_name=="p8_n9":
#     #     allowed_weights = [0.00, 0.50, 0.75, 1.00, 1.25, 2.00]
#
#     # plot_weights = [wm for wm in allowed_weights]
#     # 1. Grab every single weight available in your loaded data
#     plot_weights = list(data.keys())
#
#     # 2. Stretch the turbo colormap across the total number of weights
#     # Using 0.1 to 0.9 prevents the very darkest and lightest colors from blending into the background
#     colors = cm.turbo(np.linspace(0.1,0.9, len(plot_weights)))
#
#     for wm, color in zip(plot_weights, colors):
#         plot_data = data[wm]
#         # label_str = f"${wm:.2f}$" if wm in [0.00, 0.25, 0.50, 1.00, 1.50, 2.00, 5.00, 10.00] else None
#         label_str = f"${wm:.2f}$"
#         ax.plot(plot_data['energies'], plot_data['vals'], label=label_str, color=color, linewidth=1.5, marker="")
#
#     ax.xlabel(r"$E$")
#     if mode=="TDOS" and sys_name=="p8_n9":
#         ax.ylabel(rf"$\rho_t(E)$", rotation=0, labelpad=30)
#         ax.ylim(0,0.2)
#         ax.xlim(-0.03,0.03)
#
#     elif mode=="TDOS" and sys_name=="p10_n7":
#         ax.ylabel(rf"$\rho_t(E)$", rotation=0, labelpad=30)
#         ax.ylim(0,0.15)
#         ax.xlim(-0.025,0.025)
#
#     elif mode=="ADOS":
#         ax.ylabel(rf"$\rho_a(E)$", rotation=0, labelpad=30)
#         ax.ylim(0,0.06)
#         ax.xlim(-0.04,0.04)
#
#     if sys_name=='p10_n7':
#         ax.title(rf"{mode} v/s Energy $\{{ 10, 3\}}$")
#     elif sys_name=='p8_n9':
#         ax.title(rf"{mode} v/s Energy $\{{ 8, 3\}}$")
#     # ax.legend(loc="upper right")
#     ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
#
#     out_dir = os.path.join(PLOT_BASE_ADOS if mode=="ADOS" else PLOT_BASE_TDOS, sys_name)
#     save_path = os.path.join(out_dir, f"{mode}_vs_E_{sys_name}.svg")
#     plt.savefig(save_pathdpi=300)
#     print(f"[{sys_name}] Saved: {save_path}")
#     plt.close()


def plot_energy_resolved(sys_name, cfg, mode="TDOS"):
    # Fig 1 Insets: DOS vs Energy
    data = get_energy_resolved_data(sys_name, cfg, mode)

    wm_list = cfg['wm_list_tdos'] if mode == "TDOS" else cfg['wm_list_ados']
    plot_weights = list(data.keys())
    colors = cm.turbo(np.linspace(0.1, 0.9, len(plot_weights)))

    for wm, color in zip(plot_weights, colors):
        plot_data = data[wm]
        label_str = f"${wm:.2f}$"
        ax.plot(plot_data['energies'], plot_data['vals'], label=label_str, color=color, linewidth=1.5, marker="")

    ax.set_xlabel(r"$E$")
    if mode=="TDOS" and sys_name=="p8_n9":
        ax.set_ylabel(rf"$\rho_t(E)$", rotation=0, labelpad=30)
        ax.set_ylim(0,0.2)
        ax.set_xlim(-0.03,0.03)
    elif mode=="TDOS" and sys_name=="p10_n7":
        ax.set_ylabel(rf"$\rho_t(E)$", rotation=0, labelpad=30)
        ax.set_ylim(0,0.15)
        ax.set_xlim(-0.025,0.025)
    elif mode=="ADOS":
        ax.set_ylabel(rf"$\rho_a(E)$", rotation=0, labelpad=30)
        ax.set_ylim(0,0.06)
        ax.set_xlim(-0.04,0.04)

    if sys_name=='p10_n7':
        ax.set_title(rf"{mode} v/s Energy $\{{ 10, 3\}}$")
    elif sys_name=='p8_n9':
        ax.set_title(rf"{mode} v/s Energy $\{{ 8, 3\}}$")

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    out_dir = os.path.join(PLOT_BASE_ADOS if mode=="ADOS" else PLOT_BASE_TDOS, sys_name)
    save_path = os.path.join(out_dir, f"{mode}_vs_E_{sys_name}.svg")
    fig.savefig(save_path)
    print(f"[{sys_name}] Saved: {save_path}")
    plt.close(fig)

def plot_convergence(sys_name, cfg, mode="TDOS", type="invNm"):
    # Fig 2a-c: Convergence vs Nm or 1/Nm
    extrap_matrix = get_convergence_data(sys_name, cfg, mode)
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

        # ax.figure(figsize=(10, 6))
        fixed_box_figure()
        # we generate colors specific only to the number of lines in current plot
        colors = cm.viridis(np.linspace(0, 1, len(group_data)))

        for c_idx, (w_idx, wm) in enumerate(group_data):
            y_vals = extrap_matrix[:, w_idx]
            c = colors[c_idx]
            if mode =='TDOS':
                label_str = f'$W_M = {wm:.2f}$'
            else:
                label_str = f'$W_M = {wm:.2f}$' if w_idx % max(1, len(wm_list)//5) == 0 else None

            if type == "invNm":
                slope, intercept, r, p, err = linregress(x_vals[-4:], y_vals[-4:])
                x_fit = np.linspace(0, max(x_vals), 100)
                ax.plot(x_vals, y_vals, 's-', color=c, alpha=0.8, markersize=6)
                ax.plot(x_fit, slope * x_fit + intercept, '--', color=c, alpha=0.5, label=label_str)
                ax.plot(0, intercept, '*', color=c, markersize=6, markeredgecolor='black', alpha=0.9)
            else:
                ax.plot(x_vals, y_vals, 's-', color=c, alpha=0.8, label=label_str, markersize=6)

        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        if type == "invNm": ax.axvline(0, color='black', linewidth=1)

        ax.xlabel(r"$1/N_m$" if type == "invNm" else r"$N_m$")

        if mode == 'TDOS':
            yname = r'$\rho_t(0)$'
        else:
            yname = r'$\rho_a(0)$'
        ax.ylabel(yname, rotation=0, labelpad=30)

        # naming convention for titlr
        title_suffix = r" ($W_M \leq 2.0$)" if group_name == "W_le_2" else r" ($W_M > 2.0$)"
        if sys_name == 'p10_n7':
            base_title = rf"{mode} Extrapolation $\{{10,3\}}$" if type == "invNm" else rf"{mode} Convergence $\{{10,3\}}$"
        elif sys_name == 'p8_n9':
            base_title = rf"{mode} Extrapolation $\{{8,3\}}$" if type == "invNm" else rf"{mode} Convergence $\{{8,3\}}$"
        ax.title(base_title + title_suffix)

        if type == "invNm":
            ax.xlim(-0.0002, max(x_vals) * 1.1)
        else:
            ax.xticks(ticks=MOMENT_LIST, fontsize=10)
            ax.xlim(400, 16500)

        # scalilng y axis to current group
        group_y_max = np.max([extrap_matrix[:, idx] for idx, _ in group_data])
        ax.ylim(0 if type == "Nm" else None, group_y_max * 1.1)

        ax.legend(loc="upper right")

        # suffixes for proper naming of output files
        out_dir = os.path.join(PLOT_BASE_ADOS if mode == "ADOS" else PLOT_BASE_TDOS, sys_name)
        save_path = os.path.join(out_dir, f"{mode}_{type}_{sys_name}_{group_name}.svg")
        plt.savefig(save_path)
        print(f"[{sys_name}] Saved: {save_path}")
        plt.close()


def plot_comprehensive_scaling(sys_name, cfg):
    """Generates individual independent PNG figures for each scaling panel."""
    if sys_name != 'p10_n7':
        return

    print(f"[{sys_name}] Generating Individual Scaling Figures...")

    ados_data = np.array(get_ados_data(sys_name, cfg))
    tdos_raw = get_tdos_data(sys_name, cfg)
    tdos_data = np.array(tdos_raw[-1])

    w_vals_a = np.array(cfg['wm_list_ados'])
    w_vals_t = np.array(cfg['wm_list_tdos'])

    # critical points
    W_c1 = 0.75       # Semimetal-Metal Transition (ADOS)
    W_c1_tdos = 0.75  # Semimetal-Metal Transition (TDOS closest point)
    W_c2 = 6.50       # Metal-Anderson Insulator Transition

    out_dir = os.path.join(PLOT_BASE_TDOS, sys_name)
    os.makedirs(out_dir, exist_ok=True)
    #panel (a) from paper, energy scaling $\alpha_a$ at W_c,1
    try:
        fig, ax = fixed_box_figure()
        er_data = get_energy_resolved_data(sys_name, cfg, mode="ADOS", num_points=3000)
        w0_data = er_data[W_c1]
        e_full, ados_full = w0_data['energies'], w0_data['vals']

        mask = (e_full >= 0.002) & (e_full <= 0.04)
        fit_power_law(
            ax, e_full[mask], ados_full[mask],
            xlabel=r"$|E|$", ylabel=r"$\rho_a(E)$",
            title=r"Average DOS Energy Scaling ($W=0.75$)",
            color='tab:purple', param_name=r"\alpha_a",
            y_offset=0.0
        )
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
            xlabel=r"$\delta = (W - W_{c,1})/W_{c,1}$", ylabel=r"$\rho_a(0)$",
            title=rf"ADOS Order Parameter ($\beta_a$, $W_{{c,1}}={W_c1}$)",
            color='tab:red', param_name=r"\beta_a",
            y_offset=offset_b
        )
        save_path = os.path.join(out_dir, f"Scaling_Panel_B_ADOS_{sys_name}.svg")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"[{sys_name}] Panel B failed: {e}")

    # panel(c) TDOS, \beta_t at metal-insulator
    # commented out right now since we lack data

    # try:
    #     fig, ax = ax.subplots(figsize=(7, 5))
    #     offset_c = tdos_data[0, w_vals_t == W_c1_tdos][0] if W_c1_tdos in w_vals_t else np.min(tdos_data[0])
    #     delta_c = (w_vals_t - W_c1_tdos) / W_c1_tdos
    #     mask_c = (w_vals_t >= W_c1_tdos) & (w_vals_t <= 2.0)
    #
    #     fit_power_law(
    #         ax, delta_c[mask_c], tdos_data[0, mask_c],
    #         xlabel=r"$\delta = (W - W_{c,1})/W_{c,1}$", ylabel=r"$\rho_t^{(0)}(0)$",
    #         title=rf"TDOS Order Parameter ($\beta_t$, $W_{{c,1}}={W_c1_tdos}$)",
    #         color='tab:blue', param_name=r"\beta_t",
    #         y_offset=offset_c
    #     )
    #     # ax.tight_layout()
    #     save_path = os.path.join(out_dir, f"Scaling_Panel_C_TDOS_{sys_name}.svg")
    #     plt.savefig(save_path)
    #     plt.close()
    # except Exception as e:
    #     print(f"[{sys_name}] Panel C failed: {e}")

    # panel (d), TDOS \beta_A (Anderson Localization)
    try:
        fig, ax = fixed_box_figure()
        offset_d = tdos_data[0, w_vals_t == W_c2][0] if W_c2 in w_vals_t else np.min(tdos_data[0])
        delta_d = (W_c2 - w_vals_t) / W_c2
        mask_d = (w_vals_t <= W_c2) & (w_vals_t >=4.75)

        fit_power_law(
            ax, delta_d[mask_d], tdos_data[0, mask_d],
            xlabel=r"$\delta_A = (W_{c,2} - W)/W_{c,2}$", ylabel=r"$\rho_t^{(0)}(0)$",
            title=rf"Anderson Order Parameter ($\beta_A$, $W_{{c,2}}={W_c2}$)",
            color='tab:orange', param_name=r"\beta_A",
            y_offset=offset_d
        )
        # ax.tight_layout()
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
    tdos_data = np.array(tdos_raw[-1])
    w_vals_t = np.array(cfg['wm_list_tdos'])

    # metal-insulator critical point for {8,3}
    W_c2 = 6.00

    out_dir = os.path.join(PLOT_BASE_TDOS, sys_name)
    os.makedirs(out_dir, exist_ok=True)

    try:
        fig, ax = fixed_box_figure()
        offset_d = tdos_data[0, w_vals_t == W_c2][0] if W_c2 in w_vals_t else np.min(tdos_data[0])

        # Approaching the transition from the metallic side: W < W_{c2}
        delta_d = (W_c2 - w_vals_t) / W_c2
        mask_d = (w_vals_t <= W_c2) & (w_vals_t >= 4.75)

        fit_power_law(
            ax, delta_d[mask_d], tdos_data[0, mask_d],
            xlabel=r"$\delta_A = (W_{c,2} - W)/W_{c,2}$", ylabel=r"$\rho_t^{(0)}(0)$",
            title=rf"Anderson Order Parameter ($\beta_A$, $W_{{c,2}}={W_c2}$)",
            color='tab:orange', param_name=r"\beta_A",
            y_offset=offset_d
        )
        save_path = os.path.join(out_dir, f"Scaling_Panel_D_Anderson_{sys_name}.svg")
        plt.savefig(save_path)
        plt.close()
        print(f"[{sys_name}] Saved Anderson Scaling Panel D!")
    except Exception as e:
        print(f"[{sys_name}] Panel D failed: {e}")

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
            plot_convergence(sys_name, cfg, mode="ADOS", type="invNm")


        # Execute TDOS functions
        if cfg['tdos_dir'] is not None:
            plot_base_dos_vs_W(sys_name, cfg, mode="TDOS")
            plot_tdos_plaquettes(sys_name, cfg)
            plot_energy_resolved(sys_name, cfg, mode="TDOS")
            plot_convergence(sys_name, cfg, mode="TDOS", type="Nm")
            plot_convergence(sys_name, cfg, mode="TDOS", type="invNm")
            plot_comprehensive_scaling(sys_name, cfg)
            plot_anderson_scaling_p8(sys_name, cfg)


    print("\nAll systems processed, cached, and plotted successfully!")
