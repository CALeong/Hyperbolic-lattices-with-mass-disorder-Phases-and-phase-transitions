import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
from KPM.measure import calculate_ADOS_from_moments, calculate_LDOS_from_moments


"""
General configuration and parameters for our code. We will save directory paths here pointing toward where the existing data is stored. We also specify 'cache' folders, where we save computed TDOS and ADOS values so that we do not have to recompute these values over and over while making plotting tweaks.
"""
P_VAL = 10 # For the {10,3} system
NUM_CHUNKS = 7
# Moments to evaluate for the DOS(0) v/s N_m extrapolation plot
MOMENT_LIST = [512, 1024, 2048, 4096, 8192, 16384]
MOMENT_TARGET = 16384 # Default max moment for our analyses

# Paths to directories with numerical moment data
ADOS_DIR = './numericals/ADOS_p10_n7'
TDOS_DIR = './numericals/LDOS_p10_n7'

# Cache cirectory stores computed ADOS/TDOS values to access quickly while plotting
CACHE_DIR = './analysis_cache'

# Cache subdirectories to store ADOS/TDOS v/s E for different weights as different files
# This allows us to add/remove new weights without recomputing everything fully
CACHE_DIR_ADOS_E = os.path.join(CACHE_DIR, 'ADOS_vs_E')
CACHE_DIR_TDOS_E = os.path.join(CACHE_DIR, 'TDOS_vs_E')

PLOT_DIR_ADOS = './numericals/plots/ADOS' # save directory for ADOS plots
PLOT_DIR_TDOS = './numericals/plots/TDOS' # save directory for TDOS plots

# Safely create all output directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR_ADOS_E, exist_ok=True)
os.makedirs(CACHE_DIR_TDOS_E, exist_ok=True)
os.makedirs(PLOT_DIR_ADOS, exist_ok=True)
os.makedirs(PLOT_DIR_TDOS, exist_ok=True)

# ADOS Run Parameters
WM_LIST_ADOS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75, 1.00] # list of weights
EVAL_LIST_ADOS = [3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.50, 3.50] # corresponding energy limits
DIR_NAMES_ADOS = ['025', '030', '035', '040', '045', '050', '075', '100'] # specific directory names

# TDOS Run Parameters
WM_LIST_TDOS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00] # list of weights
EVAL_LIST_TDOS = [3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.50, 4.00, 5.00, 5.50, 5.75, 6.00, 6.25, 6.50, 7.00, 7.5] # corresponding energy limits
DIR_NAMES_TDOS = ['000', '010', '020', '030', '040', '050', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000'] #specific directory names


"""
Main computation functions. These functions exist to carry out the following analyses (compute and save data for these):
1. get_ados_vs_w_data: ADOS(E=0) v/s W
2. get_tdos_vs_w_data: TDOS(E=0) v/s W
3. get_ados_vs_e_data: ADOS v/s E upto given energy range for all W
4. get_tdos_vs_e_data: TDOS v/s E upto given energy range for all W
5.
"""
def get_ados_vs_w_data():
    cache_file = f"{CACHE_DIR}/ados_vs_w_{MOMENT_TARGET}.npy"
    if os.path.exists(cache_file):
        return np.load(cache_file)

    print("Computing ADOS vs W at E=0...")
    ados_results = []
    for ind, subdir in enumerate(DIR_NAMES_ADOS):
        current_ados = []
        for i in range(10): # 10 ADOS realizations
            file_path = f"{ADOS_DIR}/{subdir}/mass_{i}.npy"
            if os.path.exists(file_path):
                moments = np.load(file_path)[:MOMENT_TARGET]
                val = calculate_ADOS_from_moments(moments, np.array([0.0]), -EVAL_LIST_ADOS[ind], EVAL_LIST_ADOS[ind])
                current_ados.append(val)
        ados_results.append(np.mean(current_ados))

    np.save(cache_file, ados_results)
    return np.array(ados_results)

def get_tdos_vs_w_data():
    cache_file = f"{CACHE_DIR}/tdos_vs_w_{MOMENT_TARGET}.npz"
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data['global_tdos'], data['plaquette_tdos']

    print("Computing TDOS vs W at E=0...")
    global_tdos = []
    plaquette_tdos = np.zeros((NUM_CHUNKS, len(WM_LIST_TDOS)))

    for ind, subdir in enumerate(DIR_NAMES_TDOS):
        runs_log_ldos = []
        for i in range(60): # 60 TDOS realizations
            file_path = f"{TDOS_DIR}/{subdir}/LDOS_mass_{i}.npy"
            if os.path.exists(file_path):
                data = np.load(file_path)[:MOMENT_TARGET, :]
                data = data / data[0, :].reshape(1, -1)
                chunked = np.split(data, NUM_CHUNKS, axis=1)

                run_chunks = []
                for cd in chunked:
                    ldos_vals = calculate_LDOS_from_moments(cd, np.array([0.0]), -EVAL_LIST_TDOS[ind], EVAL_LIST_TDOS[ind])
                    safe_ldos = np.clip(ldos_vals, a_min=1e-14, a_max=None)
                    run_chunks.append(np.log(safe_ldos).reshape(-1))
                runs_log_ldos.append(run_chunks)

        # Collapse arrays
        runs_stack = np.array(runs_log_ldos) # Shape: (runs, chunks, sites)
        plaquette_tdos[:, ind] = np.exp(np.mean(runs_stack, axis=(0, 2)))
        global_tdos.append(np.exp(np.mean(runs_stack)))

    np.savez(cache_file, global_tdos=global_tdos, plaquette_tdos=plaquette_tdos)
    return np.array(global_tdos), plaquette_tdos

def get_ados_vs_e_data(target_indices, num_points=500):
    """Computes and caches ADOS across an energy range for specific W_M indices."""
    print("Fetching ADOS vs E data for selected weights...")
    results = {}

    for ind in target_indices:
        wm = WM_LIST_ADOS[ind]
        subdir = DIR_NAMES_ADOS[ind]
        e_bound = EVAL_LIST_ADOS[ind]

        # --- NEW: Route to dedicated ADOS_vs_E cache folder ---
        cache_file = f"{CACHE_DIR_ADOS_E}/ados_vs_e_W{wm}_{MOMENT_TARGET}.npz"

        if os.path.exists(cache_file):
            print(f"  -> Loading W_M = {wm} from cache...")
            data = np.load(cache_file)
            results[wm] = {'energies': data['energies'], 'ados': data['ados']}
            continue

        print(f"  -> Computing W_M = {wm} from scratch...")
        energies = np.linspace(-e_bound, e_bound, num_points)
        current_ados_runs = []

        for i in range(10): # 10 ADOS realizations
            file_path = f"{ADOS_DIR}/{subdir}/mass_{i}.npy"
            if os.path.exists(file_path):
                moments = np.load(file_path)[:MOMENT_TARGET]
                val = calculate_ADOS_from_moments(moments, energies, -e_bound, e_bound)
                current_ados_runs.append(val)

        if current_ados_runs:
            mean_ados = np.mean(current_ados_runs, axis=0)
            np.savez(cache_file, energies=energies, ados=mean_ados)
            results[wm] = {'energies': energies, 'ados': mean_ados}

    return results

def get_tdos_vs_e_data(target_indices, num_points=500):
    """Computes and caches global TDOS across an energy range for specific W_M indices."""
    print("Fetching TDOS vs E data for selected weights...")
    results = {}

    for ind in target_indices:
        wm = WM_LIST_TDOS[ind]
        subdir = DIR_NAMES_TDOS[ind]
        e_bound = EVAL_LIST_TDOS[ind]

        # --- NEW: Route to dedicated TDOS_vs_E cache folder ---
        cache_file = f"{CACHE_DIR_TDOS_E}/tdos_vs_e_W{wm}_{MOMENT_TARGET}.npz"

        if os.path.exists(cache_file):
            print(f"  -> Loading W_M = {wm} from cache...")
            data = np.load(cache_file)
            results[wm] = {'energies': data['energies'], 'tdos': data['tdos']}
            continue

        print(f"  -> Computing W_M = {wm} from scratch...")
        energies = np.linspace(-e_bound, e_bound, num_points)
        runs_log_ldos = []

        for i in range(60): # 60 TDOS realizations
            file_path = f"{TDOS_DIR}/{subdir}/LDOS_mass_{i}.npy"
            if os.path.exists(file_path):
                data = np.load(file_path)[:MOMENT_TARGET, :]
                data = data / data[0, :].reshape(1, -1)
                chunked = np.split(data, NUM_CHUNKS, axis=1)

                run_chunks = []
                for cd in chunked:
                    ldos_vals = calculate_LDOS_from_moments(cd, energies, -e_bound, e_bound)
                    safe_ldos = np.clip(ldos_vals, a_min=1e-14, a_max=None)
                    run_chunks.append(np.log(safe_ldos))

                runs_log_ldos.append(np.array(run_chunks))

        if runs_log_ldos:
            runs_stack = np.array(runs_log_ldos)
            mean_log_tdos = np.mean(runs_stack, axis=(0, 1, 2))
            physical_tdos = np.exp(mean_log_tdos)

            np.savez(cache_file, energies=energies, tdos=physical_tdos)
            results[wm] = {'energies': energies, 'tdos': physical_tdos}

    return results

def get_ados_convergence_data():
    """Computes and caches ADOS at E=0 across multiple moment cuts."""
    cache_file = f"{CACHE_DIR}/ados_convergence_matrix.npy"
    if os.path.exists(cache_file):
        return np.load(cache_file)

    print("Computing ADOS extrapolation data (slicing master arrays)...")
    extrap_matrix = np.zeros((len(MOMENT_LIST), len(WM_LIST_ADOS)))

    for ind, subdir in enumerate(DIR_NAMES_ADOS):
        runs_per_moment = {Nm: [] for Nm in MOMENT_LIST}

        for i in range(10): # 10 ADOS realizations
            file_path = f"{ADOS_DIR}/{subdir}/mass_{i}.npy"
            if os.path.exists(file_path):
                full_moments = np.load(file_path)

                # Slice the array in memory for each moment target
                for Nm in MOMENT_LIST:
                    sliced_moments = full_moments[:Nm]
                    val = calculate_ADOS_from_moments(sliced_moments, np.array([0.0]), -EVAL_LIST_ADOS[ind], EVAL_LIST_ADOS[ind])
                    runs_per_moment[Nm].append(val)

        for m_idx, Nm in enumerate(MOMENT_LIST):
            if runs_per_moment[Nm]:
                extrap_matrix[m_idx, ind] = np.mean(runs_per_moment[Nm])

    np.save(cache_file, extrap_matrix)
    return extrap_matrix

def get_moments_convergence_data():
    """Computes and caches global TDOS at E=0 across multiple moment cuts."""
    cache_file = f"{CACHE_DIR}/tdos_convergence_matrix.npy"
    if os.path.exists(cache_file):
        return np.load(cache_file)

    print("Computing TDOS extrapolation data (slicing master arrays)...")

    # We want a 2D array: Rows = moment sizes, Columns = W_M values
    extrap_matrix = np.zeros((len(MOMENT_LIST), len(WM_LIST_TDOS)))

    for ind, subdir in enumerate(DIR_NAMES_TDOS):
        # Dictionary to hold the chunks for each moment cut
        runs_per_moment = {Nm: [] for Nm in MOMENT_LIST}

        for i in range(60): # 60 TDOS realizations
            file_path = f"{TDOS_DIR}/{subdir}/LDOS_mass_{i}.npy"
            if os.path.exists(file_path):
                # Load the full 16k array once per realization
                full_data = np.load(file_path)
                full_data = full_data / full_data[0, :].reshape(1, -1)

                # Slice the array in memory for each moment target
                for Nm in MOMENT_LIST:
                    sliced_data = full_data[:Nm, :]
                    chunked = np.split(sliced_data, NUM_CHUNKS, axis=1)

                    run_chunks = []
                    for cd in chunked:
                        ldos_vals = calculate_LDOS_from_moments(cd, np.array([0.0]), -EVAL_LIST_TDOS[ind], EVAL_LIST_TDOS[ind])
                        safe_ldos = np.clip(ldos_vals, a_min=1e-14, a_max=None)
                        run_chunks.append(np.log(safe_ldos).reshape(-1))

                    runs_per_moment[Nm].append(np.array(run_chunks))

        # Collapse the data and store physical TDOS
        for m_idx, Nm in enumerate(MOMENT_LIST):
            if runs_per_moment[Nm]:
                runs_stack = np.array(runs_per_moment[Nm]) # Shape: (runs, chunks, sites)
                mean_log = np.mean(runs_stack) # Average across all spatial and realization axes
                extrap_matrix[m_idx, ind] = np.exp(mean_log)

    np.save(cache_file, extrap_matrix)
    return extrap_matrix
# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_ados_vs_W():
    """Fig 1a: ADOS at E=0 as a function of W"""
    ados_vals = get_ados_vs_w_data()

    plt.figure(figsize=(8, 5))
    plt.plot(WM_LIST_ADOS, ados_vals, 'o-', color='tab:blue', linewidth=2)
    plt.xlabel(r"Disorder Strength ($W_M$)")
    plt.ylabel(r"Global Average DOS, $\rho(E=0)$")
    plt.title("ADOS at E=0 vs Disorder Strength")
    plt.grid(alpha=0.3)
    # plt.tight_layout()

    save_path = os.path.join(PLOT_DIR_ADOS, "Fig1a_ADOS_vs_W.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

def plot_tdos_vs_W():
    """Fig 1b: TDOS at E=0 as a function of W"""
    global_tdos, _ = get_tdos_vs_w_data()

    plt.figure(figsize=(8, 5))
    plt.plot(WM_LIST_TDOS, global_tdos, 's-', color='tab:red', linewidth=2)
    plt.xlabel(r"Disorder Strength ($W_M$)")
    plt.ylabel(r"Global Typical DOS, $\rho_t(E=0)$")
    plt.title("TDOS at E=0 vs Disorder Strength")
    plt.xlim(0, 10.5)
    plt.grid(alpha=0.3)
    # plt.tight_layout()

    save_path = os.path.join(PLOT_DIR_TDOS, "Fig1b_TDOS_vs_W.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

def plot_tdos_plaquettes_vs_W():
    """Fig 2d: TDOS at E=0 as a function of W for different plaquettes"""
    _, plaquette_tdos = get_tdos_vs_w_data()

    plt.figure(figsize=(10, 6))
    colors = cm.viridis(np.linspace(0, 1, NUM_CHUNKS))

    for i in range(NUM_CHUNKS):
        plt.plot(WM_LIST_TDOS, plaquette_tdos[i, :], 's-',
                 label=f"Generation {i}", color=colors[i], alpha=0.8)

    plt.xlabel(r"Disorder Strength ($W_M$)")
    plt.ylabel(r"Typical DOS, $\rho_t(E=0)$")
    plt.title("Generation-Resolved TDOS at Zero Energy")
    plt.xlim(0, 10.5)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(alpha=0.3)
    # plt.tight_layout()

    save_path = os.path.join(PLOT_DIR_TDOS, "Fig2d_TDOS_Plaquettes.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

def plot_ados_vs_E():
    """Fig 1a Inset: ADOS as a function of E"""
    # Select indices for: Semimetal (0.25), Transition (0.45), Metal (1.00)
    target_indices = range(len(WM_LIST_ADOS))

    data = get_ados_vs_e_data(target_indices)

    plt.figure(figsize=(8, 5))
    # colors = ['tab:blue', 'tab:orange', 'tab:green']
    colors = cm.viridis(np.linspace(0, 1, len(target_indices)))
    # for (wm, plot_data), color in zip(data.items(), colors):
    #     plt.plot(plot_data['energies'], plot_data['ados'],
    #              label=f"$W_M = {wm}$", color=color, linewidth=1.5, marker="", ls='-')
    for (wm, plot_data), color in zip(data.items(), colors):
        # Only label a few key weights: Deep Semimetal, Critical Edge, Metal, Deep Metal
        label_str = f"$W_M = {wm:.2f}$" if wm in [0.25, 0.40, 0.50, 1.00] else None

        plt.plot(plot_data['energies'], plot_data['ados'],
                 label=label_str, color=color, linewidth=1.5, marker="", ls='-')

    plt.xlabel(r"Energy ($E$)")
    plt.ylabel(r"Global Average DOS, $\rho(E)$")
    plt.title("Energy-Resolved ADOS")

    # Zoom in tightly around the Dirac node (E=0) to show the gap filling
    plt.xlim(-1.0, 1.0)
    plt.ylim(0, None)

    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    # plt.tight_layout()

    save_path = os.path.join(PLOT_DIR_ADOS, "Fig1a_Inset_ADOS_vs_E.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

def plot_tdos_vs_E():
    """Fig 1b Inset: TDOS as a function of E"""
    # Select indices for: Semimetal (0.20), Transition/Metal (1.00), Deep Metal (5.00)
    target_indices = range(len(WM_LIST_TDOS))

    data = get_tdos_vs_e_data(target_indices)

    plt.figure(figsize=(8, 5))
    # colors = ['tab:blue', 'tab:orange', 'tab:green']
    colors = cm.viridis(np.linspace(0, 1, len(target_indices)))
    # for (wm, plot_data), color in zip(data.items(), colors):
    #     plt.plot(plot_data['energies'], plot_data['tdos'],
    #              label=f"$W_M = {wm}$", color=color, linewidth=1.5, marker='', ls='-')
    for (wm, plot_data), color in zip(data.items(), colors):
        # Only label key checkpoints across the massive 15-weight sweep
        label_str = f"$W_M = {wm:.2f}$" if wm in [0.10, 0.50, 1.00, 5.00, 10.00] else None

        plt.plot(plot_data['energies'], plot_data['tdos'],
                 label=label_str, color=color, linewidth=1.5, marker="", ls='-')

    plt.xlabel(r"Energy ($E$)")
    plt.ylabel(r"Global Typical DOS, $\rho_t(E)$")
    plt.title("Energy-Resolved TDOS (Mobility Edges)")

    # Zoom in tightly around the Dirac node
    plt.xlim(-1.0, 1.0)
    plt.ylim(0, None)

    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    # plt.tight_layout()

    save_path = os.path.join(PLOT_DIR_TDOS, "Fig1b_Inset_TDOS_vs_E.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()

# =============================================================================
# 1. TDOS vs N_m (Standard Convergence)
# =============================================================================
def plot_tdos_vs_Nm():
    """TDOS as a function of raw moments (N_m)"""
    extrap_matrix = get_moments_convergence_data()

    plt.figure(figsize=(10, 6))
    colors = cm.viridis(np.linspace(0, 1, len(WM_LIST_TDOS)))

    for w_idx, wm in enumerate(WM_LIST_TDOS):
        y_vals = extrap_matrix[:, w_idx]
        label_str = f'$W_M = {wm:.2f}$' if wm in [0.1, 0.4, 0.5, 1.0, 5.0, 10.0] else None
        plt.plot(MOMENT_LIST, y_vals, 's-', color=colors[w_idx], alpha=0.8, label=label_str)

    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel(r"Number of Moments ($N_m$)")
    plt.ylabel(r"Global Typical DOS, $\rho_t(E=0)$")
    plt.title("TDOS Convergence vs Number of Moments")

    plt.xticks(ticks=MOMENT_LIST)
    plt.xlim(400, 16500)
    plt.ylim(0, np.max(extrap_matrix) * 1.1)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Disorder Strengths")
    plt.grid(alpha=0.3)

    save_path = os.path.join(PLOT_DIR_TDOS, "Fig2_TDOS_vs_Nm.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

# =============================================================================
# 2. TDOS vs 1/N_m (Extrapolation)
# =============================================================================
def plot_tdos_vs_invNm():
    """TDOS 1/Nm Extrapolation to infinite resolution"""
    extrap_matrix = get_moments_convergence_data()
    inv_N = 1.0 / np.array(MOMENT_LIST)

    plt.figure(figsize=(10, 6))
    colors = cm.viridis(np.linspace(0, 1, len(WM_LIST_TDOS)))

    for w_idx, wm in enumerate(WM_LIST_TDOS):
        y_vals = extrap_matrix[:, w_idx]
        c = colors[w_idx]

        fit_idx = -4
        slope, intercept, r_value, p_value, std_err = linregress(inv_N[fit_idx:], y_vals[fit_idx:])
        x_fit = np.linspace(0, max(inv_N), 100)
        y_fit = slope * x_fit + intercept

        label_str = f'$W_M = {wm:.2f}$' if wm in [0.1, 0.4, 0.5, 1.0, 5.0, 10.0] else None

        plt.plot(inv_N, y_vals, 'o', color=c, alpha=0.8)
        plt.plot(x_fit, y_fit, '--', color=c, alpha=0.5, label=label_str)
        plt.plot(0, intercept, '*', color=c, markersize=8, markeredgecolor='black', alpha=0.9)

    plt.axvline(0, color='black', linewidth=1)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel(r"Inverse Moments ($1/N_m$)")
    plt.ylabel(r"Global Typical DOS, $\rho_t(E=0)$")
    plt.title("TDOS Extrapolation to Infinite Resolution")

    plt.xlim(-0.0002, max(inv_N) * 1.1)
    plt.ylim([None, np.max(extrap_matrix) * 1.1])

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Disorder Strengths")
    plt.grid(alpha=0.3)

    save_path = os.path.join(PLOT_DIR_TDOS, "Fig2_TDOS_vs_invNm.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

# =============================================================================
# 3. ADOS vs N_m (Standard Convergence)
# =============================================================================
def plot_ados_vs_Nm():
    """ADOS as a function of raw moments (N_m)"""
    extrap_matrix = get_ados_convergence_data()

    plt.figure(figsize=(10, 6))
    colors = cm.viridis(np.linspace(0, 1, len(WM_LIST_ADOS)))

    for w_idx, wm in enumerate(WM_LIST_ADOS):
        y_vals = extrap_matrix[:, w_idx]
        label_str = f"$W_M = {wm:.2f}$" if wm in [0.25, 0.40, 0.50, 1.00] else None
        plt.plot(MOMENT_LIST, y_vals, 's-', color=colors[w_idx], alpha=0.8, label=label_str)

    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel(r"Number of Moments ($N_m$)")
    plt.ylabel(r"Global Average DOS, $\rho(E=0)$")
    plt.title("ADOS Convergence vs Number of Moments")

    plt.xticks(ticks=MOMENT_LIST)
    plt.xlim(400, 16500)
    plt.ylim(0, np.max(extrap_matrix) * 1.1)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Disorder Strengths")
    plt.grid(alpha=0.3)

    save_path = os.path.join(PLOT_DIR_ADOS, "Fig_ADOS_vs_Nm.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

# =============================================================================
# 4. ADOS vs 1/N_m (Extrapolation)
# =============================================================================
def plot_ados_vs_invNm():
    """ADOS 1/Nm Extrapolation to infinite resolution"""
    extrap_matrix = get_ados_convergence_data()
    inv_N = 1.0 / np.array(MOMENT_LIST)

    plt.figure(figsize=(10, 6))
    colors = cm.viridis(np.linspace(0, 1, len(WM_LIST_ADOS)))

    for w_idx, wm in enumerate(WM_LIST_ADOS):
        y_vals = extrap_matrix[:, w_idx]
        c = colors[w_idx]

        fit_idx = -4
        slope, intercept, r_value, p_value, std_err = linregress(inv_N[fit_idx:], y_vals[fit_idx:])
        x_fit = np.linspace(0, max(inv_N), 100)
        y_fit = slope * x_fit + intercept

        label_str = f"$W_M = {wm:.2f}$" if wm in [0.25, 0.40, 0.50, 1.00] else None

        plt.plot(inv_N, y_vals, 'o', color=c, alpha=0.8)
        plt.plot(x_fit, y_fit, '--', color=c, alpha=0.5, label=label_str)
        plt.plot(0, intercept, '*', color=c, markersize=8, markeredgecolor='black', alpha=0.9)

    plt.axvline(0, color='black', linewidth=1)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel(r"Inverse Moments ($1/N_m$)")
    plt.ylabel(r"Global Average DOS, $\rho(E=0)$")
    plt.title("ADOS Extrapolation to Infinite Resolution")

    plt.xlim(-0.0002, max(inv_N) * 1.1)
    plt.ylim([None, np.max(extrap_matrix) * 1.1])

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Disorder Strengths")
    plt.grid(alpha=0.3)

    save_path = os.path.join(PLOT_DIR_ADOS, "Fig_ADOS_vs_invNm.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()
# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    plot_ados_vs_W()
    plot_tdos_vs_W()
    plot_tdos_plaquettes_vs_W()
    plot_ados_vs_E()
    plot_tdos_vs_E()
    plot_moments_convergence()
    plot_ados_convergence()
