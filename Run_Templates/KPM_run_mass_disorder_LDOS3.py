    # Import dependencies
import numpy as np
from Lattice.Hamiltonians_Sublattice_Basis import hamiltonian_hyperbolic_q3_sublattice_basis
from KPM.parallel_compute import run_KPM_LDOS_parallel_mass_disorder_hyperbolic_q3_sublattice_basis
from KPM.parallel_compute import generate_random_seeds_for_parallel_jobs
from datetime import datetime

if __name__ == '__main__':
    # Define system and run parameters
    pval = 8 # 8 for fermi liquid hyperbolic, 10 for {10,3}
    nval = 9 # for p=8, n=9 has 10^5 sites; for p = 10, n =7 works
    number_parallel_jobs = 20
    clean_hamiltonian = hamiltonian_hyperbolic_q3_sublattice_basis(pval, nval)
    Wm_list = np.round(np.repeat(9.50, number_parallel_jobs), 3)
    num_moments = 8192
    eigenspectrum_max = 7.0
    eigenspectrum_min = -eigenspectrum_max

    # Define where to save moments from KPM computation
    import os

    # Define the directory and filename
    save_directory = './numericals/LDOS_p8_n9/950'
    save_name = 'LDOS_mass'

    # Create the directory if it does not exist
    os.makedirs(save_directory, exist_ok=True)

    # Construct the full path (optional, for convenience)
    full_path = os.path.join(save_directory, save_name)

    print(f"Directory created at: {save_directory}")
    save_name_index_list = np.arange(2*number_parallel_jobs, 3*number_parallel_jobs).astype(str)
    rng_seeds = generate_random_seeds_for_parallel_jobs(int(datetime.now().strftime('%Y%m%d%H%M%S')),
                                                        number_parallel_jobs)

    # Run the parallel KPM computations using above defined parameters
    run_KPM_LDOS_parallel_mass_disorder_hyperbolic_q3_sublattice_basis(number_parallel_jobs,
                                                                       pval,
                                                                       nval,
                                                                       clean_hamiltonian,
                                                                       Wm_list,
                                                                       num_moments,
                                                                       eigenspectrum_min,
                                                                       eigenspectrum_max,
                                                                       save_directory,
                                                                       save_name,
                                                                       save_name_index_list,
                                                                       rng_seeds)
