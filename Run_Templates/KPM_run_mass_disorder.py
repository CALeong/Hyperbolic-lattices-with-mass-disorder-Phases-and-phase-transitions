# Import dependencies
import numpy as np
import os
import shutil  # <-- Added for copying files

from Lattice.Hamiltonians_Sublattice_Basis import hamiltonian_hyperbolic_q3_sublattice_basis
from KPM.parallel_compute import run_KPM_ADOS_parallel_mass_disorder, generate_random_seeds_for_parallel_jobs
from datetime import datetime

if __name__ == '__main__':
    # Define system and run parameters
    pval = 10
    nval = 7
    number_parallel_jobs = 10
    clean_hamiltonian = hamiltonian_hyperbolic_q3_sublattice_basis(pval, nval)
    Wm_list = np.round(np.repeat(0.55, number_parallel_jobs), 3)
    num_moments = 8192  #running 16384 currently
    num_random_vectors = 12
    random_vector_dim = clean_hamiltonian.shape[0]
    eigenspectrum_min = -3.50
    eigenspectrum_max = 3.50

    # Define where to save moments from KPM computation
    save_directory = './numericals/ADOS_p10_n7/055'
    save_name = 'mass'

    # 1. Create the directory (and any necessary parent directories)
    os.makedirs(save_directory, exist_ok=True)
    print(f"Created directory: {save_directory}")

    # 2. Save a copy of THIS script to the save directory
    # __file__ gets the path of the current script
    current_script_path = os.path.abspath(__file__) 
    shutil.copy(current_script_path, save_directory)
    print(f"Saved a copy of the script to: {save_directory}")

    save_name_index_list = np.arange(number_parallel_jobs).astype(str)
    rng_seeds = generate_random_seeds_for_parallel_jobs(int(datetime.now().strftime('%Y%m%d%H%M%S')),
                                                        number_parallel_jobs)

    # Run the parallel KPM computations using above defined parameters
    run_KPM_ADOS_parallel_mass_disorder(number_parallel_jobs,
                                        clean_hamiltonian,
                                        Wm_list,
                                        num_moments,
                                        num_random_vectors,
                                        random_vector_dim,
                                        eigenspectrum_min,
                                        eigenspectrum_max,
                                        save_directory,
                                        save_name,
                                        save_name_index_list,
                                        rng_seeds)
