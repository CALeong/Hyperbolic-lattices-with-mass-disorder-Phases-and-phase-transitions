# Import dependencies
import numpy as np
from Lattice.General_Hamiltonian import general_hyperbolic_q3_hamiltonian
from KPM.parallel_compute import run_KPM_ADOS_parallel_bond_disorder, generate_random_seeds_for_parallel_jobs
from datetime import datetime

if __name__ == '__main__':
    # Define system and run parameters
    pval = 10
    nval = 8
    number_parallel_jobs = 10
    clean_hamiltonian = general_hyperbolic_q3_hamiltonian(pval, nval)
    Wb_list = np.round(np.repeat(0.5, number_parallel_jobs), 1)
    num_moments = 16384
    num_random_vectors = 12
    random_vector_dim = clean_hamiltonian.shape[0]
    eigenspectrum_min = -3.12
    eigenspectrum_max = 3.12

    # Define where to save moments from KPM computation
    save_directory = ''
    save_name = ''
    save_name_index_list = np.arange(number_parallel_jobs).astype(str)
    rng_seeds = generate_random_seeds_for_parallel_jobs(int(datetime.now().strftime('%Y%m%d%H%M%S')),
                                                        number_parallel_jobs)

    # Run the parallel KPM computations using above defined parameters
    run_KPM_ADOS_parallel_bond_disorder(number_parallel_jobs,
                                        clean_hamiltonian,
                                        Wb_list,
                                        num_moments,
                                        num_random_vectors,
                                        random_vector_dim,
                                        eigenspectrum_min,
                                        eigenspectrum_max,
                                        save_directory,
                                        save_name,
                                        save_name_index_list,
                                        rng_seeds)
