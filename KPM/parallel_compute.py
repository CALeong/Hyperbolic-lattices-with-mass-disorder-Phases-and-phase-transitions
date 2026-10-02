from joblib import parallel_config, Parallel, delayed
from KPM.KPM import moments_ADOS_general, moments_LDOS_general
import numpy as np
from KPM.measure import rescale_operator_unity_window
import scipy.sparse
from Lattice.Disorder import mass_disorder, bond_disorder
from Lattice.General_Hamiltonian import number_points_q3_general_from_repeating_pattern
import pickle
from Lattice.General_Hamiltonian_Strain import sublattice_label_q3


def random_vectors_arr_generate(num_rand_vecs, vec_dim):
    return np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(vec_dim, num_rand_vecs))


def run_KPM_ADOS_parallel_mass_disorder(n_jobs, sparse_ham, W_list, number_moments,
                                        num_rand_vecs, rand_vec_dim, eigval_min, eigval_max,
                                        save_dir, save_name, save_index_list, seed_list):

    def run_routine(rescaled_sparse_ham, num_moments, random_vecs_arr, savedir, savename, saveindex):
        result = moments_ADOS_general(rescaled_sparse_ham, num_moments, random_vecs_arr)
        np.save(savedir + '/' + savename + '_' + saveindex, result)

    with parallel_config(backend='loky'):
        Parallel(n_jobs=n_jobs)(
            delayed(run_routine)
            (rescale_operator_unity_window(sparse_ham + mass_disorder(W, sparse_ham.shape[0], spec_seed),
                                           eigval_min=eigval_min, eigval_max=eigval_max
                                           ),
            number_moments,
            random_vectors_arr_generate(num_rand_vecs, rand_vec_dim),
            save_dir,
            save_name,
            save_index)
            for (W, save_index, spec_seed) in zip(W_list, save_index_list, seed_list)
        )

    metadata_dict = {
        '(Wm, random seed, save_index)': zip(W_list, seed_list, save_index_list),
        'Number random vectors': num_rand_vecs,
        'Number moments up to': number_moments,
        'Eigenvalue extrema': (eigval_min, eigval_max)
    }

    sf = open(save_dir + '/' + 'metadata.pkl', 'wb')
    pickle.dump(metadata_dict, sf)
    sf.close()


def run_KPM_ADOS_parallel_bond_disorder(n_jobs, sparse_ham, W_list, number_moments,
                                        num_rand_vecs, rand_vec_dim, eigval_min, eigval_max,
                                        save_dir, save_name, save_index_list, seed_list):

    def run_routine(rescaled_sparse_ham, num_moments, random_vecs_arr, savedir, savename, saveindex):
        result = moments_ADOS_general(rescaled_sparse_ham, num_moments, random_vecs_arr)
        np.save(savedir + '/' + savename + '_' + saveindex, result)

    with parallel_config(backend='loky'):
        Parallel(n_jobs=n_jobs)(
            delayed(run_routine)
            (rescale_operator_unity_window(sparse_ham + bond_disorder(W, sparse_ham, spec_seed),
                                           eigval_min=eigval_min, eigval_max=eigval_max
                                           ),
            number_moments,
            random_vectors_arr_generate(num_rand_vecs, rand_vec_dim),
            save_dir,
            save_name,
            save_index)
            for (W, save_index, spec_seed) in zip(W_list, save_index_list, seed_list)
        )

    metadata_dict = {
        '(Wb, random seed, save_index)': zip(W_list, seed_list, save_index_list),
        'Number random vectors': num_rand_vecs,
        'Number moments up to': number_moments,
        'Eigenvalue extrema': (eigval_min, eigval_max)
    }

    sf = open(save_dir + '/' + 'metadata.pkl', 'wb')
    pickle.dump(metadata_dict, sf)
    sf.close()


def sites_on_each_gen(pval, num_levels):
    sites_per_level, total_num_sites = number_points_q3_general_from_repeating_pattern(pval, num_levels)
    sites_per_level = np.append(np.array([0]), sites_per_level)
    sites_on_each_gen = {}
    for n in range(1, num_levels + 1):
        sites_on_each_gen['n={}'.format(n - 1)] = np.arange(np.sum(sites_per_level[:n]),
                                                            np.sum(sites_per_level[:n + 1]))
    return sites_on_each_gen


def search_for_unique_plaquettes_each_gen(pval, num_levels):
    gen_sites_dict = sites_on_each_gen(pval, num_levels)
    plaquette_indices = np.zeros((1, pval))
    offset_low = 0
    offset_high = 0
    for n in range(num_levels - 1):
        lower_gen_sites = gen_sites_dict['n={}'.format(n)][int(offset_low):int(offset_low + 2)]
        higher_gen_sites = gen_sites_dict['n={}'.format(n + 1)][int(offset_high):int(offset_high + (pval - 2))]
        plaquette_indices = np.vstack((plaquette_indices,
                                       np.concatenate((lower_gen_sites, higher_gen_sites))))

        offset_low = (n + 1) * (len(gen_sites_dict['n={}'.format(n + 1)]) / pval) + 1
        if n != num_levels - 2:
            offset_high = (n + 1) * (len(gen_sites_dict['n={}'.format(n + 2)]) / pval)

    plaquette_indices = plaquette_indices.astype(np.int64)
    return np.concat((np.arange(0, pval, dtype=np.int64), plaquette_indices[1:, :].reshape(-1)))


def search_for_unique_plaquettes_each_gen_sublattice_basis(pval, num_levels):
    plaquettes_each_gen_sites = search_for_unique_plaquettes_each_gen(pval, num_levels)
    asites, bsites = sublattice_label_q3(pval, num_levels)
    sublattice_basis = np.concat((asites, bsites)).astype(np.int64)
    site_label_map = np.argsort(sublattice_basis)

    plaquettes_each_gen_sites_sublattice_basis = np.array([])
    for i in plaquettes_each_gen_sites:
        plaquettes_each_gen_sites_sublattice_basis = np.append(plaquettes_each_gen_sites_sublattice_basis,
                                                               site_label_map[i])

    return plaquettes_each_gen_sites_sublattice_basis


def run_KPM_LDOS_parallel_mass_disorder_hyperbolic_q3_sublattice_basis(n_jobs, pval, nval, sparse_ham, W_list,
                                                                       number_moments, eigval_min, eigval_max,
                                                                       save_dir, save_name, save_index_list, seed_list):

    def run_routine(clean_sparse_ham, Wm, disorder_seed, num_moments, local_sites_sampled,
                    savedir, savename, saveindex):
        rescaled_sparse_ham = rescale_operator_unity_window(clean_sparse_ham
                                                            + mass_disorder(Wm, clean_sparse_ham.shape[0], disorder_seed),
                                                            eigval_min=eigval_min, eigval_max=eigval_max)
        result = moments_LDOS_general(rescaled_sparse_ham, num_moments, local_sites_sampled)
        np.save(savedir + '/' + savename + '_' + saveindex, result)

    with parallel_config(backend='loky'):
        Parallel(n_jobs=n_jobs)(
            delayed(run_routine)
            (sparse_ham,
             W,
             spec_seed,
             number_moments,
             search_for_unique_plaquettes_each_gen_sublattice_basis(pval, nval),
             save_dir,
             save_name,
             save_index)
            for (W, save_index, spec_seed) in zip(W_list, save_index_list, seed_list)
        )

    metadata_dict = {
        '(Wm, random seed, save_index)': zip(W_list, seed_list, save_index_list),
        'Sites LDOS computed for (sublattice basis)': search_for_unique_plaquettes_each_gen(pval, nval),
        'Number moments up to': number_moments,
        'Eigenvalue extrema': (eigval_min, eigval_max)
    }

    sf = open(save_dir + '/' + 'metadata.pkl', 'wb')
    pickle.dump(metadata_dict, sf)
    sf.close()


def generate_random_seeds_for_parallel_jobs(original_seed, number_of_jobs):
    seeds_generator = np.random.SeedSequence(original_seed)
    seeds = seeds_generator.spawn(number_of_jobs)
    return seeds


def run_KPM_ADOS_parallel_mass_disorder_separate_random_vecs(n_jobs, sparse_ham, W_list, number_moments,
                                                             num_rand_vecs, rand_vec_dim, eigval_min, eigval_max,
                                                             save_dir, save_name, save_index_list, seed_list):

    def run_routine(rescaled_sparse_ham, num_moments, number_rand_vecs, random_vec_dim, savedir, savename, saveindex):
        random_vecs_arr = random_vectors_arr_generate(number_rand_vecs, random_vec_dim)
        result = moments_ADOS_general(rescaled_sparse_ham, num_moments, random_vecs_arr)
        np.save(savedir + '/' + savename + '_' + saveindex, result)

    with parallel_config(backend='loky'):
        Parallel(n_jobs=n_jobs)(
            delayed(run_routine)
            (rescale_operator_unity_window(sparse_ham + mass_disorder(W, sparse_ham.shape[0], spec_seed),
                                           eigval_min=eigval_min, eigval_max=eigval_max
                                           ),
            number_moments,
            num_rand_vecs,
            rand_vec_dim,
            save_dir,
            save_name,
            save_index)
            for (W, save_index, spec_seed) in zip(W_list, save_index_list, seed_list)
        )
