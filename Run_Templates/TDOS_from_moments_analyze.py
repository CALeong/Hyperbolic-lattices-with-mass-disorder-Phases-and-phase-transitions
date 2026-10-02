# Import dependencies
import numpy as np
from KPM.measure import calculate_LDOS_from_moments


# User inputs
data_dir = './numericals/LDOS_p8_n9/075'  # Path to the directory holding all the moments data
data_name_template = 'LDOS_mass_{}.npy'  # Name of data file with {} for save index to be inserted later
pval = 8  # number of sides of fundamental polygon
total_num_dr = 60  # total number of disorder realization
num_chunks = 9  # Number of plaquettes considered
bad_val = 0
evalue = 3.4
########################################################################################################################

# Results holder (Rows = plaquette, Cols = sites on plaquette)
all_tdos = np.zeros((num_chunks, pval), dtype=np.float64)
for i in range(total_num_dr):  # Iterate over all disorder realizations
    # Load in data
    data = np.load(data_dir + '/' + data_name_template.format(i))

    # if/else check for diverging data due to bad eigenvalue bounds
    if np.any(np.isnan(data)) or np.max(np.abs(data) > 1.0):
        total_num_dr += -1  # bad dataset, do nothing and reduce total number of disorder realizations by one
    else:
        # Split data for each plaquette and compute log(LDOS) values for each site on each plaquette
        data = data / data[0, :].reshape(1, -1)  # Normalize LDOS for each site
        chunked_data = np.split(data, num_chunks, axis=1)
        for cdi, cd in enumerate(chunked_data):
            ldos_values = calculate_LDOS_from_moments(cd, np.array([0]), -evalue, evalue) #change eigenvalue bounds here
            bad_values = ldos_values[ldos_values <= 0]
            bad_val += len(bad_values)
            # print(f"Found {len(bad_values)} invalid values:")
            # print(bad_values)
            safe_ldos = np.clip(ldos_values, a_min=1e-14, a_max=None)
            log_ldos_values = np.log(safe_ldos)
            all_tdos[cdi, :] += log_ldos_values.reshape(-1)

print(f"There are {bad_val} bad values.")
# Complete average over disorder realizations
all_tdos = all_tdos / total_num_dr

# Average over all sites on plaquette and take exp, thereby giving TDOS of plaquette
all_tdos = np.exp(np.average(all_tdos, axis=1))
print('TDOS each plaquette (center to edge): ', all_tdos)

# Average TDOS of each plaquette to give total TDOS of system
whole_tdos = np.average(all_tdos)
print('TDOS all sampled sites: ', whole_tdos)

