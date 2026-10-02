# Import dependencies
import numpy as np
from KPM.measure import calculate_ADOS_from_moments
import matplotlib.pyplot as plt

# Define run parameters
data_directory = './numericals/ADOS_p10_n8'
data_subdirectory_list = ['025', '030','035','040','045', '050', '055']
wm_list = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]  # Need to enter numerical values with ith value corresponding to ith value in subdir list manually
data_name_template = 'mass' + '_{}'
data_indices = np.arange(10)

energy_values = np.array([0.0])
eigenspectrum_bounds_list = [3.12, 3.12, 3.12, 3.12, 3.12, 3.12, 3.50]  # list of eigval bounds, ith element corresponds to ith subdir in list above

# Load in moments and calculate ADOS from them
all_ados = np.array([])
for subdir, ev_bound in zip(data_subdirectory_list, eigenspectrum_bounds_list):
    current_ados = np.array([])
    for i in data_indices:
        moments = np.load(data_directory + '/' + subdir + '/' + data_name_template.format(i) + '.npy')
        current_ados = np.append(current_ados, calculate_ADOS_from_moments(moments, energy_values, -ev_bound, ev_bound))
    all_ados = np.append(all_ados, np.average(current_ados))

plt.scatter(wm_list, all_ados)
plt.ylim([0, None])
plt.show()


