# Import dependencies
import numpy as np
from KPM.measure import calculate_ADOS_from_moments
import matplotlib.pyplot as plt
import os

# Define run parameters
data_directory = './numericals/ADOS_p10_n8/055'
data_name_template = 'mass' + '_{}'
data_indices = np.arange(10)

energy_values = np.linspace(-3, 3, 101)
eigenspectrum_min = -3.50
eigenspectrum_max = 3.50

# Load in moments and calculate ADOS from them
all_ados = np.zeros((len(data_indices), len(energy_values)), dtype=np.float64)
for i in data_indices:
    moments = np.load(data_directory + '/' + data_name_template.format(i) + '.npy')
    all_ados[i, :] = calculate_ADOS_from_moments(moments, energy_values, eigenspectrum_min, eigenspectrum_max)

final_ados = np.average(all_ados, axis=0)

# 1. Save the raw NumPy data so you don't have to recalculate it later
save_data_path = os.path.join(data_directory, 'final_ADOS_averaged.npy')
np.save(save_data_path, final_ados)
print(f"Raw data safely saved to: {save_data_path}")

save_plot_path = os.path.join(data_directory, 'ADOS_plot.png')
plt.ylim(bottom = 0)
plt.plot(energy_values, final_ados)
plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
print(f"Plot image safely saved to: {save_plot_path}")

# plt.show()


