from Lattice.General_Hamiltonian import number_points_q3_general_from_repeating_pattern

p_val = 8
n_val = 9
R = 12 # number of random vectors for stochastic trace
_, D = number_points_q3_general_from_repeating_pattern(p_val, n_val) # total number of sites
print(f"Number of points in our lattice: {D}")
