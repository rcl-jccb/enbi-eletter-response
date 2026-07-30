import os
import ecologicalequations as ee
import json
import sys




input_params = json.loads(sys.argv[1])

N_types = int(input_params['N_types'])
N_substances = int(input_params['N_substances'])
#E_levels = int(input_params['E_levels'])
E_values_per_level = [float(x) for x in list(input_params['E_values_per_level'])]
E_distribution = str(input_params['E_distribution'])
E_dist_exponent = int(input_params['E_dist_exponent'])
#E_dist_info = [int(x) for x in list(input_params['E_dist_info'])]
D_mat_how = str(input_params['D_mat_how'])
D_mat_proportion = float(input_params['D_mat_proportion'])
D_mat_variance = float(input_params['D_mat_variance'])
init_D_mat_i_how = str(input_params['init_D_mat_i_how'])
init_D_mat_i_prob = float(input_params['init_D_mat_i_prob'])
delta = float(input_params['delta'])
h_value_per_E_level = [float(x) for x in list(input_params['h_value_per_E_level'])]
max_growth_data = float(input_params['max_growth_data'])
k_data = float(input_params['k_data'])
evo_rate = float(input_params['evo_rate'])
S_molecular_mass = float(input_params['S_molecular_mass'])
colon_volume = float(input_params['colon_volume'])
uptake_type = str(input_params['uptake_type'])
mutation_dist = str(input_params['mutation_dist'])
mutation_size = float(input_params['mutation_size'])
init_B_mutant_rel_to_thres = float(input_params['init_B_mutant_rel_to_thres'])
sample_freq = int(input_params['sample_freq']) # IN MUTATION TRIALS
tofile_freq = int(input_params['tofile_freq'])
ending_time = float(input_params['ending_time'])
gamma_value = float(input_params['gamma_value'])
tradeoff_constant = float(input_params['tradeoff_constant'])
tradeoff_exponent = float(input_params['tradeoff_exponent'])
tradeoff_maintenance = float(input_params['tradeoff_maintenance'])
tradeoff_function = str(input_params['tradeoff_function'])
extinction_threshold_relative = float(input_params['extinction_threshold_relative'])
tradeoff_only_used_pathways = bool(input_params['tradeoff_only_used_pathways'])
nan_mutating = bool(input_params['nan_mutating'])
all_mutants_considered = bool(input_params['all_mutants_considered'])
max_N_types = int(input_params['max_N_types'])
path = str(input_params['path'])
random_seed = int(input_params['random_seed'])
continue_old = bool(input_params['continue_old'])
invasion_period_h = float(input_params['invasion_period_h'])
D_values_per_E_level = [float(x) for x in list(input_params['D_values_per_E_level'])]
n_subs_per_E_level = [int(x) for x in list(input_params['n_subs_per_E_level'])]
D_prohibited_E_level_connections = [list(map(int, x)) for x in list(input_params['D_prohibited_E_level_connections'])]
max_number_of_pathways_per_type = int(input_params['max_number_of_pathways_per_type'])
use_P_mat = bool(input_params['use_P_mat'])

'''
N_types = 1
N_substances = 40
E_levels = 5
E_distribution = 'power law'
E_dist_exponent = -1
D_mat_how = 'E proportional'
D_mat_proportion = 1
D_mat_variance = None
init_D_mat_i_how = 'pathways given'
init_D_mat_i_prob = 0.1
delta = 1/24
h_value = 2
max_growth_data = 1e-10
k_data = 1e-4
evo_rate = 1e-9
S_molecular_mass = 100
colon_volume = 0.41
uptake_type = 'Monod'
mutation_dist = 'uniform'
mutation_size = 2
init_B_mutant_rel_to_thres = 10
sample_freq = 100 # IN MUTATION TRIALS
ending_time = 7*24
gamma_value = 1e13
tradeoff_constant = 1.5
extinction_threshold_relative = 1e-5
tradeoff_only_used_pathways = True
nan_mutating = False
all_mutants_considered = True
gut_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(gut_dir, 'results', 'pruebas')
path = path
random_data = os.urandom(4)
random_seed = int.from_bytes(random_data, byteorder="big")
'''

params = {'N_types': N_types, 'N_substances':N_substances, 'E_distribution': E_distribution, 'E_values_per_level': E_values_per_level,
		'E_dist_exponent': E_dist_exponent, 'D_mat_how': D_mat_how, 'D_mat_proportion': D_mat_proportion, 'max_N_types': max_N_types,
		'D_mat_variance': D_mat_variance, 'init_D_mat_i_how': init_D_mat_i_how, 'init_D_mat_i_prob': init_D_mat_i_prob, 
		'delta': delta, 'h_value_per_E_level': h_value_per_E_level, 'max_growth_data':max_growth_data, 'k_data':k_data, 'evo_rate': evo_rate, 
		'uptake_type': uptake_type, 'mutation_size': mutation_size, 'init_B_mutant_rel_to_thres': init_B_mutant_rel_to_thres, 
		'mutation_dist':mutation_dist, 'gamma_value': gamma_value, 'tradeoff_constant': tradeoff_constant, 'tradeoff_exponent': tradeoff_exponent,
        'tradeoff_maintenance': tradeoff_maintenance, 'tradeoff_function': tradeoff_function, 'ending_time': ending_time, 
        'tradeoff_only_used_pathways': tradeoff_only_used_pathways, 'random_seed':random_seed, 'path': path, 'use_P_mat': use_P_mat,
        'extinction_threshold_relative': extinction_threshold_relative,  'nan_mutating': nan_mutating, 
        'all_mutants_considered': all_mutants_considered, 'S_molecular_mass': S_molecular_mass, 'colon_volume': colon_volume, 
        'sample_freq': sample_freq, 'tofile_freq': tofile_freq, 'continue_old': continue_old, 'invasion_period_h': invasion_period_h,
        'D_values_per_E_level': D_values_per_E_level, 'n_subs_per_E_level': n_subs_per_E_level, 'max_number_of_pathways_per_type': max_number_of_pathways_per_type,
        'D_prohibited_E_level_connections': D_prohibited_E_level_connections}
ee.full_model_evolution(params)
