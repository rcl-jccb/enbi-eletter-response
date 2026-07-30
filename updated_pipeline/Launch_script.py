from os import system
import json
import os
import numpy as np

N_types = 1
N_substances = 30
E_values_per_level = sorted(np.linspace(0,3,30), reverse=True)#[2.8, 3, 2, 1, 0]
D_values_per_E_level = E_values_per_level #[8.8, 6, 6, 1, 0]
n_subs_per_E_level = [1]*30#[1, 6, 12, 5, 12] # If  E_distribution == 'given', this must sum up to N_substances and have as many entrances as E_levels
D_prohibited_E_level_connections = []#[[0,1], [0,2], [1,2]]
h_value_per_E_level = [3] + [0]*9 + [0] + [0]*9 + [0] + [0]*9
E_D_type = 1
E_distribution = 'given'
E_dist_exponent = -1
D_mat_how = 'E proportional'
D_mat_proportion = 1
D_mat_variance = 0
init_D_mat_i_how = 'pathways given'
init_D_mat_i_prob = 0.1
max_number_of_pathways_per_type = 10
delta = 1/24
max_growth_data = 1e-10
k_data = 1e-4
evo_rate = 1e-10
S_molecular_mass = 100
colon_volume = 0.41
uptake_type = 'Monod'
mutation_dist = 'gaussian'
mutation_size = 1
init_B_mutant_rel_to_thres = 1
extinction_threshold_relative = 1e-4
max_N_types = 1e10 # This is actually just to put an np.inf essentially
invasion_period_h = 1.5
gamma_value = 8e13
tradeoff_constant = 1
tradeoff_exponent = 1
tradeoff_maintenance = 0
tradeoff_function = 'linear'
tradeoff_only_used_pathways = True
use_P_mat = False
nan_mutating = False
all_mutants_considered = False
sample_freq = 24*7 #Ahora esta en horas
tofile_freq = 24*sample_freq  
ending_time = 50*12*30*24
gut_dir = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', f'{tradeoff_function}_{tradeoff_constant}_{tradeoff_maintenance}_{tradeoff_exponent}')
continue_old = False

j = 0
for tradeoff_constant in [6]:#[0.1,0.5,1,1.5]:#[0.5, 1, 5, 10]:
    for h in [2]:
        e_max=3
        E_values_per_level = sorted(np.linspace(0,e_max,N_substances), reverse=True)#[2.8, 3, 2, 1, 0]
        D_values_per_E_level = E_values_per_level #[8.8, 6, 6, 1, 0]
        n_subs_per_E_level = [1]*N_substances#[1, 6, 12, 5, 12] # If  E_distribution == 'given', this must sum up to N_substances and have as many entrances as E_levels
        D_prohibited_E_level_connections = []#[[0,1], [0,2], [1,2]]
        h_value_per_E_level = [h] + [0]*(N_substances-1)
        
        for invasion_period_h in [1.5]:#[0, 1, 2.5, 5]:#[0, 5, 8]:
            #rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{0}', 'changing_tradeoff', f'')
            for i in range(6000+j,6000+j+10):
                path = os.path.join(rootpath, f'{i}')
                #Initialize random generator of python
                random_data = os.urandom(4)
                random_seed = int.from_bytes(random_data, byteorder="big")
                #random_seed = 2552109842
                program_features = json.dumps({'N_types': N_types, 'N_substances':N_substances, 'E_distribution': E_distribution, 'tradeoff_maintenance': tradeoff_maintenance,
                                   'E_dist_exponent': E_dist_exponent, 'D_mat_how': D_mat_how, 'D_mat_proportion': D_mat_proportion, 'invasion_period_h': invasion_period_h, 
                                   'D_mat_variance': D_mat_variance, 'init_D_mat_i_how': init_D_mat_i_how, 'init_D_mat_i_prob': init_D_mat_i_prob, 
                                   'delta': delta, 'h_value_per_E_level': h_value_per_E_level, 'max_growth_data': max_growth_data, 'k_data': k_data, 'evo_rate': evo_rate, 
                                   'uptake_type': uptake_type, 'mutation_size': mutation_size, 'init_B_mutant_rel_to_thres': init_B_mutant_rel_to_thres,
                                   'mutation_dist': mutation_dist, 'gamma_value': gamma_value, 'tradeoff_constant': tradeoff_constant, 'tradeoff_exponent': tradeoff_exponent,
                                   'tradeoff_function': tradeoff_function, 'ending_time': ending_time, 'colon_volume': colon_volume, 'sample_freq': sample_freq, 
                                   'tofile_freq': tofile_freq, 'path': path, 'use_P_mat': use_P_mat,
                                   'tradeoff_only_used_pathways': tradeoff_only_used_pathways, 'extinction_threshold_relative': extinction_threshold_relative,  
                                   'nan_mutating': nan_mutating, 'all_mutants_considered': all_mutants_considered, 'S_molecular_mass': S_molecular_mass,
                                   'random_seed':random_seed, 'max_N_types':max_N_types, 'continue_old': continue_old, 'max_number_of_pathways_per_type':max_number_of_pathways_per_type,
                                   'E_values_per_level': E_values_per_level, 'D_values_per_E_level': D_values_per_E_level, 'n_subs_per_E_level': n_subs_per_E_level,
                                   'D_prohibited_E_level_connections': D_prohibited_E_level_connections} ).replace('"', '\\\"')
                system(f"slanzarv --time=48:0:0 -J new_{h}_{i}_gut -m 2000 -partition=AMD,short,metis --nomail python ./python/main.py \\\'{program_features}\\\'")
            j = j + 10
