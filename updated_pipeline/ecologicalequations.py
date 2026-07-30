import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import normalize
import contextlib
import sys
import os
import scipy.stats as stats
import subprocess
import pickle
import random
import pandas as pd
import math
import os
import json
import sys
import seaborn as sns
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import minimize
import traceback
from string import ascii_uppercase

class NumpyEncoder(json.JSONEncoder):
	# seen at: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def N_eqns(B_ivec, S_avec, P_mat_ivec, E_mat, E_conversion_constant, gamma_ivec, delta_ivec, max_growth_ivec_avec, k_ivec_avec, uptake_type='Monod'): #, probio_ivec) For the future
    '''
    avec means vector with same dimension as substances
    ivec means vector with same dimension as types
    Takes:
        - B_ivec: the biomass vector for types i
        - S_avec: the numpy vector with the concentrations of each substances
        - P_mat_ivec (=pathway_preferences_matrix_ivec): a vector where each value is a pathway matrix. A pathway preferences matrix=P is an SxS matrix (with S dim of substances) 
        where each entrance tell us the percentage of resource b (row) that is metabolized into resource c (column). For example: [[0,0.5,0.5], [0,0,1], [0,0,0]] 
        so S1 is metabolized half in S2 and half in S3, S2 is metabolized into S3 and S3 is not metabolized. So pathway_preferences_matrix_ivec has P_i as values.
        - E_mat (=pathway_energy_matrix): is an SxS matrix that tell us the energy obtained from metabolizing resource b (the rows) into resource c (the colums). So each
        entrance is E_bc = Energy(R_b)-Energy(R_c).
        - E_conversion_constant: transforms mol ATP/mol of S (which are the E_mat units) to mol/ATP/(g/L of S)
        - gamma_ivec: the numpy vector with the conversion constant from energy to biomass of the different types i.
        - delta_ivec: the numpy vector with washed out rate for types 
        - uptake_type: ['linear', 'Monod'] are the possible uptake types
        - max_growth_ivec_avec: an ivec that for each value has an avec. SO tell us the max growth of type i for each a resource. 
        Ex: [[2,0,1], [0,0,2]] so type 0 feeds on substances a=0 and a=2 and type 1 only on substance 2.
        - k_ivec_avec: half saturation constant for type i and substance a. Same as max_growth_ivec_avec. Only used for Monod uptake type
        - probio_ivec: the numpy vector with the probiotic inflow rate
    Gives:
        An i_vector (each element is a type) with elements being the right hand side of the differential equation for the change in types biomass
    '''
    # We try to use numpy vectorize because it runs internally on c and is way faster than looping in python

    S_avec_reshaped = S_avec.reshape(1,S_avec.size) # This reshapes subs vector from [1, 2] -> [[1,2]] (is needed for vectorized multiplication later)
    if uptake_type == 'linear':
        uptake_ivec_avec = max_growth_ivec_avec * S_avec_reshaped      
    elif uptake_type == 'Monod':
        uptake_ivec_avec = max_growth_ivec_avec * S_avec_reshaped/(k_ivec_avec + S_avec_reshaped)
        
    uptake_ivec_avec = uptake_ivec_avec.reshape((B_ivec.size,S_avec.size,1))
    consumption_ivec = P_mat_ivec*E_mat*E_conversion_constant*uptake_ivec_avec
    consumption_ivec = np.reshape(np.apply_over_axes(np.sum, consumption_ivec, [1,2]), B_ivec.size)
    
    # Here we simply sum the elements of each matrix and reshape it so that we get an i_vec which is what we needed.
    
    return B_ivec*(gamma_ivec*consumption_ivec-delta_ivec) #+probio_ivec

def get_birth_rates(B_ivec, S_avec, P_mat_ivec, E_mat, E_conversion_constant, gamma_ivec, max_growth_ivec_avec, k_ivec_avec, uptake_type='Monod'): 
    '''
    avec means vector with same dimension as substances
    ivec means vector with same dimension as types
    Takes:
        - B_ivec: the biomass vector for types i
        - S_avec: the numpy vector with the concentrations of each substances
        - P_mat_ivec (=pathway_preferences_matrix_ivec): a vector where each value is a pathway matrix. A pathway preferences matrix=P is an SxS matrix (with S dim of substances) 
        where each entrance tell us the percentage of resource b (row) that is metabolized into resource c (column). For example: [[0,0.5,0.5], [0,0,1], [0,0,0]] 
        so S1 is metabolized half in S2 and half in S3, S2 is metabolized into S3 and S3 is not metabolized. So pathway_preferences_matrix_ivec has P_i as values.
        - E_mat (=pathway_energy_matrix): is an SxS matrix that tell us the energy obtained from metabolizing resource b (the rows) into resource c (the colums). So each
        entrance is E_bc = Energy(R_b)-Energy(R_c).
        - E_conversion_constant: transforms mol ATP/mol of S (which are the E_mat units) to mol/ATP/(g/L of S)
        - gamma_ivec: the numpy vector with the conversion constant from energy to biomass of the different types i.
        - uptake_type: ['linear', 'Monod'] are the possible uptake types
        - max_growth_ivec_avec: an ivec that for each value has an avec. SO tell us the max growth of type i for each a resource. 
        Ex: [[2,0,1], [0,0,2]] so type 0 feeds on substances a=0 and a=2 and type 1 only on substance 2.
        - k_ivec_avec: half saturation constant for type i and substance a. Same as max_growth_ivec_avec. Only used for Monod uptake type
        - probio_ivec: the numpy vector with the probiotic inflow rate
    Gives:
        An i_vector (each element is a type) with elements being the right hand side of the differential equation for the change in types biomass
    '''
    # We try to use numpy vectorize because it runs internally on c and is way faster than looping in python

    S_avec_reshaped = S_avec.reshape(1,S_avec.size) # This reshapes subs vector from [1, 2] -> [[1,2]] (is needed for vectorized multiplication later)
    if uptake_type == 'linear':
        uptake_ivec_avec = max_growth_ivec_avec * S_avec_reshaped      
    elif uptake_type == 'Monod':
        uptake_ivec_avec = max_growth_ivec_avec * S_avec_reshaped/(k_ivec_avec + S_avec_reshaped)
        
    uptake_ivec_avec = uptake_ivec_avec.reshape((B_ivec.size,S_avec.size,1))
    consumption_ivec = P_mat_ivec*E_mat*E_conversion_constant*uptake_ivec_avec
    consumption_ivec = np.reshape(np.apply_over_axes(np.sum, consumption_ivec, [1,2]), B_ivec.size)
    # Here we simply sum the elements of each matrix and reshape it so that we get an i_vec which is what we needed.
    birth_rate_ivec = B_ivec*(gamma_ivec*consumption_ivec)
    if np.isnan(birth_rate_ivec).any():
        print(birth_rate_ivec)
        idxs = np.argwhere(np.isnan(birth_rate_ivec)).flatten()
        print(B_ivec[idxs])
        print(gamma_ivec[idxs])
        print(consumption_ivec[idxs])
        print(P_mat_ivec[idxs])
        print(uptake_ivec_avec[idxs,:])
        raise
    elif (birth_rate_ivec<0).any():
        print(birth_rate_ivec)
        idxs = np.argwhere(birth_rate_ivec<0).flatten()
        print(B_ivec[idxs])
        print(gamma_ivec[idxs])
        print(consumption_ivec[idxs])
        raise
        
    return birth_rate_ivec #+probio_ivec

def get_growth_rate(S_avec, new_type_P_mat, E_mat, E_conversion_constant, new_type_gamma, new_type_delta, new_type_max_growth_avec, new_type_k_avec, uptake_type='Monod'):
    if uptake_type == 'linear':
        uptake_avec = new_type_max_growth_avec * S_avec    
    elif uptake_type == 'Monod':
        uptake_avec = new_type_max_growth_avec * S_avec/(new_type_k_avec + S_avec)
    uptake_avec = uptake_avec.reshape((S_avec.size,1))
    growth_rate = np.sum(new_type_gamma*new_type_P_mat*E_mat*uptake_avec*E_conversion_constant)-new_type_delta
    return growth_rate

def S_eqns(B_ivec, S_avec, P_mat_ivec, h_avec, delta_avec, max_growth_ivec_avec, k_ivec_avec, uptake_type='Monod'):
    '''
    avec means vector with same dimension as substances
    ivec means vector with same dimension as types
    Takes:
        - B_ivec: the numpy vector with biomass abundances for each of the types i
        - S_avec: the numpy vector with the concentrations of each substances
        - P_mat_ivec (=pathway_preferences_matrix_ivec): a vector where each value is a pathway matrix. A pathway preferences matrix=P is an SxS matrix (with S dim of substances) 
        where each entrance tell us the percentage of resource b (row) that is metabolized into resource c (column). For example: [[0,0.5,0.5], [0,0,1], [0,0,0]] 
        so S1 is metabolized half in S2 and half in S3, S2 is metabolized into S3 and S3 is not metabolized. So pathway_preferences_matrix_ivec has P_i as values.
        - h_avec: the numpy vector with inflow rate of substances
        - delta_avec: the numpy vector with washed out rate for resources (here we could model different gastrointestinal times) 
        - uptake_type: ['linear', 'Monod'] are the possible uptake types
        - max_growth_ivec_avec: an ivec that for each value has an avec. SO tell us the max growth of type i for each a resource. 
        Ex: [[2,0,1], [0,0,2]] so type 0 feeds on substances a=0 and a=2 and type 1 only on substance 2.
        - k_ivec_avec: half saturation constant for type i and substance a. Same as max_growth_ivec_avec. Only used for Monod uptake type
    Gives:
        An i_vector (each element is a type) with elements being the right hand side of the differential equation for the change in types biomass
    '''
    # We try to use numpy vectorize because it runs internally on c and is way faster than looping in python

    S_avec_reshaped = S_avec.reshape(1,S_avec.size) # This reshapes subs vector from [1, 2] -> [[1,2]] (is needed for vectorized multiplication later)
    if uptake_type == 'linear':
        uptake_ivec_avec = max_growth_ivec_avec * S_avec_reshaped      
    elif uptake_type == 'Monod':
        uptake_ivec_avec = max_growth_ivec_avec * S_avec_reshaped/(k_ivec_avec + S_avec_reshaped)
        
    uptake_ivec_avec = uptake_ivec_avec.reshape((B_ivec.size,S_avec.size,1))

    B_ivec_reshaped = np.reshape(B_ivec, (B_ivec.shape[0],1,1)) # This reshapes B ivector from [2,3] -> [[[2]], [[3]]] (is needed for vectorized multiplication later)

    # Now we calculate the crossfeeding term for types, in a vectorized way (so we do them all at once). 
    aux_ivec_mat = P_mat_ivec*B_ivec_reshaped*uptake_ivec_avec
    # Each matrix element of p_mat_ivec gets multiplied element by element with the B_ivec_reshaped and then each row of that matrix gets multiplied (row-wise) by the elements of subs_avec.

    crossfeeding_avec = np.reshape(np.apply_over_axes(np.sum, aux_ivec_mat, [0,1]), aux_ivec_mat.shape[2])
    # Here we simply sum the elements in the i-dimension which is the 0 axis of the ndarray (so we sum over types) and over the rows of each of the matrices so the 
    # 1 axis (rows of the matrices), and reshape it so that we get an a_vec with dimension equal to the 2 axis dimension which is what we needed. In ex: [24, 24]

    # Now the reduction of substances due to consumption from types:
    consumption_avec = np.reshape(np.apply_over_axes(np.sum, aux_ivec_mat, [0,2]), aux_ivec_mat.shape[1])
    # It is actually as cross feeding but we sum over the i-dimension (0 axis) and the 2 axis (the columns of the matrices)

    return h_avec + crossfeeding_avec - consumption_avec - delta_avec*S_avec 


def CR_model(t, X, N_types, P_mat_ivec, E_mat, E_conversion_constant, gamma_ivec, delta_ivec, h_avec, delta_avec, max_growth_ivec_avec, k_ivec_avec, uptake_type):
    '''
    avec means vector with same dimension as substances
    ivec means vector with same dimension as types
    This function is needed to be able to calculate the time evolution of the differential equations. It needs time as a first parameter, X which are 
    the evolving variables as second and then the needed parameters for the definition of the equations (those are defined above). See: solve_ivp documentation
    at https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html to understand better.
    '''
    B_ivec = X[:N_types]
    S_avec = X[N_types:]
    
    return np.concatenate((N_eqns(B_ivec, S_avec, P_mat_ivec, E_mat, E_conversion_constant, gamma_ivec, delta_ivec, max_growth_ivec_avec, k_ivec_avec, uptake_type), 
                           S_eqns(B_ivec, S_avec, P_mat_ivec, h_avec, delta_avec, max_growth_ivec_avec, k_ivec_avec, uptake_type)))

def eco_advance(init_state, init_time, end_time, N_types, P_mat_ivec, E_mat, E_conversion_constant, gamma_ivec, delta_ivec, h_avec, delta_avec, max_growth_ivec_avec, k_ivec_avec, uptake_type):
    '''
    avec means vector with same dimension as substances
    ivec means vector with same dimension as types
    Here we integrate the differential equations from an init_state, avector with the state of the system, i.e. [B_0, B_1, ..., B_N_types, S_a, S_b, ..., S_S]
    at time init_time, until end_time. We return the state at end_time. The other function parameters are needed for the differential equations (see doc above)
    '''
    tries=0
    atol_exp=-16
    rtol_exp=-8
    while tries < 8:
        sol = solve_ivp(CR_model, [init_time, end_time], init_state, method='BDF', atol=10**atol_exp, rtol=10**rtol_exp, args=(N_types, P_mat_ivec, E_mat, E_conversion_constant, gamma_ivec, delta_ivec, h_avec, delta_avec, max_growth_ivec_avec, k_ivec_avec, uptake_type)) #method='Radau', atol=1e-16, rtol=1e-8,) #) #, method='LSODA', atol=1e-8, rtol=1e-4)

        #sol_t = sol.t # This gives me the time points at whcih the solution is computed
        #B_ivec_t = sol.y[:N_types,:] # This gives me the temporal solution for the biomass of the N_types (for sol_times). Ex: [type 1 ->[value at init_time ..., value at end_time], type 2 -> [...], ...]
        #S_avec_t = sol.y[N_types:,:] # Same as above but for substances
        #return sol_t, B_ivec_t, S_avec_t
        B_ivec = sol.y[:N_types, -1] # I only want the last temporal value => This gives me an ivector with the values of biomass for the different types at end_time
        S_avec = sol.y[N_types:, -1] 
        S_avec = np.where((S_avec < 1e-15) & (S_avec>0), 0, S_avec)
        if np.any(S_avec<0):
            pass
            #print("Substances cannot get below zero so an error must have occured. Here are the details")
            #print(S_avec)
            #idxs = np.argwhere(np.isnan(S_avec)).flatten()
            #print(idxs)
            #print(f"We try with improved precision")

        elif np.isnan(B_ivec).any():
            pass
            '''print("Biomasses cannot be nan so an error must have occured. Here are the details:")
            print(B_ivec)
            idxs = np.argwhere(np.isnan(B_ivec)).flatten()
            print(idxs)
            print(f"We try with improved precision")'''

        elif (B_ivec<0).any():
            pass
            '''
            print("Biomasses cannot get below zero so an error must have occured. Here are the details:")
            print(B_ivec)
            idxs = np.argwhere(np.isnan(B_ivec)).flatten()
            print(idxs)
            print(f"We try with improved precision")
            '''
        else: # Nothing happened :)
            break
        atol_exp -= 10
        rtol_exp = -13
        tries += 1
    return B_ivec, S_avec
    
    

def define_energies(N_substances, E_level_values, E_distribution,  E_dist_exponent=None, n_subs_per_E_level=[]):
    '''
    avec means vector with same dimension as substances
    ivec means vector with same dimension as types
    Takes:
        - N_substances: the number of substances
        - E_level_values: the actual Energy for each energy level [15,7,2,0] 15 will be energy of E_level 0, etc. 
        - n_subs_per_E_level: the discrete number of E_levels to whcih substances can belong to
        - distribution: this denotes how substances gets distributed in the different energy levels. Availables are:
            · forcing uniform: Here we force each E level to have same amount of substances. And if N_busbstances is not divisible by E_levels we assign the 
            rest of the substances to random E levels.
            · uniform: We use a uniform distribution so each substance has same prob of belonging to the different energy levels
            · power law: Substances get distributed in the different E_levels according to ~ x^exponent. 
        - exponent: if the distribution is power law, this is the exponent
    Returns:
        -E_avec with the energy levels for each substance as values. It is returned sorted in descending level.
    '''
    def truncated_power_law(a, m):
        # Gotten from https://stackoverflow.com/questions/24579269/sample-a-truncated-integer-power-law-in-python
        x = np.arange(1, m+1, dtype='float') # x = [1,2,...,m] m+1 is NOT there
        pmf = x**a
        pmf = pmf/pmf.sum() # We Normalize the probability mass function
        return stats.rv_discrete(values=(range(1, m+1), pmf)) # This let's us define discrete probability functions. See more at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html
    E_avec = []
    E_levels = len(E_level_values)
    N_substances = int(N_substances) # We need to ensure is an int. 1e5 is considered float for example
    if E_distribution == 'given':
        if np.sum(n_subs_per_E_level) != N_substances or len(n_subs_per_E_level) != E_levels:
            print(f'E_dist_info given: {n_subs_per_E_level} must sum up to {N_substances} (N_substances) and have {E_levels} (E_levels) entrances.')
            raise
        for E, n_subs_in_level in zip(E_level_values, n_subs_per_E_level):
            E_avec.extend([E for _ in range(n_subs_in_level)]) # Remember 0 energy level has maximum of energy 
        E_avec = np.array(E_avec)
    elif E_distribution == 'forcing uniform': # Here we force each E level to have same amount of substances. And if N_busbstances is not divisible by E_levels we assign the rest of the subs to random E levels
        N_subs_per_Elevel_truncated = N_substances//E_levels
        E_distribution_vec = [N_subs_per_Elevel_truncated]*E_levels # We distribute equally substances per energy level
        remaining_subs = N_substances-N_subs_per_Elevel_truncated*E_levels # If N_substances is not divisible by E_levels, there are substances remaining
        if remaining_subs>0: # for the substances remaining we assign them randomly an E_level (without repetition)
            remaining_levels = random.sample(range(E_levels), remaining_subs)
            for i in remaining_levels:
                E_distribution_vec[i] += 1
        for E_level, N_subs_per_Elevel_real in enumerate(E_distribution_vec): # Here we construct the Energy_avec, i.e. the vector that gives the energy level for each substance
            E_avec.extend([E_level]*N_subs_per_Elevel_real)
        E_avec = np.array(E_avec)
        E_avec = np.flip(np.sort(E_avec)) # We want a descending order so the most energetic resources first!
    elif E_distribution == 'uniform':
        E_avec = np.random.randint(0, E_levels, size=N_substances)
        E_avec = np.flip(np.sort(E_avec)) # We want a descending order so the most energetic resources first!
    elif E_distribution == 'power law': # Actually uniform should be the same as power law with zero exponent!!
        if E_dist_exponent is None:
            print(f'For power law distribution, an exponent argument needs to be given')
            raise
        E_avec = truncated_power_law(E_dist_exponent, E_levels).rvs(size=N_substances) # Here we only sample the power law x^-Exponent for E_levels integers values and size of the sample=N substances
        E_avec = E_avec-1 # The distribution is from 1 to E_levels+1 but we want it from 0 to E_levels. So we substract 1 element wise because E_avec is numpy vector
        # We are using same probs although because what we wanted was a distribution for E_levels values according to a power law. 
        # Those E_levels values can have the number that they want. They could be 30, 31, 32, 33 but will be distributed as a discrete power law from 1 to 5 (5 is not included remember)
        E_avec = np.flip(np.sort(E_avec)) # We want a descending order so the most energetic resources first!
    else:
        print(f'Distribution name: {E_distribution} is not considered. The ones allowed are: [uniform, power law, forcing uniform, given]')
        raise
    return E_avec 

def get_E_mat_from_E_avec(E_avec):
    return np.subtract.outer(E_avec, E_avec) # This gets me the matrix of the differences Ei-Ej for each entrance i,j. See: https://stackoverflow.com/questions/26053914/how-to-construct-a-matrix-of-all-possible-differences-of-a-vector-in-numpy
def get_D_mat_from_D_avec(D_avec):
    return np.subtract.outer(D_avec, D_avec) # This gets me the matrix of the differences Ei-Ej for each entrance i,j. See: https://stackoverflow.com/questions/26053914/how-to-construct-a-matrix-of-all-possible-differences-of-a-vector-in-numpy

def get_E_level_to_subs_idx_dict(n_subs_per_E_level):
    E_level_to_subs_idx_dict = {}
    suma = 0
    for E_level, n_subs_in_E_level in enumerate(n_subs_per_E_level):
        E_level_to_subs_idx_dict[E_level] = list(range(suma,suma+n_subs_in_E_level))
        suma += n_subs_in_E_level
    return E_level_to_subs_idx_dict

def define_difficulties(E_mat, how='E proportional', proportion=1, variance=1, D_values_per_E_level=[], D_prohibited_connections=[], 
                        E_level_to_subs_idx_dict={}):
    '''
    avec means vector with same dimension as substances
    ivec means vector with same dimension as types
    This function defines the difficulties for the pathways from the E-mat. So far only three ways are allowed:
    1) Direct proportionality between E_mat and D_mat with proportion argument given
    2) Average proportionality meaning D_mat = gaussian with average E_mat(a,b)*proportion and variance argent given
    3) how=given. For this we need:
    - D_values_per_E_level: defines the maximum difficulty of that energy level. [15, 10, 4, 2, 0] where each entrance corresponds to an energy level (that can have many substances inside)
    - D_prohibited_connections: if we want to avoid connections between energy levels. This is a list where each value is a tuple with E_level_a and
    E_level_b (E_a>E_b), and this pathways will be prohibited so the difficulty has to be np.nan
    - E_level_to_subs_idx_dict: {E_level_a: [0,1,2], ...} where 0,1,2 are the indices of substances belonging to level a
    '''
    if how == 'given':
        D_avec = []
        for E_level, Difficulty in enumerate(D_values_per_E_level):
            n_subs_in_level = len(E_level_to_subs_idx_dict[E_level])
            D_avec.extend([Difficulty for _ in range(n_subs_in_level)]) # Remember 0 energy level has maximum of energy 
        D_mat = get_D_mat_from_D_avec(D_avec).astype(float)
        for E_level_a, E_level_b in D_prohibited_connections:
            rows = E_level_to_subs_idx_dict[E_level_a]
            cols = E_level_to_subs_idx_dict[E_level_b]
            for row in rows:
                D_mat[row, cols] = np.nan
        D_mat = np.where(D_mat > 0, D_mat, np.nan)
    elif how == 'E proportional': # The matrix is just proportional to the E_mat
        D_mat = np.where(E_mat > 0, E_mat*proportion, np.nan) # where E_mat>0 => E_mat*proportion; otherwise we put np.nan
    elif how == 'E proportional random': # The matrix is on average proportional to the E_mat, but hfluctuates gaussianly around that value
        #D_mat = np.full(E_mat.shape, np.inf) # Initialize to infinity
        D_mat = np.zeros(E_mat.shape)
        for a in range(E_mat.shape[0]):
            for b in range(E_mat.shape[1]):
                if E_mat[a,b]>0: 
                    # Only change the pathways that gives a positive energy (those are already a subset of the upper triangular part of the matrix)
                    val = round(np.random.normal(E_mat[a,b]*proportion, variance))
                    if val > 0: 
                        D_mat[a,b] = val
                    else: # If the new value for difficulty is zero or negative we take the older value because negative would mean impossibility to make
                        D_mat[a,b] = E_mat[a,b]*proportion
        D_mat = np.where(D_mat > 0, D_mat, np.nan)
    else:
        print(f'Difficulties how parameter: {how} is not considered. The ones allowed are: [E proportional, E proportional random]')
        raise
    return D_mat



def initialize_difficulties(N_types, D_mat, how, prob=None, given_paths_ivec=None):
    '''
    avec means vector with same dimension as substances
    ivec means vector with same dimension as types
    This function defines the init state of the learned difficulties for each type. So far there are two ways of initializing:
        0) "pathways given". Pathways are given for each type. So we have a given_paths_ivec which has a vector for each value containing tuples of row
        and col which tell us which pathways are activated.
        1) "1 pathway". Each type gets assigned one pathway randomly and the others set to 0.
        2) "1 pathway + random uniform". We add at random more pathways but mantain the one per type at least. So each learned difficulty for the type have 
        a certain prob to be initiated. This prob is uniform.
        3) "1 pathway + random proportional". Same as before but the probability of assigning the pathway is inverse to the difficulty of the pathway.
        4) "full". All pathways are filled.
    Returns:
        - D_mat_ivec: which is the vector for the learned difficulties matrices of each type. [D_mat of type 1, D_mat of type 2, ....]
    '''
    init_D = np.where(np.isnan(D_mat), D_mat, 0) #np.zeros(D_mat.shape)
    D_mat_ivec = [] # The ivec for the difficulty matrix of each vector.
    for i in range(N_types):
        D_mat_i = init_D.copy() # D_mat_i is the D_mat of type i (so learned difficulties for type i)
        if how == "full":
            for a in range(D_mat.shape[0]):
                for b in range(D_mat.shape[1]):
                    if b>a:
                        D_mat_i[a,b] = D_mat[a,b]
        elif how == 'pathways given':
            for row, col in given_paths_ivec[i]:
                D_mat_i[row,col] = D_mat[row,col]
        elif how in ['1 pathway', '1 pathway + random uniform', '1 pathway + random proportional']: # For all the other cases we need at least one pathway, so we compute that first
           
            # Here we assign at least 1 pathway to each type. We do it by mapping the upper triangular matrix to a 1 dimensional vector. So with just one random value
            # that stands as the index in that one-dimensional vector (hereon k), we can obtain the index in the upper triangular matrix. With regular matrices this 
            # formula is easy row = k//columns of D_mat; col = k%columns of D_mat. For an upper triangular it gets more complicated. 
            # See: https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix and 
            #      https://dotnettutorials.net/lesson/upper-triangular-matrix-column-major-mapping/
            n = np.sqrt(D_mat.size) 
            if n != int(n):
                print('The matrix must be squared for this method of initializing difficulties to work!!')
                raise
            n = int(n)
            ending_condition=False
            while not ending_condition:
                k = np.random.randint(0,n*(n-1)/2)
                row = int(n - 2 - int(np.sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5))
                col = int(k + row + 1 - n*(n-1)/2 + (n-row)*((n-row)-1)/2)
                if not np.isnan(D_mat[row,col]): # It is possible that we get a pathway of two substances belonging to same energy level, then that route is not available.
                    D_mat_i[row,col] = D_mat[row,col]
                    ending_condition=True
            if how == '1 pathway + random uniform':
                for aux_idx in range(int(n*(n-1)/2)): 
                    # Since we have an upper triangular squared matrix we only need to choose a value (aux_idx) from 0 to n*(n-1)/2. That value can be mapped to the 
                    # row (a) and col (b) of the matrix as shown in: https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix 
                    # Here is better this because we only compute a and b when the prob is satisfied and that occur few times, so it's better than nested for loops
                    if np.random.uniform(0,1) <= prob: # and aux_idx != k: 
                        a = int(n - 2 - int(np.sqrt(-8*aux_idx + 4*n*(n-1)-7)/2.0 - 0.5))
                        b = int(aux_idx + a + 1 - n*(n-1)/2 + (n-a)*((n-a)-1)/2)
                        D_mat_i[a,b] = D_mat[a,b]
            elif how == '1 pathway + random proportional':
                # Here we opt for 2 for loops because we would have to compute a,b for all values because the probability is inversily proportional to D_mat[a,b]
                # Therefore we simply go for 2 loops and only compute values for b>a
                for a in range(D_mat.shape[0]):
                    for b in range(D_mat.shape[1]):
                        if b>a:
                            proportion = D_mat[a,b]
                            if not np.isnan(proportion) and np.random.uniform(0,1) <= prob/proportion:
                                D_mat_i[a,b] = D_mat[a,b]
        else:
            print(f'How parameter: {how} is not considered. The ones allowed are: [full, pathways given, 1 pathway, 1 pathway + random uniform, 1 pathway + random proportional]')
            raise
            
        D_mat_ivec.append(D_mat_i)
    
    return np.array(D_mat_ivec)

def get_p_mat_from_d_mat(D_mat_i, D_mat):
    '''
    Here we get the P_mat_ivec, i.e. the P_mat ivector where each element is the P_mat (pathway mat) of each type. We do that by comparing the learned difficulties
    by each type, D_mat_ivec with the fixed difficulties of the pathways D_mat.
    '''
    P_mat_i_raw = (D_mat_i>=D_mat).astype(int) # We get a 1 at elements where D_mat_i >= D_mat or 0 otherwise
    P_mat_i = normalize(P_mat_i_raw, axis=1, norm='l1') 
    # We normalize p_mat_i row_wise so that each row sum up to 1
    # See rogueleaderr answer at: https://stackoverflow.com/questions/8904694/how-to-normalize-a-2-dimensional-numpy-array-in-python-less-verbose
    return np.array(P_mat_i)

def get_p_mat_ivec_from_d_mat_ivec(D_mat_ivec, D_mat):
    '''
    Here we get the P_mat_ivec, i.e. the P_mat ivector where each element is the P_mat (pathway mat) of each type. We do that by comparing the learned difficulties
    by each type, D_mat_ivec with the fixed difficulties of the pathways D_mat.
    '''
    P_mat_ivec = []
    for D_mat_i in D_mat_ivec:
        P_mat_ivec.append(get_p_mat_from_d_mat(D_mat_i, D_mat))
    return np.array(P_mat_ivec)

def initialize_state(N_types, N_substances, init_B=1e7, init_S=1):
    '''
    Here we initialize the types biomass and substances concentrations. So far we do it all equal to 1.
    '''
    B_ivec = np.full(N_types, init_B)
    S_avec = np.full(N_substances, init_S)
    types_identity_dict = {}
    for idx, type in enumerate(range(N_types)):
        types_identity_dict.update({type: [(0, idx)]})
    types_ivec = np.array(range(N_types))
    return B_ivec, S_avec, types_identity_dict, types_ivec

def tradeoff(D_mat_i, D_mat, P_mat_i, tradeoff_constant, tradeoff_only_used_pathways, tradeoff_exponent, tradeoff_maintenance, tradeoff_function='fraction', use_P_mat=False):
    if tradeoff_only_used_pathways:
        aux_matrix = D_mat_i*(D_mat_i>=D_mat).astype(int) # This gets me a matrix that has a 0 for unacquired pathways and the value of the difficultie of the pathway if it is acquired
        n_pathways = np.count_nonzero(aux_matrix[~np.isnan(aux_matrix)]) # This count the number of pathways
    else:
        aux_matrix = D_mat_i
    if use_P_mat:
        aux_matrix = aux_matrix*P_mat_i
    if tradeoff_function == 'fraction':
        gamma = 1/(1+tradeoff_constant*(tradeoff_maintenance*n_pathways+np.nansum(aux_matrix**tradeoff_exponent))) # The gamma constant which tell us how much energy is actually converted to biomass is inversively related to the actual difficulty used for pathways
    elif tradeoff_function == 'exponential':
        gamma = np.exp(-tradeoff_constant*(tradeoff_maintenance*n_pathways+np.nansum(aux_matrix**tradeoff_exponent))) 
    elif tradeoff_function == 'exponential_nl':
        gamma = np.exp(-tradeoff_constant*(tradeoff_maintenance*n_pathways+np.nansum(aux_matrix)**tradeoff_exponent)) 
    elif tradeoff_function == 'fraction_nl':
        gamma = 1/(1+tradeoff_constant*(tradeoff_maintenance*n_pathways+np.nansum(aux_matrix)**tradeoff_exponent))
    elif tradeoff_function == 'linear':
        gamma = 1/(tradeoff_constant*np.nansum(aux_matrix)**tradeoff_exponent)
    else:
        print(f'Tradeoff function: {tradeoff_function} is not permitted! Permitted ones are [fraction, exponential]')
        raise
    return gamma

def get_gamma_ivec(D_mat_ivec, D_mat, P_mat_ivec, gamma_value, tradeoff_constant, tradeoff_only_used_pathways, tradeoff_exponent, tradeoff_maintenance, tradeoff_function, use_P_mat):
    gamma_ivec = np.array([gamma_value*tradeoff(D_mat_i, D_mat, P_mat_i, tradeoff_constant, tradeoff_only_used_pathways, tradeoff_exponent, tradeoff_maintenance, tradeoff_function, use_P_mat) for D_mat_i, P_mat_i in zip(D_mat_ivec,P_mat_ivec)])
    return gamma_ivec

def define_h_avec(h_value_per_E_level, n_subs_per_E_level):
    h_avec = []
    for h, n_subs_in_level in zip(h_value_per_E_level, n_subs_per_E_level):
        h_avec.extend([h for _ in range(n_subs_in_level)]) # Remember 0 energy level has maximum of energy 
    
    return np.array(h_avec)

def define_uptake_parameters(max_growth_data, k_data, N_types, N_substances, how='1 value'):
    '''
    Here we define the uptake parameters. So far we only consider 1 value for all types and substances so we can simply give that as the value for
    max_growth_ivec_avec and the k_ivec_avec. Maybe in the future we can make differences for the different types or substances!
    '''
    if how == '1 value':
        max_growth_ivec_avec = np.full((N_types,1), max_growth_data)
        k_ivec_avec = np.full((N_types,1), k_data)
    # In this way I get ivec of [[k], [k], [k]] (so it has N_types values because is an ivec and then the avec dimension is just 1. We checked in this way
    # it works up in the equations without having to fill [k, ...k] for all the substances as an avec. It works due to broadcasting)
    return max_growth_ivec_avec, k_ivec_avec

def fixed_time_evo(birth_rate_ivec, evo_rate, delta_t):
    '''
    Here we obtain the types that mutate in a delta_t of time with mutation_rates being simply the birth rates multiplied by their evolutionary rate. Evo_rate could be an ivec but for now is a constant (XXX)

    '''
    individual_prob_ivec = birth_rate_ivec*evo_rate*delta_t
    types_mutating_vec = np.argwhere(individual_prob_ivec >= np.random.uniform(0,1, size=individual_prob_ivec.size)).flatten() # flatten is needed cuz this gives a vector of vectors otherwise
    # This gets me the indices of the values that are higher or equal than the random numbers. So that we get those that are mutating with prob equal to the
    # mutation_rate*delta_t. OJO!! I dont know if using the same random value as a threshold for all the ivec_values would work. To be sure we directly
    # use different random values for each.
    return delta_t, types_mutating_vec

def gillespie_evo(birth_rate_ivec, evo_rate):
    '''
    Here we obtain the increment of time for the next mutation (delta_t) and the index of the type that is mutating (idx_mutating). We do so in the standard way of the
    Gillespie algorith where the types rates are simply the birth rates multiplied by their evolutionary rate. Evo_rate could be an ivec but for now is a constant (XXX)
    '''
    individual_rates_ivec = birth_rate_ivec*evo_rate
    total_rate = np.sum(individual_rates_ivec)
    idx_mutating = np.random.choice(np.arange(birth_rate_ivec.size), p=individual_rates_ivec/total_rate)
    # In this way we choose a type with probabilities according to their birth rate * eco_rates
    delta_t = -np.log(1-np.random.uniform(0,1))/total_rate
    # Here we compute the delta_t according to an exponential distribution

    return delta_t, [idx_mutating] # I return an array for compatibility with fixed_time_evo function, so that both can be interexchanged with the rest of the code being the same

def mutation(idx_mutating_vec, mutation_dist, mutation_size, init_B_mutant_relative, gamma_ivec, delta_ivec, D_mat, D_mat_ivec, P_mat_ivec, B_ivec, 
             max_growth_ivec_avec, k_ivec_avec, N_types, init_N_types, new_types_appeared,  gamma_value, tradeoff_constant, tradeoff_exponent,
             tradeoff_maintenance, tradeoff_function, t, types_identity_dict, types_ivec, max_number_of_pathways_per_type, S_avec,
             E_mat, E_conversion_constant, uptake_type, use_P_mat,
             tradeoff_only_used_pathways=True, nan_mutating=False, all_mutants_considered=True, invasion=False, pathways_ids=None):
    '''
    avec means vector with same dimension as substances
    ivec means vector with same dimension as types
    This function performs the mutation. 
    It takes:
        - idx_mutating_vec: a vector in which elements are the index in the ivecs of the types that mutate
        - mutation_size: the size of the step for the mutations in the difficulties. (XXX) we have to discuss yet if mutations are gaussian or uniform
        - init_B_mutant_relative: mutant's init relative biomass. (XXX) we have to discuss how to include mutants
        - delta_ivec: an ivec with the outflow dilution rate for the types.
        - gamma_ivec: an ivec with the conversion constant from energy to biomass.
        - D_mat: the difficulty matrix for the pathways
        - D_mat_ivec: an ivec (types vectors) with the learned difficulties matrices as elements (for each type)
        - P_mat_ivec: an ivec with the pathway matrices as elements. Those matrices tell us which pathways can be used.
        - B_ivec: biomass ivector where each element is the biomass of a type
        - max_growth_ivec_avec: an ivec that for each value has an avec. SO tell us the max growth of type i for each a resource. 
        Ex: [[2,0,1], [0,0,2]] so type 0 feeds on substances a=0 and a=2 and type 1 only on substance 2.
        - k_ivec_avec: half saturation constant for type i and substance a. Same as max_growth_ivec_avec. Only used for Monod uptake type
        - N_types: the number of types
        - init_N_types: is the initial number of types
        - new_types_appeared: if a mutant appeared or not
        - gamma value: the conversion from mol of atp to biomass
        - max_N_types: maximum number of types we allow to consider
        - tradeoff_constant: the conversion constant for the tradeoff
        - tradeoff_function: the function used for the tradeoff
        - tradeoff_only_used_pathway: if the tradeoff includes all the acquired difficulties for the pathways or only the ones for pathways that are being used
        - nan_mutating: if we allow non realizable pathways to mutate
        - all_mutants_considered: if we consider all the mutations as giving raise to a new mutant (so a different type is included every time a mutation happens)
        or only those that create a new pathway. The first is biologically more sensible BUT computationally more expensive.
        - invasion: whether we have an invader (so completely new type with pathways at random) or not
        - types_identity_dict: a dictionary in which its keys are the types and the value is a vector with tuples of time and index in the ivec at that time. Ex:
        {0: [(0,t_0_appearing), (np.nan, t_0_disappearing)],
        1: [(1, t_1_appearing), (0, t_0_disappearing), (np.nan, t_1_disappearing)],
        ...} 
        We put np.nan in the index when they disappear so we can have a control of when they disappear, because otherwise we wouldn't know if they were disappear.
        Also, not only times of appearing in the population but the times in which their index change are recorded because they are needed to reconstruct
        the temporal evolution of each type.
        - types_ivec: tell us the type that is at each index at the particular time. Ex: [0, 2, 4, 5, 10, 12, ...] so type 0 at idx 0, type 2 at idx 1, etc
    It returns the modified gamma_ivec, delta_ivec, D_mat_ivec, P_mat_ivec, B_ivec, N_types, new_types_appeared, types_identity_dict, types_ivec
    '''
    if not nan_mutating:
        possible_mutating_Ds = np.argwhere(~np.isnan(D_mat)) # Checked
        # This gets me a 2-d array with only the indices of the difficulties that can mutate (so only realizable pathways, the rest are nan). Each row is
        # is a 1-dim vec indication row and column of the not nan indices
    else: # Here nan can also mutate
        it = np.nditer(D_mat, flags=['multi_index'])
        possible_mutating_Ds = np.array([it.multi_index for _ in it]) # This gives me the indixes of a matrix. See: https://numpy.org/doc/stable/reference/arrays.nditer.html#tracking-an-index-or-multi-index
    if not invasion: # when we have actual mutations
        for idx_mutating in idx_mutating_vec: 
            new_type = False
            mutating_D_mat = D_mat_ivec[idx_mutating].copy() # OJO! We need copies because the mutants are new types that need to be created with similar properties. We don't change the original matrices!!
            mutating_P_mat = P_mat_ivec[idx_mutating].copy()
            aux_rdn = np.random.randint(possible_mutating_Ds.shape[0]) # We choose a random element of D_mat by chosing a random row of possible_mutating_Ds
            idx = possible_mutating_Ds[aux_rdn,:] # The index, i.e. the difficulty that mutates, is chosen randomly from all the possible ones
            row = idx[0]
            col = idx[1]
            #mutating_D_mat[idx] += np.random.uniform(low=-mutation_size, high=mutation_size) # We add the mutation to that index. (XXX) We have to discuss if mutation is uniform, gaussian,...
            if not np.isnan(mutating_D_mat[row, col]): # Only if we choose to mutate a not nan that we have to compute things
                
                pre_difficulty_value = mutating_D_mat[row, col]
                required_difficulty = D_mat[row, col]
                if mutation_dist == 'gaussian':
                    mut = np.random.normal(0.0, mutation_size)
                elif mutation_dist == 'uniform':
                    mut = np.random.uniform(-mutation_size, mutation_size)
                else:
                    print(f'Given mutation distribution was {mutation_dist} which is not in: [gaussian, uniform]')

                mutating_D_mat[row, col] += mut
                #print(row,col)
                #print(mutating_D_mat[row,:])
                if pre_difficulty_value == required_difficulty and mutating_D_mat[row, col] < required_difficulty: # Pathway loss
                    mutating_P_mat[row, col] = 0
                    new_type = True
                    if np.nansum(mutating_P_mat[row,:]) > 0: # We renormalise only if there are other pathways. If there are not the row will be just 0 and renormilising would give me nans!
                        mutating_P_mat[row,:] = mutating_P_mat[row,:]/np.nansum(mutating_P_mat[row,:]) #normalize(mutating_P_mat[row,:], norm='l1') # We have to renormalise because we lost a pathway
                
                elif pre_difficulty_value < required_difficulty and mutating_D_mat[row, col] >= required_difficulty: # Pathway learned
                    mutating_P_mat[row, col] = 1
                    new_type = True
                    mutating_P_mat[row,:][mutating_P_mat[row,:] != 0] = 1 # Here we make all the pathways that are in use to be 1 in order to renormalise
                    mutating_P_mat[row,:] = mutating_P_mat[row,:]/np.nansum(mutating_P_mat[row,:]) #normalize(mutating_P_mat[row,:], norm='l1') # We have to renormalise because we learned a new pathway

                if mutating_D_mat[row, col] <= 0: # If a difficulty goes below 0 => we simply put a zero
                    mutating_D_mat[row, col] = 0

                if mutating_D_mat[row, col] >= required_difficulty: # If the learned difficulty goes above or equalises the require difficulty we set the learned difficulty value to simply the required_difficulty
                    mutating_D_mat[row, col] = required_difficulty
                if pre_difficulty_value != mutating_D_mat[row, col] and (all_mutants_considered or not tradeoff_only_used_pathways): 
                    # There is a mutant if the difficulties are different and:
                    #   - either we consider all of the variations in the difficulties as mutans (so all_mutants_considered = True)
                    #   - or we have a tradeoff that depend on all acquired difficulties not only the ones used for pathways (tradeoff_only_used_pathways = False)
                    new_type = True
            if new_type: # If we have a new mutant we add it to the types
                new_type_gamma = gamma_value*tradeoff(mutating_D_mat, D_mat, mutating_P_mat, tradeoff_constant, tradeoff_only_used_pathways, tradeoff_exponent, tradeoff_maintenance, tradeoff_function, use_P_mat)
                new_type_delta = delta_ivec[idx_mutating]
                new_type_max_growth_avec = max_growth_ivec_avec[idx_mutating]
                new_type_k_avec = k_ivec_avec[idx_mutating]
                new_type_D_mat = mutating_D_mat
                new_type_P_mat = mutating_P_mat
                new_type_size = np.minimum(init_B_mutant_relative*np.sum(B_ivec), B_ivec[idx_mutating]) # So if the actual population of the parent is smaller that the mutant introduction size we can only mutate until the parent size 
                B_ivec[idx_mutating] = B_ivec[idx_mutating] - new_type_size # OJO!! IF we are so unlucky to select a parent with only few more biomass that the threshold, this would be negative. We prevent this with both, doing that the mutating population cannot be higher that the actual parent above
                # (If we really want to substract this we would send the parent to 0 essentially and mutate only the population of the parent) DISCUSS!!
                
            else: # If we don't have a new mutant we change the whole past population with the mutation because mutant and parent would be equal ecologically
                D_mat_ivec[idx_mutating] = mutating_D_mat
                P_mat_ivec[idx_mutating] = mutating_P_mat
                
    else: # Here we have invasion
        
        # from all possible pathways we select a random number of them: n_pathways
        n_pathways = np.random.randint(1, max_number_of_pathways_per_type) # OJO! we speed up this part because bacteria cannot possibly have all pathways and we know it will get killed soon due to the tradeoff actually. This only affects the temporal scale! In the past we had: len(possible_mutating_Ds))
        identity_of_pathways = np.array([possible_mutating_Ds[pathway] for pathway in np.random.choice(len(possible_mutating_Ds), n_pathways, replace=False)])
        new_type_D_mat = np.where(np.isnan(D_mat), D_mat, 0)
        new_type_D_mat[tuple(identity_of_pathways.T)] = D_mat[tuple(identity_of_pathways.T)] # See here: https://stackoverflow.com/questions/70231320/how-to-change-element-of-matrix-by-a-list-of-index
        new_type = True
        for type_idx, D_mat_i in enumerate(D_mat_ivec):
            if np.array_equal(new_type_D_mat, D_mat_i, equal_nan=True) == True:
                new_type = False
                break
        new_type_size = init_B_mutant_relative*np.sum(B_ivec)
        if new_type == False:
            B_ivec[type_idx] += new_type_size
        else: # so there is a new type
            new_type_P_mat = get_p_mat_from_d_mat(new_type_D_mat, D_mat)
            new_type_gamma = gamma_value*tradeoff(new_type_D_mat, D_mat, new_type_P_mat, tradeoff_constant, tradeoff_only_used_pathways, tradeoff_exponent, tradeoff_maintenance, tradeoff_function, use_P_mat)
            new_type_delta = delta_ivec[-1]
            new_type_max_growth_avec = max_growth_ivec_avec[-1]
            new_type_k_avec = k_ivec_avec[-1]
            # OJO!!! If in the future we want to introduce new values we should change this. SO far all values are equal
            # new_type_delta = define_delta(delta_data, how=delta_dist)
            # new_type_max_growth_avec, new_type_k_avec = define_uptake_parameters(max_growth_data, k_data, N_types, N_substances, how=uptake_parameters_dist)
            new_type_growth_rate = get_growth_rate(S_avec, new_type_P_mat, E_mat, E_conversion_constant, new_type_gamma, new_type_delta, new_type_max_growth_avec, new_type_k_avec, uptake_type)
            if new_type_growth_rate < 0: # We don't add new type if it will get extinct
                new_type = False
        #pathways_identity = get_type_functional_identity(new_type_D_mat[~np.isnan(D_mat)], pathways_ids, D_mat)
        #print(f'The new mutant pathways identity is: {pathways_identity}')
        
    if new_type:
        gamma_ivec = np.append(gamma_ivec, new_type_gamma) # The mutant has a different gamma cuz the pathway utilization is different
        delta_ivec = np.append(delta_ivec, new_type_delta) # Unless otherwise specified mutants have same outflow dilution rate (delta) than parents
        max_growth_ivec_avec = np.append(max_growth_ivec_avec, [new_type_max_growth_avec], axis=0) # Unless otherwise specified mutants have same max_growth than parents
        k_ivec_avec = np.append(k_ivec_avec, [new_type_k_avec], axis=0) # Unless otherwise specified mutants have same half saturation constant than parents
        D_mat_ivec = np.append(D_mat_ivec, [new_type_D_mat], axis=0) # To understand why add it in this way, see: https://numpy.org/doc/stable/reference/generated/numpy.append.html
        P_mat_ivec = np.append(P_mat_ivec, [new_type_P_mat], axis=0)
        B_ivec = np.append(B_ivec, new_type_size) # OJO!!! This has to be discussed, how we include them!!! (XXX) Also if we need to change parental biomass!!!           
        N_types += 1
        new_types_appeared += 1 # The total number of types (not only active ones) will be initial number + new_types_appeared
        #total_N_types = init_N_types + new_types_appeared
        total_N_types = list(types_identity_dict.keys())[-1] + 2 # total number of types is last type number + 2 because type numbers are from 0 and we are adding a new one
        types_ivec = np.append(types_ivec, total_N_types-1)
        types_identity_dict.update({total_N_types-1: [(t, N_types-1)]}) # (appearing_time, its index in the ivec)

    return delta_ivec, gamma_ivec, D_mat_ivec, P_mat_ivec, B_ivec, max_growth_ivec_avec, k_ivec_avec, N_types, new_types_appeared, types_identity_dict, types_ivec

def extinction(B_ivec, extinction_threshold_relative, gamma_ivec, delta_ivec, D_mat_ivec, P_mat_ivec, max_growth_ivec_avec, k_ivec_avec,
               birth_rate_ivec, N_types, max_N_types, t, types_identity_dict, types_ivec, idx_mutating_vec, threshold, pathways_ids, D_mat):
    '''
    Here we remove populations whose relative abundance lay below an extinction threshold. However, in doing so we have to change all the ivec's, and also since
    we are changing the ivec's we have to keep track which species is at which index at each time. For that, we define the following objects:
    - types_identity_dict: a dictionary in which its keys are the types and the value is a vector with tuples of time and index in the ivec at that time. Ex:
    {0: [(0,t_0_appearing), (np.nan, t_0_disappearing)],
     1: [(1, t_1_appearing), (0, t_0_disappearing), (np.nan, t_1_disappearing)],
     ...} 
    We put np.nan in the index when they disappear so we can have a control of when they disappear, because otherwise we wouldn't know if they were disappear.
    Also, not only times of appearing in the population but the times in which their index change are recorded because they are needed to reconstruct
    the temporal evolution of each type.
    - types_ivec: tell us the type that is at each index at the particular time. Ex: [0, 2, 4, 5, 10, 12, ...] so type 0 at idx 0, type 2 at idx 1, etc
    '''
    if threshold == True:
        extinguising_types_idxs = np.argwhere(B_ivec/np.sum(B_ivec)<=extinction_threshold_relative).flatten() # This get me the indices of the types having less relative abundance than the threshold
        extinguising_types_idxs = np.sort(list(set(extinguising_types_idxs)-set(idx_mutating_vec))) # We extinguish only types that do not mutate, if they are mutating they don't get extinguish this time
        number_extinguising_types = np.size(extinguising_types_idxs)
    else: # So we are here for maximum_N_types threshold=False
        n_types_above_max = N_types-max_N_types
        if n_types_above_max>=0: # OJO! This comes before mutation comes so if it's equal, mutation will go over the max and we don't want that
            extinguising_types_idxs = np.argpartition(birth_rate_ivec, n_types_above_max)[:n_types_above_max] #https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
            # Now we check if there are mutating idx in extinguising idx because we will ignore the mutating ones so we nee to extinguis the following n_mutating_idxs_in_extinguising_idxs with the least birthrate
            n_mutating_idxs_in_extinguising_idxs = np.isin(idx_mutating_vec, extinguising_types_idxs).sum() # This gives me the amount of mutating vec that would go extinct
            if n_mutating_idxs_in_extinguising_idxs > 0:
                aux_extinguising = n_types_above_max + n_mutating_idxs_in_extinguising_idxs
                extinguising_types_idxs = np.argpartition(birth_rate_ivec, aux_extinguising)[:aux_extinguising]
                extinguising_types_idxs = np.sort(list(set(extinguising_types_idxs)-set(idx_mutating_vec))) # OJO! In this way if we have a mutating indx that was supposed to extinguish
            number_extinguising_types = np.size(extinguising_types_idxs)
        else:
            number_extinguising_types = 0
    if number_extinguising_types > 0: 
        print(f'EXTINGUISHING idxs: {extinguising_types_idxs}')
        '''
        pathways_identities = []
        for type_id in extinguising_types_idxs:
            pathways_identities.append(get_type_functional_identity(D_mat_ivec[type_id][~np.isnan(D_mat)], pathways_ids, D_mat))
        print(f'Their pathways identities are: {pathways_identities}')
        '''
        N_types = N_types - np.size(extinguising_types_idxs)
        B_ivec = np.delete(B_ivec, extinguising_types_idxs)
        #B_ivec_t = np.delete(B_ivec_t, extinguising_types_idxs, axis=0)
        gamma_ivec = np.delete(gamma_ivec, extinguising_types_idxs)
        delta_ivec = np.delete(delta_ivec, extinguising_types_idxs)
        max_growth_ivec_avec = np.delete(max_growth_ivec_avec, extinguising_types_idxs, axis=0)
        k_ivec_avec = np.delete(k_ivec_avec, extinguising_types_idxs, axis=0)
        D_mat_ivec = np.delete(D_mat_ivec, extinguising_types_idxs, axis=0)
        P_mat_ivec = np.delete(P_mat_ivec, extinguising_types_idxs, axis=0)
        for idx_extinguishing in extinguising_types_idxs: #estinguising_types_idxs need to come here sorted! (and it does)
            type_extinguishing = types_ivec[idx_extinguishing]
            types_identity_dict[type_extinguishing].append((t, np.nan))
        type_mutating_vec = [types_ivec[idx] for idx in idx_mutating_vec] # we get the types that mutate
        types_ivec = np.delete(types_ivec, extinguising_types_idxs) # The extinguishing types must disappear as well from types_ivec 
        idx_mutating_vec = np.searchsorted(types_ivec, type_mutating_vec) # Searches type_mutating_vec in types_ivec and returns the indices # seen at: https://stackoverflow.com/questions/12122639/find-indices-of-a-list-of-values-in-a-numpy-array
        # After changing the idx of the types, we get the new index for the mutating types (REMEMBER THIS CANNOT BE EXTINGUISHED SO WE SHOULDN'T GET AN ERROR HERE)
        for idx in range(extinguising_types_idxs[0], len(types_ivec)): # indices that lay before the first extinguishing index don't get modified! All the others do.
            type_changing = types_ivec[idx]
            types_identity_dict[type_changing].append((t, idx))
        

    
    return B_ivec, gamma_ivec, delta_ivec, D_mat_ivec, P_mat_ivec, max_growth_ivec_avec, k_ivec_avec, N_types, types_identity_dict, types_ivec, idx_mutating_vec
    
def get_given_paths_ivec(E_avec, E_levels):
    given_paths_ivec = []
    N_types = 0
    for E_level in range(E_levels-1, E_levels)[::-1]:
        types_in_higher_energy_level = np.argwhere(E_avec == np.max(E_avec)).flatten()
        types_in_lower_energy_level = np.argwhere(E_avec <= np.max(E_avec)-1).flatten()
        aux = types_in_higher_energy_level.size - types_in_lower_energy_level.size
        if aux>0:
            end_products = np.concatenate((np.random.shuffle(types_in_lower_energy_level), np.random.choice(types_in_lower_energy_level, size=aux)))
        else:
            end_products = np.random.choice(types_in_lower_energy_level, size=types_in_higher_energy_level.size, replace=False)
        for type, end_prod in zip(types_in_higher_energy_level, end_products):
            given_paths_ivec.append([[type, end_prod]])
        N_types += len(types_in_higher_energy_level) # So we get 1 type for each path
    return N_types, np.array(given_paths_ivec) # So we get 1 path for each upcoming resource!

def get_D_mat_ivec(D_ivec, D_mat):
    D_mat_ivec = []
    for D_info in D_ivec:
        D_mat_i = np.where(~np.isnan(D_mat), D_info, np.nan)
        D_mat_ivec.append(D_mat_i)
    return np.array(D_mat_ivec)

def get_last_t_data_from_file(rootpath):
    t_path = os.path.join(rootpath, "t_data.npy")
    B_path = os.path.join(rootpath, "B_data.pkl")
    S_path = os.path.join(rootpath, "S_data.npy")
    D_path = os.path.join(rootpath, "D_data.pkl")
    t_file = open(t_path, 'rb')
    B_file = open(B_path, 'rb')
    S_file = open(S_path, 'rb')
    D_file = open(D_path, 'rb')
    t_vec = []
    B_tvec = []
    S_tvec = []
    D_tvec = []
    i=0
    j=0
    while True:
        # See how to load from same file different vectors: https://numpy.org/doc/stable/reference/generated/numpy.load.html and https://numpy.org/doc/stable/reference/generated/numpy.save.html
        try: 
            t_aux = np.load(t_file)
            B_aux = pickle.load(B_file)
            S_aux = np.load(S_file)
            D_aux = pickle.load(D_file)
            time = t_aux[-1]
            B_ivec = B_aux[-1]
            S_avec = S_aux[-1]
            D_ivec = D_aux[-1] 

        except Exception as e: #EOFError: # Only when we finish reading the file we exit. SInce all vectors are saved at the same time they will have same amount of vectors so they will all end when the t_path ends
            print(traceback.format_exc())
            t_vec = t_vec[:len(D_tvec)]
            B_tvec = B_tvec[:len(D_tvec)]
            S_tvec = S_tvec[:len(D_tvec)]
            # This above is only needed when the data has not finished, because it can happen that since we are constantly adding data, that we arrive at a moment
            # where we have t_aux already dumped but not the others (Or same for B or S or D tvecs)
            break
    t_file.close()
    B_file.close()
    S_file.close()
    D_file.close()
    return time, B_ivec, S_avec, D_ivec

def get_D_mat_ivec(D_ivec, D_mat):
    '''
    Takes:
    - D_ivec: is an ivec vector (so types in each values) that has the vector with the acquired difficulties of each type as values. These are obtained
    by doing D_mat_i[~np.isnan(D_mat)], so we get all the difficulty values for the possible pathways (the ones with nan are not possible!). It is a python
    list so len() works on it. 
    - D_mat: is the pathways fixed difficulties.
    Returns:
    D_mat_ivec: an ivec (types in each entrance of the vector) that has the D_mat of those types. Here we fill the non nan values with the info from D_ivec
    and the others remain as nan.
    '''
    D_mat_ivec = np.full((len(D_ivec), D_mat.shape[0], D_mat.shape[1]), np.nan) # We create a D_mat_ivec with right dimensions (N_types, N_subs, N_subs). Len works on D_ivec because is a list, but had it been numpy and it would return 0th axis dimension which is what we want
    D_mat_ivec[:,~np.isnan(D_mat)] = D_ivec # This accomplishes D_mat_i[~np.isnan(D_mat)] = D_ivec[i] for all i's at once! (Pretty freaking cool!!)
    return np.array(D_mat_ivec)

def get_types_ivec(types_identity_dict):
    '''
    Takes:
    - types_identity_dict: a dictionary in which its keys are the types and the value is a vector with tuples of time and index in the ivec at that time. Ex:
    {0: [(t_0_appearing,0), (t_0_disappearing,np.nan)],
     1: [(t_1_appearing,1), (t_0_disappearing,0), (t_1_disappearing,np.nan)],
     ...} 
    We put np.nan in the index when they disappear so we can have a control of when they disappear, because otherwise we wouldn't know if they were disappear.
    Also, not only times of appearing in the population but the times in which their index change are recorded because they are needed to reconstruct
    the temporal evolution of each type.
    Returns:
    - types_ivec: an ivec (each value corresponds to a current type) with the types identity number. 
    Ex: [0,4,5,8,...] So in i-index 0 is type 0 in i-index 1 is type 4 and so on so forth.
    '''
    types_ivec = []
    for type, type_info_vec in types_identity_dict.items():
        if not np.isnan(type_info_vec[-1][1]):
            types_ivec.append(type)

    return types_ivec

def get_types_identities(rootpath):
    with open(os.path.join(rootpath, "identity_data.json"), "r") as f:
        identity_dict = json.load(f)
        types_identity_dict = {int(k): v for k,v in identity_dict['types_identity_dict'].items()} # We have to cast keys to int because they are str comming from json
        try:
            types_ivec = [int(v) for v in identity_dict['types_ivec']] # We cast them to int just in case
        except KeyError: # There are some realizations in whcih we didn't save types_ivec so we get a KeyError and we have to obtain in from last types_identity_dict data
            types_ivec = get_types_ivec(types_identity_dict)
       
    return types_identity_dict, types_ivec

def get_pathways_tuples(D_mat):
    D_mat_str = []
    D_mat_str = np.array([[(i,j) for j in range(D_mat.shape[1])] for i in range(D_mat.shape[0])])
    pathways_tuples = D_mat_str[~np.isnan(D_mat)]
    return pathways_tuples

def get_pathways_id(D_mat, E_avec, zeroth_Elevel_equal=True):
    '''
    OJO!! This has to change if we have substances with same energy but belonging to different levels dues to the difficulties!!
    We are not implementing that because yet we do not need that!
    Takes:
    - D_mat: the difficulty matrix so each entrance (a,b) gives me the difficulty of the metabolic route S_a->S_b
    - E_avec: avector (so substances vector) that gives me the energy of each substance
    - zeroth_E-level_equal: if the substances in the zeroth level are equal (so they enter at same rate h) bthere is no reason to distinguis between them.
    Returns:
    - pathways_ids: is a pathway vector (so a vector corresponding to the non nan values of D_mat obtained as: D_mat[~np.isnan(D_mat)]). That for each of this
    non nan pathways gives me its functional identity. So if we have that substances in the zero E_level are different from each otehr and we have 2 of those 
    substances and 4 E_levels we would have: S_A^0 -> S^1 => A0(the subindex indicates the substance, the superindex indicates E_level); S_A^0 -> S^2 => A1
    S_A^0 -> S^3 => A2; S^1 -> S^2 => A3; (Another 3 with those starting with substance B and then) S^1 -> S^3 => A4; S^2 -> S^3 => A5. The formula can be 
    obtained easily (I have it in the ipad) for S^(E_a) -> S^(E_b) => E_a*(E_levels) - sum_(k=2)^(E_a+1) k + E_b -1
    We only have (6+3) (the +3 is considering B != A) different possibilities for pathways identities because the only distinguisable thing between substances
    is their energy, i don't care the exact identity of the substance in the avec!
    '''
    pathways_tuple_info = get_pathways_tuples(D_mat)
    E_levels = len(np.unique(E_avec))
    E_levels_avec = [] #(np.max(E_avec) - E_avec).astype(int)
    level = 0
    past_E = E_avec[0]
    for E in E_avec: # We have E_avec ordered by energy! (That's needed)
        if E != past_E:
            level += 1
            past_E=E
        E_levels_avec.append(level)

    substances_in_higher_E = np.sum(E_avec == np.max(E_avec))
    
    pathways_ids = []
    pathways_id_strs = []
    for (a,b) in pathways_tuple_info:
        id_letter = None
        if not zeroth_Elevel_equal and a < substances_in_higher_E:
            id_letter = ascii_uppercase[a]
        Elevel_a = E_levels_avec[a]
        Elevel_b = E_levels_avec[b]
        number_id = int(Elevel_a*(E_levels)-np.sum(range(2,Elevel_a+1+1))+Elevel_b-1)
        identity_str = str(number_id)
        info_str = f'{Elevel_a}->{Elevel_b}'
        if id_letter is not None:
            identity_str = id_letter + identity_str
            info_str = f'S_{a}^{Elevel_a}->S^{Elevel_b}'
        pathways_ids.append(identity_str)
        pathways_id_strs.append(info_str)
    print(pathways_id_strs)
    print(pathways_ids)
    print(pathways_tuple_info)
    print(E_levels_avec)
    return np.array(pathways_ids), pathways_id_strs
    

def get_type_functional_identity(D_info, pathways_ids, pathways_id_to_str_dict, D_mat):
    D_mat_info = D_mat[~np.isnan(D_mat)]
    type_functional_identity_vec = [pathways_id_to_str_dict[id] for id in pathways_ids[D_info>=D_mat_info]]
    type_functional_identity = '|'.join(type_functional_identity_vec)
    return type_functional_identity

def full_model_evolution(params):

    continue_old = params['continue_old']
    path = params['path']
    if not os.path.exists(path):
        os.makedirs(path)
    random_seed = params['random_seed']
    np.random.seed(random_seed)
    #print(f"Numpy seed: {random_seed}")
    if continue_old: 
        added_time = params['ending_time']
        first_simulation_path = path
        k=1
        while True:
            path = os.path.join(first_simulation_path, f'advanced_{k}') # This is where the new data will be written 
            if not os.path.exists(path):
                os.makedirs(path)
                if k > 1:
                    last_existing_path = os.path.join(first_simulation_path, f'advanced_{k-1}')
                else:
                    last_existing_path = first_simulation_path
                break
            k += 1
        
        with open(os.path.join(last_existing_path, "params.json"), "r") as f:
            params = json.load(f)
        
        E_avec = np.loadtxt(os.path.join(first_simulation_path, "E_avec.txt"))
        E_mat =  get_E_mat_from_E_avec(E_avec)
        D_mat = np.loadtxt(os.path.join(first_simulation_path, "D_mat.txt"))
        t, B_ivec, S_avec, D_ivec = get_last_t_data_from_file(last_existing_path)
        D_mat_ivec = get_D_mat_ivec(D_ivec, D_mat)
        types_identity_dict, types_ivec = get_types_identities(last_existing_path)
        N_types = B_ivec.size
        N_substances = S_avec.size
        
        init_N_types = np.count_nonzero(E_avec == np.max(E_avec))
        params['N_types'] = N_types
        params['N_substances'] = N_substances
        params['path'] = path
        params['ending_time'] = t+added_time
        params['random_seed'] = random_seed # The seed changes because we want a new seed for this second part of the code, because we do not save the state of the random generator so it would makje any sense to star with same seed. Could give us flawed results
        
      
    N_types = params['N_types']
    N_substances = params['N_substances']
    #E_levels = params['E_levels']
    S_molecular_mass = params['S_molecular_mass']
    colon_volume = params['colon_volume']
    delta = params['delta']
    evo_rate = params['evo_rate']
    mutation_dist = params['mutation_dist']
    mutation_size = params['mutation_size']
    init_B_mutant_rel_to_thres = params['init_B_mutant_rel_to_thres']
    sample_freq = params['sample_freq']
    tofile_freq = params['tofile_freq']
    ending_time = params['ending_time']
    max_growth_data = params['max_growth_data']
    k_data = params['k_data']
    uptake_type = params['uptake_type']
    extinction_threshold_relative = params['extinction_threshold_relative']
    tradeoff_constant = params['tradeoff_constant']
    gamma_value = params['gamma_value']
    tradeoff_only_used_pathways = params['tradeoff_only_used_pathways']
    nan_mutating = params['nan_mutating']
    all_mutants_considered = params['all_mutants_considered']
    E_distribution = params['E_distribution']
    E_dist_exponent = params['E_dist_exponent']
    #E_dist_info = params['E_dist_info']
    init_D_mat_i_how = params['init_D_mat_i_how']
    init_D_mat_i_prob = params['init_D_mat_i_prob']
    given_paths_ivec = None
    random_seed = params['random_seed']
    max_N_types = params['max_N_types']
    use_P_mat = params['use_P_mat']
 
    try: # We need a try except because there will be past data that do not have invasion_period_h, neither the given D_mat and E_mat etc in the params!
        invasion_period_h = params['invasion_period_h']
        invasion = True
        tradeoff_exponent = params['tradeoff_exponent']
        tradeoff_maintenance = params['tradeoff_maintenance']
        tradeoff_function = params['tradeoff_function']
        D_values_per_E_level = params['D_values_per_E_level']
        n_subs_per_E_level = params['n_subs_per_E_level']
        D_prohibited_E_level_connections = params['D_prohibited_E_level_connections']
        max_number_of_pathways_per_type = params['max_number_of_pathways_per_type']
        h_value_per_E_level = params['h_value_per_E_level']
        E_values_per_level = params['E_values_per_level']
    except KeyError: # If we don't have any of the above means we won't have invasion and neither customized D and E mats nor the general tradeoff
        invasion_period_h = 0
        invasion = False  
        max_number_of_pathways_per_type = 0 
        tradeoff_exponent = 1
        tradeoff_maintenance = 0
        tradeoff_function = 'fraction'
        D_values_per_E_level = []
        n_subs_per_E_level = []
        E_levels = params['E_levels']
        if np.unique(E_avec).size != E_levels:
            print('OJO! E_levels need to match the number of different energies in order to continue.')
            # This is because here we only arrive by continuing old implementations. In those, for each E_level we had a different E_value
            raise
        for E in sorted(np.unique(E_avec), reverse=True): # 
            n_subs_per_E_level.append(np.count_nonzero(E_avec == E))
        D_prohibited_E_level_connections = []
        h_value = params['h_value']
        h_value_per_E_level = [h_value]
        h_value_per_E_level.extend([0 for _ in range(E_levels-1)])

        

    E_conversion_constant = colon_volume/S_molecular_mass
    init_B_mutant_relative = init_B_mutant_rel_to_thres*extinction_threshold_relative
    print(json.dumps(params, indent=2))

    if not continue_old: 
        E_levels = len(E_values_per_level)
        E_level_to_subs_idx_dict = get_E_level_to_subs_idx_dict(n_subs_per_E_level)   
        print(E_level_to_subs_idx_dict)
        E_avec = define_energies(N_substances, E_values_per_level, E_distribution=E_distribution, E_dist_exponent=E_dist_exponent,
                                 n_subs_per_E_level=n_subs_per_E_level) # Gives me the energies of each susbtance
        if init_D_mat_i_how == 'pathways given':
            #N_types, given_paths_ivec = get_given_paths_ivec(E_avec, E_levels)
            #print(given_paths_ivec)
            N_types = 1
            given_paths_ivec = np.array([[[0,3]]])
            np.savetxt(os.path.join(path, "given_paths_ivec.txt"), given_paths_ivec.reshape(N_types, 2))
        E_mat =  get_E_mat_from_E_avec(E_avec)
        D_mat = define_difficulties(E_mat, how='given', proportion=1, variance=None, D_values_per_E_level=D_values_per_E_level, 
                                    D_prohibited_connections=D_prohibited_E_level_connections, E_level_to_subs_idx_dict=E_level_to_subs_idx_dict)
        D_mat_ivec = initialize_difficulties(N_types, D_mat, how=init_D_mat_i_how, prob=init_D_mat_i_prob, given_paths_ivec=given_paths_ivec)    
        B_ivec, S_avec, types_identity_dict, types_ivec = initialize_state(N_types, N_substances, init_B=1e10)
        init_N_types = N_types
            
    print(E_avec)
    h_avec = define_h_avec(h_value_per_E_level, n_subs_per_E_level)
    P_mat_ivec = get_p_mat_ivec_from_d_mat_ivec(D_mat_ivec, D_mat)
    max_growth_ivec_avec, k_ivec_avec = define_uptake_parameters(max_growth_data, k_data, N_types, N_substances)
    gamma_ivec = get_gamma_ivec(D_mat_ivec, D_mat, P_mat_ivec, gamma_value, tradeoff_constant, tradeoff_only_used_pathways, tradeoff_exponent, tradeoff_maintenance, tradeoff_function, use_P_mat)
    delta_ivec = np.array([delta for _ in range(N_types)])
    delta_avec = np.array([delta for _ in range(N_substances)])
    pathways_ids, pathways_ids_strs = get_pathways_id(D_mat, E_avec, zeroth_Elevel_equal=True)
    pathways_id_to_str_dict = dict(zip(pathways_ids, pathways_ids_strs))
    t_vec = []
    B_tvec = []
    S_tvec = []
    D_tvec = []
    if not continue_old: # Only if we are in the first iteration that we add data to the t_vecs because in the other iterations the last data was already added so it would be repetitive
        B_ivec, S_avec = eco_advance(np.concatenate((B_ivec, S_avec)), 0, 1000, N_types, P_mat_ivec, E_mat, E_conversion_constant, gamma_ivec, delta_ivec, h_avec, delta_avec, max_growth_ivec_avec, k_ivec_avec, uptake_type=uptake_type)
        t=0
        t_vec.append(t)
        B_tvec.append(B_ivec)
        S_tvec.append(S_avec)
        D_ivec = D_mat_ivec[:,~np.isnan(D_mat)] # Same as: D_ivec = [D_mat_i[~np.isnan(D_mat)] for D_mat_i in D_mat_ivec]  OJO! This way of getting all the difficulties that are not nan preserves order, that's why we will be able to decode it after!!
        D_tvec.append(D_ivec)

    np.savetxt(os.path.join(path, "E_avec.txt"), E_avec)
    np.savetxt(os.path.join(path, "D_mat.txt"), D_mat)
    with open(os.path.join(path, "params.json"), "w") as outfile:
        json.dump(params, outfile, cls=NumpyEncoder)
    

    ending_condition = False
    total_mutations_trials = 0
    new_types_appeared = 0
    
    t_path = os.path.join(path, "t_data.npy")
    B_path = os.path.join(path, "B_data.pkl")
    with contextlib.suppress(FileNotFoundError):# remove a file if it already exits. See: https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
        os.remove(B_path)
    S_path = os.path.join(path, "S_data.npy")
    D_path = os.path.join(path, "D_data.pkl")
    
    last_sample = t//sample_freq
    last_sample_tofile = t//tofile_freq
    with open(t_path, 'wb') as t_file, open(S_path, 'wb') as S_file, open(D_path, 'wb') as D_file:
        while not ending_condition:
            if not invasion:
                birth_rate_ivec = get_birth_rates(B_ivec, S_avec, P_mat_ivec, E_mat, E_conversion_constant, gamma_ivec, max_growth_ivec_avec, k_ivec_avec, uptake_type=uptake_type)
                delta_t, idx_mutating_vec = gillespie_evo(birth_rate_ivec, evo_rate)
            else:
                delta_t = invasion_period_h
                idx_mutating_vec = []
            B_ivec, S_avec = eco_advance(np.concatenate((B_ivec, S_avec)), t, t+delta_t, N_types, P_mat_ivec, E_mat, E_conversion_constant, gamma_ivec, delta_ivec, h_avec, delta_avec, max_growth_ivec_avec, k_ivec_avec, uptake_type=uptake_type)
            
            # extinction threshold now
            birth_rate_ivec = get_birth_rates(B_ivec, S_avec, P_mat_ivec, E_mat, E_conversion_constant, gamma_ivec, max_growth_ivec_avec, k_ivec_avec, uptake_type=uptake_type)
            t = t+delta_t
            B_ivec, gamma_ivec, delta_ivec, D_mat_ivec, P_mat_ivec, max_growth_ivec_avec, k_ivec_avec, N_types, types_identity_dict, types_ivec, idx_mutating_vec  = extinction(B_ivec, 
                extinction_threshold_relative, gamma_ivec, delta_ivec, D_mat_ivec, P_mat_ivec, max_growth_ivec_avec, k_ivec_avec, birth_rate_ivec, N_types, max_N_types, 
                t, types_identity_dict, types_ivec, idx_mutating_vec, threshold=True, pathways_ids=pathways_ids, D_mat=D_mat)
            #extinction due to max number of types reached
            #B_ivec, gamma_ivec, delta_ivec, D_mat_ivec, P_mat_ivec, max_growth_ivec_avec, k_ivec_avec, N_types, types_identity_dict, types_ivec, idx_mutating_vec = extinction(B_ivec, 
                #extinction_threshold_relative, gamma_ivec, delta_ivec, D_mat_ivec, P_mat_ivec, max_growth_ivec_avec, k_ivec_avec, birth_rate_ivec, N_types, max_N_types, 
                #t, types_identity_dict, types_ivec, idx_mutating_vec, threshold=False)
            #mutation function now
            delta_ivec, gamma_ivec, D_mat_ivec, P_mat_ivec, B_ivec, max_growth_ivec_avec, k_ivec_avec, N_types, new_types_appeared, types_identity_dict, types_ivec = mutation(idx_mutating_vec, 
                mutation_dist, mutation_size, init_B_mutant_relative, gamma_ivec, delta_ivec, D_mat, D_mat_ivec, P_mat_ivec, B_ivec, 
                max_growth_ivec_avec, k_ivec_avec, N_types, init_N_types, new_types_appeared, gamma_value, tradeoff_constant, tradeoff_exponent, 
                tradeoff_maintenance, tradeoff_function, t, types_identity_dict, types_ivec, max_number_of_pathways_per_type, 
                S_avec, E_mat, E_conversion_constant, uptake_type, use_P_mat,
                tradeoff_only_used_pathways, nan_mutating, all_mutants_considered, invasion, pathways_ids)
            

            
            total_mutations_trials += 1
            sample = t // sample_freq
            if sample != last_sample:
            #if total_mutations_trials % sample_freq == 0:
                
                t_vec.append(t)
                B_tvec.append(B_ivec)
                S_tvec.append(S_avec)
                D_ivec = [D_mat_i[~np.isnan(D_mat)] for D_mat_i in D_mat_ivec] # OJO! This way of getting all the difficulties that are not nan conserves order, that's why we will be able to decode it after!!
                D_tvec.append(D_ivec)
                #last_sample = sample
                np.set_printoptions(precision=6, threshold=25, edgeitems=13)
                print((t, delta_t))
                print(B_ivec)
                print(S_avec)
                print(N_types)
                print(new_types_appeared)
                pathways_identities = []
                for D_mat_i in D_mat_ivec:
                    pathways_identities.append(get_type_functional_identity(D_mat_i[~np.isnan(D_mat)], pathways_ids, pathways_id_to_str_dict, D_mat))
                print(f'The identities of the types in the system are: {pathways_identities}')
                last_sample = sample
            
            sample_tofile = t // tofile_freq
            if sample_tofile != last_sample_tofile:
            #if total_mutations_trials % tofile_freq == 0:
                #np.save(B_file, np.stack(B_tvec), allow_pickle=False) # # done like this in the .npy format for efficiency. See: https://stackoverflow.com/questions/51391713/efficient-way-of-writing-numpy-arrays-to-file-in-python For how to load when saved multiple vec in same file see also: https://numpy.org/doc/stable/reference/generated/numpy.save.html
                np.save(t_file, np.stack(t_vec), allow_pickle=False) # stack is just to create a FULL numpy array instead of having a list of numpy arrays
                np.save(S_file, np.stack(S_tvec), allow_pickle=False) # stack is just to create a FULL numpy array instead of having a list of numpy arrays
                pickle.dump(D_tvec, D_file, protocol=pickle.HIGHEST_PROTOCOL)
                # I don't know why but we need to do the opening and closing here
                with open(B_path, 'ab') as B_file:
                    pickle.dump(B_tvec, B_file, protocol=pickle.HIGHEST_PROTOCOL) #np.save was giving me errors because it wasnt a pure numpy vector but a numpy vector with dtype=object because it has inhomogeneous shape
                with open(os.path.join(path, "identity_data.json"), "w") as outfile:
                    identity_dict = {'types_identity_dict': types_identity_dict, 'types_ivec': types_ivec}
                    json.dump(identity_dict, outfile, cls=NumpyEncoder)
                t_vec = []
                B_tvec = []
                S_tvec = []
                D_tvec = []
                last_sample_tofile = sample_tofile
                
                
            ending_condition = t>ending_time#t > 100*24
            # here we advance the eco dymamics until a mutatnt appears, then recompute D_mat, P_mat and continue evolving so on and forth
        # When we exit the while we save the excess of the tvecs
        np.save(t_file, t_vec)
        np.save(S_file, S_tvec)
        with open(B_path, 'ab') as B_file:
            pickle.dump(B_tvec, B_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(D_tvec, D_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, "identity_data.json"), "w") as outfile:
            identity_dict = {'types_identity_dict': types_identity_dict, 'types_ivec': types_ivec}
            json.dump(identity_dict, outfile, cls=NumpyEncoder)
    #t_file.close()
    #B_file.close()
    #S_file.close()
    #D_file.close()


