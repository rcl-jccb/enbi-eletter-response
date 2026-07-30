import os
import numpy as np
import pandas as pd
import json
from numpy.linalg import pinv
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import entropy
import dataanalyst as DA
from utils import *


def change_base_directory(current_path, old_base, new_base):
    # Normalize the path to ensure there are no platform-specific path separators
    normalized_path = os.path.normpath(current_path)
    new_path = normalized_path.replace(old_base, new_base)

    return new_path
def get_metadata_dict(rootpath, raw=False, overwrite=False, longitudinal=False, sample_size=None):
    longitudinal_str = ''
    sample_str = ''
    if longitudinal:
        longitudinal_str = '_Longitudinal'
    if sample_size is not None:
        sample_str = f'_{sample_size}'
    if raw:
        # Load the CSV file
        metadata_file = os.path.join(rootpath, 'processed', f'metadata_dict{longitudinal_str}{sample_str}.pkl')
        with open(metadata_file, 'rb') as fin:
            raw_metadata_dict = pickle.load(fin)
        return raw_metadata_dict
    else:
        wanted_file = os.path.join(rootpath, 'processed', f'postprocessed_metadata_dict{longitudinal_str}{sample_str}.pkl')
        if not overwrite and os.path.isfile(wanted_file): # here we try to load pear_pre_dict
            with open(wanted_file, 'rb') as fin:
                metadata_dict = pickle.load(fin)
        else:
            
            # Load the CSV file
            metadata_file = os.path.join(rootpath, 'processed', f'metadata_dict{longitudinal_str}{sample_str}.pkl')
            with open(metadata_file, 'rb') as fin:
                raw_metadata_dict = pickle.load(fin)
            # Initialize a new dictionary to group by 'realization'
            metadata_dict = {}

            # Iterate through the original dictionary
            for sample_id, info in raw_metadata_dict.items():
                patient_id = f"{info['diagnosis']}-{info['realization']}"
                # Create the entry for the realization if it doesn't exist
                if patient_id not in metadata_dict:
                    metadata_dict[patient_id] = []

                # Create the new dictionary structure for each sample
                sample_tuple = (patient_id, info['time_idx'], info['time'], info['diagnosis'], info['rho'])
                

                # Append the transformed sample data to the list for that realization
                metadata_dict[patient_id].append(sample_tuple)
            with open(wanted_file, 'wb') as f:
                pickle.dump(metadata_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return metadata_dict

def calculate_pathways_df(df, E_D_type=0, include_cost=True, include_presence=True):
    """
    Transforms the bacterial dataframe into a pathway-centric dataframe with Enzymatic Cost.
    Rows (pathways) are added dynamically as they are encountered.
    
    Parameters:
    - df: DataFrame, rows are bacterial functional identities (index), columns are samples, and values are biomasses.
    - E_D_type: is the Energy Difficulty/Enzymatic Cost structure of the system
    
    Returns:
    - pathways_df: DataFrame, rows are dynamically added pathway IDs, columns are samples, and values are Enzymatic Cost.
    """
    # Initialize an empty DataFrame for the pathways
    pathways_df = pd.DataFrame(0.0, columns=df.columns, index=[])
    cost = 1
    presence_series = 1
    # Iterate over each bacterial type in the DataFrame (rows)
    for bacteria_functional_identity, biomass_series in df.iterrows():
        # Convert biomass_series to a binary presence/absence series
        if include_presence:
            presence_series = (biomass_series != 0).astype(int)
        
        # For each pathway in the bacterial functional identity
        for pathway_id in bacteria_functional_identity.split('.'):
            # Calculate the cost for the current pathway
            if include_cost:
                cost = DA.calculate_enzymatic_cost_from_pathway_id(pathway_id, E_D_type)
            
            # Check if the pathway_id exists in pathways_df, if not, initialize it
            if pathway_id not in pathways_df.index:
                pathways_df.loc[pathway_id] = 0.0  # Initialize the row for this pathway with 0 for all samples

            # Add the cost * presence for each sample where the pathway is present
            pathways_df.loc[pathway_id] += cost * presence_series  # Accumulate cost only for the present samples

    return pathways_df

def get_data_df(rootpath, data_type='taxonomy', E_D_type=0, wanted_realization = None, overwrite=False, longitudinal=False, sample_size = None):
    longitudinal_str = ''
    sample_str = ''
    if longitudinal:
        longitudinal_str = '_Longitudinal'
    if sample_size is not None:
        sample_str = f'_{sample_size}'
    
    wanted_file = os.path.join(rootpath, 'processed', f'postprocessed_{data_type}_df{longitudinal_str}{sample_str}.pkl')
    if not overwrite and os.path.isfile(wanted_file): # here we try to load pear_pre_dict
        with open(wanted_file, 'rb') as fin:
            cleaned_df = pickle.load(fin)
    else:    
        if data_type == 'taxonomy':
            data_file = os.path.join(rootpath, 'processed', f'sampling_df{longitudinal_str}{sample_str}.pkl')
            with open(data_file, 'rb') as fin:
                df = pickle.load(fin)
            df = df.div(df.sum())
        elif data_type == 'coarsed_taxonomy':
            data_file = os.path.join(rootpath, 'processed', f'sampling_df{longitudinal_str}{sample_str}.pkl')
            with open(data_file, 'rb') as fin:
                df = pickle.load(fin)
            path = os.path.join(rootpath, wanted_realization)
            df = get_coarsened_species(path, df)
            df = df.div(df.sum())
        elif data_type == 'pathway':
            data_file = os.path.join(rootpath, 'processed', f'sampling_df{longitudinal_str}{sample_str}.pkl')
            with open(data_file, 'rb') as fin:
                samples_df = pickle.load(fin)
            df = calculate_pathways_df(samples_df, E_D_type)
        elif data_type == 'substances':
            data_file = os.path.join(rootpath, 'processed', f'substances_sampling_df{longitudinal_str}{sample_str}.pkl')
            with open(data_file, 'rb') as fin:
                df = pickle.load(fin)

        else:
            print(f'The data type must be either "taxonomy" or "pathway" you introduced: {data_type}')
        
        
        cleaned_df = df.dropna(axis=1) # This eliminates those cases in whcih filtered_df.sum() is basically 0 and the we would have nan's
        with open(wanted_file, 'wb') as f:
            pickle.dump(cleaned_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return cleaned_df
def get_full_data_dict(metadata_dict, data_df):
    final_dict = {}
    # Loop through each participant in the grouped dictionary
    for participant_id, samples in metadata_dict.items():
        final_dict[participant_id] = []  # Prepare a sub-dictionary for this participant
        # Loop through each sample for the participant
        for external_id, time_idx, time, diagnosis, rho in samples:
            # Only process if the sample_id exists in the DataFrame's columns
            sample_id = f'{participant_id}-{time_idx}'
            # Construct the sub-dictionary for this sample
            final_dict[participant_id].append({
                'Sample ID': sample_id,
                'Time num': time_idx,
                'Time point': time, 
                'Diagnosis': diagnosis,
                'rho': rho,
                'Data': data_df[sample_id]  # We have a df.series (so that we still have rows names)
            })

    return final_dict

def get_coarsened_species(rootpath, df):
    def coarse_bacterial_id(bacterial_id, p_id_to_category_dict):
        # Split pathway IDs, map to categories, sort, and join back
        pathway_ids = bacterial_id.split('.')

        categories = sorted(str(p_id_to_category_dict[str(pid)]) for pid in pathway_ids)
        return '.'.join(categories)
    D_mat = DA.get_D_mat(rootpath)
    E_avec = DA.get_E_avec(rootpath)
    pathways_strs = DA.get_pathways_strs(D_mat)
    pathways_ids, _ = DA.get_pathways_id(D_mat, E_avec)
    p_str_to_p_id_dict = dict(zip(pathways_strs, pathways_ids))
    _, p_str_to_category_dict, _ = DA.get_functions_categories()
    p_id_to_category_dict = {p_str_to_p_id_dict[p_str]: v[0] for p_str,v in p_str_to_category_dict.items()}
    # Step 2: Apply function to create a new column for coarsed IDs
    df['CoarsedID'] = df.index.map(lambda x: coarse_bacterial_id(x, p_id_to_category_dict))

    # Step 3: Group by coarsed IDs and sum the biomasses
    coarsed_df = df.groupby('CoarsedID').sum()
    
    return coarsed_df

def check_diagnosis_changes(metadata_dict):
    # Function to check for diagnosis changes in participants

    # This function takes a metadata dictionary where:
    # - The keys are participant IDs
    # - The values are lists of tuples, with each tuple containing ('sample_id', 'week_id', 'diagnosis')
    #
    # The function checks if there is a change in diagnosis (i.e., if a participant has multiple different diagnoses).
    # If a participant's diagnosis changes, the function prints:
    # - The participant ID
    # - Information about the sample IDs, weeks, and diagnoses for that participant.
    no_changes=True
    for participant_id, samples in metadata_dict.items():
        # Extract diagnosis information from the tuple (sample_id, week_id, diagnosis)
        diagnoses = [sample[2] for sample in samples]
        
        # Check if all diagnoses are the same for the participant
        if len(set(diagnoses)) > 1:  # If there is more than 1 unique diagnosis
            print(f"Diagnosis change detected for Participant {participant_id}")
            for sample in samples:
                print(f"Sample ID: {sample[0]}, Week: {sample[1]}, Diagnosis: {sample[2]}")
            print("\n")
            no_changes=False
    if no_changes:
        print('No diagnosis changes found in the participants.')

def find_max_species_samples(df, max_species, min_ratio=10):
    """
    Finds the maximum number of species (N) such that N species are all present in the same 10 * N samples.
    
    Parameters:
    - df: DataFrame with species as rows and samples as columns (values are abundances).
    - min_ratio: The sample-to-species ratio (default is 10).
    
    Returns:
    - best_species: List of species that meet the criterion.
    - best_samples: List of sample IDs where all selected species are present.
    - median_abundances: Series with median abundance values for the selected species.
    - filtered_df: Filtered DataFrame with the selected species and samples.
    """
    # Function to get the largest consecutive sample vector from an unsorted list of sample IDs: Diagnosis-patient_id-time_idx
    def get_consecutive_samples(samples):
        # Extract time index from sample names and sort by time
        sorted_samples = sorted(samples, key=lambda x: int(x.split('-')[-1])) # we want the time_idx to order, remember sample IDs: Diagnosis-patient_id-time_idx
        consecutive_samples = []
        temp_group = [sorted_samples[0]]
        
        for i in range(1, len(sorted_samples)):
            current_time = int(sorted_samples[i].split('-')[-1])
            previous_time = int(sorted_samples[i - 1].split('-')[-1])
            if current_time == previous_time + 1:
                temp_group.append(sorted_samples[i])
            else:
                if len(temp_group) > len(consecutive_samples):
                    consecutive_samples = temp_group
                temp_group = [sorted_samples[i]]
        
        # Final check if the last group was the longest
        if len(temp_group) > len(consecutive_samples):
            consecutive_samples = temp_group
            
        return consecutive_samples
    # Step 1: Calculate presence sets for each species
    species_presence = {species: set(df.columns[df.loc[species] > 0]) for species in df.index}
    #print(f'Species presence: {species_presence}')
    species_info = pd.DataFrame({
        'presence': (df > 0).sum(axis=1),
        'median_abundance': df.median(axis=1)
    })

    # Sort species by the number of samples in which they are present (descending order)
    sorted_species = species_info.sort_values(by=['presence', 'median_abundance'], ascending=[False, False]).index
    #sorted_species = sorted(species_presence.keys(), key=lambda sp: len(species_presence[sp]), reverse=True)
    best_species = []
    best_samples = []

    # Iteratively build combinations
    for start_idx, species in enumerate(sorted_species):
        # Initialize with the first species in the current iteration
        current_species = [species]
        current_samples = species_presence[species]

        # Check if the initial species has at least min_ratio samples
        if len(current_samples) < min_ratio:
            continue  # Skip species that don't meet the minimum sample requirement

        for next_idx in range(start_idx + 1, len(sorted_species)):
            if len(current_species) >= max_species:
                break  # Stop if we reach the maximum species limit
            next_species = sorted_species[next_idx]
            
            # Compute the intersection of samples with the next species
            new_samples = current_samples.intersection(species_presence[next_species])

            # Check if the new sample set meets the required sample count
            required_samples = min_ratio * (len(current_species) + 1)
            consecutive_samples = []
            if len(new_samples) > 0:
                consecutive_samples = get_consecutive_samples(new_samples)
            if len(consecutive_samples) >= required_samples:
                # Update species list and sample set
                current_species.append(next_species)
                current_samples = set(consecutive_samples)
            else:
                break  # Stop if adding this species doesn’t meet sample criteria
            
            # Update best result if this is the largest valid set found
            if len(current_species) > len(best_species):
                best_species = list(current_species)
                best_samples = sorted(current_samples, key=lambda x: int(x.split('-')[-1]))
    #best_samples = best_samples[:50]
    # Step 4: Build the final filtered DataFrame and calculate median abundances
    filtered_df = df.loc[best_species, best_samples]
    median_abundances = filtered_df.median(axis=1) if not filtered_df.empty else pd.Series(dtype=float)
    '''
    print(f'Best species: {best_species}')
    print(f'Best samples: {best_samples}')
    print(f'Median abundances: {median_abundances}')
    print(f'Filtered df: {filtered_df}')
    raise
    '''
    return best_species, best_samples, median_abundances, filtered_df

def process_data_for_limits_algorithm(full_data_dict, max_species, min_ratio, wanted_file, max_samples, overwrite=False):
    def get_patient_df(full_sample_data):
        combined_df = pd.DataFrame()
        data_dict = {}
        for sample in full_sample_data:
            sample_id = sample['Sample ID']  # Get the sample ID for column naming
            data_series = sample['Data']     # Get the data series with bacteria names and biomasses
            data_dict[sample_id] = data_series

        # Concatenate all series in the dictionary into a single DataFrame
        combined_df = pd.concat(data_dict, axis=1)

        # If you want to reset the DataFrame to avoid fragmentation
        combined_df = combined_df.copy()

        # Return the final DataFrame with bacteria as rows and sample IDs as columns
        return combined_df
    
    
    if not overwrite and os.path.isfile(wanted_file): # here we try to load pear_pre_dict
        with open(wanted_file, 'rb') as fin:
            processed_data_dict = pickle.load(fin)
    else:
        processed_data_dict = {}
        pd.options.display.float_format = '{:.3e}'.format
        for patient_id, full_sample_data in full_data_dict.items():
            patient_df = get_patient_df(full_sample_data)
            selected_species, selected_samples, median_abundances, filtered_df = find_max_species_samples(patient_df, max_species, min_ratio)
            selected_samples = selected_samples[:max_samples]
            num_species = len(selected_species)
            t_points = len(selected_samples)
            matrix_ivec = np.zeros((num_species, t_points - 1, num_species + 1))  # Subtract 1 because we need t and t+1
            
            processed_data_dict[patient_id] = {}
            processed_data_dict[patient_id]['Species'] = selected_species  # Store the top species names
            processed_data_dict[patient_id]['Samples'] = selected_samples  # Store the samples names
            processed_data_dict[patient_id]['Diagnosis'] = full_sample_data[0]['Diagnosis']
            filtered_df = patient_df.loc[selected_species, selected_samples]

            for t in range(t_points - 1):
                x_t = filtered_df.iloc[:,t]
                matrix_ivec[:, t, :num_species] = x_t - median_abundances
            
            for i,specie_name in enumerate(selected_species):
                x_tvec_i = filtered_df.loc[specie_name][:-1]
                x_tvec_shifted_i = filtered_df.loc[specie_name].shift(-1)[:-1]# The shift adds a np.nan value to the end
                matrix_ivec[i, :, -1] = np.log(x_tvec_shifted_i) - np.log(x_tvec_i)
            print(f'Patient: {patient_id}')
            print(f'Matrix: {matrix_ivec}')
            processed_data_dict[patient_id]['Data'] = matrix_ivec
        
        with open(wanted_file, 'wb') as f:
            pickle.dump(processed_data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    info_dict = {p_id: {'n_species': len(v['Species']), 'n_t_points': len(v['Samples']), 'Diagnosis': v['Diagnosis']}
                 for p_id, v in processed_data_dict.items()}
    return processed_data_dict, info_dict

def restricted_least_squares(inbag, outbag, test):
    """
    Performs linear regression and computes the error on the test set.
    """
    X1 = inbag[:, test]
    y1 = inbag[:, -1]
    B1 = pinv(X1) @ y1
    
    X2 = outbag[:, test]
    y2 = outbag[:, -1]
    error = root_mean_squared_error(y2, X2 @ B1) / np.var(y2)
    return [np.repeat(error, len(test)), B1, test]

def limits_algorithm(data, n, i, thresh, n_bagging):
    """
    Main function to perform feature selection and regression based on a threshold.
    - "data" is a matrix containing the data. One row of the matrix looks like {x_1(t) - u_1, ..., x_N(t) - u_N, ln x_i(t+1) - ln x_i(t) }, 
    where u_i is the median abundance of species i. ;
    - "n" is the number of species ;
    - "i" is the species that is the dependent variable of the regression ;
    - "thresh" is the prediction error threshold;
    """
    results = []

    for _ in range(n_bagging):
        excluded = list(set(range(n)) - {i})
        included = [i]

        # Randomly partition data into training and test sets
        train_idx, test_idx = train_test_split(range(len(data)), test_size=0.5)
        data1 = data[train_idx, :]
        data2 = data[test_idx, :]

        test = included
        best = restricted_least_squares(data1, data2, test)

        # Add covariates as long as prediction error decreases by thresh
        for _ in range(n-1):
            tmp2 = [restricted_least_squares(data1, data2, included + [ex])
                    for ex in excluded]
            prev = sorted(tmp2, key=lambda x: x[0][0])[0] # We sort the data based on the error and get the one with lowest error

            if 100.0 * (prev[0][0] - best[0][0]) / best[0][0] < -thresh:
                best = prev
                included = list(map(int, best[2]))
                excluded = list(set(excluded) - set(included))
            else:
                break

        # Store final regression coefficients
        B = np.zeros(n)
        B[np.array(best[2], dtype=int)] = best[1]
        results.append(B)

    return np.median(results, axis=0), np.average(results, axis=0)


def get_interaction_mat_dict_limits(rootpath, full_data_dict, max_species,  max_samples, min_ratio=10, error_threshold=1, 
                                    n_bagging=200, coarsed=False, overwrite=False):
    coarsed_str = ''
    if coarsed:
        coarsed_str = '_coarsed'
    name = f'interaction_mats{coarsed_str}_{max_species}_{max_samples}_{min_ratio}_{error_threshold}_{n_bagging}.pkl'
    limits_dir = os.path.join(rootpath, 'LIMITS')
    if not os.path.exists(limits_dir):
        os.makedirs(limits_dir)
    wanted_file = os.path.join(limits_dir, name)
    if not overwrite and os.path.isfile(wanted_file): # here we try to load pear_pre_dict
        with open(wanted_file, 'rb') as fin:
            (interaction_matrices_median_dict, interaction_matrices_avg_dict) = pickle.load(fin)
    else:
        data_dict_name = f'data_dict{coarsed_str}_{max_species}_{min_ratio}.pkl'
        data_dict_path = os.path.join(limits_dir, data_dict_name)
        processed_data_dict, info_dict = process_data_for_limits_algorithm(full_data_dict, max_species, min_ratio, data_dict_path, max_samples, overwrite)
        interaction_matrices_median_dict = {}
        interaction_matrices_avg_dict = {}
        n_patients = len(processed_data_dict)
        for j, (patient_id, data_dict) in enumerate(processed_data_dict.items()):
            c_total_median = []
            c_total_avg = []
            for i, specie_name in enumerate(data_dict['Species']):
                n_species = len(data_dict['Species'])
                matrix = data_dict['Data'][i]
                c_i_median, c_i_avg = list(limits_algorithm(matrix, n_species, i, thresh=error_threshold, n_bagging=n_bagging))
                c_total_median.append(c_i_median)
                c_total_avg.append(c_i_avg)
            interaction_matrices_median_dict[patient_id] = np.array(c_total_median)
            interaction_matrices_avg_dict[patient_id] = np.array(c_total_avg)
            print(f'Patient: {patient_id} is number {j}/{n_patients}')
        
        with open(wanted_file, 'wb') as f:
            pickle.dump((interaction_matrices_median_dict, interaction_matrices_avg_dict), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(limits_dir, "info_dict.json"), "w") as outfile:
            json.dump(info_dict, outfile, indent=2)

    return interaction_matrices_median_dict, interaction_matrices_avg_dict

def get_diagnosis_stats_dict(interaction_matrices_median_dict, interaction_matrices_avg_dict, full_data_dict):
    def compute_interaction_matrix_stats(interaction_matrices_dict, patient_diagnoses_dict):
        # Dictionary to store the computed stats for each patient
        stats_dict = {}
        
        for patient_id, matrix in interaction_matrices_dict.items():
            np.fill_diagonal(matrix, 0)
            positive_values = matrix[matrix > 0]
            negative_values = matrix[matrix < 0]
            # Calculating required statistics
            percent_positive = 100 * len(positive_values) / matrix.size
            sum_positive = positive_values.sum()
            sum_negative = negative_values.sum()
            rho = (sum_positive+sum_negative)/(sum_positive-sum_negative)  # Since sum_negative is negative, add it to sum_positive


            # Store the stats in a dictionary for each patient
            stats_dict[patient_id] = {
                'Diagnosis': patient_diagnoses_dict[patient_id],
                'percent_positive': percent_positive,
                'sum_positive': sum_positive,
                'sum_negative': sum_negative,
                'rho': rho
            }
        
        return stats_dict
    patient_diagnosis_dict = {p_id: vals[0]['Diagnosis'] for p_id,vals in full_data_dict.items()} # Assuming patients diagnosis does not change (we checked that in the data)
    stats_dict_median = compute_interaction_matrix_stats(interaction_matrices_median_dict, patient_diagnosis_dict)
    stats_dict_avg = compute_interaction_matrix_stats(interaction_matrices_avg_dict, patient_diagnosis_dict)
    
    # Initialize dictionary to collect stats by diagnosis
    diagnosis_stats_dict_median = {'Healthy': [], 'Unhealthy': []}
    diagnosis_stats_dict_avg = {'Healthy': [], 'Unhealthy': []}
    
    # Gather all stats by diagnosis
    for patient_id, stats in stats_dict_median.items():
        diagnosis = patient_diagnosis_dict[patient_id]
        diagnosis_stats_dict_median[diagnosis].append(stats)
    for patient_id, stats in stats_dict_avg.items():
        diagnosis = patient_diagnosis_dict[patient_id]
        diagnosis_stats_dict_avg[diagnosis].append(stats)
    
    # Calculate average stats for each diagnosis
    average_stats_median = {}
    for diagnosis, stats_list in diagnosis_stats_dict_median.items():
        if stats_list:  # Check to avoid division by zero if no patients with that diagnosis
            average_stats_median[diagnosis] = {
                'number': len(stats_list),
                'avg_percent_positive': np.mean([stats['percent_positive'] for stats in stats_list]),
                'avg_sum_positive': np.mean([stats['sum_positive'] for stats in stats_list]),
                'avg_sum_negative': np.mean([stats['sum_negative'] for stats in stats_list]),
                'avg_rho': np.nanmean([stats['rho'] for stats in stats_list])
            }
    average_stats_avg = {}
    for diagnosis, stats_list in diagnosis_stats_dict_avg.items():
        if stats_list:  # Check to avoid division by zero if no patients with that diagnosis
            average_stats_avg[diagnosis] = {
                'number': len(stats_list),
                'avg_percent_positive': np.mean([stats['percent_positive'] for stats in stats_list]),
                'avg_sum_positive': np.mean([stats['sum_positive'] for stats in stats_list]),
                'avg_sum_negative': np.mean([stats['sum_negative'] for stats in stats_list]),
                'avg_rho': np.nanmean([stats['rho'] for stats in stats_list])
            }
    print(average_stats_median)
    print(average_stats_avg)
    return stats_dict_median, stats_dict_avg
def mann_whitney_tests(data, measures):
    results = {}
    groups = data['Diagnosis'].unique()
    
    for measure in measures:
        # Gather data by groups for the measure
        group_data = [data[data['Diagnosis'] == group][measure] for group in groups]
        k_stat, p_value = scipy.stats.mannwhitneyu(*group_data)
        results[measure] = p_value
    
    return results
    
def kruskal_wallis_tests(data, measures):
    results = {}
    groups = data['Diagnosis'].unique()
    
    for measure in measures:
        # Gather data by groups for the measure
        group_data = [data[data['Diagnosis'] == group][measure] for group in groups]
        k_stat, p_value = scipy.stats.kruskal(*group_data)
        results[measure] = p_value
    
    return results

def kolmogorov_smirnov_tests(data, measures):
    results = {}
    groups = data['Diagnosis'].unique()
    
    for measure in measures:
        # Gather data by groups for the measure
        group_data = [data[data['Diagnosis'] == group][measure] for group in groups]
        k_stat, p_value = scipy.stats.ks_2samp(*group_data)
        results[measure] = p_value
    
    return results



def plot_stats(image_dir, filename, stats_dict):
    significant_str = ''
    fig = plt.figure() #20 cm to inches width
    gs = fig.add_gridspec(2,2)
    axs = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    data = pd.DataFrame.from_dict(stats_dict, orient='index')

    diagnosis_counts = data['Diagnosis'].value_counts()
    Healthy_count = diagnosis_counts['Healthy']
    Unhealthy_count = diagnosis_counts['Unhealthy']
    # Create a list of statistics
    statistics = ['percent_positive', 'sum_positive', 'sum_negative', 'rho']
    fig.suptitle(f'Number (Healthy, Unhealthy): {(Healthy_count, Unhealthy_count)}')
    # Plot each statistic
    p_values_mw = mann_whitney_tests(data, statistics)
    p_values_kw = kruskal_wallis_tests(data, statistics)
    p_values_ks = kolmogorov_smirnov_tests(data, statistics)
    print(f'P values {filename} MW: {p_values_mw}')
    print(f'P values {filename} KW: {p_values_kw}')
    print(f'P values {filename} KS: {p_values_kw}')
    for stat, ax in zip(statistics, axs):
        if stat != 'rho':
            continue
        my_pal = {"Healthy": '#5F60F5', "Unhealthy": "#ED3A32"}
        sns.boxplot(x='Diagnosis', y=stat, data=data, showfliers=False, ax=ax, palette=my_pal)  # showfliers=False to hide outliers
        ax.set_xlabel('Diagnosis')
        ax.set_ylabel(f'{stat}')
        # Annotate significance
        # Find the x positions of the groups
        group_labels = data['Diagnosis'].unique()
        xticks = ax.get_xticks()  # get the positions for the xticks
        if p_values_kw[stat] < 0.05:
            significant_str += '_KW'
        if p_values_ks[stat] < 0.05:
            significant_str += '_KS'
        if p_values_mw[stat] < 0.05:
            significant_str += '_MW' 
        if len(significant_str) > 0:
            x1 = xticks[group_labels.tolist().index('Healthy')]
            x2 = xticks[group_labels.tolist().index('Unhealthy')]
            # Draw line and asterisk
            y, h, col = data[stat].max() + 1, 0.5, 'k'
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            ax.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, filename+significant_str+'_new.svg')

    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.98, hspace=0.3, wspace=0.25)
    plt.savefig(image_name, format='svg', transparent=False, dpi=600)

# Function to calculate Shannon Index from relative abundance data
def calculate_shannon_index(abundance_series):
    relative_abundance = abundance_series.values
    return entropy(relative_abundance)
def calculate_nonzero_number(abundance_series):
    relative_abundance = abundance_series.values
    return np.count_nonzero(relative_abundance)
# Function to add significance marker to the plot
def add_significance_marker(ax, text, x1, x2, y, h, color='k'):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color=color)
    ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=color)

def plot_diversity_distributions(image_dir, participant_dict, diagnosis_list, aux_data_dict=None, data_type = 'Species', metric='Shannon'):

    diagnosis_metric_dict = {diagnosis: [] for diagnosis in diagnosis_list}

    for participant_id, samples in participant_dict.items():
        if aux_data_dict is not None:
            aux_samples = aux_data_dict[participant_id]
        for sample in samples:
            if metric == 'Shannon':
                metric_value = calculate_shannon_index(sample['Data'])
                ylabel = f'Shannon Index of {data_type}'
                metric_name = metric
            elif metric == 'Number':
                metric_value = calculate_nonzero_number(sample['Data'])
                ylabel = f'Number of {data_type}'
                metric_name = metric
            elif metric == 'Number/Species':
                ylabel = f'Number of {data_type} per Species'
                sample_id = sample['Sample ID']
                metric_name = 'Number_per_species' 
                # I need to make sure we have same sample_id so we need to iterate the second list of samples
                for i, sample2 in enumerate(aux_samples):
                    if sample2['Sample ID'] == sample_id:
                        pathway_number = calculate_nonzero_number(sample['Data'])
                        taxa_number = calculate_nonzero_number(aux_samples[i]['Data'])
                        metric_value = pathway_number/taxa_number
                        aux_samples.pop(i) # This shortens the list for future iterations (notice that the break later avoid shortening the list while looping)
                        break
            elif metric == 'rho':
                ylabel = r'$\mathdefault{\rho}$'
                metric_value = sample['rho']
                metric_name = metric
            else:
                print(f'Metric: {metric} is not accepted. Accepted ones are Shannon, Number, Number/Species.')
                raise
            diagnosis_metric_dict[sample['Diagnosis']].append(metric_value)

    # Separate healthy and non-healthy data
    healthy_data = diagnosis_metric_dict['Healthy']
    non_healthy_data = []
    for diag, values in diagnosis_metric_dict.items():
        if diag != 'Healthy':
            non_healthy_data.extend(values)

    # Create subplots
    # Combine the data into a DataFrame
    data = {
        'Diagnosis': ['Healthy'] * len(healthy_data) + ['Unhealthy'] * len(non_healthy_data),
        'Metric Value': healthy_data + non_healthy_data
    }
    df = pd.DataFrame(data)

    # Custom color palette and category order
    diagnosis_color_map = {
        "Healthy": '#5F60F5',  # Blue
        "Unhealthy": "#ED3A32"  # Red
    }

    category_order = ['Healthy', 'Unhealthy']

    # Create the boxplot with seaborn
    plt.figure(figsize=(3*0.393701, 2.5*0.393701))
    #ax = sns.boxplot(x='Diagnosis', y='Metric Value', data=df, order=category_order, palette=diagnosis_color_map, showfliers=False)
    ax = sns.boxplot(x='Diagnosis', y='Metric Value', hue='Diagnosis', data=df, order=category_order, 
                     palette=diagnosis_color_map, showfliers=False, legend=False, width=0.3)
    # Change the y-axis label
    ax.set_ylabel(ylabel,labelpad=-1)
    ax.set_xlabel('',labelpad=-1)
    ax.tick_params(axis='x', pad=1)
    # Add p-value using Mann-Whitney U test
    _, p_value_healthy_vs_nonhealthy = scipy.stats.mannwhitneyu(healthy_data, non_healthy_data, alternative='two-sided')
    print(f'P-value: {p_value_healthy_vs_nonhealthy}')
    # Add p-value annotation to the plot
    '''y_max = max(max(healthy_data), max(non_healthy_data)) + 0.05
    ax.text(0.5, y_max, f'p={p_value_healthy_vs_nonhealthy:.2e}', fontsize=12, verticalalignment='center')
    '''
    ax.annotate(f'p={p_value_healthy_vs_nonhealthy:.2e}', 
            xy=(0.5, 1.1),  # x=0.5 places it at the center horizontally, y=1.05 places it slightly above the plot
            xycoords='axes fraction',  # Use axes coordinates
            fontsize=12, 
            ha='center',  # Center horizontally
            va='center')  # Center vertically
    despine(ax)
    '''
    # 1st Plot: Healthy vs Non-Healthy
    axes.boxplot([healthy_data, non_healthy_data], tick_labels=['Healthy', 'Non-Healthy'])
    axes.set_ylabel(ylabel)
    axes.set_xlabel('Diagnosis')
    # Perform Mann-Whitney U test between Healthy and Non-Healthy
    _, p_value_healthy_vs_nonhealthy = scipy.stats.mannwhitneyu(healthy_data, non_healthy_data, alternative='two-sided')

    # Add p-value to the first plot
    axes.text(1.5, max(max(healthy_data), max(non_healthy_data)) + 0.05,
                f'p={p_value_healthy_vs_nonhealthy:.2e}', fontsize=12, verticalalignment='center')
    '''
    filename = f'{metric_name}_{data_type}'
    image_name = os.path.join(image_dir, filename+'.svg')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    plt.subplots_adjust(top=0.89, bottom=0.25, left=0.15, right=0.98, hspace=0.3, wspace=0.25)
    plt.savefig(image_name, format='svg', transparent=False, dpi=600)
    # Adjust layout and show the plot
    #plt.tight_layout()
    plt.show()

'''
max_species = int(sys.argv[1])
error_threshold = int(sys.argv[2])
n_bagging = int(sys.argv[3])
gut_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f = 'fraction_nl'
c = '6'
m = '0'
e = '1.1'  
E_D_type = 0
min_ratio = 10
max_samples = 500
coarsed = False
coarsed_str = ''
if coarsed:
    coarsed_str = 'coarsed_'
tradeoff_data = f'{f}_{c}_{m}_{e}'
rootpath_data = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)
rootpath_results = os.path.join(gut_dir,'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data, 'processed')

diagnosis_list = ['Healthy', 'Unhealthy']
metadata_dict = get_metadata_dict(rootpath_data, overwrite=False, longitudinal=True, sample_size=500)
taxonomy_data_df = get_data_df(rootpath_data, wanted_realization = '680', overwrite=False, data_type=f'{coarsed_str}taxonomy', longitudinal=True, sample_size=500)
full_data_dict = get_full_data_dict(metadata_dict, taxonomy_data_df)

rootpath_img = os.path.join(gut_dir,'images', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data, 
                            'longitudinal_analyses', f'{coarsed_str}{max_species}_{error_threshold}_{min_ratio}_{n_bagging}')
interaction_matrices_median_dict, interaction_matrices_avg_dict = get_interaction_mat_dict_limits(rootpath_results, full_data_dict, max_species, 
                                                                                                  max_samples, min_ratio, error_threshold, 
                                                                                                  n_bagging, coarsed=coarsed, overwrite=False)
stats_dict_median, stats_dict_avg = get_diagnosis_stats_dict(interaction_matrices_median_dict, interaction_matrices_avg_dict, full_data_dict)
plot_stats(rootpath_img, filename='stats_grouped_median', stats_dict=stats_dict_median)
plot_stats(rootpath_img, filename='stats_grouped_avg', stats_dict=stats_dict_avg)
#pathway_data_df = get_data_df(rootpath_data, data_type='pathway')
#taxonomy_full_data_dict = get_full_data_dict(metadata_dict, taxonomy_data_df)
#pathway_full_data_dict = get_full_data_dict(metadata_dict, taxonomy_data_df)
#print(taxonomy_full_data_dict)
#print(metadata_dict)
#plot_diversity_distributions(rootpath_img, taxonomy_full_data_dict, diagnosis_list, aux_data_dict = None, data_type='Species', metric='Shannon')

'''
