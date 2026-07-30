import os
import numpy as np
import pandas as pd
import json
from numpy.linalg import pinv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import entropy


def change_base_directory(current_path, old_base, new_base):
    # Normalize the path to ensure there are no platform-specific path separators
    normalized_path = os.path.normpath(current_path)
    new_path = normalized_path.replace(old_base, new_base)

    return new_path
def get_metadata_dict(rootpath, where='stool'):
    def IBS_cohort_filter(cohort_str):
        if cohort_str == 'H':
            return 'Healthy'
        elif cohort_str == 'C':
            return 'IBS-C'
        elif cohort_str == 'D':
            return 'IBS-D'
        else:
            print(f'Unrecognized cohort type: {cohort_str}')
            raise ValueError
    # where is either stool or biopsy
    # Load the CSV file
    metadata_file = os.path.join(rootpath, f'{where}_metadata.csv')
    data = pd.read_csv(metadata_file)

    # Create a dictionary where each 'Participant ID' has a list of tuples, each containing 'Sample ID', 'Timepoint', and 'diagnosis'
    metadata_dict = {}
    for idx, row in data.iterrows():
        try:
            participant_id = str(int(row['study_id']))
        except ValueError: # some data is nan so we get a ValueError when trying to convert it to int
            continue
        entry_tuple = (row['SampleID'], row['Timepoint'], IBS_cohort_filter(row['Cohort']))
        if participant_id in metadata_dict:
            metadata_dict[participant_id].append(entry_tuple)
        else:
            metadata_dict[participant_id] = [entry_tuple]
    return metadata_dict
def get_data_df(rootpath, data_type='taxonomy', taxonomy_level = 'species'):
    # Extract family by splitting the taxonomy string and finding the entry starting with f__
    def extract_family(tax_string):
        for field in tax_string.split(';'):
            if field.startswith('f__') and field != 'f__':
                return field.replace('f__', '')
        return np.nan
    # Extract genus by splitting the taxonomy string and finding the entry starting with g__
    def extract_genus(tax_string):
        for field in tax_string.split(';'):
            if field.startswith('g__') and field != 'g__':
                return field.replace('g__', '')
        return np.nan
    if data_type == 'taxonomy':
        data_file = os.path.join(rootpath,'taxonomy_stool_data.csv')
        df = pd.read_csv(data_file, index_col=0)
        #with pd.option_context('display.max_rows', None):
            #print(f'Raw df is:\n{df[df.iloc[:, 0] != 0].iloc[:, 0]}')
        if taxonomy_level == 'species':
            df = df[
                df.index.str.contains('k__Bacteria') &
                df.index.str.contains('s__') &
                ~df.index.str.contains('_unclassified')
            ]
        if taxonomy_level == 'genus':
            df = df[df.index.str.startswith('k__Bacteria')].copy()
            df.index = df.index.astype(str)           

            df['genus'] = df.index.map(extract_genus)
            # Drop rows where genus was not found
            df = df[df['genus'].notna()]
            df = df.groupby('genus').sum()
        elif taxonomy_level == 'family':
            df = df[df.index.str.startswith('k__Bacteria')].copy()
            df.index = df.index.astype(str)

            df['family'] = df.index.map(extract_family)
            df = df[df['family'].notna()]
            df = df.groupby('family').sum()

    elif data_type == 'enzyme':
        data_file = os.path.join(rootpath,'enzymes_stool_data.csv')
        df = pd.read_csv(data_file, index_col=0)
        df = df[
            df.index.str.contains('K') 
        ]
    elif data_type == 'inferred_pathway':
        data_file = os.path.join(rootpath,'enzymes_stool_data.csv')
        df = pd.read_csv(data_file, index_col=0)
        df = df[
            df.index.str.contains('K') 
        ]
        df = filter_catabolic_enzymes(rootpath, df)
        print(df)
    else:
        print(f'The data type must be either "taxonomy" or "enzyme" you introduced: {data_type}')
    
    normalized_df = df.div(df.sum())
    cleaned_df = normalized_df.dropna(axis=1) # This eliminates those cases in whcih filtered_df.sum() is basically 0 and the we would have nan's
    return cleaned_df
def get_full_data_dict(metadata_dict, data_df):
    final_dict = {}

    # Loop through each participant in the grouped dictionary
    for participant_id, samples in metadata_dict.items():
        final_dict[participant_id] = []  # Prepare a sub-dictionary for this participant
        
        # Loop through each sample for the participant
        for external_id, week_num, diagnosis in samples:
            
            # Only process if the external_id exists in the DataFrame's columns
            if external_id in data_df.columns:
                # Construct the sub-dictionary for this sample
                final_dict[participant_id].append({
                    'Sample ID': external_id,
                    'Week': week_num,
                    'Diagnosis': diagnosis,
                    'Data': data_df[external_id]  # We have a df.series (so that we still have rows names)
                })
    return final_dict


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

def process_data_for_limits_algorithm(full_data_dict, wanted_n_species, t_points_threshold):
    processed_data_dict = {}
    for patient_id, full_sample_data in full_data_dict.items():
        t_points = len(full_sample_data)
        num_species = wanted_n_species
        matrix_ivec = np.zeros((num_species, t_points - 1, num_species + 1))  # Subtract 1 because we need t and t+1
        if t_points > t_points_threshold:
            
            # Create DataFrame from the sample data
            all_data = pd.DataFrame([s['Data'] for s in full_sample_data])
            
            # Ensure only species present in all samples are considered
            valid_species = all_data.columns[(all_data.notna() & (all_data != 0)).all()]

            # Calculate the median abundance for each species across all samples
            median_abundances = all_data[valid_species].median()
            
            # Select the top 'wanted_n_species' most abundant species
            top_species = median_abundances.nlargest(wanted_n_species).index
            num_species = len(top_species)
            if num_species >= wanted_n_species:
                processed_data_dict[patient_id] = {}
                processed_data_dict[patient_id]['Species'] = top_species  # Store the top species names
                # Filter data to include only the top species
                filtered_data = all_data[top_species]
                # Initialize matrix to store the transformed data
                # Each row corresponds to a temporal point, columns are as specified
                
                
                # Populate the matrix
                median_abundances = median_abundances[top_species]
                for t in range(t_points - 1):
                    x_t = filtered_data.iloc[t]
                    matrix_ivec[:, t, :num_species] = x_t - median_abundances
            
                for i,specie_name in enumerate(top_species):
                    x_tvec_i = filtered_data[specie_name][:-1]
                    x_tvec_shifted_i = filtered_data[specie_name].shift(-1)[:-1]# The shift adds a np.nan value to the end
                    matrix_ivec[i, :, -1] = np.log(x_tvec_shifted_i) - np.log(x_tvec_i)               

                # Store the matrix in the dictionary
                processed_data_dict[patient_id]['Data'] = matrix_ivec
    info_dict = {p_id: {'n_species': len(v['Species']), 'n_t_points': len(full_data_dict[p_id]), 'Diagnosis': full_data_dict[p_id][0]['Diagnosis']}
                 for p_id, v in processed_data_dict.items()}
    print(info_dict)
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
    error = mean_squared_error(y2, X2 @ B1, squared=False) / np.var(y2)
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
    n_bagging = 100  # number of iterations for bagging

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


def get_interaction_mat_dict_limits(rootpath, full_data_dict, wanted_n_species=10, t_points_threshold=10, error_threshold=1, 
                                    n_bagging=200, overwrite=False):
    name = f'interaction_mats_{wanted_n_species}_{t_points_threshold}_{error_threshold}_{n_bagging}.pkl'
    wanted_file = os.path.join(rootpath, 'processed', name)
    if not overwrite and os.path.isfile(wanted_file): # here we try to load pear_pre_dict
        with open(wanted_file, 'rb') as fin:
            (interaction_matrices_median_dict, interaction_matrices_avg_dict) = pickle.load(fin)
    else:
        processed_data_dict, info_dict = process_data_for_limits_algorithm(full_data_dict, wanted_n_species, t_points_threshold)
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
        processed_dir = os.path.join(rootpath, 'processed')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        with open(wanted_file, 'wb') as f:
            pickle.dump((interaction_matrices_median_dict, interaction_matrices_avg_dict), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(processed_dir, "info_dict.json"), "w") as outfile:
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
            pos_neg_diff = (sum_positive+sum_negative)/(sum_positive-sum_negative)  # Since sum_negative is negative, add it to sum_positive


            # Store the stats in a dictionary for each patient
            stats_dict[patient_id] = {
                'diagnosis': patient_diagnoses_dict[patient_id],
                'percent_positive': percent_positive,
                'sum_positive': sum_positive,
                'sum_negative': sum_negative,
                'pos_neg_diff': pos_neg_diff
            }
        
        return stats_dict
    patient_diagnosis_dict = {p_id: vals[0]['Diagnosis'] for p_id,vals in full_data_dict.items()} # Assuming patients diagnosis does not change (we checked that in the data)
    stats_dict_median = compute_interaction_matrix_stats(interaction_matrices_median_dict, patient_diagnosis_dict)
    stats_dict_avg = compute_interaction_matrix_stats(interaction_matrices_avg_dict, patient_diagnosis_dict)
    
    # Initialize dictionary to collect stats by diagnosis
    diagnosis_stats_dict_median = {'nonIBD': [], 'UC': [], 'CD': []}
    diagnosis_stats_dict_avg = {'nonIBD': [], 'UC': [], 'CD': []}
    
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
                'avg_pos_neg_diff': np.mean([stats['pos_neg_diff'] for stats in stats_list])
            }
    average_stats_avg = {}
    for diagnosis, stats_list in diagnosis_stats_dict_avg.items():
        if stats_list:  # Check to avoid division by zero if no patients with that diagnosis
            average_stats_avg[diagnosis] = {
                'number': len(stats_list),
                'avg_percent_positive': np.mean([stats['percent_positive'] for stats in stats_list]),
                'avg_sum_positive': np.mean([stats['sum_positive'] for stats in stats_list]),
                'avg_sum_negative': np.mean([stats['sum_negative'] for stats in stats_list]),
                'avg_pos_neg_diff': np.mean([stats['pos_neg_diff'] for stats in stats_list])
            }
    print(average_stats_median)
    print(average_stats_avg)
    return stats_dict_median, stats_dict_avg
def mann_whitney_tests(data, measures):
    
    results = {}
    nonIBD_data_raw = data[data['diagnosis'] == 'nonIBD']
    IBD_data_raw = data[data['diagnosis'].isin(['UC', 'CD'])]
    groups = data['diagnosis'].unique()
    for measure in measures:
        results[measure] = {}
        # Test each IBD-related group against nonIBD
        nonIBD_data = nonIBD_data_raw[measure]
        IBD_data = IBD_data_raw[measure]
        
        u_stat, p_value = scipy.stats.mannwhitneyu(nonIBD_data, IBD_data, alternative='two-sided')
        results[measure][('nonIBD', 'IBD')] = p_value
        '''
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1 = groups[i]
                group2 = groups[j]
                u_stat, p_value = scipy.stats.mannwhitneyu(data[data['diagnosis'] == group1][measure],
                                                     data[data['diagnosis'] == group2][measure],
                                                     alternative='two-sided')
                results[measure][(group1, group2)] = p_value
        '''
    return results
def kruskal_wallis_tests(data, measures):
    results = {}
    groups = data['grouped_diagnosis'].unique()
    
    for measure in measures:
        # Gather data by groups for the measure
        group_data = [data[data['grouped_diagnosis'] == group][measure] for group in groups]
        k_stat, p_value = scipy.stats.kruskal(*group_data)
        results[measure] = p_value
    
    return results

def kolmogorov_smirnov_tests(data, measures):
    results = {}
    groups = data['grouped_diagnosis'].unique()
    
    for measure in measures:
        # Gather data by groups for the measure
        group_data = [data[data['grouped_diagnosis'] == group][measure] for group in groups]
        k_stat, p_value = scipy.stats.ks_2samp(*group_data)
        results[measure] = p_value
    
    return results



def plot_stats(image_dir, filename, stats_dict):
    significant_str = ''
    fig = plt.figure() #20 cm to inches width
    gs = fig.add_gridspec(2,2)
    axs = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    data = pd.DataFrame.from_dict(stats_dict, orient='index')
    data['grouped_diagnosis'] = np.where(data['diagnosis'].isin(['UC', 'CD']), 'Unhealthy', 'Healthy')
    diagnosis_counts = data['diagnosis'].value_counts()
    nonIBD_count = diagnosis_counts['nonIBD']
    UC_count = diagnosis_counts['UC']
    CD_count = diagnosis_counts['CD']
    # Create a list of statistics
    statistics = ['percent_positive', 'sum_positive', 'sum_negative', 'pos_neg_diff']
    fig.suptitle(f'Number (Healthy, Unhealthy): {(nonIBD_count, UC_count+CD_count)}')
    # Plot each statistic
    p_values = mann_whitney_tests(data, statistics)
    p_values_kw = kruskal_wallis_tests(data, statistics)
    p_values_ks = kolmogorov_smirnov_tests(data, statistics)
    print(f'P values {filename} MW: {p_values}')
    print(f'P values {filename} KW: {p_values_kw}')
    print(f'P values {filename} KS: {p_values_kw}')

    for stat, ax in zip(statistics, axs):
        my_pal = {"Healthy": '#5F60F5', "Unhealthy": "#ED3A32"}
        order = ['Healthy', 'Unhealthy']
        sns.boxplot(x='grouped_diagnosis', y=stat, data=data, showfliers=False, ax=ax, palette=my_pal, order=order)  # showfliers=False to hide outliers
        ax.set_xlabel('Diagnosis')
        ax.set_ylabel(f'{stat}')
        # Annotate significance
        # Find the x positions of the groups
        group_labels = data['grouped_diagnosis'].unique()
        xticks = ax.get_xticks()  # get the positions for the xticks
        if p_values_kw[stat] < 0.05:
            significant_str += '_KW'
        if p_values_ks[stat] < 0.05:
            significant_str += '_KS'
        for pair, p_value_mw in p_values[stat].items():
            group1, group2 = pair
            if p_value_mw < 0.05:  # Check if significant at the 0.05 level
                # Get positions
                x1 = xticks[group_labels.tolist().index(group1)]
                x2 = xticks[group_labels.tolist().index(group2)]
                '''
                if group2 == 'IBD':
                    positions = ax.get_xticks()
                    labels = ax.get_xticklabels()
                    label_dict = {label.get_text(): idx for idx, label in enumerate(labels)}
                    UC_pos = positions[label_dict['UC']]
                    CD_pos = positions[label_dict['CD']]
                    IBD_pos = (UC_pos + CD_pos) / 2
                    x2 = IBD_pos
                '''
                # Draw line and asterisk
                y, h, col = data[stat].max() + 1, 0.5, 'k'
                plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
                significant_str += '_MW'
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
def plot_diversity_distributions(image_dir, participant_dict, aux_data_dict=None, data_type = 'Species', metric='Shannon'):
    healthy_metric = []
    unhealthy_metric = []

    for participant_id, samples in participant_dict.items():
        if aux_data_dict is not None:
            aux_samples = aux_data_dict[participant_id]
        for sample in samples:
            if metric == 'Shannon':
                metric_value = calculate_shannon_index(sample['Data'])
                ylabel = f'Shannon Index of {data_type}'
            elif metric == 'Number':
                metric_value = calculate_nonzero_number(sample['Data'])
                ylabel = f'Number of {data_type}'
            elif metric == 'Number/Species':
                ylabel = f'Number of {data_type} per Species'
                sample_id = sample['Sample ID'] 
                # I need to make sure we have same sample_id so we need to iterate the second list of samples
                for i, sample2 in enumerate(aux_samples):
                    if sample2['Sample ID'] == sample_id:
                        pathway_number = calculate_nonzero_number(sample['Data'])
                        taxa_number = calculate_nonzero_number(aux_samples[i]['Data'])
                        metric_value = pathway_number/taxa_number
                        aux_samples.pop(i) # This shortens the list for future iterations (notice that the break later avoid shortening the list while looping)
                        break
            else:
                print(f'Metric: {metric} is not accepted. Accepted ones are Shannon, Number, Number/Species.')
                raise
            if sample['Diagnosis'] == 'Healthy':
                healthy_metric.append(metric_value)
            else:
                unhealthy_metric.append(metric_value)
    #Significance tests
    kruskal_stat, kruskal_p = scipy.stats.kruskal(healthy_metric, unhealthy_metric)
    mannwhitney_stat, mannwhitney_p = scipy.stats.mannwhitneyu(healthy_metric, unhealthy_metric, alternative='two-sided')
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.boxplot([healthy_metric, unhealthy_metric], tick_labels=['Healthy', 'Unhealthy'])
    plt.ylabel(ylabel)
    plt.title(f'Distribution of {ylabel} for Healthy and Unhealthy Groups')
    # Set a significance marker based on the results
    '''
    significance_text = []
    print(f"Kruskal-Wallis Test: Statistic={kruskal_stat}, p-value={kruskal_p}")
    print(f"Mann-Whitney U Test: Statistic={mannwhitney_stat}, p-value={mannwhitney_p}")
    if kruskal_p < 0.05:
        significance_text.append('kw')
    if mannwhitney_p < 0.05:
        significance_text.append('mw')

    if significance_text:
        significance_text = '-'.join(significance_text)
        y_max = max(max(healthy_metric), max(unhealthy_metric))  # Get the maximum y value for plotting
        add_significance_marker(ax, significance_text, 1, 2, y_max + 0.1, 0.05)
    else:
        significance_text=''
    '''
    significance_text = ''
    if kruskal_p < 0.05:
        significance_text += '_kw'
    if mannwhitney_p < 0.05:
        significance_text += '_mw'
    ax.text(1.1, max(max(healthy_metric), max(unhealthy_metric)) + 0.05,
         f'Kruskal p-value={kruskal_p:.3e}\nMann-Whitney p-value={mannwhitney_p:.3e}',
         fontsize=12, verticalalignment='center')
    filename = f'{metric}_{data_type}'
    image_name = os.path.join(image_dir, filename+significance_text+'.svg')

    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.98, hspace=0.3, wspace=0.25)
    plt.savefig(image_name, format='svg', transparent=False, dpi=600)


gut_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rootpath_data = os.path.join(gut_dir,'real_data','IBS_Mars_Cell')
#data_df_taxonomy = get_data_df(rootpath_data, data_type = 'taxonomy', taxonomy_level = 'family')
#df = data_df_taxonomy.copy()
'''
wanted_n_species = int(sys.argv[1])   
t_points_threshold = int(sys.argv[2])
error_threshold = float(sys.argv[3])
n_bagging = int(sys.argv[4])  
'''
'''
gut_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rootpath_data = os.path.join(gut_dir,'real_data','IBS_Mars_Cell')
rootpath_results = os.path.join(gut_dir,'results', 'real_data','IBS_Mars_Cell')
#rootpath_img = os.path.join(gut_dir,'images', 'real_data','IBD_MDB', f'{wanted_n_species}_{t_points_threshold}_{error_threshold}_{n_bagging}')
#print(f'{wanted_n_species}_{t_points_threshold}_{error_threshold}_{n_bagging}')
rootpath_img = os.path.join(gut_dir,'images', 'real_data','IBS_Mars_Cell')
metadata_dict = get_metadata_dict(rootpath_data)
#data_df_enzymes = get_data_df(rootpath_data, data_type = 'enzyme')
data_df_taxonomy = get_data_df(rootpath_data, data_type = 'taxonomy')
full_data_dict_taxonomy = get_full_data_dict(metadata_dict, data_df_taxonomy)
#full_data_dict_enzymes = get_full_data_dict(metadata_dict, data_df_enzymes)
#data_df_inferred_pathways = get_data_df(rootpath_data, data_type = 'inferred_pathway')
#full_data_dict_inferred_pathways = get_full_data_dict(metadata_dict, data_df_inferred_pathways)
plot_diversity_distributions(rootpath_img, full_data_dict_taxonomy, aux_data_dict = None, data_type='Species', metric='Shannon')
'''
'''
interaction_matrices_median_dict, interaction_matrices_avg_dict = get_interaction_mat_dict_limits(rootpath_results, full_data_dict, 
                                                                                                  wanted_n_species, t_points_threshold, 
                                                                                                  error_threshold, n_bagging, overwrite=False)
stats_dict_median, stats_dict_avg = get_diagnosis_stats_dict(interaction_matrices_median_dict, interaction_matrices_avg_dict, full_data_dict)
plot_stats(rootpath_img, filename='stats_grouped_median', stats_dict=stats_dict_median)
plot_stats(rootpath_img, filename='stats_grouped_avg', stats_dict=stats_dict_avg)
'''
