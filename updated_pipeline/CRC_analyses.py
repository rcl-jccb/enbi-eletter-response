import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import entropy


def change_base_directory(current_path, old_base, new_base):
    # Normalize the path to ensure there are no platform-specific path separators
    normalized_path = os.path.normpath(current_path)
    new_path = normalized_path.replace(old_base, new_base)

    return new_path
def get_metadata_dict(rootpath):
    # where is either stool or biopsy
    # Load the CSV file
    metadata_file = os.path.join(rootpath, f'metadata.csv')
    data = pd.read_csv(metadata_file)
    # Create a dictionary where each 'Participant ID' has a list of tuples, each containing 'Sample ID', 'week', 'diagnosis'
    metadata_dict = {}
    for idx, row in data.iterrows():
        try:
            participant_id = str(int(row['Subject_ID']))
        except ValueError: # if data is nan so we get a ValueError when trying to convert it to int
            continue
        entry_tuple = (participant_id, 0, row['Group']) # sample and participant id are the same here. Week is always 0 because not longitudinal data. We mantain this way because everything is done like this because we started with IBD, CD and IBS were we have longitudinal data.
        if participant_id in metadata_dict:
            metadata_dict[participant_id].append(entry_tuple)
        else:
            metadata_dict[participant_id] = [entry_tuple]
    return metadata_dict
def get_data_df(rootpath, data_type='taxonomy'):
    if data_type == 'taxonomy':
        data_file = os.path.join(rootpath,'taxonomy_data.csv')
        df = pd.read_csv(data_file, index_col=0)
        # Data is already filtered and also normalized actually
    elif data_type == 'taxonomy2':
        data_file = os.path.join(rootpath,'taxonomy_data2.csv')
        df = pd.read_csv(data_file, index_col=0)
        # Data is already filtered and also normalized actuallyeli
    elif data_type == 'enzyme':
        data_file = os.path.join(rootpath,'enzymes_data.csv')
        df = pd.read_csv(data_file, index_col=0)
    elif data_type == 'inferred_pathway':
        data_file = os.path.join(rootpath,'enzymes_data.csv')
        df = pd.read_csv(data_file, index_col=0)
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

# Function to calculate Shannon Index from relative abundance data
def calculate_shannon_index(abundance_series):
    relative_abundance = abundance_series.values
    return entropy(relative_abundance)
def calculate_nonzero_number(abundance_series):
    relative_abundance = abundance_series.values
    return np.count_nonzero(relative_abundance)
def calculate_simpson_index(relative_abundances):
    """
    Calculate the Simpson Diversity Index (1 - D) for a given sample's relative abundances.
    :param relative_abundances: Series with relative abundances for a single sample.
    :return: Simpson Diversity Index (1 - D).
    """
    D = (relative_abundances ** 2).sum()
    simpson_diversity_index = 1 - D
    return simpson_diversity_index
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
            elif metric == 'Simpson':
                metric_value = calculate_simpson_index(sample['Data'])
                ylabel = f'Simpson Index of {data_type}'
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
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # 1st Plot: Healthy vs Non-Healthy
    axes[0].boxplot([healthy_data, non_healthy_data], tick_labels=['Healthy', 'Non-Healthy'])
    axes[0].set_ylabel(ylabel)
    axes[0].set_xlabel('Diagnosis')
    # Perform Mann-Whitney U test between Healthy and Non-Healthy
    _, p_value_healthy_vs_nonhealthy = scipy.stats.mannwhitneyu(healthy_data, non_healthy_data, alternative='two-sided')

    # Add p-value to the first plot
    axes[0].text(1.5, max(max(healthy_data), max(non_healthy_data)) + 0.05,
                f'p={p_value_healthy_vs_nonhealthy:.2e}', fontsize=12, verticalalignment='center')

    # 2nd Plot: Boxplot for all diagnoses
    all_diagnosis_labels = list(diagnosis_metric_dict.keys())
    all_diagnosis_data = [diagnosis_metric_dict[diag] for diag in all_diagnosis_labels]

    axes[1].boxplot(all_diagnosis_data, tick_labels=all_diagnosis_labels)
    axes[1].set_ylabel(ylabel)
    axes[1].set_xlabel('Diagnosis')

    # Perform pairwise Mann-Whitney U test for significance and add markers
    y_max = max([max(values) for values in diagnosis_metric_dict.values()])  # Max value for placing markers
    y_offset = 0.1  # Offset for plotting significance markers
    for i in range(len(all_diagnosis_data)):
        for j in range(i+1, len(all_diagnosis_data)):
            # Perform Mann-Whitney U test
            _, p_value = scipy.stats.mannwhitneyu(all_diagnosis_data[i], all_diagnosis_data[j], alternative='two-sided')

            # Check if p-value is significant
            if p_value < 0.05:
                # Plot significance marker between the boxplots
                x1, x2 = i + 1, j + 1
                y = y_max + y_offset * (j - i)
                axes[1].plot([x1, x1, x2, x2], [y, y + y_offset, y + y_offset, y], lw=1.5, color='black')
                axes[1].text((x1 + x2) * 0.5, y + y_offset, f'p={p_value:.2e}', ha='center', va='bottom', color='black')

    
    filename = f'{metric_name}_{data_type}'
    image_name = os.path.join(image_dir, filename+'2.svg')

    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.98, hspace=0.3, wspace=0.25)
    plt.savefig(image_name, format='svg', transparent=False, dpi=600)
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
'''
gut_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rootpath_data = os.path.join(gut_dir,'real_data','CRC_Yachida_NatMed')
rootpath_results = os.path.join(gut_dir,'results', 'real_data','CRC_Yachida_NatMed')
rootpath_img = os.path.join(gut_dir,'images', 'real_data','CRC_Yachida_NatMed')
diagnosis_list = ['Healthy', 'MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV', 'HS']
metadata_dict = get_metadata_dict(rootpath_data)
#data_df_enzymes = get_data_df(rootpath_data, data_type = 'enzyme')
#data_df_taxonomy = get_data_df(rootpath_data, data_type = 'taxonomy2')
#data_df_inferred_pathways = get_data_df(rootpath_data, data_type = 'inferred_pathway')
#full_data_dict_inferred_pathways = get_full_data_dict(metadata_dict, data_df_inferred_pathways)
#full_data_dict_enzymes = get_full_data_dict(metadata_dict, data_df_enzymes)
data_df_taxonomy = get_data_df(rootpath_data, data_type = 'taxonomy2')
full_data_dict_taxonomy = get_full_data_dict(metadata_dict, data_df_taxonomy)
plot_diversity_distributions(rootpath_img, full_data_dict_taxonomy, diagnosis_list, aux_data_dict = None, 
                             data_type='Species', metric='Shannon')
'''
'''
interaction_matrices_median_dict, interaction_matrices_avg_dict = get_interaction_mat_dict_limits(rootpath_results, full_data_dict, 
                                                                                                  wanted_n_species, t_points_threshold, 
                                                                                                  error_threshold, n_bagging, overwrite=False)
stats_dict_median, stats_dict_avg = get_diagnosis_stats_dict(interaction_matrices_median_dict, interaction_matrices_avg_dict, full_data_dict)
plot_stats(rootpath_img, filename='stats_grouped_median', stats_dict=stats_dict_median)
plot_stats(rootpath_img, filename='stats_grouped_avg', stats_dict=stats_dict_avg)
'''
