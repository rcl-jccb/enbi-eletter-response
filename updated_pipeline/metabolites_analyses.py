import os
import numpy as np
import pandas as pd
import json
from numpy.linalg import pinv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import entropy, mannwhitneyu
import traceback


def change_base_directory(current_path, old_base, new_base):
    # Normalize the path to ensure there are no platform-specific path separators
    normalized_path = os.path.normpath(current_path)
    new_path = normalized_path.replace(old_base, new_base)

    return new_path
def get_metadata_dict(rootpath, disease_name):
    def get_temporal_point(row, disease_name):
        if disease_name in ['CRC-Yachida', 'CRC-Sinha', 'CRC-Kim', 'IBD-Franzosa', 'IBD-Jacobs', 'ESRD-Wang']:
            week = 0
        elif disease_name == 'IBDMDB':
            week = row['week_num']
        elif disease_name == 'IBS-Mars':
            week = row['Timepoint']
        return week
    # where is either stool or biopsy
    # Load the CSV file
    metadata_file = os.path.join(rootpath, f'metadata_{disease_name}.tsv')
    data = pd.read_csv(metadata_file, sep = '\t')
    # Create a dictionary where each 'Participant ID' has a list of tuples, each containing 'Sample ID', 'week', 'diagnosis'
    metadata_dict = {}
    for idx, row in data.iterrows():
        participant_id = str(row['Subject'])
        sample_id = str(row['Sample'])
        week= get_temporal_point(row,disease_name)
        diagnosis = str(row['Study.Group'])
        entry_tuple = (sample_id, week, diagnosis) # sample and participant id are the same here. Week is always 0 because not longitudinal data. We mantain this way because everything is done like this because we started with IBD, CD and IBS were we have longitudinal data.
        if participant_id in metadata_dict:
            metadata_dict[participant_id].append(entry_tuple)
        else:
            metadata_dict[participant_id] = [entry_tuple]
    return metadata_dict

def get_data_df(rootpath, disease_name, high_confidence_wanted=False):
    data_file = os.path.join(rootpath,f'metabolites_{disease_name}.tsv')
    df = pd.read_csv(data_file, index_col=0, sep = '\t').T
    df.columns = df.columns.astype(str)
    df = df.fillna(0)
    if high_confidence_wanted:
        map_file = os.path.join(rootpath,f'metabolites-map_{disease_name}.tsv')
        map_df = pd.read_csv(map_file, index_col=0, sep = '\t', dtype={"High.Confidence.Annotation": str})
        wanted_metabolites = map_df.index[map_df["High.Confidence.Annotation"] == 'TRUE']
        df = df.loc[wanted_metabolites]

    return df


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

def get_full_data_dict_IBS_Lijuan(gut_dir, positive, fecal):
    if positive:
        str2 = 'positive'
    else:
        str2 = 'negative'
    if fecal:
        str1 = 'fecal'
    else:
        str1 = 'serum'
    rootpath = os.path.join(gut_dir, 'real_data', 'IBS_Metabolome_Lijuan')
    data_file = os.path.join(rootpath, f'{str1}_{str2}_metabolite_profiling.tsv')
    df = pd.read_csv(data_file, index_col=0, sep = '\t')
    sample_columns = df.filter(regex='CON|IBS').columns.tolist()
    df = df[sample_columns]
    print(df)
    final_dict = {}
    for sample_id in sample_columns:
        # Determine diagnosis based on sample ID
        diagnosis = 'IBS' if 'IBS' in sample_id else 'Healthy'

        # Extract participant_id (assume it's the same as sample_id)
        participant_id = sample_id

        # Initialize participant in the dictionary if not already present
        if participant_id not in final_dict:
            final_dict[participant_id] = []

        # Add sample details to the participant's list
        final_dict[participant_id].append({
            'Sample ID': sample_id,
            'Week': 0,  # Set week to 0 as specified
            'Diagnosis': diagnosis,
            'Data': df[sample_id]  # Extract column as Series
        })
    return final_dict



def metabolites_boxplots(rootpath_img, disease, full_data_dict, unhealthy_diagnoses, high_confidence_wanted=False):
    """
    Generate boxplots for number of non-zero concentrations and total concentrations.
    Includes one boxplot for "Healthy," one combined "Unhealthy," and one for each specific diagnosis.

    Parameters:
    - full_data_dict: Dictionary with participants' data as described.
    - unhealthy_diagnoses: List of diagnoses considered "Unhealthy".
    """
    def filter_diagnosis(diagnosis):
        if diagnosis in ['H', 'nonIBD', 'Control', '0', 'Normal']:
            filtered_diagnosis = 'Healthy'
        else:
            filtered_diagnosis = diagnosis
        return filtered_diagnosis
    def calculate_whisker(data):
        """
        Calculate the top whisker position based on the 1.5*IQR rule.
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        whisker = q3 + 1.5 * iqr
        return whisker
    # Prepare data for analysis
    data_rows = []
    
    for participant_id, samples in full_data_dict.items():
        for sample in samples:
            sample_id = sample['Sample ID']
            diagnosis = filter_diagnosis(sample['Diagnosis'])
            data = sample['Data']
            
            # Compute metrics
            num_non_zero = (data > 0).sum()  # Count of non-zero substances
            total_concentration = data.sum()  # Total concentration
            
            # Append to the data list
            data_rows.append({
                'Participant ID': participant_id,
                'Sample ID': sample_id,
                'Diagnosis': diagnosis,
                'Num Non-Zero': num_non_zero,
                'Total Concentration': total_concentration
            })
    
    # Create DataFrame
    analysis_df = pd.DataFrame(data_rows)
    
    
    # Plot combined boxplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax0 = axes[0]
    ax1 = axes[1]
    
    # Determine the order of categories
    combined_order = ['Healthy', 'Unhealthy'] + [diag for diag in unhealthy_diagnoses if diag in analysis_df['Diagnosis'].unique()]
    analysis_df = pd.concat([analysis_df, analysis_df[analysis_df['Diagnosis'].isin(unhealthy_diagnoses)].assign(Diagnosis='Unhealthy')])
    # Boxplot for Number of Non-Zero Concentrations
    sns.boxplot(
        data=analysis_df,
        x='Diagnosis',
        y='Num Non-Zero',
        palette='Set2',
        order=combined_order,
        ax=ax0,
        showfliers=False
    )
    ax0.set_xticks(ax0.get_xticks())
    ax0.set_xlabel('Diagnosis')
    ax0.set_ylabel('Number of Substances')
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45)
    
    # Boxplot for Total Concentration
    sns.boxplot(
        data=analysis_df,
        x='Diagnosis',
        y='Total Concentration',
        palette='Set2',
        order=combined_order,
        ax=ax1,
        showfliers=False
    )
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_xlabel('Diagnosis')
    ax1.set_ylabel('Total concentration')
    ax1.set_xticklabels(ax0.get_xticklabels(), rotation=45)

    p_values = {}
    healthy_data = analysis_df[analysis_df['Diagnosis'] == 'Healthy']['Num Non-Zero']
    
    for group in combined_order[1:]:
        group_data = analysis_df[analysis_df['Diagnosis'] == group]['Num Non-Zero']
        if not group_data.empty:  # Ensure group has data
            _, p = mannwhitneyu(healthy_data, group_data, alternative='two-sided')
            p_values[group] = p
    print(p_values)
    significance_level = 0.05
    vertical_spacing = 0.05  # Space between p-values
    
    for idx, (group, p) in enumerate(p_values.items(), start=0):
        ax0.text(
            0.15 + 0.33*idx%3,  # Center horizontally
            1.03 + idx//3 * vertical_spacing,  # Stack vertically
            f'p-value {group}: {p:.2e}',
            ha='center',
            va='bottom',
            fontsize=10,
            transform=ax0.transAxes  # Use axis-relative coordinates for placement
        )
    if not os.path.exists(rootpath_img):
        os.makedirs(rootpath_img)
    image_name = os.path.join(rootpath_img, f'metabolites_{disease}_{"high_confidence" if high_confidence_wanted else ""}.png')
    
    
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.98, hspace=0.3, wspace=0.25)
    plt.savefig(image_name, format='png', transparent=False, dpi=600)
    plt.show()
    
'''
gut_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rootpath_data = os.path.join(gut_dir,'real_data','Metabolome_Borenstein')
disease_to_diagnosis_list = {'IBDMDB': ['CD', 'UC'], 'IBS-Mars': ['C', 'D'], 'CRC-Yachida': ['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV', 'HS'],
                             'CRC-Kim': ['Adenoma', 'Carcinoma'], 'IBD-Franzosa': ['CD', 'UC'], 'IBD-Jacobs': ['CD', 'UC'], 'CRC-Sinha': ['1'], 
                             'ESRD-Wang': ['ESRD']}
for disease in ['IBD-Franzosa', 'IBD-Jacobs', 'IBDMDB', 'IBS-Mars', 'CRC-Yachida', 'CRC-Kim', 'CRC-Sinha', 'ESRD-Wang']:
    rootpath_img = os.path.join(gut_dir,'images', 'real_data', 'Metabolome_Borenstein')
    diagnosis_list = disease_to_diagnosis_list[disease]
    metadata_dict = get_metadata_dict(rootpath_data, disease)
    data_df = get_data_df(rootpath_data, disease, high_confidence_wanted=True)
    full_data_dict = get_full_data_dict(metadata_dict, data_df)
    metabolites_boxplots(rootpath_img, disease, full_data_dict, diagnosis_list, high_confidence_wanted=True)
'''
'''
interaction_matrices_median_dict, interaction_matrices_avg_dict = get_interaction_mat_dict_limits(rootpath_results, full_data_dict, 
                                                                                                  wanted_n_species, t_points_threshold, 
                                                                                                  error_threshold, n_bagging, overwrite=False)
stats_dict_median, stats_dict_avg = get_diagnosis_stats_dict(interaction_matrices_median_dict, interaction_matrices_avg_dict, full_data_dict)
plot_stats(rootpath_img, filename='stats_grouped_median', stats_dict=stats_dict_median)
plot_stats(rootpath_img, filename='stats_grouped_avg', stats_dict=stats_dict_avg)
'''
