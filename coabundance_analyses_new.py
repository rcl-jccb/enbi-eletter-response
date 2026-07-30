
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
from julia import FlashWeave as FW

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import pyreadr

# Activate pandas <-> R DataFrame conversion
pandas2ri.activate()

import igraph as ig
import os
import json
import pandas as pd
import numpy as np
import itertools
import sys
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import pickle
from itertools import combinations
import biom
import signal
from distutils.util import strtobool
import ModelData_analyses as MDA
from sklearn.model_selection import train_test_split

def change_base_directory(current_path, old_base, new_base):
    # Normalize the path to ensure there are no platform-specific path separators
    normalized_path = os.path.normpath(current_path)
    new_path = normalized_path.replace(old_base, new_base)
    return new_path

def save_calculation(out_file, full_results):
    """Function to save the last calculation to a file."""
    with open(out_file, 'w') as json_file:
        json.dump(full_results, json_file)
    print("Last calculation saved before termination.")

def signal_handler(signum, frame, out_file, full_results):
    """Handler for termination signals."""
    print(f"Signal {signum} received. Saving the last calculation...")
    save_calculation(out_file, full_results)
    # Exit gracefully
    exit(0)

# Wrapper function to create a signal handler with specific parameters
def create_signal_handler(out_file, full_results):
    return lambda signum, frame: signal_handler(signum, frame, out_file, full_results)


def get_metadata_dict_IBD(rootpath):
    # Load the CSV file
    metadata_file = os.path.join(rootpath, 'metadata.csv')
    data = pd.read_csv(metadata_file)

    # Filter the data to include only entries where 'data_type' is 'metagenomics'
    filtered_data = data[data['data_type'] == 'metagenomics']
    # Create a dictionary where each 'Participant ID' has a list of tuples, each containing 'External ID', 'week_num', and 'diagnosis'
    metadata_dict = {}
    for idx, row in filtered_data.iterrows():
        participant_id = row['Participant ID']
        entry_tuple = (row['External ID'], row['week_num'], row['diagnosis'])
        if participant_id in metadata_dict:
            metadata_dict[participant_id].append(entry_tuple)
        else:
            metadata_dict[participant_id] = [entry_tuple]
    return metadata_dict

def get_metadata_dict_CD(rootpath):
    metadata_file = os.path.join(rootpath, 'CD_metadata.txt')
    df = pd.read_csv(metadata_file, delim_whitespace=True)
    df['sampleID'] = df['sampleID'].astype(str)
    df['subjectID'] = np.where(df['measurementid'] <= 28.0, df['subjectID'].astype(str)+'-H', df['subjectID'].astype(str)+'-CD')
    df['Diagnosis'] = np.where(df['measurementid']<=28.0, 'Healthy', 'CD')
    metadata_dict = df.set_index('sampleID')[['subjectID', 'measurementid', 'Diagnosis']].to_dict('index')
    return metadata_dict
def get_metadata_dict_IBS(rootpath, where='stool'):
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
def get_metadata_dict_CRC(rootpath):
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


def get_metadata_dict_CD_Elife(rootpath):
    def filter_diagnosis(diagnosis):
        if diagnosis == 'true_control':
            return 'Healthy'
        elif diagnosis == 'CDI':
            return 'Unhealthy'
    # Load the CSV file
    metadata_file = os.path.join(rootpath, f'metadata.csv')
    data = pd.read_csv(metadata_file)
    # Create a dictionary where each 'Participant ID' has a list of tuples, each containing 'Sample ID', 'week', 'diagnosis'
    metadata_dict = {}
    for idx, row in data.iterrows():
        if row['group'] == 'mixed_control':
            continue
        participant_id = row['Samples']
        entry_tuple = (participant_id, 0, filter_diagnosis(row['group'])) # sample and participant id are the same here. Week is always 0 because not longitudinal data. We mantain this way because everything is done like this because we started with IBD, CD and IBS were we have longitudinal data.
        if participant_id in metadata_dict:
            metadata_dict[participant_id].append(entry_tuple)
        else:
            metadata_dict[participant_id] = [entry_tuple]
    return metadata_dict


def get_metadata_df_CD_ElifeStudy(rootpath):
    def filter_diagnosis(diagnosis):
        if diagnosis == 'true_control':
            return 'Healthy'
        elif diagnosis == 'mixed_control':
            return 'Diarrhea'
        elif diagnosis == 'CDI':
            return 'CDI'
    # Load the CSV file
    data_rootpath = os.path.dirname(rootpath)
    metadata_file = os.path.join(data_rootpath, f'metadata.csv')
    df = pd.read_csv(metadata_file)
    df.rename(columns={"Samples": "sample_id"}, inplace=True)
    df["Cohort"] = df["group"].astype(str).apply(filter_diagnosis) + ':' + df["Study"].astype(str)
    return df

def count_samples_by_rho_range(data_dict):
    # Initialize a dictionary to store counts for each range
    rho_ranges = {f"{i/10}-{(i+1)/10}": 0 for i in range(10)}

    # Iterate through each sample and check the 'rho' value
    for sample in data_dict.values():
        rho = sample.get('rho', None)
        if rho is not None:
            # Find the appropriate range for the current 'rho' value
            for i in range(10):
                lower_bound = i / 10
                upper_bound = (i + 1) / 10
                if lower_bound <= rho < upper_bound:
                    rho_ranges[f"{lower_bound}-{upper_bound}"] += 1
                    break
                # Include the upper bound (1.0) in the last range
                elif i == 9 and rho == 1.0:
                    rho_ranges[f"{lower_bound}-{upper_bound}"] += 1

    return rho_ranges
def get_metadata_dict_ModelData(rootpath, num_samples_per_diagnosis, overwrite=False):
    wanted_file = os.path.join(rootpath, 'processed', f'postprocessed_metadata_dict2_{num_samples_per_diagnosis}.pkl')
    if not overwrite and os.path.isfile(wanted_file): # here we try to load pear_pre_dict
        with open(wanted_file, 'rb') as fin:
            metadata_dict = pickle.load(fin)
    else:
        # Load the CSV file
        metadata_file = os.path.join(rootpath, 'processed', f'metadata_dict_{num_samples_per_diagnosis}.pkl')
        with open(metadata_file, 'rb') as fin:
            raw_metadata_dict = pickle.load(fin)
        # Initialize a new dictionary to group by 'realization'
        metadata_dict = {}
        #bin_centers = np.array([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.85]) #np.array([-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        #bin_edges = np.concatenate([[-np.inf], (bin_centers[:-1] + bin_centers[1:]) / 2, [1], [np.inf]])
        #bin_edges = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 1]
        # Iterate through the original dictionary
        for sample_id, info in raw_metadata_dict.items():
            realization = info['realization']  # Extract realization (sample_id in the original dict)

            # Create the entry for the realization if it doesn't exist
            if realization not in metadata_dict:
                metadata_dict[realization] = []
            rho = info['rho']
            
            
            if rho<=-0.25:
                diagnosis = 'Healthy_0'
            elif rho>-0.25 and rho<=0:
                diagnosis = 'Healthy_1'
            elif rho>0 and rho<=0.25:
                diagnosis = 'Unhealthy_0'
            elif rho>0.25 and rho<=0.5:
                diagnosis = 'Unhealthy_1'
            elif rho>0.5 and rho<0.75:
                diagnosis = 'Unhealthy_2'
            elif rho>0.75:
                diagnosis = 'Unhealthy_3'
            
            #bin_index = np.digitize(rho, bin_edges) - 1
            #diagnosis = float(bin_centers[bin_index])
            # Create the new dictionary structure for each sample
            sample_tuple = (sample_id, info['time'], diagnosis)#, info['rho'])
            

            # Append the transformed sample data to the list for that realization
            metadata_dict[realization].append(sample_tuple)
        with open(wanted_file, 'wb') as f:
            pickle.dump(metadata_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return metadata_dict

def get_metadata_dict_HealthyPark(rootpath):
    morinaga_df = pd.read_csv(os.path.join(rootpath, 'MORINAGA_data.csv'), index_col=0)
    nibiohn_df = pd.read_csv(os.path.join(rootpath, 'NIBIOHN_data.csv'), index_col=0)
    morinaga_samples = list(morinaga_df.index)
    nibiohn_samples = list(nibiohn_df.index)
    metadata_df = pd.DataFrame({"sample_id": morinaga_samples + nibiohn_samples,
                                "Cohort": ["morinaga"] * len(morinaga_samples) + ["nibiohn"] * len(nibiohn_samples)})
    return metadata_df

def get_metadata_dict_BarleyGoto(rootpath):
    df = pd.read_csv(os.path.join(rootpath, 'abundance_data.csv'), index_col=0)
    samples = list(df.columns)
    metadata_df = pd.DataFrame({"sample_id": samples, "Cohort": ["control" if "C" in s else "test" if "T" in s else "Unknown" for s in samples]})
    return metadata_df

def get_data_df_BarleyGoto(rootpath):
    df = pd.read_csv(os.path.join(rootpath, 'abundance_data.csv'), index_col=0)
    # Normalize each sample (column) so the sum is 1
    normalized_df = df.div(df.sum(axis=0), axis=1)
    
    return normalized_df

def get_data_df_HealthyPark(rootpath):
    morinaga_df = pd.read_csv(os.path.join(rootpath, 'MORINAGA_data.csv'), index_col=0)
    nibiohn_df = pd.read_csv(os.path.join(rootpath, 'NIBIOHN_data.csv'), index_col=0)
    # Combine both DataFrames (align on columns, fill missing with 0)
    combined_df = pd.concat([morinaga_df, nibiohn_df], axis=0).fillna(0)

    # Transpose: bacteria as rows, samples as columns
    transposed_df = combined_df.T

    # Normalize each sample (column) so the sum is 1
    normalized_df = transposed_df.div(transposed_df.sum(axis=0), axis=1)
    
    return normalized_df


def get_data_df_ModelData(rootpath, num_samples_per_diagnosis, coarsed=False, aux_realization=None, overwrite=False):
    if coarsed:
        coarsed_str = '_coarsed'
    else:
        coarsed_str = ''
    wanted_file = os.path.join(rootpath, 'processed', f'postprocessed{coarsed_str}_taxonomy_df2_{num_samples_per_diagnosis}.pkl')
    if not overwrite and os.path.isfile(wanted_file): # here we try to load pear_pre_dict
        with open(wanted_file, 'rb') as fin:
            cleaned_df = pickle.load(fin)
    else:    
        data_file = os.path.join(rootpath, 'processed', f'sampling_df_{num_samples_per_diagnosis}.pkl')
        with open(data_file, 'rb') as fin:
            df = pickle.load(fin)
        if coarsed:
            path = os.path.join(rootpath, aux_realization)
            df = MDA.get_coarsened_species(path, df)
            
        df = df.div(df.sum())

        cleaned_df = df.dropna(axis=1) # This eliminates those cases in whcih filtered_df.sum() is basically 0 and the we would have nan's
        with open(wanted_file, 'wb') as f:
            pickle.dump(cleaned_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return cleaned_df


def get_data_df_CD_Elife(rootpath, study_country=False):
    if study_country:
        rootpath = os.path.dirname(rootpath)
    data_file = os.path.join(rootpath,'species_data.csv')
    df = pd.read_csv(data_file, index_col=0)
    filtered_df = df[
            ~df.index.str.contains('-1') &
            ~df.index.str.contains('unknown') &
            ~df.index.str.contains('meta_mOTU')
        ]
    normalized_df = filtered_df.div(df.sum())
    cleaned_df = normalized_df.dropna(axis=1) # This eliminates those cases in whcih filtered_df.sum() is basically 0 and the we would have nan's
    return cleaned_df

def get_data_df_CRC(rootpath, CRC_type):
    data_file = os.path.join(rootpath,f'taxonomy_data{CRC_type}.csv')
    df = pd.read_csv(data_file, index_col=0)
    # Data is already filtered and also normalized actually
    
    normalized_df = df.div(df.sum())
    cleaned_df = normalized_df.dropna(axis=1) # This eliminates those cases in whcih filtered_df.sum() is basically 0 and the we would have nan's
    return cleaned_df

def get_data_df_IBS(rootpath, taxonomy_level='species'):
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
    data_file = os.path.join(rootpath,'taxonomy_stool_data.csv')
    df = pd.read_csv(data_file, index_col=0)
    if taxonomy_level.lower() == 'species':
        df = df[
            df.index.str.contains('k__Bacteria') &
            df.index.str.contains('s__') &
            ~df.index.str.contains('_unclassified')
        ]
    if taxonomy_level.lower() == 'genus':
        df = df[df.index.str.startswith('k__Bacteria')].copy()
        df.index = df.index.astype(str)           

        df['genus'] = df.index.map(extract_genus)
        # Drop rows where genus was not found
        df = df[df['genus'].notna()]
        df = df.groupby('genus').sum()
    elif taxonomy_level.lower() == 'family':
        df = df[df.index.str.startswith('k__Bacteria')].copy()
        df.index = df.index.astype(str)

        df['family'] = df.index.map(extract_family)
        df = df[df['family'].notna()]
        df = df.groupby('family').sum()
    normalized_df = df.div(df.sum())
    cleaned_df = normalized_df.dropna(axis=1) # This eliminates those cases in whcih filtered_df.sum() is basically 0 and the we would have nan's
    return cleaned_df

def get_data_df_IBD(rootpath, taxonomy_level='species'):
    data_file = os.path.join(rootpath,'taxonomic_profiles.biom')
    with open(data_file, 'r') as file:
        biom_data = json.load(file)

    # Extract matrix shape
    matrix_shape = biom_data['shape']

    # Initialize a matrix of zeros
    data_matrix = np.zeros(matrix_shape)

    # Fill in the non-zero entries from the "data" key
    for entry in biom_data['data']:
        row, column, value = entry
        data_matrix[row, column] = value

    # Extract row and column IDs
    row_ids = [row['id'] for row in biom_data['rows']]
    column_ids = [col['id'] for col in biom_data['columns']]
    if taxonomy_level.lower() == 'species':
        # Create a DataFrame from the matrix
        df = pd.DataFrame(data_matrix, index=row_ids, columns=column_ids)
        filtered_df = df[
            df.index.str.contains('k__Bacteria') &
            df.index.str.contains('s__') &
            ~df.index.str.contains('t__') &
            ~df.index.str.contains('_unclassified')
        ]
    elif taxonomy_level.lower() == 'genus':
        # Create a DataFrame from the matrix
        df = pd.DataFrame(data_matrix, index=row_ids, columns=column_ids)
        filtered_df = df[
            df.index.str.contains('k__Bacteria') &
            df.index.str.contains('g__') &
            ~df.index.str.contains('s__') &
            ~df.index.str.contains('_unclassified')
        ]
    elif taxonomy_level.lower() == 'family':
        # Create a DataFrame from the matrix
        df = pd.DataFrame(data_matrix, index=row_ids, columns=column_ids)
        filtered_df = df[
            df.index.str.contains('k__Bacteria') &
            df.index.str.contains('f__') &
            ~df.index.str.contains('g__') &
            ~df.index.str.contains('_unclassified')
        ]
    normalized_df = filtered_df.div(filtered_df.sum())
    return normalized_df

def get_data_df_CD(rootpath):
    data_file = os.path.join(rootpath,'CD_data.txt')
    df = pd.read_csv(data_file, sep='\t', index_col=0)
    normalized_df = df.div(df.sum(axis=0), axis=1)
    return normalized_df

def get_metadata_dict(rootpath, disease, num_samples_per_diagnosis=None, overwrite=False):
    if disease in ['IBD', 'IBD-taxonomy']:
        metadata_dict = get_metadata_dict_IBD(rootpath)
    elif disease == 'CD':
        metadata_dict = get_metadata_dict_CD(rootpath)
    elif disease in ['IBS', 'IBS-taxonomy']:
        metadata_dict = get_metadata_dict_IBS(rootpath)
    elif disease == 'CRC':
        metadata_dict = get_metadata_dict_CRC(rootpath)
    elif disease == 'CD_Elife':
        metadata_dict = get_metadata_dict_CD_Elife(rootpath)
    elif disease == 'CD_Elife_Study':
        metadata_dict = get_metadata_df_CD_ElifeStudy(rootpath) # THIS IS A DF NOT A DICT, but it does not matter
    elif disease == 'ModelData':
        metadata_dict = get_metadata_dict_ModelData(rootpath, num_samples_per_diagnosis, overwrite)
    elif disease == 'Healthy_Park':
        metadata_dict = get_metadata_dict_HealthyPark(rootpath) # THIS IS A DF NOT A DICT, but it does not matter
    elif disease == 'Barley_Goto':
        metadata_dict = get_metadata_dict_BarleyGoto(rootpath) # THIS IS A DF NOT A DICT, but it does not matter
    

    return metadata_dict
def get_data_df(rootpath, disease, CRC_type=2, taxonomy_level='Species', num_samples_per_diagnosis=None, coarsed=False, 
                aux_realization=None, overwrite=False):
    if disease in ['IBD', 'IBD-taxonomy']:
        data_df = get_data_df_IBD(rootpath, taxonomy_level)
    elif disease == 'CD':
        data_df = get_data_df_CD(rootpath)
    elif disease in ['IBS', 'IBS-taxonomy']:
        data_df = get_data_df_IBS(rootpath, taxonomy_level)
    elif disease == 'CRC':
        data_df = get_data_df_CRC(rootpath, CRC_type)
    elif disease in ['CD_Elife_Study']:
        data_df = get_data_df_CD_Elife(rootpath, study_country=True)
    elif disease == 'CD_Elife':
        data_df = get_data_df_CD_Elife(rootpath, study_country=False)
    elif disease == 'ModelData':
        data_df = get_data_df_ModelData(rootpath, num_samples_per_diagnosis, coarsed, aux_realization, overwrite)
    elif disease == 'Healthy_Park':
        data_df = get_data_df_HealthyPark(rootpath)
    elif disease == 'Barley_Goto':
        data_df = get_data_df_BarleyGoto(rootpath)  
    return data_df

def filter_first_fraction_samples(dataframe, sample_fraction_to_keep):
    """
    Filters the first fraction of the samples (columns) and eliminates rows with all zero values after the selection.
    
    :param dataframe: Input pandas DataFrame with bacteria as index and samples as columns.
                    Each cell contains normalized counts.
    :param sample_fraction_to_keep: Fraction of the samples to keep (between 0 and 1), starting from the first columns.
    :return: A new pandas DataFrame with the reduced number of samples (columns) and cleaned rows (bacteria).
    """
    # Step 1: Determine how many samples to keep based on the fraction provided
    num_samples_to_keep = int(len(dataframe.columns) * sample_fraction_to_keep)
    
    # Step 2: Select the first 'num_samples_to_keep' columns (samples)
    filtered_df = dataframe.iloc[:, :num_samples_to_keep]
    
    # Step 3: Delete rows with all zeros
    cleaned_df = filtered_df[(filtered_df != 0).any(axis=1)]
    
    return cleaned_df
def filter_random_fraction_samples(dataframe, sample_fraction_to_keep, random_state=None, samples_to_keep=None):
    """
    Filters a random fraction of the samples (columns) and eliminates rows with all zero values after the selection.
    
    :param dataframe: Input pandas DataFrame with bacteria as index and samples as columns.
                    Each cell contains normalized counts.
    :param sample_fraction_to_keep: Fraction of the samples to keep (between 0 and 1), randomly selected.
    :param random_state: Seed for the random number generator (optional) for reproducibility.
    :return: A new pandas DataFrame with the reduced number of samples (columns) and cleaned rows (bacteria).
    """
    # Step 1: Determine how many samples to keep based on the fraction provided
    if samples_to_keep == None:
        num_samples_to_keep = int(len(dataframe.columns) * sample_fraction_to_keep)
    else:
        num_samples_to_keep = samples_to_keep
    # Step 2: Randomly select 'num_samples_to_keep' columns (samples)
    filtered_df = dataframe.sample(n=num_samples_to_keep, axis=1, random_state=random_state)
    
    # Step 3: Delete rows with all zeros
    cleaned_df = filtered_df[(filtered_df != 0).any(axis=1)]
    
    return cleaned_df
def filter_most_abundant_bacteria(dataframe, bacteria_fraction_to_keep):
    """
    Filters the most abundant bacteria based on a specified fraction and eliminates rows
    with all zero values after the selection.
    
    :param dataframe: Input pandas DataFrame with bacteria as index and samples as columns.
                    Each cell contains normalized counts.
    :param bacteria_fraction_to_keep: Fraction of the most abundant bacteria to keep (between 0 and 1).
    :return: A new pandas DataFrame with the reduced number of bacteria (rows) and cleaned samples.
    """
    # Step 1: Sum the bacterial counts across all samples (columns) for each bacterium (row)
    bacteria_sums = dataframe.sum(axis=1)
    
    # Step 2: Determine how many bacteria to keep based on the fraction provided
    num_bacteria_to_keep = int(len(dataframe.index) * bacteria_fraction_to_keep)
    
    # Step 3: Sort bacteria by their total counts in descending order
    sorted_bacteria = bacteria_sums.sort_values(ascending=False)
    
    # Step 4: Select the indices (bacteria names) of the most abundant bacteria
    bacteria_to_keep = sorted_bacteria.head(num_bacteria_to_keep).index
    
    # Step 5: Filter the original dataframe to keep only the most abundant bacteria
    filtered_df = dataframe.loc[bacteria_to_keep]
    
    # Step 6: Normalize the filtered dataframe so that each column sums to 1
    normalized_df = filtered_df.div(filtered_df.sum(axis=0))
    
    # Step 7: Drop columns (samples) where the sum is zero after normalization
    cleaned_df = normalized_df.dropna(axis=1)

    return cleaned_df


def split_samples_by_diagnosis(metadata_dict, data_df, disease, bacteria_fraction_to_keep=1, sample_fraction_to_keep=1, 
                               random_sampling=False, random_state=None, samples_to_keep=None):
    # Lists to store the external IDs of 'Healthy' and 'Other' samples
    healthy_diagnosis_dict = {'IBD': ['nonIBD'], 'IBD-taxonomy': ['nonIBD'], 'CD': ['Healthy'], 'IBS': ['Healthy'], 'IBS-taxonomy': ['Healthy'], 'CRC': ['Healthy'],
                              'CD_Elife': ['Healthy'], 'ModelData': ['Healthy_0', 'Healthy_1']}#'ModelData': ['Healthy']}
    unhealthy_diagnosis_dict = {'IBD': ['CD', 'UC'], 'IBD-taxonomy': ['CD', 'UC'], 'IBS': ['IBS-C', 'IBS-D'], 'IBS-taxonomy': ['IBS-C', 'IBS-D'],
                                'CRC': ['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV'],#, 'HS'],
                                'ModelData': ['Unhealthy_0', 'Unhealthy_1', 'Unhealthy_2', 'Unhealthy_3']} #[0.2, 0.4, 0.6, 0.85]}#}
    healthy_samples = []
    unhealthy_samples = []
    specific_unhealthy = False
    specific_healthy = False
    if disease in unhealthy_diagnosis_dict.keys(): # Not all diseases have specific subtypes for unhealthy
        specific_unhealthy_samples_dict = {diagnosis: [] for diagnosis in unhealthy_diagnosis_dict[disease]}
        specific_unhealthy = True
    if len(healthy_diagnosis_dict[disease])>1: 
        specific_healthy_samples_dict = {diagnosis: [] for diagnosis in healthy_diagnosis_dict[disease]}
        specific_healthy = True
    
    # Iterate through the metadata_dict to categorize the external IDs
    for participant_id, sample_info in metadata_dict.items():
        for external_id, week_num, diagnosis in sample_info:
            #diagnosis = float(diagnosis)
            if diagnosis in healthy_diagnosis_dict[disease]:
                healthy_samples.append(external_id)
                if specific_healthy:
                    specific_healthy_samples_dict[diagnosis].append(external_id)
            else:
                unhealthy_samples.append(external_id)
                if specific_unhealthy and (diagnosis in unhealthy_diagnosis_dict[disease]):
                    specific_unhealthy_samples_dict[diagnosis].append(external_id)

    # Create a dictionary to map original bacterial names to simple numbers
    bacteria_name_to_number = {name: i for i, name in enumerate(data_df.index)}
    # Filter the data_df to get two DataFrames: one for healthy samples and one for the rest. In this wayh if healthy samples have columns not present
    # in the df they get ignored. (df[healthy_samples] would raise an error)
    healthy_df = data_df.filter(items=healthy_samples)
    unhealthy_df = data_df.filter(items=unhealthy_samples)
    all_dfs_dict = {}
    all_dfs_dict = {'Healthy': healthy_df, 'Unhealthy': unhealthy_df}
    if specific_unhealthy:
        specific_unhealthy_df_dict = {diagnosis: data_df.filter(items=sample_list) for diagnosis, sample_list in specific_unhealthy_samples_dict.items()}
        all_dfs_dict.update(specific_unhealthy_df_dict)
    if specific_healthy:
        specific_healthy_df_dict = {diagnosis: data_df.filter(items=sample_list) for diagnosis, sample_list in specific_healthy_samples_dict.items()}
        all_dfs_dict.update(specific_healthy_df_dict)
    for k,df in all_dfs_dict.items():
        new_df = df.copy()
        if sample_fraction_to_keep < 1 or samples_to_keep is not None:
            if random_sampling:
                new_df = filter_random_fraction_samples(new_df, sample_fraction_to_keep, random_state=random_state, samples_to_keep=samples_to_keep)
            else:
                new_df = filter_first_fraction_samples(new_df, sample_fraction_to_keep)
        if bacteria_fraction_to_keep < 1:
            new_df = filter_most_abundant_bacteria(new_df, bacteria_fraction_to_keep)
        new_df.index = [bacteria_name_to_number[name] for name in new_df.index]
        new_df = new_df.dropna(axis='columns')
        new_df = new_df.loc[(new_df!=0).any(axis=1)]
        all_dfs_dict[k] = new_df
    
    return all_dfs_dict, bacteria_name_to_number

def split_Cohort(metadata_df, species_abundance_df, split_by):
    # --- Ensure Run IDs match ---
    metadata_df["sample_id"] = metadata_df["sample_id"].astype(str)
    species_abundance_df.columns = species_abundance_df.columns.astype(str)
    if split_by == 'Cohort':
        splitted_dfs = {
            cohort: species_abundance_df.filter(items=metadata_df[metadata_df["Cohort"] == cohort]["sample_id"], axis=1)
            for cohort in metadata_df["Cohort"].dropna().unique()
        }

    splitted_dfs = {group: df.loc[(df != 0).any(axis=1)] for group, df in splitted_dfs.items()}
    return splitted_dfs


def stats(matrix):
    np.fill_diagonal(matrix, 0)
    print(matrix)
    positive_values = matrix[matrix > 0]
    negative_values = matrix[matrix < 0]
    # Calculating required statistics
    percent_positive = 100 * len(positive_values) / (len(positive_values) + len(negative_values))
    sum_positive = positive_values.sum()
    sum_negative = negative_values.sum()
    pos_neg_diff = sum_positive + sum_negative
    non_zero_number = len(positive_values) + len(negative_values)
    asymmetry = np.linalg.norm(matrix - matrix.T, 'fro')/(2*np.linalg.norm(matrix, 'fro'))
    asymmetry2 = np.linalg.norm(matrix - matrix.T, 'fro')/np.linalg.norm(matrix + matrix.T, 'fro')
    asymmetry3 = np.linalg.norm(matrix - matrix.T, 'fro')
    return [percent_positive, sum_positive, sum_negative, pos_neg_diff/(sum_positive-sum_negative), non_zero_number, asymmetry, asymmetry2, asymmetry3]

def get_stats_from_network(data, beem_static):
    if not beem_static:
        #G = ig.Graph.Read_GML(gml_file)#x.read_gml(gml_file)
        signed_weights = []
        ''' THERE ARE NO SELF-LOOPS (we checked this)
        # Print the self-loops
        self_loops = [edge for edge in G.es if edge.source == edge.target]
        if self_loops:
            print("Self-loops found:")
            for edge in self_loops:
                print(f"Self-loop at vertex {edge.source} (weight: {edge['weight']})")
        else:
            print("No self-loops found in the graph.")
        #here data is a network
        
        for edge in data.es:
            weight = edge['weight']
            signed_weights.append(weight)
        signed_weights = np.array(signed_weights)
        positive_values = set(adj_matrix[adj_matrix > 0])
        print(set(signed_weights[signed_weights>0])==positive_values) # we checked they are the same
        '''
        
        adj_matrix = np.array(data.get_adjacency(attribute='weight').data)
        stats_values = stats(np.array(adj_matrix))
        
    else:
        # here data is a pandas dataframe
        stats_values = stats(data.to_numpy())

    return stats_values

def preprocess_data_for_beemstatic(df, prevalence_threshold=0.3, abundance_threshold=0.001):
    """
    Filters the dataset for BEEM-Static by removing species that are:
    1. Present in fewer than prevalence_threshold * total_samples.
    2. Have a maximum relative abundance lower than abundance_threshold.

    Parameters:
    - df: DataFrame with bacterial species as rows and samples as columns.
    - prevalence_threshold: Fraction of samples in which a species must be present (default 30%).
    - abundance_threshold: Minimum relative abundance threshold (default 0.1%).

    Returns:
    - filtered_df: Preprocessed DataFrame ready for BEEM-Static.
    """

    # Compute presence across samples (species must be > 0 in at least threshold * total_samples)
    presence = (df > 0).sum(axis=1) / df.shape[1]

    # Compute mean relative abundance for each species
    mean_abundance = df.mean(axis=1)
    #max_abundance = df.max(axis=1)

    # Filter species based on both criteria
    filtered_species = df.index[(presence >= prevalence_threshold) & (mean_abundance > abundance_threshold)]
    filtered_df = df.loc[filtered_species]
    # Find duplicate row indices
    duplicate_indices = filtered_df.index[filtered_df.index.duplicated()].unique()
    
    # Print the duplicates if they exist
    if len(duplicate_indices) > 0:
        print("Duplicate row names found:")
        print(duplicate_indices)
    else:
        print("No duplicate row names found.")
    filtered_df = filtered_df[~filtered_df.index.duplicated(keep='first')]
    return filtered_df

def infer_network_beem_static(subset_df, rootpath, wanted_diagnosis, iter_str, ncpu, scaling_bs, iters_bs, alpha_bs, lambda_bs, 
                              prevalence_threshold_bs, abundance_threshold_bs):
    """
    Runs BEEM-Static in R from Python and extracts the interaction network.

    Parameters:
    - subset_df: DataFrame with bacterial species as rows and samples as columns.
    - rootpath: Path to store temporary files.
    - wanted_diagnosis: Diagnosis name for file naming.
    - iter_str: String identifier for the iteration.
    - ncpu: Number of CPU cores to use for parallel processing.

    Returns:
    - inferred_network: Pandas DataFrame containing inferred interaction edges.
    """

    # Define file paths
    data_file = os.path.join(rootpath, f'aux_subsampled_data_beem_static_{wanted_diagnosis}_{iter_str}.csv')
    results_file = os.path.join(rootpath, f'aux_subsampled_results_beem_static_{wanted_diagnosis}_{iter_str}.csv')
    subset_df = preprocess_data_for_beemstatic(subset_df, prevalence_threshold_bs, abundance_threshold_bs)
    print(subset_df)
    # Save subset_df as CSV for BEEM-Static
    subset_df.to_csv(data_file, index=True)
    
    # Load R and run BEEM-Static
    ro.r(f"""
    library(devtools)
    library(beemStatic)

    # Load data
    data_file <- "{data_file}"
    results_file <- "{results_file}"
    abundance_data <- read.csv(data_file, row.names=1)

    # Run BEEM-Static using func.EM
    res <- func.EM(abundance_data, ncpu={ncpu}, scaling={scaling_bs}, max.iter={iters_bs}, alpha={alpha_bs}, lambda.choice={lambda_bs})

    # Extract interaction matrix
    inferred_network <- beem2param(res)$b.est

    # Save results
    write.csv(inferred_network, results_file, row.names=TRUE)
    """)

    # Load the inferred network back into Python
    inferred_network = pd.read_csv(results_file, index_col=0)
    print(inferred_network)
    # Cleanup temporary files
    if os.path.isfile(data_file):
        os.remove(data_file)
    if os.path.isfile(results_file):
        pathlib.Path(results_file).unlink()

    return inferred_network

def infer_network(subset_df, rootpath, sensitive, heterogeneous, wanted_diagnosis, iter_str):
    data_file = os.path.join(rootpath, f'aux_subsampled_data{wanted_diagnosis}{iter_str}.csv')
    netw_file = os.path.join(rootpath, f'aux_subsampled_network{wanted_diagnosis}{iter_str}.gml')
    with open(data_file, 'w') as d_f:
        # ---- PATCH: numeric sample-ID parsing artefact in the FlashWeave junction ----
        # FlashWeave's CSV loader (load_dlm / hasrowids in FlashWeave/src/io.jl) only
        # strips the sample-ID row when the first row-label is a String. Here the CSV is
        # written taxa x samples and read back with transposed=True, so after FlashWeave's
        # internal transpose the sample IDs form the first column of the data matrix. If
        # those sample IDs are purely numeric (e.g. the Yachida CRC cohort), then
        # data[1,1] is a number, hasrowids() returns false, that row is NOT stripped, and
        # the sample IDs are kept as an extra variable named "OTU_id" -- i.e. the sample
        # IDs get inferred as a phantom bacterium, corrupting the network (extra node +
        # spurious edges), which shifts rho / ENBI.
        # Fix: prefix the sample IDs with a non-numeric token so hasrowids() recognises
        # them as row labels and drops them. Sample IDs are only observation labels, so
        # this does not change the inference over the real taxa; it just removes the
        # phantom node. (rename() returns a new frame, so the caller's subset_df is intact.)
        subset_df.rename(columns=lambda c: f's_{c}').to_csv(d_f, index_label='OTU_id')

    FW.save_network(netw_file, FW.learn_network(data_file, sensitive=sensitive, heterogeneous=heterogeneous, transposed=True))
    G = ig.Graph.Read_GML(netw_file)
    if os.path.isfile(data_file):
        os.remove(data_file)
    if os.path.isfile(netw_file):
        pathlib.Path(netw_file).unlink()
    return G

def get_bootstrap_measurements(subset_dir, data_df, out_file, subset_fraction,  sensitive=True, heterogeneous=False, 
                               max_iters=20, max_attempts=50, wanted_diagnosis='', iter_str='', beem_static=False,
                               scaling_bs=1000, iters_bs=30, alpha_bs=1, lambda_bs=1,
                               prevalence_threshold_bs=0.3, abundance_threshold_bs=0.001):
    '''
    data_df: has OTU's as index and the columns are different samples
    '''
    def hash_subset(subset_df):
        # The hash() method returns the hash value of an object if it has one. 
        # Hash values are just integers that are used to compare quickly.
        return hash(tuple(subset_df.columns))
    data_size = len(data_df.columns) # Number of all samples
    subset_size = int(data_size*subset_fraction)
    full_results = {}
    #all_subsets = list(itertools.combinations(data_df.columns, int(subset_fraction*data_size)))
    #sampled_subsets = np.random.choice(len(all_subsets), size=min(max_iters, len(all_subsets)), replace=False)
    full_results[subset_fraction] = []
    seen_hashes = set()
    
    for _ in range(max_iters):
        attempts = 0
        unique_subset = False
        while not unique_subset and attempts < max_attempts:
            subset_df = data_df.sample(n=subset_size, axis=1, replace=False)  # Sample without replacement
            subset_hash = hash_subset(subset_df)
            if subset_hash not in seen_hashes:
                unique_subset = True
                seen_hashes.add(subset_hash) # It's faster to compare the hashes instead of the full sets
                if beem_static:
                    try:
                        data = infer_network_beem_static(subset_df, subset_dir, wanted_diagnosis, iter_str, 4, scaling_bs, 
                                                     iters_bs, alpha_bs, lambda_bs, prevalence_threshold_bs, abundance_threshold_bs)
                    except:
                        attempts += 1
                        continue
                else:
                    print(subset_df)
                    data = infer_network(subset_df, subset_dir, sensitive, heterogeneous, wanted_diagnosis, iter_str)
                
                subset_stats = get_stats_from_network(data, beem_static)
                full_results[subset_fraction].append(subset_stats)
            attempts += 1
        
        if attempts == max_attempts:
            print("Warning: Maximum attempts reached, could not find a unique subset.")
            break
    with open(out_file, 'w') as json_file:
        json.dump(full_results, json_file)
    subset_stats_avg = {}
    subset_stats_std = {}
    for fraction, stats_vec in full_results.items():
        subset_stats_avg[fraction] = list(np.mean(np.array(stats_vec), axis=0))
        subset_stats_std[fraction] = list(np.std(np.array(stats_vec), axis=0))
    return subset_stats, subset_stats_avg, subset_stats_std

def get_stats_dict(bootstrap_dir, max_iters, overwrite=False):
    def filter_json_files(folder_path, target_iters, target_fraction):
        """
        Filters .json files based on specific iters and fraction values.
        
        
        :param folder_path: Path to the folder containing the .json files.
        :param target_iters: The target iters value to filter.
        :param target_fraction: The target fraction value to filter.
        :return: List of filenames (without .json extension) that match the target iters and fraction.
        """
        # Get all .json files in the folder
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        print(json_files)
        # Filter files based on the iters and fraction
        filtered_files_dict = {}
        for file in json_files: # we know the files have a name structure like: [diagnosis]_stats_bootstrap_[fraction]_[iters]
            # Remove the .json extension
            file_name = os.path.splitext(file)[0]
            # Split the filename using underscores
            parts = file_name.split('_')
            
            # Ensure the file has enough parts to match the structure
            if len(parts) >= 5:
                fraction = parts[-2]  # Assuming fraction is the second last part
                iters = parts[-1]  # Assuming iters is the last part
                
                # Check if this file matches the target iters and fraction
                if fraction == target_fraction: # and iters == target_iters:
                    diagnosis = '_'.join(parts[0:-4]) # This is needed because some diagnosis have _, for example Stage_I_II
                    filtered_files_dict[diagnosis] = file_name
            else:
                print(f'The filename {file} does not follow the expected structure of [diagnosis]_stats_bootstrap_[fraction]_[iters]')
                raise ValueError

        print(filtered_files_dict)
        return filtered_files_dict
    def is_decimal_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    wanted_file = os.path.join(bootstrap_dir, 'processed', 'stats_dict2.pkl')
    if not overwrite and os.path.isfile(wanted_file): # here we try to load B_types_dict
        with open(wanted_file, 'rb') as fin:
            stats_dict = pickle.load(fin)
    else:
        '''healthy_diagnosis_dict = {'IBD': 'nonIBD', 'CD': 'Healthy', 'IBS': 'Healthy', 'CRC': 'Healthy'}
        unhealthy_diagnosis_dict = {'IBD': ['CD', 'UC'], 'IBS': ['IBS-C', 'IBS-D'], 'CRC': ['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV', 'HS']}
        healthy_samples = []
        unhealthy_samples = []
        specific_unhealthy = False
        if disease in unhealthy_diagnosis_dict.keys():
            specific_unhealthy_samples_dict = {diag: [] for diag in unhealthy_diagnosis_dict[disease]}
            specific_unhealthy = True
        '''
        stats_dict = {}
        for subset_fraction in os.listdir(bootstrap_dir): # Here we should have the subset_fractions           
            full_path = os.path.join(bootstrap_dir, str(subset_fraction))
            if os.path.isdir(full_path) and is_decimal_number(subset_fraction):
                stats_dict[subset_fraction] = {}
                files_dict = filter_json_files(full_path, str(max_iters), str(subset_fraction))
                for diagnosis,filename in files_dict.items():
                    with open(os.path.join(full_path, f'{filename}.json'), 'r') as file:
                        data = json.load(file)[subset_fraction]
                    diagnosis_subset_dict = {diagnosis: data[:min(len(data),max_iters)]}
                    stats_dict[subset_fraction].update(diagnosis_subset_dict)
        processed_dir = os.path.join(bootstrap_dir, 'processed')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        with open(wanted_file, 'wb') as f:
            pickle.dump(stats_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return dict(sorted(stats_dict.items()))
def p_value_tests(bootstrap_dir, data_dict, measures, overwrite=False):
    def p_value_function(p_value_indicator, data1, data2):
        if p_value_indicator == 'MW':
            k_stat, p_val = scipy.stats.mannwhitneyu(data1, data2)
        elif p_value_indicator == 'KW':
            k_stat, p_val = scipy.stats.kruskal(data1, data2)
        elif p_value_indicator == 'KS':
            k_stat, p_val = scipy.stats.ks_2samp(data1, data2)
        return k_stat, p_val

    wanted_file = os.path.join(bootstrap_dir, 'processed', 'p_values_dict2.pkl')
    if not overwrite and os.path.isfile(wanted_file):  # Load if exists
        with open(wanted_file, 'rb') as fin:
            results = pickle.load(fin)
    else:
        results = {}
        for subset_fraction in data_dict.keys():
            data = data_dict[subset_fraction]
            results[subset_fraction] = {}

            # Get all pairwise combinations of subsets
            subset_names = list(data.keys())
            subset_pairs = list(combinations(subset_names, 2))

            for p_value_indicator in ['MW', 'KW', 'KS']:
                results[subset_fraction][p_value_indicator] = {}

                for subset1, subset2 in subset_pairs:
                    # Combine subset names into a key like 'name1-name2'
                    pair_key = f'{subset1}-{subset2}'

                    # Get the data for each subset
                    data1 = np.array(data[subset1])
                    data2 = np.array(data[subset2])

                    # Initialize a list for storing p-values for all measures
                    p_values_for_measures = []

                    for measure_index, measure in enumerate(measures):
                        # Gather data by groups for the measure
                        measure_data1 = data1[:, measure_index]
                        measure_data2 = data2[:, measure_index]

                        # Perform the p-value test
                        k_stat, p_value = p_value_function(p_value_indicator, measure_data1, measure_data2)

                        # Store the p-value for this measure
                        p_values_for_measures.append(p_value)

                    # Store the p-values for this pair of subsets
                    results[subset_fraction][p_value_indicator][pair_key] = p_values_for_measures

        # Save the results
        processed_dir = os.path.join(bootstrap_dir, 'processed')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        with open(wanted_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return results
def p_value_tests_old(data_dict, measures, overwrite=False):
    def p_value_function(p_value_indicator, healthy_data, unhealthy_data):
        if p_value_indicator == 'MW':
            k_stat, p_val = scipy.stats.mannwhitneyu(healthy_data, unhealthy_data)
        elif p_value_indicator == 'KW':
            k_stat, p_val = scipy.stats.kruskal(healthy_data, unhealthy_data)
        elif p_value_indicator == 'KS':
            k_stat, p_val = scipy.stats.ks_2samp(healthy_data, unhealthy_data)
        return k_stat, p_val
    wanted_file = os.path.join(bootstrap_dir, 'processed', 'p_values_dict.pkl')
    if not overwrite and os.path.isfile(wanted_file): # here we try to load B_types_dict
        with open(wanted_file, 'rb') as fin:
            results = pickle.load(fin)
    else:
        results = {}
        for subset_fraction in data_dict.keys():
            data = data_dict[subset_fraction]
            results[subset_fraction] = {}
            healthy_data = np.array(data['Healthy'])
            unhealthy_data = np.array(data['Unhealthy'])
            for p_value_indicator in ['MW', 'KW', 'KS']:
                results[subset_fraction][p_value_indicator] = {}
                for measure_index, measure in enumerate(measures):
                    # Gather data by groups for the measure
                    healthy_measure = healthy_data[:, measure_index]
                    unhealthy_measure = unhealthy_data[:, measure_index]
                    k_stat, p_value = p_value_function(p_value_indicator, healthy_measure, unhealthy_measure)
                    results[subset_fraction][p_value_indicator][measure] = p_value
        processed_dir = os.path.join(bootstrap_dir, 'processed')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        with open(wanted_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return results

def create_boxplots_with_significance_multiple_pairs(bootstrap_dir, image_dir, max_iters, unhealthy_color_map=None, overwrite=False):
    measures = ['percent_positive', 'sum_positive', 'sum_negative', 'pos_neg_diff']
    stats_dict = get_stats_dict(bootstrap_dir, max_iters, overwrite)
    
    p_value_dict = p_value_tests(bootstrap_dir, stats_dict, measures, overwrite)

    # Set up figure and axes
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    ax = axs[0]

    # Define the custom order for the boxplot categories
    category_order = ['Healthy', 'Unhealthy'] 
    diagnosis_color_map = {"Healthy": '#5F60F5',  # Blue
                           "Unhealthy": "#ED3A32",  # Red
                           }
    print(unhealthy_color_map)
    if unhealthy_color_map is not None:
        category_order = category_order + list(unhealthy_color_map.keys())
        diagnosis_color_map.update(unhealthy_color_map)
    print(category_order)
    significance = False
    h = 1.5
    h2 = 0.05
    legend_loc = 'upper center'

     # Set for storing unique legend handles and labels
    unique_legend_handles = {}
    for measure_index, measure in enumerate(measures):
        if measure_index in [1, 2]:  # Skip the 2nd and 3rd measures
            continue
        elif measure_index == 3:
            ax = axs[1]
            h = 0.025
            h2 = 0.001

        sns.set_theme(style="whitegrid")

        # Prepare data for the current measure
        data = []

        for subset_fraction, groups in stats_dict.items():

            if subset_fraction not in ['0.8','0.9']:
                continue
            for group, group_data in groups.items():
                for sample in group_data:
                    value = sample[measure_index]  # Extract the specific measure for each sample
                    data.append([subset_fraction, group, value])

        # Create DataFrame for plotting
        df = pd.DataFrame(data, columns=['Subset Fraction', 'Group', measure])
        print(df)
        # Iterate over each subset_fraction to check if it has other subsets
        for subset_fraction, groups in stats_dict.items():
            if subset_fraction not in ['0.8','0.9']:
                continue
            # Get the available groups for the specific subset_fraction
            available_groups = list(groups.keys())
            print(available_groups)

            # Filter the predefined category order to match only the available groups
            filtered_category_order = [cat for cat in category_order if cat in available_groups]
            print(filtered_category_order)
            # Create DataFrame for the specific subset fraction
            df_fraction = df[df['Subset Fraction'] == subset_fraction].copy()  # Ensure .copy() to avoid SettingWithCopyWarning

            # Ensure that the 'Group' column follows the filtered order
            df_fraction['Group'] = pd.Categorical(df_fraction['Group'], categories=filtered_category_order, ordered=True)

            # Create boxplot for this subset fraction
            sns.boxplot(x='Subset Fraction', y=measure, hue='Group', data=df_fraction, palette=diagnosis_color_map, ax=ax, showfliers=False)
            # Remove the automatic legend for each plot
            if ax.get_legend():
                ax.legend_.remove()
            '''
            # Get xticks for this specific subset_fraction
            xticks = ax.get_xticks()

            # Add significance annotations using precomputed p-values for each subset_fraction
            for i, subset1 in enumerate(subset_names):
                for j in range(i + 1, len(subset_names)):  # Compare with the remaining subsets
                    subset2 = subset_names[j]

                    # Skip comparisons involving Unhealthy with any subset other than Healthy
                    if ('Unhealthy' in [subset1, subset2]) and ('Healthy' not in [subset1, subset2]):
                        continue  # Skip this comparison

                    # Get p-value from precomputed p_value_dict (Mann-Whitney U test)
                    p_value = p_value_dict[subset_fraction]['MW'].get(f'{subset1}-{subset2}', [None])[measure_index]
                    
                    if p_value is None:
                        continue  # If there's no p-value, skip the comparison

                    # Ensure that xticks are long enough for comparison
                    if i < len(xticks) and j < len(xticks):
                        # Prepare p-value significance
                        significant_str = ''
                        if p_value < 0.05:    
                            significant_str += '*'
                            significance = True
                        
                        # Avoid cluttering markers
                        if len(significant_str) > 0:
                            x1 = xticks[i]
                            x2 = xticks[j]
                            y = max(df_fraction[df_fraction['Group'] == subset1][measure]) + h * (j - i) / 4  # Adjust height
                            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
                            ax.text((x1 + x2) / 2, y + h / 2, f"{significant_str}", ha='center', va='bottom', color='k', fontsize=8)
            '''
        # Collect unique legend handles
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in unique_legend_handles:
                unique_legend_handles[label] = handle  # Store only unique labels and their corresponding handles
    
    # Add the combined legend above the first subplot (outside, horizontally extended) with unique entries
    fig.legend(unique_legend_handles.values(), unique_legend_handles.keys(), loc= 'center', ncols = 4, fontsize=10)

    # Adjust layout to make room for the legend
    plt.subplots_adjust(top=0.97, bottom=0.1, left=0.1, right=0.98, hspace=0.45, wspace=0.25)

    # Save the plot
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    print(image_dir)
    image_name = os.path.join(image_dir, f'{max_iters}_{significance}1.png')
    plt.savefig(image_name, format='png', transparent=False, dpi=600)

def merge_bootstrap_iterations(folder_path, diagnosis, subset_fraction): 
    """
    Merges all JSON files in a folder with specific patterns into one large dictionary.
    
    :param folder_path: The path to the folder containing the JSON files.
    :param output_file_prefix: The prefix of the output file, default is Healthy_stats_boostrap_0.8
    :return: None. Writes the merged dictionary to a new JSON file.
    """
    
    # Initialize an empty dictionary to hold the merged data
    merged_data = {}

    files_to_delete = []
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and 'it' in filename and filename.startswith(f'{diagnosis}_stats_boostrap'):
            file_path = os.path.join(folder_path, filename)
            
            # Load the JSON content from the file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Loop through each key in the dictionary
            for subset_fraction, vectors in data.items():
                # If the key doesn't exist in merged_data, initialize an empty list for it
                if subset_fraction not in merged_data:
                    merged_data[subset_fraction] = []

                # Extend the list with new vectors from the current file
                merged_data[subset_fraction].extend(vectors)
            
            # Add file to the list of files to delete
            files_to_delete.append(file_path)

    # Calculate the total number of vectors across all subsets
    total_vectors = sum(len(vectors) for vectors in merged_data.values())

    # Define the output file name
    output_file = os.path.join(folder_path, f"{diagnosis}_stats_boostrap_{subset_fraction}_{total_vectors}.json")
    
    # Save the merged data to a new JSON file
    with open(output_file, 'w') as out_file:
        json.dump(merged_data, out_file, indent=2)

    print(f"Merged data saved to {output_file}. Total vectors: {total_vectors}")
    
    # Delete the original files
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")
    
    # -----------------------------
    # Part 2: Group parameter files by value
    # -----------------------------
    params_files = [
        f for f in os.listdir(folder_path)
        if f.startswith(f"{diagnosis}_params_{subset_fraction}") and f.endswith('.json')
    ]

    params_groups = {}
    params_files_to_delete = []

    for filename in params_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            param_data = json.load(f)

        # Get iteration number from filename
        parts = filename.split('_')
        it_part = [p for p in parts if p.startswith("it")]
        if not it_part:
            continue
        iteration = it_part[0].replace("it", "")

        # Use stringified dict as key
        key = json.dumps(param_data, sort_keys=True)
        params_groups.setdefault(key, []).append(iteration)
        params_files_to_delete.append(file_path)

    # Create readable output: keys = list of iterations, value = params
    merged_params_dict = {
        "_".join(sorted(its)): json.loads(param_json)
        for param_json, its in params_groups.items()
    }

    output_params_file = os.path.join(folder_path, f"{diagnosis}_merged_params_{subset_fraction}.json")
    with open(output_params_file, 'w') as f:
        json.dump(merged_params_dict, f, indent=2)

    print(f"Merged params saved to {output_params_file}")
    
    # Delete original param files
    for file_path in params_files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")
    
'''
# BOXPLOTS
gut_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f = 'fraction_nl'
c = '6'
m = '0'
e = '1.1'  
E_D_type = 0
tradeoff_data = f'{f}_{c}_{m}_{e}'

subset_fraction = float(sys.argv[1])
max_iters = int(sys.argv[2])
disease = str(sys.argv[3])
wanted_diagnosis = str(sys.argv[4])
sensitive = bool(strtobool(sys.argv[5]))
heterogeneous = bool(strtobool(sys.argv[6]))
bacteria_fraction_to_keep = float(sys.argv[7])
sample_fraction_to_keep = float(sys.argv[8])
iter_str = str(sys.argv[9])
coarsed = bool(strtobool(sys.argv[10]))
try:
    num_samples_per_diagnosis = int(sys.argv[11])
except:
    num_samples_per_diagnosis = np.inf
try:
    sample_length = int(sys.argv[12])
except:
    sample_length = None


disease_to_dir = {'IBD': 'IBD_MDB', 'CD': 'CD_MDSINE_Genome_Biology', 'IBS': 'IBS_Mars_Cell', 'CRC': 'CRC_Yachida_NatMed',
                  'CD_Elife': 'CD_Ferretti_Elife' 
                  'ModelData': os.path.join('ModelData', 'invasion', f'E_D_{E_D_type}', tradeoff_data)}


unhealthy_color_map_CRC = {
                            "MP": "#ffa500",  # Orange
                            "Stage_0": "#32CD32",  # Lime green
                            "Stage_I_II": "#8A2BE2",  # Purple
                            "Stage_III_IV": "#FFD700",  # Gold
                            "HS": "#008080"  # Teal (new color for HS)
}# This must be ordered in the order we want the categories to appear
unhealthy_color_map_IBS = {"IBS-C": '#3BB273',  # Green
                           "IBS-D": '#FFA500'  # Orange
                          }
unhealthy_color_map_IBD = {"UC": '#3BB273',  # Green
                           "CD": '#FFA500'  # Orange
                          }
unhealthy_color_map_Model = {"Unhealthy_I": "#F48A8A",
                             "Unhealthy_II": "#C10000"
                            }
unhealthy_color_map_Model = {"Unhealthy_0": "#F48A8A",
                             "Unhealthy_1": "#E03D3D",
                             "Unhealthy_2": "#A70000",
                            }
unhealthy_color_map_dict = {'IBD': unhealthy_color_map_IBD, 'IBS': unhealthy_color_map_IBS, 'CRC': unhealthy_color_map_CRC,
                             'CD_Elife': None, 'ModelData': unhealthy_color_map_Model}

if sensitive:
    str2 = 'S'
else:
    str2 = 'F'
if heterogeneous:
    str1 = '_HE'
else:
    str1 = '_NHE'
CRC_type = None
CRC_str = ''
if disease == 'CRC':
    CRC_type = 3
    CRC_str = str(CRC_type)
rootpath_results = os.path.join(gut_dir, 'real_data', disease_to_dir[disease])
if disease == 'ModelData':
    rootpath_data = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)
    bootstrap_dir = os.path.join(rootpath_results, f'bootstrap{str1}-{str2}', 
                          f'BacFrac_{bacteria_fraction_to_keep}_FullsetFrac_{sample_fraction_to_keep}', f'coarsed_{coarsed}', 
                          f'samples_{num_samples_per_diagnosis}')
    image_dir = os.path.join(gut_dir, 'images', 'real_data', disease_to_dir[disease], f'bootstrap{str1}-{str2}', 
                         f'BacFrac_1.0_FullsetFrac_{sample_fraction_to_keep}', f'coarsed_{coarsed}', f'samples_{num_samples_per_diagnosis}')
else:
    rootpath_data = rootpath_results
    bootstrap_dir = os.path.join(rootpath_results, f'bootstrap{CRC_str}{str1}-{str2}', 
                          f'BacFrac_{bacteria_fraction_to_keep}_FullsetFrac_{sample_fraction_to_keep}')
    #bootstrap_dir = os.path.join(rootpath_results, f'bootstrap{str1}-{str2}', 
    #                             f'BacFrac_{bacteria_fraction_to_keep}_FullsetLength_{sample_length}')
    
    image_dir = os.path.join(gut_dir, 'images', 'real_data', disease_to_dir[disease], f'bootstrap{CRC_str}{str1}-{str2}')

create_boxplots_with_significance_multiple_pairs(bootstrap_dir, image_dir, max_iters, unhealthy_color_map=unhealthy_color_map_dict[disease],
                                                 overwrite=True)
'''
'''
rootpath_data = os.path.join(gut_dir,'real_data', disease_to_dir[disease])
for fraction in [0.01]:
    bootstrap_dir = os.path.join(rootpath_data, 'bootstrap', 'bacterial_fractions', f'{fraction}')
    image_dir = os.path.join(gut_dir, 'images', 'real_data', disease_to_dir[disease], 'bootstrap', f'{fraction}')
    create_boxplots_with_significance_multiple_pairs(bootstrap_dir, image_dir, max_iters, unhealthy_color_map=None)
'''


# OBTAIN COABUNDANCE NETWORKS
input_params = json.loads(sys.argv[1])
subset_fraction = float(input_params["subset_fraction"])
max_iters = int(input_params["max_iters_inside"])
disease = str(input_params["disease"])
try:
    wanted_diagnosis = float(input_params["diagnosis"])
except:
    wanted_diagnosis = str(input_params["diagnosis"])
sensitive = bool(strtobool(str(input_params["sensitive"])))
heterogeneous = bool(strtobool(str(input_params["heterogeneous"])))
bacteria_fraction_to_keep = float(input_params["bacteria_fraction_to_keep"])
sample_fraction_to_keep = float(input_params["sample_fraction_to_keep"])
iter_str = str(input_params["iteration"])
coarsed = bool(strtobool(str(input_params["coarsed"])))
try:
    num_samples_per_diagnosis = int(input_params["num_samples_per_diagnosis"])
except:
    num_samples_per_diagnosis = np.inf
try:
    sample_length = int(input_params["sample_length"])
except:
    sample_length = None
algorithm = str(input_params["algorithm"])
try:
    scaling_bs = float(input_params["scaling_bs"])
except:
    scaling_bs = None
try:
    iters_bs = int(input_params["iters_bs"])
except:
    iters_bs = None
alpha_bs = float(input_params["alpha_bs"])
lambda_bs = float(input_params["lambda_bs"])
prevalence_threshold_bs = float(input_params["prevalence_threshold_bs"])
abundance_threshold_bs = float(input_params["abundance_threshold_bs"])
split_by = str(input_params["split_by"])
taxonomy_level = str(input_params["taxonomy_level"])

beem_static = False
if algorithm.lower() in ['beem', 'beem static', 'bs']:
    algorithm_str = 'beem_static'
    beem_static = True
elif algorithm.lower() in ['fw', 'flashweave']:
    algorithm_str = 'flashweave'
params_names = ['subset_fraction', 'max_iters', 'disease', 'wanted_diagnosis', 'sensitive', 'heterogeneous', 'bacteria_fraction_to_keep',
                'sample_fraction_to_keep', 'iter_str', 'coarsed', 'num_samples_per_diagnosis', 'sample_length', 'algorithm', 'scaling_bs',
                'iters_bs', 'alpha_bs', 'lambda_bs', 'prevalence_threshold_bs', 'abundance_threshold_bs', 'split_by',
                'taxonomy_level']
params_dict = dict(zip(params_names, sys.argv[1:]))

print(params_dict)

f = 'fraction_nl'
c = '6'
m = '0'
e = '1.1'  
E_D_type = 0
tradeoff_data = f'{f}_{c}_{m}_{e}'
#samples_to_keep = 1899

disease_to_dir = {'IBD': 'IBD_MDB', 'CD': 'CD_MDSINE_Genome_Biology', 'IBS': 'IBS_Mars_Cell', 'CRC': 'CRC_Yachida_NatMed', 
                  'CD_Elife': 'CD_Ferretti_Elife','ModelData': os.path.join('ModelData', 'invasion', f'E_D_{E_D_type}', tradeoff_data),
                  'Healthy_Park': 'Healthy_Park_BMCMicrobiology', 'IBD-taxonomy': 'IBD_MDB', 'IBS-taxonomy': 'IBS_Mars_Cell',
                  'Barley_Goto': 'Barley_Goto_Nutrients', 'CD_Elife_Study': os.path.join('CD_Ferretti_Elife', 'Study') 
                  }
if sensitive:
    str2 = 'S'
else:
    str2 = 'F'
if heterogeneous:
    str1 = '_HE'
else:
    str1 = '_NHE'
CRC_type = None
CRC_str = ''
if disease == 'CRC':
    CRC_type = 2
    CRC_str = str(CRC_type)
if beem_static:
    str1 = f'_alpha_{alpha_bs}'
    str2 = f'lambda_{lambda_bs}'
    CRC_str= ''

gut_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


rootpath_results = os.path.join(gut_dir, 'real_data', disease_to_dir[disease])
if disease == 'ModelData':
    rootpath_data = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)
    subset_dir = os.path.join(rootpath_results, f'{algorithm_str}', f'bootstrap{str1}-{str2}', 
                          f'BacFrac_{bacteria_fraction_to_keep}_FullsetFrac_{sample_fraction_to_keep}', f'coarsed_{coarsed}', 
                          f'samples_{num_samples_per_diagnosis}',  f'{subset_fraction}')
else:
    rootpath_data = rootpath_results
    subset_dir = os.path.join(rootpath_results, f'{algorithm_str}', f'bootstrap{CRC_str}{str1}-{str2}', 
                          f'BacFrac_{bacteria_fraction_to_keep}_FullsetFrac_{sample_fraction_to_keep}', f'{subset_fraction}')
    if disease in ['IBD-taxonomy', 'IBS-taxonomy']:
        subset_dir = os.path.join(rootpath_results, f'taxonomy_level_{taxonomy_level}', f'{algorithm_str}', f'bootstrap{CRC_str}{str1}-{str2}', 
                          f'BacFrac_{bacteria_fraction_to_keep}_FullsetFrac_{sample_fraction_to_keep}', f'{subset_fraction}')
    #subset_dir = os.path.join(rootpath_results, f'bootstrap{CRC_str}{str1}-{str2}', 
    #                          f'BacFrac_{bacteria_fraction_to_keep}_FullsetLength_{sample_length}', f'{subset_fraction}')

print(rootpath_data)
metadata_dict = get_metadata_dict(rootpath_data, disease, num_samples_per_diagnosis=num_samples_per_diagnosis, overwrite=True, )
data_df = get_data_df(rootpath_data, disease, taxonomy_level=taxonomy_level, CRC_type=CRC_type, 
                      num_samples_per_diagnosis=num_samples_per_diagnosis, coarsed=coarsed, aux_realization='680',
                      overwrite=True)

if not os.path.exists(subset_dir):
    os.makedirs(subset_dir)


if disease in ['Healthy_Park', 'Barley_Goto', 'CD_Elife_Study']:
    all_dfs_dict = split_Cohort(metadata_df=metadata_dict, species_abundance_df=data_df,split_by=split_by)
else:
    all_dfs_dict, bacteria_to_OTUid_dict = split_samples_by_diagnosis(metadata_dict, data_df, disease, bacteria_fraction_to_keep, 
                                                                    sample_fraction_to_keep, random_sampling=True, random_state=42, 
                                                                    samples_to_keep=sample_length)

#for diagnosis,df in all_dfs_dict.items():
#    print(f'Diagnosis: {diagnosis} has the following df:')
#    print(df)
#raise
diagnoses = all_dfs_dict.keys() if wanted_diagnosis == 'all' else [wanted_diagnosis]

for diag in diagnoses:
    #merge_bootstrap_iterations(subset_dir,diag,subset_fraction)
    #raise
    outfile = os.path.join(subset_dir, f'{diag}_stats_boostrap_{subset_fraction}_{max_iters}_it{iter_str}.json')
    params_file = os.path.join(subset_dir, f'{diag}_params_{subset_fraction}_{max_iters}_it{iter_str}.json')
    overwrite = True

    if not os.path.isfile(outfile) or overwrite:
        df = all_dfs_dict[diag]
        stats_full, stats_avg, stats_std = get_bootstrap_measurements(
            subset_dir, df, outfile, subset_fraction,
            max_iters=max_iters, sensitive=sensitive,
            heterogeneous=heterogeneous, wanted_diagnosis=diag,
            iter_str=iter_str, beem_static=beem_static,
            iters_bs=iters_bs, scaling_bs=scaling_bs,
            alpha_bs=alpha_bs, lambda_bs=lambda_bs,
            prevalence_threshold_bs=prevalence_threshold_bs,
            abundance_threshold_bs=abundance_threshold_bs
        )
        print(f'Stats for diagnosis: {diag}, with alpha: {alpha_bs} and lambda: {lambda_bs}')
        print(stats_avg)
        print(stats_std)

    with open(params_file, 'w') as out_file:
        json.dump(params_dict, out_file, indent=2)