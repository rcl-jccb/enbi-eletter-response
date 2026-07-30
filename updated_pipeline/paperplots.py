import sys
import os
import numpy as np
import json 
from os import system
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_hex
import pandas as pd
from utils import *
import seaborn as sns
import pickle
import random
#import pyreadr
#import rpy2.robjects as ro
from scipy.stats import entropy
import dataanalyst as DA
import ModelData_analyses as MDA
import IBDMDB_analyses as IBD
import CD_Elife_analyses as CDI
import IBS_analyses as IBS
import CRC_analyses as CRC
#import CD_GenBio_analyses as CD
import metabolites_analyses as MET
from scipy.stats import mannwhitneyu, kruskal, ks_2samp, expon
from matplotlib.patches import Patch
from scipy.special import gamma, digamma, polygamma, erfc
from scipy.stats import gamma as gamma_dist
from scipy.stats import loggamma
from scipy.optimize import fsolve, minimize
from matplotlib.ticker import MaxNLocator
from itertools import combinations
from matplotlib.colors import to_rgb
from matplotlib.colors import LinearSegmentedColormap
import colormaps as cmaps
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, to_tree
import networkx as nx
from collections import defaultdict
from sklearn.metrics import (roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score)
from scipy.stats import percentileofscore

def change_base_directory(current_path, old_base, new_base):
    # Normalize the path to ensure there are no platform-specific path separators
    normalized_path = os.path.normpath(current_path)
    new_path = normalized_path.replace(old_base, new_base)
    return new_path

def calculate_shannon_index(abundance_series):
    relative_abundance = abundance_series.values
    return entropy(relative_abundance)
def calculate_nonzero_number(abundance_series):
    relative_abundance = abundance_series.values
    return np.count_nonzero(relative_abundance)

def get_tvec(folder_path, sample_hours, wanted_tvec = 't'):
    #with open(os.path.join(folder_path, "params.json"), "r") as f: 
    #    params = json.load(f)
    #    sample_freq = params['sample_freq'] #This just serves us to know the unit of the data! (The samples are in sample_freq units! So sample=24 is 24 hours only if sample_freq=1, but if sample_freq = 0.5 then sample is 0.5*24=12 hours)
    #    sample = sample_hours//sample_freq
    if wanted_tvec == 't':
        t_path_joined = os.path.join(folder_path, f"t_data_joined_{sample_hours}.npy") # existing_sample=sample if joine_data non existing or unuseful
        with open(t_path_joined, 'rb') as t_file_joined:
            specific_tvec = np.load(t_file_joined)
    elif wanted_tvec == 'S':
        S_path_joined = os.path.join(folder_path, f"S_data_joined_{sample_hours}.npy")
        with open(S_path_joined, 'rb') as S_file_joined:
            specific_tvec = np.load(S_file_joined)
    elif wanted_tvec == 'B':
        B_path_joined = os.path.join(folder_path, f"B_data_joined_{sample_hours}.pkl")
        with open(B_path_joined, 'rb') as B_file_joined:
            specific_tvec = pickle.load(B_file_joined)
    elif wanted_tvec == 'D':
        D_path_joined = os.path.join(folder_path, f"D_data_joined_{sample_hours}.pkl")
        with open(D_path_joined, 'rb') as D_file_joined:
            specific_tvec = pickle.load(D_file_joined)
        
    return specific_tvec
def get_B_types_dict(folder_path, sample_hours):
    wanted_file = os.path.join(folder_path, 'processed', f'B_types_dict_{sample_hours}.pkl')
    with open(wanted_file, 'rb') as fin:
        B_types_dict = pickle.load(fin)
    return B_types_dict
def get_enzymatic_cost_shannon_D_data(folder_path, sample_hours):
    wanted_file = os.path.join(folder_path, 'processed', f'functions_data_{sample_hours}.pkl')
    with open(wanted_file, 'rb') as fin:
        N_pathways_tvec, N_enzymes_tvec, shannon_tvec, rel_D_B_tvec, rel_D_tvec, D_data_dict, N_types_tvec = pickle.load(fin)
    return N_enzymes_tvec, shannon_tvec, D_data_dict
def get_cf_cp_mat_tvecs(folder_path, sample_hours):
    wanted_file = os.path.join(folder_path, 'processed', f'cf_cp_data_{sample_hours}.pkl')
    with open(wanted_file, 'rb') as fin:
        crossfeeding_ijmat_tvec, competition_ijmat_tvec = pickle.load(fin)
    return crossfeeding_ijmat_tvec, competition_ijmat_tvec

def get_rel_B_df(folder_path, sample_hours, t_vec, B_types_dict, overwrite=False):
    wanted_file = os.path.join(folder_path, 'processed', f'rel_B_df_{sample_hours}.pkl')
    if not overwrite and os.path.isfile(wanted_file): 
        with open(wanted_file, 'rb') as fin:
            rel_B_df = pickle.load(fin)
    else:
        B_df = pd.DataFrame({'t': t_vec})
        if len(t_vec) != len(set(t_vec)):
            print("The t_vec has values repeated. This would cause problems in the merging of the dataframes right after in the code. Solve that before proceeding")
            raise
        duplicate_columns = False
        
        for Type, (t_init, t_end, type_functional_identity, B_type_t) in B_types_dict.items():
            wanted_t = t_vec[(t_vec>=t_init) & (t_vec<t_end)]
            aux_df = pd.DataFrame({'t': wanted_t, f'{type_functional_identity}': B_type_t})
            if type_functional_identity in B_df.columns: # If the type_functional_identity is already inside we will get duplicate columns in the merge
                duplicate_columns = True
            B_df = B_df.merge(aux_df, how='left', on='t')

            # Handle duplicate columns with the same type_functional_identity               
            if duplicate_columns:
                col_x = f'{type_functional_identity}_x'
                col_y = f'{type_functional_identity}_y'
                #print(B_df[['t',col_x]][B_df[col_x].notna()])
                # Combine the two columns into one
                B_df[type_functional_identity] = B_df[col_x].fillna(0) + B_df[col_y].fillna(0)
                # Drop the original `_x` and `_y` columns
                B_df = B_df.drop(columns=[col_x, col_y])
                duplicate_columns = False
        
        B_df = B_df.fillna(0) # Because for duplicate columns we used 0 instead of nans
        excluded_columns = ['t']
        other_cols = B_df.columns.difference(excluded_columns)
        B_df['total_B'] = B_df[other_cols].sum(axis=1, skipna=True) # We sum all columns without including t_vec obviously!
        B_df[other_cols] = B_df[other_cols].div(B_df['total_B'], axis=0) # Remember here other cols does not include 'total_B' because it has been created after
        rel_B_df = B_df.copy()
        rel_B_df = rel_B_df.fillna(0)
        rel_B_df = rel_B_df.drop('total_B', axis=1)
        processed_dir = os.path.join(folder_path, 'processed')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
            
        with open(wanted_file, 'wb') as f:
            pickle.dump(rel_B_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rel_B_df

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
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json') and 'params' not in f]
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

        return filtered_files_dict
    def is_decimal_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    wanted_file = os.path.join(bootstrap_dir, 'processed', 'stats_dict.pkl')
    if not overwrite and os.path.isfile(wanted_file): # here we try to load B_types_dict
        with open(wanted_file, 'rb') as fin:
            stats_dict = pickle.load(fin)
    else:
        stats_dict = {}
        for subset_fraction in os.listdir(bootstrap_dir): # Here we should have the subset_fractions           
            full_path = os.path.join(bootstrap_dir, str(subset_fraction))
            if os.path.isdir(full_path) and is_decimal_number(subset_fraction):
                stats_dict[subset_fraction] = {}
                files_dict = filter_json_files(full_path, str(max_iters), str(subset_fraction))
                for diagnosis,filename in files_dict.items():
                    with open(os.path.join(full_path, f'{filename}.json'), 'r') as file:
                        aux = json.load(file)
                        if len(aux) == 0:
                            continue
                        data = aux[subset_fraction]
                    diagnosis_subset_dict = {diagnosis: data[:min(len(data),max_iters)]}
                    stats_dict[subset_fraction].update(diagnosis_subset_dict)
        processed_dir = os.path.join(bootstrap_dir, 'processed')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        with open(wanted_file, 'wb') as f:
            pickle.dump(stats_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return dict(sorted(stats_dict.items()))

def get_sampled_metric_df(rootpath, data_type, metric, sample_size):
        def filter_diagnosis(diagnosis):
            if diagnosis == 'Healthy':
                return 'H'
            elif diagnosis == 'Unhealthy':
                return 'U'
            else:
                print(f'Diagnosis: {diagnosis} is not valid')
                raise
        metadata_dict = MDA.get_metadata_dict(rootpath,overwrite=False, sample_size=sample_size)
        data_df = MDA.get_data_df(rootpath, data_type, overwrite=False, sample_size=sample_size)
        full_data_df = MDA.get_full_data_dict(metadata_dict, data_df)
        diagnosis_metric_dict = {diagnosis: [] for diagnosis in ['H', 'U']}
        for participant_id, samples in full_data_df.items():
            for sample in samples:
                if metric == 'Shannon':
                    metric_value = calculate_shannon_index(sample['Data'])
                elif metric == 'Enzymes':
                    metric_value = sample['Data'].sum()
                elif metric == 'rho':
                    metric_value = sample['rho']
                elif metric == 'Substances':
                    metric_value =  np.count_nonzero(sample['Data'])
                else:
                    print(f'Metric: {metric} is not accepted. Accepted ones are Shannon, Number, Number/Species.')
                    raise
                diagnosis_metric_dict[filter_diagnosis(sample['Diagnosis'])].append(metric_value)

        # Separate healthy and non-healthy data
        healthy_data = diagnosis_metric_dict['H']
        non_healthy_data = []
        for diag, values in diagnosis_metric_dict.items():
            if diag != 'H':
                non_healthy_data.extend(values)

        # Create subplots
        # Combine the data into a DataFrame
        data = {
            'Diagnosis': ['H'] * len(healthy_data) + ['U'] * len(non_healthy_data),
            'Metric Value': healthy_data + non_healthy_data
        }
        _, p_value = mannwhitneyu(healthy_data, non_healthy_data, alternative='two-sided')
        df = pd.DataFrame(data)
        return df, p_value
def get_realization_data(folder_path, sample_hours, full): 
    t_vec= get_tvec(folder_path, sample_hours, wanted_tvec='t')
    B_types_dict = get_B_types_dict(folder_path, sample_hours)
    S_tvec = None
    shannon_tvec = None
    enzymatic_cost_tvec = None
    D_data_dict = None
    rho_tvec = None
    if full:
        S_tvec = get_tvec(folder_path, sample_hours, wanted_tvec='S')
        enzymatic_cost_tvec, shannon_tvec, D_data_dict = get_enzymatic_cost_shannon_D_data(folder_path, sample_hours)
        crossfeeding_ijmat_tvec, competition_ijmat_tvec = get_cf_cp_mat_tvecs(folder_path, sample_hours)
        cf_tvec = np.array([np.sum(cf) for cf in crossfeeding_ijmat_tvec])
        cp_tvec = np.array([(np.sum(cp)-np.trace(cp)) for cp in competition_ijmat_tvec])
        rho_tvec = (cf_tvec-cp_tvec)/(cf_tvec+cp_tvec)
    
    return t_vec, S_tvec, B_types_dict, shannon_tvec, enzymatic_cost_tvec, D_data_dict, rho_tvec

def filter_tvecs(t_vec, shannon_tvec, enzymatic_cost_tvec, rho_tvec, th_years):
    rho_tvec = rho_tvec[t_vec>=th_years]
    shannon_tvec = shannon_tvec[t_vec>=th_years]
    enzymatic_cost_tvec = enzymatic_cost_tvec[t_vec>=th_years]
    t_vec = t_vec[t_vec>=th_years]
    return t_vec, shannon_tvec, enzymatic_cost_tvec, rho_tvec


def generate_shades(base_color, n_shades):
    '''
    base_rgb = np.array(to_rgb(base_color))
    lightness_factors = np.linspace(1.0, 0.5, n_shades)  # from bright to dark
    return [tuple(base_rgb * factor) for factor in lightness_factors]
    '''
    random.seed(1)  # Replace 42 with any integer you like
    base_rgb = list(np.array(to_rgb(base_color)))
    alphas = np.linspace(0.6, 1, n_shades)
    random.shuffle(alphas)
    return [(*base_rgb, alpha) for alpha in alphas]

def get_color_mapping_for_group1_using_group10(group1_dict, group10_dict, palette=cmaps.safe):
    # Map fine category (group 1) → coarse category (group 10)
    group1_to_group10 = {p_str: group10_dict[p_str] for p_str in group1_dict}

    # Reverse map: coarse category → list of fine categories
    coarse_to_fine = {}
    for p_str, coarse_cat in group1_to_group10.items():
        fine_cat = group1_dict[p_str]
        if coarse_cat not in coarse_to_fine:
            coarse_to_fine[coarse_cat] = []
        coarse_to_fine[coarse_cat].append(fine_cat)

    # Assign base colors to coarse categories
    coarse_categories = sorted(set(group1_to_group10.values()), key=int)
    #coarse_palette = sns.color_palette(palette_name, len(coarse_categories))
    coarse_palette = spread_colormap(palette, total_items=len(coarse_categories), resolution=palette.N)
    coarse_color_map = {coarse_cat: coarse_palette[i] for i, coarse_cat in enumerate(coarse_categories)}

    # Now assign shades to each fine category based on its coarse category
    final_color_mapping = {}
    for coarse_cat, fine_cats in coarse_to_fine.items():
        shades = generate_shades(coarse_color_map[coarse_cat], len(fine_cats))
        for fine_cat, shade in zip(sorted(fine_cats, key=str), shades):
            final_color_mapping[fine_cat] = shade

    return final_color_mapping

def get_color_mapping_from_pathway_classification(pathway_class_csv, pathway_list, palette='deep'):
    # Load your custom CSV
    mapping_df = pd.read_csv(pathway_class_csv, header=None, names=['pathway', 'category'])

    # Keep only the first category if multiple are present
    mapping_df['category'] = mapping_df['category'].str.split(' and ').str[0]

    # Filter only pathways you actually use
    mapping_df = mapping_df[mapping_df['pathway'].isin(pathway_list)]

    # Build category → list of pathways
    cat_to_pathways = mapping_df.groupby('category')['pathway'].apply(list).to_dict()

    # Assign a base color per category
    category_palette = sns.color_palette(palette, len(cat_to_pathways))
    category_color_map = {cat: category_palette[i] for i, cat in enumerate(cat_to_pathways)}

    # Now assign shades (alpha) per pathway
    final_color_mapping = {}
    for cat, base_color in category_color_map.items():
        pathways = sorted(cat_to_pathways[cat])
        shades = generate_shades(base_color, len(pathways))
        for path, shade in zip(pathways, shades):
            final_color_mapping[path] = shade

    return final_color_mapping, mapping_df

def spread_colormap(palette, total_items, resolution, jump=1):
    """
    Spread colors evenly across a sequential colormap without clustering repeats.

    Args:
        palette: A matplotlib colormap (e.g., cmaps.grads_default)
        total_items: Number of values you need
        resolution: Number of distinct colors to extract from the colormap
        jum: If i need to make a jump for each color

    Returns:
        A list of RGB or RGBA colors
    """
    base_colors = [palette(i / (resolution - 1)) for i in range(resolution)]
    
    # Spread repeated colors
    colors = []
    for i in range(total_items):
        color_index = (int(i*jump) % resolution)
        colors.append(base_colors[color_index])
    return colors

def get_color_mapping_from_classification_tabsep(pathway_class_tsv, pathway_list, palette=cmaps.grads_default):
    # Define important subclasses
    important_subclasses = {
        "Amino Acid Biosynthesis", "Carbohydrate Biosynthesis", "Carbohydrate Degradation", 
        "Cell Structure Biosynthesis", "Cofactor, Carrier, and Vitamin Biosynthesis", "Fatty Acid and Lipid Biosynthesis",
        "Nucleoside and Nucleotide Biosynthesis", "Nucleoside and Nucleotide Degradation",
    }

    # Load classification file
    df = pd.read_csv(pathway_class_tsv, sep="\t", header=0, names=["pathway", "class", "subclass"])
    
    # Strip quotes from both pathway list and df pathway column
    df['pathway'] = df['pathway'].str.strip('"')
    cleaned_pathways = [p.strip('"') for p in pathway_list]

    # Check that all pathway names match
    df_set = set(df['pathway'])
    list_set = set(cleaned_pathways)
    if df_set != list_set:
        missing_in_df = list_set - df_set
        missing_in_list = df_set - list_set
        raise ValueError(f"Mismatch in pathway names:\nMissing in classification file: {missing_in_df}\nMissing in list: {missing_in_list}")

    # Choose base color per subclass if important, otherwise fall back to class
    df['category'] = df.apply(
        lambda row: row['subclass'] if row['subclass'] in important_subclasses else row['class'], axis=1
    )

    # Group pathways by category
    cat_to_pathways = df.groupby('category')['pathway'].apply(list).to_dict()

    # Assign base colors
    #category_palette = sns.color_palette(palette, len(cat_to_pathways))
    category_palette = [palette(i / (len(cat_to_pathways) - 1)) for i in range(len(cat_to_pathways))]
    category_palette = spread_colormap(palette, total_items=len(cat_to_pathways), resolution=palette.N)
    category_color_map = {cat: category_palette[i] for i, cat in enumerate(cat_to_pathways)}

    # Assign color shades to each pathway
    final_color_mapping = {}
    for cat, base_color in category_color_map.items():
        pathways = sorted(cat_to_pathways[cat])
        shades = generate_shades(base_color, len(pathways))
        for path, shade in zip(pathways, shades):
            final_color_mapping[path] = shade

    return final_color_mapping, df, category_color_map

def get_color_mapping_classification_final(classification_tsv, pathway_list, healthy_df, unhealthy_df, palette=cmaps.antique):
    important_subclasses = {
        "Amino Acid Biosynthesis", "Carbohydrate Biosynthesis", "Carbohydrate Degradation",
        "Cell Structure Biosynthesis", "Cofactor, Carrier, and Vitamin Biosynthesis", 
        "Fatty Acid and Lipid Biosynthesis", "Nucleoside and Nucleotide Biosynthesis", 
        "Nucleoside and Nucleotide Degradation"
    }

    keep_named_classes = ["Biosynthesis", "Generation of Precursor Metabolites and Energy"]

    classification_df = pd.read_csv(classification_tsv, sep="\t", header=0, names=["pathway", "class", "subclass"])
    classification_df['pathway'] = classification_df['pathway'].str.strip('"')
    cleaned_pathways = [p.strip('"') for p in pathway_list]

    df_set = set(classification_df['pathway'])
    list_set = set(cleaned_pathways)
    if df_set != list_set:
        missing_in_df = list_set - df_set
        missing_in_list = df_set - list_set
        raise ValueError(f"Mismatch in pathway names:\nMissing in classification file: {missing_in_df}\nMissing in list: {missing_in_list}")

    # Assign category
    def assign_category(row):
        if row['subclass'] in important_subclasses:
            return row['subclass']
        elif row['class'] in keep_named_classes:
            return row['class']
        else:
            return "OTHER"

    classification_df['category'] = classification_df.apply(assign_category, axis=1)

    # Calculate category-level mean abundances
    healthy_means = healthy_df.mean(axis=0)
    unhealthy_means = unhealthy_df.mean(axis=0)
    classification_df['mean_healthy'] = classification_df['pathway'].map(healthy_means)
    classification_df['mean_unhealthy'] = classification_df['pathway'].map(unhealthy_means)
    classification_df['delta'] = classification_df['mean_healthy'] - classification_df['mean_unhealthy']

    # Order categories by summed delta
    category_order = (
        classification_df.groupby('category')['delta']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Assign base colors using ordered categories
    category_palette = spread_colormap(palette, total_items=len(category_order), resolution=palette.N)
    category_color_map = {cat: category_palette[i] for i, cat in enumerate(category_order)}

    # Build pathway color map
    final_color_mapping = {}
    for cat in category_order:
        pathways = classification_df[classification_df['category'] == cat].sort_values(by='pathway')['pathway'].tolist()
        shades = generate_shades(category_color_map[cat], len(pathways))
        for path, shade in zip(pathways, shades):
            final_color_mapping[path] = shade

    return final_color_mapping, classification_df, category_color_map, category_order

def assign_pathway_colors(healthy_df, unhealthy_df, seed=42):
    blue = '#1771a2'
    red = '#cd1d2e'
    grey = '#555555'

    # Drop 't' if present, then sum across samples for each pathway
    healthy_pathways_total = healthy_df.drop(columns=['t'], errors='ignore').sum(axis=0)
    unhealthy_pathways_total = unhealthy_df.drop(columns=['t'], errors='ignore').sum(axis=0)

    healthy_present = set(healthy_pathways_total[healthy_pathways_total > 0].index)
    unhealthy_present = set(unhealthy_pathways_total[unhealthy_pathways_total > 0].index)

    shared = sorted(healthy_present & unhealthy_present)
    healthy_only = sorted(healthy_present - unhealthy_present)
    unhealthy_only = sorted(unhealthy_present - healthy_present)
    print(healthy_only)
    print(unhealthy_only)
    def generate_alpha_colors(base_color, pathways):
        base_rgb = mcolors.to_rgb(base_color)
        alphas = np.linspace(0.4, 1.0, len(pathways))
        random.seed(seed)
        random.shuffle(alphas)
        return {p: (*base_rgb, alpha) for p, alpha in zip(pathways, alphas)}

    color_map = {}
    color_map.update(generate_alpha_colors(grey, shared))
    color_map.update(generate_alpha_colors(blue, healthy_only))
    color_map.update(generate_alpha_colors(red, unhealthy_only))

    all_pathways_order = shared + healthy_only + unhealthy_only
    return color_map, all_pathways_order

def assign_pathway_colors_with_significance(healthy_df, unhealthy_df, seed=42, min_diff=0.0, alpha=0.05):
    """
    Assigns a unique color (with varying alpha) to each pathway based on its presence in healthy and/or unhealthy samples.
    If a shared pathway is significantly different in presence between healthy and unhealthy groups, it gets a unique color.

    Args:
        healthy_df (pd.DataFrame): Pathways x Samples (columns), with values as abundances. Includes 't' column.
        unhealthy_df (pd.DataFrame): Same structure as healthy_df.
        seed (int): Random seed for consistent alpha shuffling.
        min_diff (float): Minimum difference in mean presence to consider a pathway differentially present.
        alpha (float): Significance threshold.

    Returns:
        dict: pathway_name -> rgba color mapping.
        list: Ordered list of all pathways to be used for consistent plotting.
    """

    blue = '#1771a2'
    red = '#cd1d2e'
    grey = '#555555'
    yellow = '#e5c610'  # soft yellow that matches blue and red

    # Drop time columns if present
    healthy_df = healthy_df.drop(columns=['t'], errors='ignore')
    unhealthy_df = unhealthy_df.drop(columns=['t'], errors='ignore')

    # Convert to binary presence/absence
    healthy_bin = (healthy_df > 0).astype(int)
    unhealthy_bin = (unhealthy_df > 0).astype(int)

    all_pathways = set(healthy_df.columns) | set(unhealthy_df.columns)

    # Identify shared and specific pathways based on presence across any time/sample
    shared = []
    healthy_only = []
    unhealthy_only = []
    significant_shared = []

    for p in all_pathways:
        h_present = healthy_bin[p].sum() > 0
        u_present = unhealthy_bin[p].sum() > 0

        if h_present and u_present:
            # Check significance
            h_vals = healthy_bin[p]
            u_vals = unhealthy_bin[p]
            diff = abs(h_vals.mean() - u_vals.mean())
            if diff >= min_diff:
                _, p_val = mannwhitneyu(h_vals, u_vals, alternative='two-sided')
                if p_val < alpha:
                    significant_shared.append(p)
                else:
                    shared.append(p)
            else:
                shared.append(p)
        elif h_present:
            healthy_only.append(p)
        elif u_present:
            unhealthy_only.append(p)
        # else neither — skip

    # Create alpha-based color shading
    def generate_alpha_colors(base_color, pathways):
        base_rgb = mcolors.to_rgb(base_color)
        alphas = np.linspace(0.4, 1.0, len(pathways))
        random.seed(seed)
        random.shuffle(alphas)
        return {p: (*base_rgb, alpha) for p, alpha in zip(pathways, alphas)}

    color_map = {}
    color_map.update(generate_alpha_colors(grey, shared))
    color_map.update(generate_alpha_colors(yellow, significant_shared))
    color_map.update(generate_alpha_colors(blue, healthy_only))
    color_map.update(generate_alpha_colors(red, unhealthy_only))

    ordered_pathways = sorted(significant_shared) + sorted(shared) + sorted(healthy_only) + sorted(unhealthy_only)

    return color_map, ordered_pathways

def prepare_df(df, ordered_pathways):
    df = df.drop(columns=['t'], errors='ignore')
    df = df[ordered_pathways] if set(ordered_pathways).issubset(df.columns) else df
    return df.copy()
def prepare_presence_df_for_plot(df, ordered_pathways):
    df = df.drop(columns=['t'], errors='ignore')
    df = df[ordered_pathways].copy()

    # Binarize: presence (1) or absence (0)
    binary_df = (df > 0).astype(float)

    # Normalize rows to sum to 1
    normalized_df = binary_df.div(binary_df.sum(axis=1), axis=0).fillna(0)

    return normalized_df
def remove_statistically_similar_pathways(healthy_df, unhealthy_df, p_threshold=0.8):
    """
    Removes pathways from both dataframes that are statistically similar in abundance.
    
    Args:
        healthy_df (pd.DataFrame): DataFrame with samples as rows and pathways as columns.
        unhealthy_df (pd.DataFrame): Same structure as healthy_df.
        p_threshold (float): Threshold above which p-values are considered similar.
    
    Returns:
        pd.DataFrame, pd.DataFrame, list: Filtered healthy and unhealthy DataFrames, and removed pathways list.
    """
    shared_pathways = set(healthy_df.columns) & set(unhealthy_df.columns)
    pathways_to_remove = []

    for p in shared_pathways:
        h_vals = healthy_df[p].values
        u_vals = unhealthy_df[p].values
        _, pval = mannwhitneyu(h_vals, u_vals, alternative='two-sided')
        if pval > p_threshold:
            pathways_to_remove.append(p)

    healthy_df_filtered = healthy_df.drop(columns=pathways_to_remove)
    unhealthy_df_filtered = unhealthy_df.drop(columns=pathways_to_remove)

    return healthy_df_filtered, unhealthy_df_filtered, pathways_to_remove

def assign_colors_by_difference(healthy_df, unhealthy_df, p_value_threshold=0.05, seed=42):
    blue = '#1771a2'
    red = '#cd1d2e'
    grey = '#555555'
    yellow = '#f4c542'  # Statistically significant

    rng = np.random.default_rng(seed)

    # Exclude 't' column
    healthy_data = healthy_df.drop(columns=['t'], errors='ignore')
    unhealthy_data = unhealthy_df.drop(columns=['t'], errors='ignore')

    # Compute mean abundance across samples
    mean_healthy = healthy_data.mean(axis=0)
    mean_unhealthy = unhealthy_data.mean(axis=0)

    # Compute difference
    abundance_diff = mean_healthy - mean_unhealthy  # Positive: more in healthy
    sorted_pathways = abundance_diff.sort_values()

    # Compute percentiles
    q25 = sorted_pathways.quantile(0.25)
    q75 = sorted_pathways.quantile(0.75)

    # Simulated p-values (replace with real test in practice)
    p_values = {col: mannwhitneyu(healthy_data[col], unhealthy_data[col], alternative='two-sided').pvalue
                for col in healthy_data.columns}

    color_map = {}
    for i, pathway in enumerate(sorted_pathways.index):
        diff = sorted_pathways[pathway]
        p_val = p_values[pathway]

        # Priority: statistically significant
        if p_val < p_value_threshold:
            color_map[pathway] = yellow

        # Overrepresented in unhealthy (bottom 25%)
        elif diff < q25:
            rel_rank = (diff - sorted_pathways.min()) / (q25 - sorted_pathways.min())
            alpha = 0.2 + 0.8 * rel_rank
            color_map[pathway] = (*to_rgb(red), alpha)

        # Overrepresented in healthy (top 25%)
        elif diff > q75:
            rel_rank = (diff - q75) / (sorted_pathways.max() - q75)
            alpha = 0.2 + 0.8 * rel_rank
            color_map[pathway] = (*to_rgb(blue), alpha)

        # Middle 50% → grey with varying alpha
        else:
            if diff <= 0:  # 25th to 50th percentile → low to high alpha
                rel_rank = (diff - q25) / (0 - q25)
                alpha = 0.2 + 0.8 * rel_rank
            else:          # 50th to 75th percentile → high to low alpha
                rel_rank = (q75 - diff) / (q75 - 0)
                alpha = 0.2 + 0.8 * rel_rank
            color_map[pathway] = (*to_rgb(grey), alpha)

    return color_map, sorted_pathways.index.tolist()

def assign_colors_abundance_difference(healthy_df, unhealthy_df, p_value_threshold=0.05):
    blue = '#1771a2'
    red = '#cd1d2e'
    grey = '#555555'

    # Remove 't' column
    healthy_df = healthy_df.drop(columns=['t'], errors='ignore')
    unhealthy_df = unhealthy_df.drop(columns=['t'], errors='ignore')

    # Compute mean abundance differences
    mean_healthy = healthy_df.mean(axis=0)
    mean_unhealthy = unhealthy_df.mean(axis=0)
    mean_diff = mean_healthy - mean_unhealthy  # positive = more in healthy

    # Sort pathways by mean difference
    sorted_pathways = mean_diff.sort_values()
    
    # Compute p-values with Mann-Whitney U test
    p_values = {
        col: mannwhitneyu(healthy_df[col], unhealthy_df[col], alternative='two-sided').pvalue
        for col in healthy_df.columns
    }
    p_values = pd.Series(p_values)
    # Get index lists
    significant = p_values[p_values < p_value_threshold].index
    #nonsignificant = p_values[p_values >= p_value_threshold].index
    nonsignificant = sorted_pathways.index.difference(significant)

    color_map = {}

    # Handle significant pathways
    sig_positive = sorted_pathways[(sorted_pathways > 0) & (sorted_pathways.index.isin(significant))]
    sig_negative = sorted_pathways[(sorted_pathways <= 0) & (sorted_pathways.index.isin(significant))]

    if not sig_positive.empty:
        min_diff = sig_positive.min()
        max_diff = sig_positive.max()
        for pathway, diff in sig_positive.items():
            alpha = 0.2 + 0.8 * (diff - min_diff) / (max_diff - min_diff) if max_diff > min_diff else 1.0
            color_map[pathway] = (*to_rgb(blue), alpha)

    if not sig_negative.empty:
        min_diff = sig_negative.min()
        max_diff = sig_negative.max()
        for pathway, diff in sig_negative.items():
            alpha = 0.2 + 0.8 * (max_diff - diff) / (max_diff - min_diff) if max_diff > min_diff else 1.0
            color_map[pathway] = (*to_rgb(red), alpha)

    # Handle nonsignificant with gradient gray
    nonsig_pos = sorted_pathways[(sorted_pathways > 0) & (sorted_pathways.index.isin(nonsignificant))]
    nonsig_neg = sorted_pathways[(sorted_pathways <= 0) & (sorted_pathways.index.isin(nonsignificant))]

    if not nonsig_pos.empty:
        min_diff = nonsig_pos.min()
        max_diff = nonsig_pos.max()
        for pathway, diff in nonsig_pos.items():
            alpha = 0.2 + 0.8 * (diff - min_diff) / (max_diff - min_diff) if max_diff > min_diff else 0.4
            color_map[pathway] = (*to_rgb(grey), alpha)

    if not nonsig_neg.empty:
        min_diff = nonsig_neg.min()
        max_diff = nonsig_neg.max()
        for pathway, diff in nonsig_neg.items():
            alpha = 0.2 + 0.8 * (max_diff - diff) / (max_diff - min_diff) if max_diff > min_diff else 0.4
            color_map[pathway] = (*to_rgb(grey), alpha)
    
    default_grey = to_rgb(grey)
    for pathway in sorted_pathways.index:
        if pathway not in color_map:
            color_map[pathway] = (*default_grey, 0.6)
            print(f'Pathway: {pathway} with p-value: {p_values[pathway]} and mean difference of : {sorted_pathways[pathway]} did not have a color')
    return color_map, sorted_pathways.index.tolist()

def plot_model3(gut_dir, rootpath, rootpath_img, big_realization_number, small_realization_numbers, sample_hours, healthy_index=0, IBD_index=0, overwrite=False):
    '''
    This will make the plot for the:
    - Biomass vs time
    - Shannon Index vs time
    - Nº of pathways vs time
    - Crossfeeding-competition vs time
    all sharing the time axis.
    '''
    
    def get_IBD_data(gut_dir, healthy_index=0, IBD_index=0):
        def calculate_shannon_index(abundance_series):
            relative_abundance = abundance_series.values
            return entropy(relative_abundance)
        def calculate_nonzero_number(abundance_series):
            relative_abundance = abundance_series.values
            return np.count_nonzero(relative_abundance)
        def get_metric(participant_dict,metric):
            healthy_metric = []
            unhealthy_metric = []
            for participant_id, samples in participant_dict.items():                
                for sample in samples:
                    if metric == 'Shannon':
                        metric_value = calculate_shannon_index(sample['Data'])
                    elif metric == 'Number':
                        metric_value = calculate_nonzero_number(sample['Data'])
                    else:
                        print(f'Metric:{metric} not allowed')
                    if sample['Diagnosis'] == 'nonIBD':
                        healthy_metric.append(metric_value)
                    else:
                        unhealthy_metric.append(metric_value)
            
            return healthy_metric, unhealthy_metric
        def get_dfs(participant_dict, healthy_index, IBD_index, min_samples=10):
            """
            Returns the DataFrames for a selected healthy and unhealthy participant
            based on the number of samples they have (most to least), with optional indexing.
            
            Args:
                participant_dict (dict): Dictionary with participant_id → list of sample dicts.
                min_samples (int): Minimum number of samples a participant must have to be considered.
                healthy_index (int): Rank (by number of samples) of healthy participant to choose.
                IBD_index (int): Rank (by number of samples) of unhealthy participant to choose.
                
            Returns:
                healthy_df, unhealthy_df (DataFrames)
            """
            healthy_participants = []
            unhealthy_participants = []

            for participant_id, samples in participant_dict.items():
                if len(samples) < min_samples:
                    continue
                if samples[0]['Diagnosis'] == 'nonIBD':
                    healthy_participants.append((participant_id, len(samples)))
                else:
                    unhealthy_participants.append((participant_id, len(samples)))

            # Sort by number of samples (descending)
            healthy_participants.sort(key=lambda x: x[1], reverse=True)
            unhealthy_participants.sort(key=lambda x: x[1], reverse=True)
            try:
                healthy_id = healthy_participants[healthy_index][0]
                unhealthy_id = unhealthy_participants[IBD_index][0]
            except IndexError:
                raise ValueError("Index out of range. Fewer participants meet the `min_samples` condition than expected.")

            # Construct dataframes
            healthy_df = pd.concat([sample['Data'] for sample in participant_dict[healthy_id]], axis=1).T  
            unhealthy_df = pd.concat([sample['Data'] for sample in participant_dict[unhealthy_id]], axis=1).T   
            healthy_df['t'] = [sample['Week'] for sample in participant_dict[healthy_id]]
            unhealthy_df['t'] = [sample['Week'] for sample in participant_dict[unhealthy_id]]

            return healthy_df, unhealthy_df
        rootpath_data = os.path.join(gut_dir,'real_data','IBD_MDB')
        rootpath_metabolites = os.path.join(gut_dir, 'real_data', 'Metabolome_Borenstein')
        metadata_dict1 = IBD.get_metadata_dict(rootpath_data)
        metadata_dict2 = MET.get_metadata_dict(rootpath_metabolites, 'IBDMDB')
        data_df_taxonomy = IBD.get_data_df(rootpath_data, data_type = 'taxonomy')
        data_df_pathways = IBD.get_data_df(rootpath_data, data_type = 'pathway')
        data_df_enzymes = IBD.get_data_df(rootpath_data, data_type = 'enzyme')
        data_df_substances = MET.get_data_df(rootpath_metabolites, 'IBDMDB', high_confidence_wanted=True)
        full_data_dict_taxonomy = IBD.get_full_data_dict(metadata_dict1, data_df_taxonomy)
        full_data_dict_pathways = IBD.get_full_data_dict(metadata_dict1, data_df_pathways)
        full_data_dict_enzymes = IBD.get_full_data_dict(metadata_dict1, data_df_enzymes)
        full_data_dict_substances = MET.get_full_data_dict(metadata_dict2, data_df_substances)
        
        healthy_shannon, unhealthy_shannon = get_metric(full_data_dict_taxonomy, metric='Shannon')
        healthy_enzymes, unhealthy_enzymes = get_metric(full_data_dict_enzymes, metric='Number')
        healthy_substances, unhealthy_substances = get_metric(full_data_dict_substances, metric='Number')
        _, p_shannon = mannwhitneyu(healthy_shannon, unhealthy_shannon, alternative='two-sided')
        _, p_enzymes = mannwhitneyu(healthy_enzymes, unhealthy_enzymes, alternative='two-sided')
        _, p_substances = mannwhitneyu(healthy_substances, unhealthy_substances, alternative='two-sided')
        #healthy_df, unhealthy_df = get_dfs(full_data_dict_taxonomy)
        #healthy_df, unhealthy_df = get_dfs(full_data_dict_pathways)
        healthy_df, unhealthy_df = get_dfs(full_data_dict_pathways, healthy_index, IBD_index)
        return_data = (healthy_shannon, unhealthy_shannon, p_shannon, healthy_enzymes, unhealthy_enzymes,
                       p_enzymes, healthy_substances, unhealthy_substances, p_substances, healthy_df, unhealthy_df)
        return return_data

    fig = plt.figure(figsize = (18.4*0.393701, 14*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(12,9, height_ratios=[0.10, 0.02, 0.045, 0.02, 0.10, 0.04, 0.055, 0.17, 0.055, 0.17, 0.055, 0.17],
                                width_ratios=[0.284, 0.069, 0.132, 0.02, 0.132, 0.069, 0.132, 0.02, 0.132])
    ax00 = fig.add_subplot(gs[0:5,0])
    ax10 = fig.add_subplot(gs[7,0], sharex=ax00)
    ax20 = fig.add_subplot(gs[9,0], sharex=ax00)
    ax30 = fig.add_subplot(gs[11,0], sharex=ax00)
    ax01 = fig.add_subplot(gs[0:5,2])#fig.add_subplot(gs[0:2,2:4])
    ax11 = fig.add_subplot(gs[0:5,4])
    #ax21 = fig.add_subplot(gs[3:5,2:4], sharex=ax01)
    ax31 = fig.add_subplot(gs[7,2:5])#, sharey=ax10)
    ax41 = fig.add_subplot(gs[9,2:5])#, sharey=ax20)
    ax51 = fig.add_subplot(gs[11,2:5])#, sharey=ax30)
    ax02 = fig.add_subplot(gs[0:5,6])#, sharey=ax10)
    ax22 = fig.add_subplot(gs[0:5,8])#, sharey=ax10)
    ax32 = fig.add_subplot(gs[7,6:])#, sharey=ax10)
    ax42 = fig.add_subplot(gs[9,6:])#, sharey=ax10)
    ax52 = fig.add_subplot(gs[11,6:])#, sharey=ax10)

    #get data for big realization
    path = os.path.join(rootpath, f'{big_realization_number}')
    t_vec, S_tvec, B_types_dict, shannon_tvec, enzymatic_cost_tvec, _, _ = get_realization_data(path, sample_hours, full=True)
    t_vec = t_vec/24/365 #(time in years)
    N_substances =  np.count_nonzero(S_tvec, axis=1)
    N_substances_rolling = pd.Series(N_substances).rolling(window=100, min_periods=1).mean()
    # get data for IBD:
    healthy_shannon, unhealthy_shannon, p_shannon, healthy_enzymes, unhealthy_enzymes, p_enzymes, healthy_substances, \
    unhealthy_substances, p_substances, healthy_df, unhealthy_df = get_IBD_data(gut_dir, healthy_index, IBD_index)
    
    
    #t_vec, shannon_tvec, enzymatic_cost_tvec, rho_tvec = filter_tvecs(t_vec, shannon_tvec, enzymatic_cost_tvec, rho_tvec, th_years=5)

    '''
    palette = sns.color_palette("tab20c", n_colors=len(B_types_dict.keys()))
    color_mapping = {Type: palette[i % len(palette)] for i, Type in enumerate(B_types_dict.keys())}
    # Plot the Biomasses vs time
    for Type, (t_init, t_end, type_functional_identity, B_type_t) in B_types_dict.items():
        wanted_t = t_vec[(t_vec>=t_init/24/365) & (t_vec<t_end/24/365)]
        ax00.plot(wanted_t,B_type_t,label=Type,color=color_mapping[Type])
    '''
    custom_colors = [
        "#2461AA",  # blue
        "#EC9627",  # orange
        "#58BF91",  # mint green
        "#D84752",  # adjusted red
        "#F8CB17",  # yellow
        "#90429A",  # purple
        "#2E5C3F",  # dark green
        "#B36F41",  # brown
        "#DC79B3",  # pink
        "#4AB3B7",  # lighter blue
        "#000000"   # black
    ]

    # Create a ListedColormap
    custom_cmap = ListedColormap(custom_colors, name='my_custom_palette')
    # 1. Get both fine and coarse mappings
    _, p_str_to_cat_dict1 = get_functions_categories(group_size=1, num_substances=30)
    _, p_str_to_cat_dict10 = get_functions_categories(group_size=10, num_substances=30)

    # 2. Get your color mapping
    final_color_mapping_model = get_color_mapping_for_group1_using_group10(p_str_to_cat_dict1, p_str_to_cat_dict10, 
                                                                     palette=custom_cmap)
    
    category_df, _ = get_taxonomy_and_functions_data(path, sample_hours, p_str_to_cat_dict1)
    category_df['t'] = category_df['t'] / (24 * 365)
    category_df.columns = category_df.columns.map(str)
    #category_df = category_df[category_df['t'] >= 3].reset_index(drop=True)

    for col in category_df.columns:
        if col != 't' and col not in final_color_mapping_model:
            print(f"Missing color for column: {col}")
    
    
    category_df = category_df.iloc[::10].reset_index(drop=True)
    
    healthy_df_model = category_df[(category_df['t']>=48) & (category_df['t']<58)].copy()
    unhealthy_df_model = category_df[(category_df['t']>=23) & (category_df['t']<33)].copy()
    # Define the list of pathway columns (as strings)
    pathway_cols = [col for col in category_df.columns if col != 't']
    
    healthy_df_model.plot(x='t', kind='bar', stacked=True, width=1, ax=ax01, legend=False,
                    color=[final_color_mapping_model[str(col)] for col in pathway_cols])
    unhealthy_df_model.plot(x='t', kind='bar', stacked=True, width=1, ax=ax11, legend=False,
                    color=[final_color_mapping_model[str(col)] for col in pathway_cols])
    
    

    # Plot Shannon vs t
    ax10.plot(t_vec, shannon_tvec, color='dimgrey')
    # Plot Enzymatic cost vs time
    ax20.plot(t_vec, enzymatic_cost_tvec, color='dimgrey')
    # Plot Number of substances (rolling average)
    ax30.plot(t_vec,N_substances, color='dimgrey', alpha=0.4)
    ax30.plot(t_vec,[int(x) for x in N_substances_rolling], color='dimgrey')
    
    print('Fase 1')
    #get data for small realizations and plot them
    for number, ax in zip(small_realization_numbers,[ax01]): #,ax21]):
        path = os.path.join(rootpath, f'{number}')
        t_vec, _, B_types_dict, _,_,_,_ = get_realization_data(path, sample_hours, full=False)
        t_vec = t_vec/24/365 #(time in years)
        # Plot the Biomasses vs time
        palette = sns.color_palette("tab20c", n_colors=len(B_types_dict.keys()))
        color_mapping = {Type: palette[i % len(palette)] for i, Type in enumerate(B_types_dict.keys())}
        for Type, (t_init, t_end, type_functional_identity, B_type_t) in B_types_dict.items():
            wanted_t = t_vec[(t_vec>=t_init/24/365) & (t_vec<t_end/24/365)]
            ax.plot(wanted_t,B_type_t,label=Type,color=color_mapping[Type])
    
    print('Fase 2')
    # get data for shannon, number of pathways and rho
    # Custom color palette and category order
    diagnosis_color_map = {
        "H": '#1771a2ff',  # Blue
        "U": "#cd1d2eff",  # Red
        "Healthy": '#1771a2ff',
        "IBD": "#cd1d2eff"
    }
    
    category_order = ['H', 'U']
    
    shannon_df, p_value_shannon = get_sampled_metric_df(rootpath, data_type='taxonomy', metric='Shannon', sample_size=200)
    sns.boxplot(x='Diagnosis', y='Metric Value', hue='Diagnosis', data=shannon_df, order=category_order, 
                palette=diagnosis_color_map, showfliers=False, legend=False, width=0.3, ax=ax31)
    pathways_df, p_value_enzymes = get_sampled_metric_df(rootpath, data_type='pathway', metric='Enzymes', sample_size=200)
    sns.boxplot(x='Diagnosis', y='Metric Value', hue='Diagnosis', data=pathways_df, order=category_order, 
                palette=diagnosis_color_map, showfliers=False, legend=False, width=0.3, ax=ax41)
    N_substances_df, p_value_substances = get_sampled_metric_df(rootpath, data_type='substances', metric='Substances', sample_size=200)
    sns.boxplot(x='Diagnosis', y='Metric Value', hue='Diagnosis', data=N_substances_df, order=category_order, 
                palette=diagnosis_color_map, showfliers=False, legend=False, width=0.3, ax=ax51)
    
    print('Fase 3')
    
    
    IBD_shannon_df = pd.DataFrame({'Metric Value': healthy_shannon + unhealthy_shannon, 
                                'Diagnosis': ['Healthy'] * len(healthy_shannon) + ['IBD'] * len(unhealthy_shannon)})
    IBD_enzymes_df = pd.DataFrame({'Metric Value': healthy_enzymes + unhealthy_enzymes, 
                                'Diagnosis': ['Healthy'] * len(healthy_enzymes) + ['IBD'] * len(unhealthy_enzymes)})
    IBD_substances_df = pd.DataFrame({'Metric Value': healthy_substances + unhealthy_substances, 
                                'Diagnosis': ['Healthy'] * len(healthy_substances) + ['IBD'] * len(unhealthy_substances)})
    category_order2 = ['Healthy', 'IBD']
    sns.boxplot(x='Diagnosis', y='Metric Value', hue='Diagnosis', data=IBD_shannon_df, order=category_order2, 
                palette=diagnosis_color_map, showfliers=False, legend=False, width=0.3, ax=ax32)
    sns.boxplot(x='Diagnosis', y='Metric Value', hue='Diagnosis', data=IBD_enzymes_df, order=category_order2, 
                palette=diagnosis_color_map, showfliers=False, legend=False, width=0.3, ax=ax42)
    sns.boxplot(x='Diagnosis', y='Metric Value', hue='Diagnosis', data=IBD_substances_df, order=category_order2, 
                palette=diagnosis_color_map, showfliers=False, legend=False, width=0.3, ax=ax52)
    all_bacterial_types = [col for col in healthy_df if col != 't'] + [col for col in unhealthy_df if col != 't']
    palette = sns.color_palette("tab20", n_colors=len(all_bacterial_types))
    color_mapping = {bac: palette[i % len(palette)] for i, bac in enumerate(sorted(all_bacterial_types))}
    healthy_df.plot(x='t', kind='bar', stacked=True, width=1, ax=ax02, legend=False, color=color_mapping)
    unhealthy_df.plot(x='t', kind='bar', stacked=True, width=1, ax=ax22, legend=False, color=color_mapping)
    
    healthy_df = healthy_df.copy().T
    unhealthy_df = unhealthy_df.copy().T
    healthy_df.columns = healthy_df.iloc[0]
    unhealthy_df.columns = unhealthy_df.iloc[0]
    healthy_df = healthy_df.drop('t')
    unhealthy_df = unhealthy_df.drop('t')

    # Get color mapping
    pathway_list = list(healthy_df.index)
    classification_csv = os.path.join(gut_dir, 'real_data', 'IBD_MDB', 'Pathway_Classification.csv')
    final_color_mapping, mapping_df = get_color_mapping_from_pathway_classification(classification_csv, pathway_list)

    # Sort pathway names by category (using mapping_df)
    sorted_pathways = mapping_df.sort_values(by='category')['pathway'].tolist()
    healthy_df = healthy_df.loc[sorted_pathways]
    unhealthy_df = unhealthy_df.loc[sorted_pathways]

    # Plot Healthy
    healthy_df.T.plot(kind='bar', stacked=True, width=1, ax=ax02, legend=False,
                    color=[final_color_mapping[p] for p in healthy_df.index])

    # Plot Unhealthy
    unhealthy_df.T.plot(kind='bar', stacked=True, width=1, ax=ax22, legend=False,
                        color=[final_color_mapping[p] for p in unhealthy_df.index])
    
    
    #healthy_df, unhealthy_df, removed_pathways = remove_statistically_similar_pathways(healthy_df, unhealthy_df, p_threshold=0.8)
    #print(f'Removed pathways number: {len(removed_pathways)}')
    #print(f'Removed pathways: {len(removed_pathways)}')
    #color_map, order = assign_pathway_colors_with_significance(healthy_df, unhealthy_df, seed=42, min_diff=0.0, alpha=0.05)#assign_pathway_colors(healthy_df, unhealthy_df)
    #color_map, order = assign_colors_by_difference(healthy_df, unhealthy_df, p_value_threshold=0.05, seed=42)
    #color_map, order = assign_colors_abundance_difference(healthy_df, unhealthy_df, p_value_threshold=0.05)
    pathway_list =  list(set(healthy_df.columns).union(unhealthy_df.columns) - {'t'})
    classification_tsv = os.path.join(gut_dir, 'real_data', 'IBD_MDB', 'Pathway_Classification_Good.tsv')
    
    final_color_mapping, mapping_df, category_color_map, category_order = get_color_mapping_classification_final(classification_tsv, pathway_list, 
                                                                          healthy_df, unhealthy_df, palette=custom_cmap)
    #get_color_mapping_from_classification_tabsep(classification_tsv, pathway_list, palette=cmaps.grads_default)
    healthy_len = len(healthy_df.index)
    unhealthy_len = len(unhealthy_df.index)
    print(f'Category color dict: {category_color_map}')
    #raise
    #order = mapping_df.sort_values(by='category')['pathway'].tolist()
    # Assign category order
    mapping_df['category'] = pd.Categorical(mapping_df['category'], categories=category_order, ordered=True)

    # Now get the final ordered list of pathways
    order = mapping_df.sort_values(['category', 'pathway'])['pathway'].tolist()
    healthy_prepped = prepare_df(healthy_df, order) #prepare_presence_df_for_plot(healthy_df, order) 
    unhealthy_prepped = prepare_df(unhealthy_df, order) #prepare_presence_df_for_plot(unhealthy_df, order)
    healthy_prepped.plot(kind='bar', stacked=True, color=[final_color_mapping[p] for p in healthy_prepped.columns],
                        legend=False, ax=ax02, width=1)
    unhealthy_prepped.plot(kind='bar', stacked=True, color=[final_color_mapping[p] for p in unhealthy_prepped.columns], 
                        legend=False, ax=ax22, width=1)
    
    '''
    ax01.set_xlabel(r'Time', labelpad=2, fontsize=10)
    ax00.set_xlabel(r'Time', labelpad=2, fontsize=10)
    #ax21.set_xlabel(r'Time', labelpad=2, fontsize=10)
    
    ax00.set_ylabel(r'Biomass', labelpad=2, fontsize=10)
    ax10.set_ylabel(r'Shannon', labelpad=2, fontsize=10)
    ax20.set_ylabel(r'Enzymes', labelpad=2, fontsize=10)
    ax30.set_ylabel(r'Substances', labelpad=2, fontsize=10)
    
    for ax in [ax00,ax10,ax20,ax30,ax01,ax31,ax41,ax51,ax32,ax42,ax52]:
        despine(ax)      

    ax30.set_xlabel('Time', fontsize=10)
    #for ax in [ax31,ax41,ax51,ax61]:
    #    ax.tick_params(labelleft=False)
    for ax in [ax31,ax41,ax51, ax32, ax42, ax52]:
        ax.set_xlabel('')
        ax.set_ylabel('')
    for ax in [ax31,ax41, ax32, ax42]:
        ax.set_xticklabels([])
    for ax in [ax00,ax10,ax20,ax30]:
        ax.tick_params(axis='both', labelsize=8)
    for ax in [ax31,ax41,ax51,ax32,ax42,ax52]:
        ax.tick_params(axis='y', which='major', labelsize=8)
        ax.tick_params(axis='x', which='major', labelsize=8)
    for ax in [ax02,ax22]:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)  
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel('Time', fontsize=10, labelpad=2)
    ax02.set_ylabel('Biomass fraction', fontsize=10)
    
    #ax21.tick_params(labelbottom=True)
    for i,ax in enumerate([ax00, ax10, ax20, ax30]):
        # Set the x-axis limits to show only data from 3 to 63
        ax.set_xlim(3, 60)

        # Manually set the tick positions (corresponding to the original data values)
        tick_positions = [3, 13, 23, 33, 43, 53]  # These are the actual data points in the original data

        # Set the new tick labels to start from 0
        tick_labels = [0, 10, 20, 30, 40, 50]  # These are the custom labels you want to show

        # Update the x-ticks and their corresponding labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        if i>0 and i<3:
            ax.tick_params(labelbottom=False)
    
    for ax,p in zip([ax31,ax41,ax51],[p_value_shannon,p_value_enzymes,p_value_substances]):
        print(p)
        ax.text(0,1.1, f'p-value={p:.3e}', fontsize=6, verticalalignment='center', transform=ax.transAxes)
    for ax,p in zip([ax32,ax42,ax52],[p_shannon,p_enzymes,p_substances]):
        print(p)
        ax.text(0,1.1, f'p-value={p:.3e}', fontsize=6, verticalalignment='center', transform=ax.transAxes)
    
    
    ax00.set_yticks([0,1e12,2e12, 3e12])
    ax00.set_ylim([0,3.4e12])
    
    #ax01.set_yticks([0,1e12,2e12, 3e12])
    #ax01.set_yticks([0,1e12,2e12])
    #ax21.set_yticks([0,2e12,4e12])
    ax10.set_ylim((0.8,3.4))
    ax10.set_yticks([1,2,3])
    ax20.set_ylim([15,100])
    ax20.set_yticks([25,50,75,100])
    ax30.set_ylim([16,32])
    ax30.set_yticks([18,24,30])
    ax51.set_ylim([10,31])
    ax51.set_yticks([10,20,30])
    ax02.set_ylim([0,1])
    ax22.set_ylim([0,1])
    ax42.set_ylim([470,1530])
    ax42.set_yticks([500,1000,1500])


    #ax40.tick_params(axis='both', labelsize=8)
    #ax21.tick_params(axis='both', labelsize=8)

    #ax00.set_xticks([0,20,40,60])
    #ax01.set_xticks([0,20,40,60])
    #ax10.set_yticks([0,1,2,3])
    
    #ax20.set_yticks([0,60,120])
    '''
    for ax in [ax02, ax22]:
        ax.set_xlabel("Time", fontsize=10)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='both', which='both', left=False, labelleft=False, bottom=False, labelbottom=False) 
        ax.set_ylim([0,1])
    for ax, (l_lim, h_lim) in zip([ax01,ax11], [(48,58),(23,33)]):  # or [ax00, ax01, ...] if multiple plots
        # 1. Define your desired tick labels (e.g., years)
        desired_ticks = np.arange(l_lim, h_lim+1, 5)
        wanted_tvec = category_df['t']
        wanted_tvec = wanted_tvec[(wanted_tvec>=l_lim) & (wanted_tvec<=h_lim)]
        # 3. Compute scaled tick positions (indices on x-axis)
        data_min = wanted_tvec.min()
        data_max = wanted_tvec.max()
        scaled_tick_positions = ((desired_ticks - data_min) / (data_max - data_min)) * (len(wanted_tvec) - 1)
        scaled_tick_positions = scaled_tick_positions.astype(int)  # Must be integer positions

        # 4. Apply to the axes
        ax.set_xticks(scaled_tick_positions)
        ax.set_xticklabels([int(x-3) for x in desired_ticks], rotation=0, ha='center')
        ax.set_xlabel("Time", fontsize=10)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)  
        ax.set_yticks([])
        ax.set_ylim([0,1])
   
    ax01.set_ylabel("Pathway fraction", fontsize=10)
    ax02.set_ylabel("Pathway fraction", fontsize=10)

    #rootpath_img = os.path.join(rootpath_img, 'model_fig_IBD_participants')
    if not os.path.exists(rootpath_img):
        os.makedirs(rootpath_img)

    longname = f'model_fig_grouped_pathways_classification_good_spread_{healthy_index}_{IBD_index}_{healthy_len}_{unhealthy_len}_final'
    image_name = os.path.join(rootpath_img, longname)

    plt.subplots_adjust(top=0.99, bottom=0.085, left=0.07, right=0.98, hspace=0.0, wspace=0.0)
    #plt.savefig(image_name, format='svg', transparent=False, dpi=600)
    
    image_name = os.path.join(rootpath_img, image_name+'.png')
    plt.savefig(image_name, format='png', transparent=False, dpi=1200)
    #plt.show()
    


def get_df_from_stats(stats_dict, unhealthy_list, healthy_list, subset_fraction, healthy_normalized):
    def filter_diagnosis(diagnosis):
        if diagnosis == 'healthy':
            diagnosis = 'Healthy'
        elif diagnosis == 'unhealthy':
            diagnosis = 'Unhealthy'
        return diagnosis
    category_order = ['Healthy'] 

    if healthy_list is not None:
        category_order += healthy_list
    category_order += ['Unhealthy']
    if unhealthy_list is not None:
        category_order += unhealthy_list

    # Prepare data for the current measure
    data = []
    groups = stats_dict[subset_fraction]
    groups = {filter_diagnosis(k): v for k,v in groups.items()}

    for group, group_data in groups.items():
        for sample in group_data:
            value = sample[3]  # Extract the specific measure for each sample, they are: ['percent_positive', 'sum_positive', 'sum_negative', 'pos_neg_diff']
            
            data.append([subset_fraction, group, value])

    # Create DataFrame for plotting
    df = pd.DataFrame(data, columns=['Subset Fraction', 'Group', 'rho'])
    # Normalize data if requested
    if healthy_normalized:
        # Calculate the median of the Healthy group
        healthy_median = df[df['Group'] == 'Healthy']['rho'].median()
        
        # Subtract the median from all values
        df['rho'] = df['rho'] - healthy_median

    # Iterate over each subset_fraction to check if it has other subsets
    # Get the available groups for the specific subset_fraction
    available_groups = list(groups.keys())
    # Filter the predefined category order to match only the available groups
    filtered_category_order = [cat for cat in category_order if cat in available_groups]
    # Create DataFrame for the specific subset fraction
    df_fraction = df[df['Subset Fraction'] == subset_fraction].copy()  # Ensure .copy() to avoid SettingWithCopyWarning

    # Ensure that the 'Group' column follows the filtered order
    df_fraction['Group'] = pd.Categorical(df_fraction['Group'], categories=filtered_category_order, ordered=True)
    return df_fraction
def get_data(bootstrap_dir, unhealthy_list=None, healthy_list=None, subset_fraction='0.8', max_iters=500, healthy_normalized=False, overwrite=False):
    stats_dict = get_stats_dict(bootstrap_dir, max_iters, overwrite=overwrite)
    return get_df_from_stats(stats_dict, unhealthy_list, healthy_list, subset_fraction, healthy_normalized)

def get_model_net_interaction(rootpath, specific=False):
    
    metadata_dict = MDA.get_metadata_dict(rootpath, overwrite=False, sample_size=200)
    data_df = MDA.get_data_df(rootpath, data_type='taxonomy', overwrite=False, sample_size=200)
    full_data_df = MDA.get_full_data_dict(metadata_dict, data_df)
    
    if not specific:
        diagnosis_rho_dict = {diagnosis: [] for diagnosis in ['H', 'U']}
    else:
        diagnosis_rho_dict = {diagnosis: [] for diagnosis in ['H', 'U', 'U0', 'U1', 'U2', 'U3']}
    
    for participant_id, samples in full_data_df.items():
        for sample in samples:
            rho = sample['rho']
            if rho <= 0:
                diagnosis_rho_dict['H'].append(rho)
            else:
                diagnosis_rho_dict['U'].append(rho)
                if specific:
                    if rho <= 0.25:
                        diagnosis_rho_dict['U0'].append(rho)
                    elif rho > 0.25 and rho <= 0.5:
                        diagnosis_rho_dict['U1'].append(rho)
                    elif rho > 0.5 and rho < 0.75:
                        diagnosis_rho_dict['U2'].append(rho)
                    elif rho > 0.75:
                        diagnosis_rho_dict['U3'].append(rho)
    
    # Prepare data for DataFrame
    data = {'Group': [], 'rho': []}
    
    for key, values in diagnosis_rho_dict.items():
        data['Group'].extend([key] * len(values))
        data['rho'].extend(values)
    
    df = pd.DataFrame(data)
    
    # Compute Mann-Whitney U test only between Healthy and Unhealthy
    _, p_value = mannwhitneyu(diagnosis_rho_dict['H'], diagnosis_rho_dict['U'], alternative='two-sided')
    
    return df, p_value

def plot_interaction_fig(gut_dir):
    '''def get_stats_dict(bootstrap_dir):
        try:
            wanted_file = os.path.join(bootstrap_dir, 'processed', 'stats_dict2.pkl')
            with open(wanted_file, 'rb') as fin:
                stats_dict = pickle.load(fin)
        except:
            wanted_file = os.path.join(bootstrap_dir, 'processed', 'stats_dict.pkl')
            with open(wanted_file, 'rb') as fin:
                stats_dict = pickle.load(fin)
        return stats_dict
    '''
    

    CRC_bootstrap_path = os.path.join(gut_dir, 'real_data', 'CRC_Yachida_NatMed', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    IBS_bootstrap_path = os.path.join(gut_dir, 'real_data', 'IBS_Mars_Cell', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    IBD_bootstrap_path = os.path.join(gut_dir, 'real_data', 'IBD_MDB', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    CD_bootstrap_path = os.path.join(gut_dir, 'real_data', 'CD_Ferretti_Elife', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    Model_bootstrap_path = os.path.join(gut_dir, 'real_data', 'ModelData', 'invasion', 'E_D_0', 'fraction_nl_6_0_1.1',
                                        'bootstrap_HE-S', 'BacFrac_1.0_FullsetFrac_1.0', 'coarsed_True', 'samples_1000')
    Model_rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)
    Model_bootstrap_path2 = os.path.join(gut_dir, 'real_data', 'ModelData', 'invasion', 'E_D_0', 'fraction_nl_6_0_1.1',
                                        'bootstrap_HE-S', 'BacFrac_1.0_FullsetFrac_1.0', 'coarsed_True', 'samples_1000')
    #bootstrap2 is where I have data inferred for the different bins of net interactions. In order to do the inferred vs net plot
    base_color_map = {'Healthy': '#0078B9',
                      'H': '#0078B9',
                      'Unhealthy': '#EA0017',
                      'U': '#EA0017'}

    CRC_color_map = {"MP": '#F9B3B3',  # Lightest red
                     "Stage_0": "#F48A8A",  # Light red
                     "Stage_I_II": "#E03D3D",  # Dark red
                     "Stage_III_IV": "#A70000",  # Darkest Red
                     "HS": "#D4C2E5"  # Purple
                     }
    #Model_color_map = {"Unhealthy_I": "#F48A8A",
    #                   "Unhealthy_II": "#C10000"
    #                 }
    Model_color_map = {"Healthy_0": "#66AEDD",
                       "Healthy_1": "#005080",
                       "Unhealthy_0": '#F9B3B3',
                       "Unhealthy_1": "#F48A8A",
                       "Unhealthy_2": "#E03D3D",
                       "Unhealthy_3": "#A70000"
                    }
    CRC_color_map.update(base_color_map)
    Model_color_map.update(base_color_map)
    
    
    fig = plt.figure(figsize = (12.1*0.393701, 10.5*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(7,2, height_ratios=[0.38, 0.12, 0.10333, 0.1, 0.10333, 0.1, 0.10333], width_ratios=[0.3, 0.7])
    
    ax00 = fig.add_subplot(gs[0,0])
    ax01 = fig.add_subplot(gs[0,1])
    ax10 = fig.add_subplot(gs[2,0])
    ax20 = fig.add_subplot(gs[4,0])
    ax30 = fig.add_subplot(gs[6,0])
    ax11 = fig.add_subplot(gs[2:,1])

    CRC_df = get_data(CRC_bootstrap_path,['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV'],None, subset_fraction='0.8', healthy_normalized=True)
    Model_df = get_data(Model_bootstrap_path, ['Unhealthy_0', 'Unhealthy_1', 'Unhealthy_2', 'Unhealthy_3'], None, '0.8', healthy_normalized=True)

    IBD_df = get_data(IBD_bootstrap_path, None, None, '0.8', healthy_normalized=True)
    IBS_df = get_data(IBS_bootstrap_path, None, None, '0.8', healthy_normalized=True)
    CD_df = get_data(CD_bootstrap_path, None, None, '0.8', healthy_normalized=True)


    sns.boxplot(x='Group', y='rho', hue='Group', data=Model_df, palette=Model_color_map, ax=ax01, width=0.55, showfliers=False)
    sns.boxplot(x='Group', y='rho', hue='Group', data=CRC_df, palette=CRC_color_map, ax=ax11, width=0.55, showfliers=False)
    sns.boxplot(x='Group', y='rho', hue='Group', data=IBD_df, palette=base_color_map, ax=ax10, width=0.4, showfliers=False)
    sns.boxplot(x='Group', y='rho', hue='Group', data=IBS_df, palette=base_color_map, ax=ax20, width=0.4, showfliers=False)
    sns.boxplot(x='Group', y='rho', hue='Group', data=CD_df, palette=base_color_map, ax=ax30, width=0.4, showfliers=False)
    diagnosis_color_map = {
        "H": '#5F60F5',  # Blue
        "U": "#ED3A32"  # Red
    }
    
    # Plot net interaction
    net_interactions_df, p_value = get_model_net_interaction(Model_rootpath)
    category_order = ['H', 'U']
    sns.boxplot(x='Group', y='rho', hue='Group', data=net_interactions_df, order=category_order, 
                palette=base_color_map, showfliers=False, legend=False, width=0.35, ax=ax00)

    
    for ax in [ax00, ax01, ax10, ax20, ax30, ax11]:
        despine(ax)
        if ax.get_legend():
            ax.legend_.remove()
        ax.set_xlabel('')
        ax.set_ylabel('')
    for ax in [ax20, ax01]:
        ax.set_ylabel('Ecological Balance', labelpad=1.3, fontsize=10)
    
    ax11.set_ylabel('Ecological Balance', labelpad=-1.8, fontsize=10)
    ax00.set_ylabel('Net Interaction', labelpad=1.3, fontsize=10)
    ax11.set_xticks([0, 1, 2, 3, 4, 5])
    ax11.set_xticklabels(['Healthy', 'CRC', 'CRC-0', 'CRC-1', 'CRC-2', 'CRC-3'], rotation=30)
    ax01.set_xticks([0, 1, 2, 3, 4, 5])
    ax01.set_xticklabels(['FS', 'CS', 'CS-0', 'CS-1', 'CS-2', 'CS-3'], rotation=30)
    for ax in [ax00, ax10, ax20, ax30]:
        ax.set_xticks([0,1])
    ax00.set_xticklabels(['FS', 'CS'])
    ax10.set_xticklabels(['Healthy', 'IBD'])
    ax20.set_xticklabels(['Healthy', 'IBS'])
    ax30.set_xticklabels(['Healthy', 'CDI'])
    ax20.set_yticks([0,0.03])
    ax30.set_yticks([-0.1,0.1])
    '''
    for ax in [ax20,ax21,ax22]:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['H', 'U'])
    '''
    '''
    
    ax1.set_yticks([0.5,0.6,0.7,0.8,0.9,1.0])
    ax2.set_yticks([0.65,0.75,0.85])
    ax3.set_yticks([0.84,0.87,0.9])
    ax4.set_yticks([0.3,0.4,0.5,0.6])
    ax0.set_ylim((0.15,0.54))
    ax1.set_ylim((0.55,0.95))
    ax2.set_ylim((0.65,0.85))
    ax3.set_ylim((0.84,0.9))
    ax4.set_ylim((0.3,0.6))
    '''

    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'interactions_fig.svg')

    plt.subplots_adjust(top=0.98, bottom=0.112, left=0.13, right=0.99, hspace=0.0, wspace=0.35)
    #plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    plt.show()

def plot_interaction_fig2(gut_dir):
    def print_statistical_tests(data, dataset_name, x_col='Group', y_col='rho'):
        """Computes and prints statistical test results for specified category pairs."""
        if dataset_name=='Model_net':
            pairs = [('H', 'U'), ('U0', 'U1'), ('U0', 'U2'), ('U0', 'U3'), ('U1', 'U2'), ('U1', 'U3'), ('U2', 'U3')]
        elif dataset_name=='Model_ENBI':
            pairs = [('Healthy', 'Unhealthy'), ('Unhealthy_0', 'Unhealthy_1'), ('Unhealthy_0', 'Unhealthy_2'), 
                     ('Unhealthy_0', 'Unhealthy_3'), ('Unhealthy_1', 'Unhealthy_2'), ('Unhealthy_1', 'Unhealthy_3'), 
                     ('Unhealthy_2', 'Unhealthy_3')]
        elif dataset_name=='CRC': 
            pairs = [('Healthy', 'Unhealthy'), ('MP', 'Stage_0'), ('MP', 'Stage_I_II'), ('MP', 'Stage_III_IV'), 
                     ('Stage_0', 'Stage_I_II'), ('Stage_0', 'Stage_III_IV'), ('Stage_I_II', 'Stage_III_IV')]
        elif dataset_name in ['IBD', 'IBS', 'CDI']:
            pairs = [('Healthy', 'Unhealthy')]
        for pair in pairs:
            group1 = data[data[x_col] == pair[0]][y_col]
            group2 = data[data[x_col] == pair[1]][y_col]
            if not group1.empty and not group2.empty:
                stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                delta = cliffs_delta(group1, group2)
                print(f"{pair[0]} vs {pair[1]}: Cliffs delta={delta}, p-value={p_value}")

    def analyze_interaction_datasets(datasets):
        """Runs statistical analysis on multiple datasets and prints results."""
        for dataset_name, data in datasets.items():
            print(f"\nStatistical tests for {dataset_name}:")
            print_statistical_tests(data, dataset_name)
    CRC_bootstrap_path = os.path.join(gut_dir, 'real_data', 'CRC_Yachida_NatMed', 'bootstrap2_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    IBS_bootstrap_path = os.path.join(gut_dir, 'real_data', 'IBS_Mars_Cell', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    IBD_bootstrap_path = os.path.join(gut_dir, 'real_data', 'IBD_MDB', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    CD_bootstrap_path = os.path.join(gut_dir, 'real_data', 'CD_Ferretti_Elife', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    Model_bootstrap_path = os.path.join(gut_dir, 'real_data', 'ModelData', 'invasion', 'E_D_0', 'fraction_nl_6_0_1.1',
                                        'bootstrap_HE-S', 'BacFrac_1.0_FullsetFrac_1.0', 'coarsed_True', 'samples_1000')
    Model_rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)
    Model_bootstrap_path2 = os.path.join(gut_dir, 'real_data', 'ModelData', 'invasion', 'E_D_0', 'fraction_nl_6_0_1.1',
                                        'bootstrap_HE-S', 'BacFrac_1.0_FullsetFrac_1.0', 'coarsed_True', 'samples_1000')
    #bootstrap2 is where I have data inferred for the different bins of net interactions. In order to do the inferred vs net plot
    base_color_map = {'Healthy': '#0078B9',
                      'H': '#0078B9',
                      'Unhealthy': '#EA0017',
                      'U': '#EA0017'}

    CRC_color_map = {"MP": '#F9B3B3',  # Lightest red
                     "Stage_0": "#F48A8A",  # Light red
                     "Stage_I_II": "#E03D3D",  # Dark red
                     "Stage_III_IV": "#A70000",  # Darkest Red
                     "HS": "#D4C2E5"  # Purple
                     }
    #Model_color_map = {"Unhealthy_I": "#F48A8A",
    #                   "Unhealthy_II": "#C10000"
    #                 }
    Model_color_map = {"Healthy_0": "#66AEDD",
                       "Healthy_1": "#005080",
                       "Unhealthy_0": '#F9B3B3',
                       "Unhealthy_1": "#F48A8A",
                       "Unhealthy_2": "#E03D3D",
                       "Unhealthy_3": "#A70000",
                       "U0": '#F9B3B3',
                       "U1": "#F48A8A",
                       "U2": "#E03D3D",
                       "U3": "#A70000"
                      }
    CRC_color_map.update(base_color_map)
    Model_color_map.update(base_color_map)
    
    
    fig = plt.figure(figsize = (12.1*0.393701, 12*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(4,5, height_ratios=[0.27, 0.27, 0.27, 0.19], width_ratios=[0.26, 0.11, 0.26, 0.11, 0.26])
    
    ax0 = fig.add_subplot(gs[0,:])
    ax1 = fig.add_subplot(gs[1,:])
    ax2 = fig.add_subplot(gs[2,:])
    ax30 = fig.add_subplot(gs[3,0])
    ax31 = fig.add_subplot(gs[3,2])
    ax32 = fig.add_subplot(gs[3,4])

    CRC_df = get_data(CRC_bootstrap_path,['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV'],None, subset_fraction='0.8', healthy_normalized=True)
    Model_df = get_data(Model_bootstrap_path, ['Unhealthy_0', 'Unhealthy_1', 'Unhealthy_2', 'Unhealthy_3'], None, '0.8', healthy_normalized=True)

    IBD_df = get_data(IBD_bootstrap_path, None, None, '0.8', healthy_normalized=True)
    IBS_df = get_data(IBS_bootstrap_path, None, None, '0.8', healthy_normalized=True)
    CD_df = get_data(CD_bootstrap_path, None, None, '0.8', healthy_normalized=True)


    sns.boxplot(x='Group', y='rho', hue='Group', data=Model_df, palette=Model_color_map, ax=ax1, width=0.35, showfliers=False)
    sns.boxplot(x='Group', y='rho', hue='Group', data=CRC_df, palette=CRC_color_map, ax=ax2, width=0.35, showfliers=False)
    sns.boxplot(x='Group', y='rho', hue='Group', data=IBD_df, palette=base_color_map, ax=ax30, width=0.4, showfliers=False)
    sns.boxplot(x='Group', y='rho', hue='Group', data=IBS_df, palette=base_color_map, ax=ax31, width=0.4, showfliers=False)
    sns.boxplot(x='Group', y='rho', hue='Group', data=CD_df, palette=base_color_map, ax=ax32, width=0.4, showfliers=False)
    diagnosis_color_map = {
        "H": '#5F60F5',  # Blue
        "U": "#ED3A32"  # Red
    }
    
    # Plot net interaction
    net_interactions_df, p_value = get_model_net_interaction(Model_rootpath, specific=True)
    category_order = ['H', 'U', 'U0', 'U1', 'U2', 'U3']
    sns.boxplot(x='Group', y='rho', hue='Group', data=net_interactions_df, order=category_order, 
                palette=Model_color_map, showfliers=False, legend=False, width=0.35, ax=ax0)

    datasets = {
    'Model_net': net_interactions_df,
    'Model_ENBI': Model_df,
    'CRC': CRC_df,
    'IBD': IBD_df,
    'IBS': IBS_df,
    'CDI': CD_df,
    }

    #analyze_interaction_datasets(datasets)

    for ax in [ax0, ax1, ax2, ax30, ax31, ax32]:
        despine(ax)
        if ax.get_legend():
            ax.legend_.remove()
        ax.set_xlabel('')
        ax.set_ylabel('')
    for ax in [ax1, ax2]:
        ax.set_ylabel('ENBI', labelpad=1.5, fontsize=10)
    
    ax30.set_ylabel('ENBI', labelpad=1, fontsize=10)
    ax0.set_ylabel('Net Interaction', labelpad=1.3, fontsize=10)
    ax2.set_xticks([0, 1, 2, 3, 4, 5])
    ax2.set_xticklabels(['Healthy', 'CRC', 'CRC-0', 'CRC-1', 'CRC-2', 'CRC-3'], rotation=0)
    ax1.set_xticks([0, 1, 2, 3, 4, 5])
    ax1.set_xticklabels(['FS', 'CS', 'CS-0', 'CS-1', 'CS-2', 'CS-3'], rotation=0)
    for ax in [ax30, ax31, ax32]:
        ax.set_xticks([0,1])
    ax0.set_xticks([0, 1, 2, 3, 4, 5])
    ax0.set_xticklabels(['FS', 'CS', 'CS-0', 'CS-1', 'CS-2', 'CS-3'])
    ax30.set_xticklabels(['Healthy', 'IBD'])
    ax31.set_xticklabels(['Healthy', 'IBS'])
    ax32.set_xticklabels(['Healthy', 'CDI'])
    ax31.set_yticks([0,0.03])
    ax32.set_yticks([0.0,0.1])
    
    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'interactions_fig2.svg')

    plt.subplots_adjust(top=0.98, bottom=0.07, left=0.13, right=0.99, hspace=0.55, wspace=0.0)
    plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    #plt.show()

def plot_rho_LIMITS(gut_dir, E_D_type, tradeoff_data, avg=True):
    def extract_data_from_files(LIMITS_path, metadata_dict, model=False):
        """
        Extracts interaction matrices data from files in the specified directory.
        
        Parameters:
        - LIMITS_path: Path to the directory containing the interaction matrix files.

        Returns:
        - data_dict: Dictionary containing interaction matrices for 'median' and 'avg' calculations.
        """
        def filter_diagnosis(diagnosis):
            if diagnosis in ['CD', 'UC', 'IBD', 'Unhealthy']:
                diagnosis = 'Unhealthy'
            elif diagnosis in ['nonIBD', 'Healthy']:
                diagnosis = 'Healthy'
            else:
                print(f'Diagnosis {diagnosis} is not valid.')
                raise
            return diagnosis
        data_dict_median = {}
        data_dict_avg = {}

        for filename in os.listdir(LIMITS_path):
            if filename.startswith("interaction_mats") and filename.endswith(".pkl") and "coarsed" not in filename:
                splitted_name =  filename[:-4].split('_')
                if not model:
                    _, _, max_species, max_samples, error_threshold, n_bagging = splitted_name
                elif model and len(splitted_name) == 7: 
                    _, _, max_species, max_samples, min_ratio, error_threshold, n_bagging = splitted_name
                else:
                    continue
                if max_species == '10':
                    continue

                error_threshold = float(error_threshold)
                if error_threshold<1:
                    continue
                file_path = os.path.join(LIMITS_path, filename)
                with open(file_path, 'rb') as fin:
                    interaction_matrices_median_dict, interaction_matrices_avg_dict = pickle.load(fin)
                # Initialize dictionary for each max_species if not already present
                for data_dict, matrices in zip((data_dict_median, data_dict_avg), 
                                            (interaction_matrices_median_dict, interaction_matrices_avg_dict)):
                    if max_species not in data_dict:
                        data_dict[max_species] = {'Healthy': {}, 'Unhealthy': {}}
                    
                    for patient_id, matrix in matrices.items():
                        if model: # in the model, the metadata dict has diagnosis as 4th
                            diagnosis = filter_diagnosis(metadata_dict[patient_id][0][3])
                        else:
                            diagnosis = filter_diagnosis(metadata_dict[patient_id][0][2])
                        if patient_id not in data_dict[max_species][diagnosis]:
                            data_dict[max_species][diagnosis][patient_id] = []
                        # Append the matrix and error_threshold for later processing
                        data_dict[max_species][diagnosis][patient_id].append((matrix, error_threshold))
        
        return data_dict_median, data_dict_avg
    def process_interaction_matrices(data_dict):
        """
        Processes the interaction matrices to calculate rho and averages across error_thresholds for each max_species.
        Parameters:
        - data_dict: Dictionary containing interaction matrices grouped by max_species and patient groups.
        Returns:
        - final_data: Dictionary containing averaged quantities for each max_species and patient group.
        """
        final_data = {}

        for max_species_key, max_species_dict in data_dict.items(): # data_dict[max_species]['Healthy' if '-H' in patient_id else 'Unhealthy'].append((matrix, error_threshold))
            final_data[max_species_key] = {'Healthy': {}, 'Unhealthy': {}}
            for diagnosis in ['Healthy', 'Unhealthy']:
                for patient_id, matrices_vec in max_species_dict[diagnosis].items():
                    rhos = []
                    errors = []
                    for matrix, error_threshold in matrices_vec:
                        # Set diagonal to 0
                        np.fill_diagonal(matrix, 0)
                        # Calculate sum of positive and negative values
                        pos_sum = matrix[matrix > 0].sum()
                        neg_sum = matrix[matrix < 0].sum()
                        rho = (pos_sum + neg_sum) / (pos_sum - neg_sum)
                        rhos.append(rho)
                        errors.append(error_threshold)
                    #print(f'Patient: {patient_id} has {rhos}')
                    #if len(rhos) == 2:
                    #    print(f'Max species {max_species_key} for patient {patient_id} has only 2 rhos for errors {errors}')
                    if np.isnan(np.array(rhos)).all():
                        print(f'{patient_id} is full of nans for max_species {max_species_key}')
                    else:
                        final_data[max_species_key][diagnosis][patient_id] = np.nanmean(rhos)
        
        return final_data
    def get_rho_data(LIMITS_path, metadata_dict, model=False):
        data_dict_median, data_dict_avg = extract_data_from_files(LIMITS_path, metadata_dict, model)
        final_data_median = process_interaction_matrices(data_dict_median)
        final_data_avg = process_interaction_matrices(data_dict_avg)
        #final_data_median = {}
        return final_data_median, final_data_avg
    
    def prepare_data_for_plotting(final_data_dict):
        """
        Converts the nested dictionary structure into a DataFrame for plotting.
        
        Parameters:
        - final_data_dict: Nested dictionary with keys as max_species, then dictionaries for 'healthy' and 'unhealthy' groups,
                        and values as rho values for each patient_id.
                        
        Returns:
        - plot_df: DataFrame ready for plotting with columns 'max_species', 'diagnosis', 'patient_id', and 'rho'.
        """
        data_rows = []
        for max_species, diagnosis_data in final_data_dict.items():
            for diagnosis, patient_data in diagnosis_data.items():
                for patient_id, rho in patient_data.items():
                    # Add rows for specific max_species
                    data_rows.append({
                        'max_species': max_species,
                        'diagnosis': diagnosis,
                        'patient_id': patient_id,
                        'rho': rho
                    })
                    # Add rows for combined category "Combined"
                    data_rows.append({
                        'max_species': 'Combined',
                        'diagnosis': diagnosis,
                        'patient_id': patient_id,
                        'rho': rho
                    })

        # Convert to DataFrame
        plot_df = pd.DataFrame(data_rows)
        # Sort by max_species for plotting in ascending order
        plot_df['max_species'] = pd.Categorical(plot_df['max_species'], 
                                                categories=sorted(final_data_dict.keys()) + ['Combined'],
                                                ordered=True)
        plot_df = plot_df.sort_values(by='max_species')
        
        return plot_df
    def perform_pvalue_tests(plot_df, id_str, test, significant_p_value=1):
        """
        Performs a Mann-Whitney U test for each max_species between the healthy and unhealthy groups
        and prints those with p-values < 0.05.
        
        Parameters:
        - plot_df: DataFrame containing 'max_species', diagnosis', and 'rho' columns.
        """
        significant_results = []
        
        for max_species in plot_df['max_species'].unique():
            # Split data into healthy and unhealthy groups for the current max_species
            healthy_rho = plot_df[(plot_df['max_species'] == max_species) & (plot_df['diagnosis'] == 'Healthy')]['rho']
            unhealthy_rho = plot_df[(plot_df['max_species'] == max_species) & (plot_df['diagnosis'] == 'Unhealthy')]['rho']
            # Perform Mann-Whitney U test
            if test.lower() in ['mw', 'mann-whitney']:
                u_stat, p_value = mannwhitneyu(healthy_rho, unhealthy_rho, alternative='two-sided')
            elif test.lower() in ['kw', 'kruskal-wallis']:
                h_stat, p_value = kruskal(healthy_rho, unhealthy_rho)
            elif test.lower() in ['ks', 'kolmogorov-smirnov']:
                k_stat, p_value = ks_2samp(healthy_rho, unhealthy_rho)

            # Check if p-value is below the threshold
            if p_value < significant_p_value:
                significant_results.append((max_species, p_value))
        
        # Print significant results
        if significant_results:
            print(f"For {id_str} and test {test} significant max_species values (p < 0.05) with Mann-Whitney U test:")
            for max_species, p_value in significant_results:
                print(f"max_species: {max_species}, p-value: {p_value}")
        else:
            print(f"For {id_str} and test {test} no significant results with p < 0.05.")
    IBD_rootpath_data = os.path.join(gut_dir,'real_data','IBD_MDB')
    IBD_metadata_dict = IBD.get_metadata_dict(IBD_rootpath_data)
    #CD_rootpath_data = os.path.join(gut_dir,'real_data','CD_MDSINE_Genome_Biology')
    #CD_metadata_dict = CD.get_metadata_dict2(CD_rootpath_data)
    Model_rootpath_data = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)
    Model_metadata_dict = MDA.get_metadata_dict(Model_rootpath_data, overwrite=False, longitudinal=True, sample_size=500)

    #CD_LIMITS_path = os.path.join(gut_dir, 'results', 'real_data', 'CD_MDSINE_Genome_Biology', 'LIMITS')
    IBD_LIMITS_path = os.path.join(gut_dir, 'results', 'real_data', 'IBD_MDB', 'LIMITS')
    Model_LIMITS_path = os.path.join(Model_rootpath_data,'processed','LIMITS')
    
    #CD_final_data_median, CD_final_data_avg = get_rho_data(CD_LIMITS_path, CD_metadata_dict)
    IBD_final_data_median, IBD_final_data_avg = get_rho_data(IBD_LIMITS_path, IBD_metadata_dict)
    Model_final_data_median, Model_final_data_avg = get_rho_data(Model_LIMITS_path, Model_metadata_dict, model=True)
    
    #CD_median_df = prepare_data_for_plotting(CD_final_data_median)
    IBD_median_df = prepare_data_for_plotting(IBD_final_data_median)
    Model_median_df = prepare_data_for_plotting(Model_final_data_median)
    #CD_avg_df = prepare_data_for_plotting(CD_final_data_avg)
    IBD_avg_df = prepare_data_for_plotting(IBD_final_data_avg)
    Model_avg_df = prepare_data_for_plotting(Model_final_data_avg)
    
    fig = plt.figure(figsize = (20*0.393701, 8.5*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(2,1, height_ratios=[1/2,1/2],
                                width_ratios=[1])
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[1,0])
    #ax2 = fig.add_subplot(gs[2,0])
    base_color_map = {'Healthy': '#0078B9', 'Unhealthy': '#EA0017'}
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    if avg:        
        sns.boxplot(data=Model_avg_df, ax=ax0, x='max_species', y='rho', hue='diagnosis', palette=base_color_map, 
                    hue_order=['Healthy', 'Unhealthy'], showfliers=False, width=0.6, dodge=True)
        sns.boxplot(data=IBD_avg_df, ax=ax1, x='max_species', y='rho', hue='diagnosis', palette=base_color_map, 
                    hue_order=['Healthy', 'Unhealthy'], showfliers=False, width=0.6, dodge=True)
        #sns.boxplot(data=CD_avg_df, ax=ax2, x='max_species', y='rho', hue='diagnosis', palette=base_color_map, hue_order=['Healthy', 'Unhealthy'], showfliers=False)
        for test in ['mw']:
            perform_pvalue_tests(IBD_avg_df, 'IBD', test)
            perform_pvalue_tests(Model_avg_df, 'Model', test)
        image_name = os.path.join(image_dir, f'rho_LIMITS_avg_all_errors_final.svg')
    else:
        sns.boxplot(data=Model_median_df, ax=ax0, x='max_species', y='rho', hue='diagnosis', palette=base_color_map, 
                    hue_order=['Healthy', 'Unhealthy'], showfliers=False, width=0.6, dodge=True)
        sns.boxplot(data=IBD_median_df, ax=ax1, x='max_species', y='rho', hue='diagnosis', palette=base_color_map, 
                    hue_order=['Healthy', 'Unhealthy'], showfliers=False, width=0.6, dodge=True)
        #sns.boxplot(data=CD_median_df, ax=ax2, x='max_species', y='rho', hue='diagnosis', palette=base_color_map, hue_order=['Healthy', 'Unhealthy'], showfliers=False)
        for test in ['mw', 'ks']:
            perform_pvalue_tests(IBD_median_df, 'IBD', test)
            perform_pvalue_tests(Model_median_df, 'Model', test)
        image_name = os.path.join(image_dir, f'rho_LIMITS_median_all_errors.svg')

    
    for ax in [ax0, ax1]:
        despine(ax)
        ax.set_ylabel(r'$\mathdefault{\rho}$', labelpad=0, fontsize=10)
        ax.set_xlabel('Max species')
        #ax.tick_params(axis='x', rotation=45)
        if ax.get_legend():
            ax.legend_.remove()
        #ax.legend(loc='upper right', labels=['Healthy', 'Unhealthy'])
        ax.set_ylim([-1,1])
    
    ax0.set_xlabel('')  # Remove x-axis label
    ax0.set_xticklabels([])  # Remove x tick labels
    ax1.set_xlabel('Number of species')

    
    plt.subplots_adjust(top=0.88, bottom=0.15, left=0.06, right=0.99, hspace=0.45, wspace=0.0)
    plt.savefig(image_name, format='svg', transparent=False, dpi=600)
    #plt.show()

def get_functions_categories(group_size, num_substances, p_str_to_p_id_dict=None):
    """
    Genera un diccionario con categorías de rutas entre sustancias agrupadas en grupos.
    
    :param group_size: Tamaño de cada grupo (excepto el primer grupo que es solo la sustancia 0).
    :param num_substances: Número total de sustancias.
    :return: Diccionario con las rutas categorizadas.
    """
    # Crear los grupos
    groups = []
    groups.append([0])  # Primer grupo con solo la sustancia 0
    for start in range(1, num_substances, group_size):
        end = min(start + group_size, num_substances)
        groups.append(list(range(start, end)))

    # Crear las categorías
    p_categories_to_str_dict = {}
    category_id = 0

    # Paso 1: Saltos desde el grupo 0 hacia todos los demás grupos
    for j in range(1, len(groups)):
        p_categories_to_str_dict[str(category_id)] = [
            f'{x}->{y}' for x in groups[0] for y in groups[j]
        ]
        category_id += 1

    # Paso 2: Procesar los demás grupos en orden
    for i in range(1, len(groups)):
        # Saltos dentro del mismo grupo
        p_categories_to_str_dict[str(category_id)] = [
            f'{x}->{y}' for x in groups[i] for y in groups[i] if x < y
        ]
        category_id += 1

        # Saltos hacia otros grupos
        for j in range(i + 1, len(groups)):  # Solo hacia grupos posteriores
            p_categories_to_str_dict[str(category_id)] = [
                f'{x}->{y}' for x in groups[i] for y in groups[j]
            ]
            category_id += 1
    p_str_to_category_dict = get_inverse_dict(p_categories_to_str_dict)
    p_str_to_category_dict = {k:v[0] for k,v in p_str_to_category_dict.items()}
    p_id_to_category_dict = None
    if p_str_to_p_id_dict is not None:
        p_id_to_category_dict = {p_str_to_p_id_dict[p_str]: v for p_str,v in p_str_to_category_dict.items()}
    
    return p_id_to_category_dict, p_str_to_category_dict

def get_taxonomy_and_functions_data(folder_path, sample_hours, p_str_to_category_dict, overwrite=False):
    wanted_file = os.path.join(folder_path, 'processed', f'taxonomy_and_functions_data_{sample_hours}.pkl')
    t_vec = get_tvec(folder_path, sample_hours, wanted_tvec='t')
    D_mat = DA.get_D_mat(folder_path)
    pathways_strs = DA.get_pathways_strs(D_mat)
    if not overwrite and os.path.isfile(wanted_file): 
        with open(wanted_file, 'rb') as fin:
            (rel_D_B_tvec, rel_B_df) = pickle.load(fin)
    else:
        # Get necessary objects
        B_tvec = get_tvec(folder_path, sample_hours, wanted_tvec='B')
        D_tvec = get_tvec(folder_path, sample_hours, wanted_tvec='D')
        
        pathways_D = D_mat[~np.isnan(D_mat)]
        B_types_dict = get_B_types_dict(folder_path, sample_hours)
        
        # Get taxonomy
        B_df = pd.DataFrame({'t': t_vec})
        if len(t_vec) != len(set(t_vec)):
            print("The t_vec has values repeated. This would cause problems in the merging of the dataframes right after in the code. Solve that before proceeding")
            raise
        duplicate_columns = False
        
        for Type, (t_init, t_end, type_functional_identity, B_type_t) in B_types_dict.items():
            wanted_t = t_vec[(t_vec>=t_init) & (t_vec<t_end)]
            aux_df = pd.DataFrame({'t': wanted_t, f'{type_functional_identity}': B_type_t})
            if type_functional_identity in B_df.columns: # If the type_functional_identity is already inside we will get duplicate columns in the merge
                duplicate_columns = True
            B_df = B_df.merge(aux_df, how='left', on='t')

            # Handle duplicate columns with the same type_functional_identity               
            if duplicate_columns:
                col_x = f'{type_functional_identity}_x'
                col_y = f'{type_functional_identity}_y'
                #print(B_df[['t',col_x]][B_df[col_x].notna()])
                # Combine the two columns into one
                B_df[type_functional_identity] = B_df[col_x].fillna(0) + B_df[col_y].fillna(0)
                # Drop the original `_x` and `_y` columns
                B_df = B_df.drop(columns=[col_x, col_y])
                duplicate_columns = False
        
        B_df = B_df.fillna(0) # Because for duplicate columns we used 0 instead of nans
        excluded_columns = ['t']
        other_cols = B_df.columns.difference(excluded_columns)
        B_df['total_B'] = B_df[other_cols].sum(axis=1, skipna=True) # We sum all columns without including t_vec obviously!
        B_df[other_cols] = B_df[other_cols].div(B_df['total_B'], axis=0) # Remember here other cols does not include 'total_B' because it has been created after
        rel_B_df = B_df.copy()
        rel_B_df = rel_B_df.fillna(0)
        rel_B_df = rel_B_df.drop('total_B', axis=1)

        # Get functions
        aux_D_B_tvec = []
        for B_ivec, D_ivec in zip(B_tvec, D_tvec): # OJO!! we have to go through D_tvec because each D_ivec inside has different dimension
            aux_D = np.where(D_ivec>=pathways_D, D_ivec, 0) # This gives me a (N_type, N_pathways) vector where for each entrance we have a vector with difficulties different that 0 only in the actual pathways of the type. # This works because D_ivec has (N_type,N_Pathways) and D_mat has (N_pathways) (Remember N_type changes in time)        
            aux_D_B = aux_D*B_ivec.reshape(B_ivec.shape[0],1)
            aux_D_B_tvec.append(aux_D_B.sum(axis=0))
            
        aux_D_B_tvec = np.array(aux_D_B_tvec)
        sum_D_B_over_types = (aux_D_B_tvec.sum(axis=1).reshape(aux_D_B_tvec.shape[0],1))
        sum_D_B_over_types = np.where(sum_D_B_over_types>0, sum_D_B_over_types, 1)
        # With the np.where we avoid if there are pathways that are not realised for any of the types, then when dividing by it we won't get an invalid number. Since it's not realised by any of the types in the numerator there will be a zero as well!
        rel_D_B_tvec = aux_D_B_tvec/sum_D_B_over_types

        processed_dir = os.path.join(folder_path, 'processed')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        with open(wanted_file, 'wb') as f:
            pickle.dump((rel_D_B_tvec, rel_B_df), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    category_data_dict = {'t': t_vec}
    for i, p_str in enumerate(pathways_strs):
        p_category = p_str_to_category_dict[p_str]
        if p_category not in category_data_dict.keys():
            category_data_dict[p_category] = rel_D_B_tvec[:,i].copy()
        else:
            category_data_dict[p_category] += rel_D_B_tvec[:,i].copy()
    category_data_df = pd.DataFrame(category_data_dict)
    
    return category_data_df, rel_B_df

def plot_functional_redundancy(rootpath, folder1, folder2, folder3, folder_list, sample_hours):
    
        
    
    def generate_professional_colors(num_colors):
        """
        Generate a professional and visually appealing color palette.
        
        :param num_colors: Number of colors required.
        :return: A list of color hex codes.
        """
        if num_colors <= 10:
            # Use a qualitative color palette for up to 10 categories
            palette = sns.color_palette("deep", n_colors=num_colors)
        elif num_colors <= 20:
            # Use a larger qualitative palette for 11-20 categories
            palette = sns.color_palette("tab20", n_colors=num_colors)
        else:
            # Use a perceptually uniform continuous colormap for >20 categories
            #cmap = plt.colormaps["tab20c" if num_colors <= 40 else "viridis"]
            #palette = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
            palette = sns.color_palette("deep", n_colors=num_colors)
        # Convert to hex for consistent use
        return [to_hex(color) for color in palette]

        
    '''
            fig = plt.figure(figsize = (18.4*0.393701, 15*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(5,5, height_ratios=[0.22,0.22,0.22,0.04,0.3], width_ratios=[0.2475,0.01,0.2475,0.2475,0.2475])
    
    ax00 = fig.add_subplot(gs[0,0])
    ax10 = fig.add_subplot(gs[1,0], sharex=ax00)
    ax20 = fig.add_subplot(gs[2,0], sharex=ax00)
    ax01 = fig.add_subplot(gs[0,2])
    ax11 = fig.add_subplot(gs[1,2], sharex=ax01)
    ax21 = fig.add_subplot(gs[2,2], sharex=ax01)
    ax02 = fig.add_subplot(gs[0,3])
    ax12 = fig.add_subplot(gs[1,3], sharex=ax02)
    ax22 = fig.add_subplot(gs[2,3], sharex=ax02)
    ax03 = fig.add_subplot(gs[0,4])
    ax13 = fig.add_subplot(gs[1,4], sharex=ax03)
    ax23 = fig.add_subplot(gs[2,4], sharex=ax03)
    ax30 = fig.add_subplot(gs[4,0])
    ax31 = fig.add_subplot(gs[4,2])
    ax32 = fig.add_subplot(gs[4,3])
    ax33 = fig.add_subplot(gs[4,4])

    folder1_path = os.path.join(rootpath,folder1)
    folder2_path = os.path.join(rootpath,folder2)
    folder3_path = os.path.join(rootpath,folder3)
    folder_paths_list = [os.path.join(rootpath,folder) for folder in folder_list]
    
    
    color_mappings = []
    # Plot functions
    for i,(axes1,axes2,axes3,group_size) in enumerate([(ax01,ax11,ax21,1), (ax02,ax12,ax22,4), (ax03,ax13,ax23,10)]):
        _, p_str_to_category_dict = get_functions_categories(group_size, num_substances=30)
        print(f'Number of categories with coarse group-size {group_size} is: {len(set(p_str_to_category_dict.values()))}')
        categories_df1, rel_B_df1 = get_taxonomy_and_functions_data(folder1_path, sample_hours, p_str_to_category_dict, overwrite=False)
        categories_df2, rel_B_df2 = get_taxonomy_and_functions_data(folder2_path, sample_hours, p_str_to_category_dict, overwrite=False)
        categories_df3, rel_B_df3 = get_taxonomy_and_functions_data(folder3_path, sample_hours, p_str_to_category_dict, overwrite=False)
        
        if i == 0 : # Taxonomy data are just plotted once because they do not depend on coarse graining
            dataframes = [rel_B_df1, rel_B_df2, rel_B_df3]
            all_bacterial_types = set(rel_B_df1.columns).union(rel_B_df2.columns).union(rel_B_df3.columns) - {'t'}
            palette = sns.color_palette("tab20c", n_colors=len(all_bacterial_types))
            color_mapping = {func_id: palette[i % len(palette)] for i, func_id in enumerate(sorted(all_bacterial_types))}
            color_mappings.append(color_mapping)
            for idx,df in enumerate(dataframes):
                df = df.iloc[::20].reset_index(drop=True)
                df['t'] = df['t'] / (24 * 365)
                dataframes[idx] = df
            rel_B_df1, rel_B_df2, rel_B_df3 = dataframes
            #taxonomy_palette = sns.color_palette("tab20", n_colors=20)
            #taxonomy_cmap = [to_hex(color) for color in taxonomy_palette]
            rel_B_df1.plot(x='t', kind='bar', stacked=True, width=1, ax=ax00, legend=False, color=color_mapping)
            print('Plot1 :)')
            rel_B_df2.plot(x='t', kind='bar', stacked=True, width=1, ax=ax10, legend=False, color=color_mapping)
            print('Plot2 :)')
            rel_B_df3.plot(x='t', kind='bar', stacked=True, width=1, ax=ax20, legend=False, color=color_mapping)
            print('Plot3 :)')
            # For the xticks for the stacked bar plots. We assume all data have same times
            desired_ticks = np.arange(0, 51, 10)  # Integers from 0 to 50
            data_min = rel_B_df1['t'].min()
            data_max = rel_B_df1['t'].max()
            # Map desired_ticks to the x-axis scale
            scaled_tick_positions = (desired_ticks - data_min) / (data_max - data_min) * len(rel_B_df1)-1
            # This works because: When creating a stacked bar plot with pandas or matplotlib, tick positions refer to the 
            # indices of the bars plotted on the x-axis, not the actual data values.
            for ax in [ax00,ax10,ax20]:
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.xaxis.set_minor_locator(plt.NullLocator())


        dataframes = [categories_df1, categories_df2, categories_df3]
        for idx,df in enumerate(dataframes):
            df = df.iloc[::20].reset_index(drop=True)
            df['t'] = df['t'] / (24 * 365)
            dataframes[idx] = df
        categories_df1, categories_df2, categories_df3 = dataframes
        categories_df2['t'] = categories_df2['t'] / (24 * 365)
        categories_df3['t'] = categories_df3['t'] / (24 * 365)
        
        categories = [col for col in categories_df1.columns if col != 't']
        n_categories = len(categories)
        categories_color_list = generate_professional_colors(n_categories)
        
        categories_color_mapping = dict(zip(categories, categories_color_list))
        color_mappings.append(categories_color_mapping)
        categories_df1.plot(x='t', kind='bar', stacked=True, width=1, ax=axes1, legend=False, color=categories_color_list)
        categories_df2.plot(x='t', kind='bar', stacked=True, width=1, ax=axes2, legend=False, color=categories_color_list)
        categories_df3.plot(x='t', kind='bar', stacked=True, width=1, ax=axes3, legend=False, color=categories_color_list)
        print(f'Coarsed type: {i}')
        # Plot taxonomy
        
        # Remove automatic pandas ticks by disabling major and minor ticks
        for ax in [axes1,axes2,axes3]:
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.xaxis.set_minor_locator(plt.NullLocator())
    
    transversal_taxonomy,transversal_coarsed1,transversal_coarsed2,transversal_coarsed3 = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    p_id_to_category_dict1, p_str_to_category_dict1 = get_functions_categories(group_size=1, num_substances=30)
    p_id_to_category_dict2, p_str_to_category_dict2 = get_functions_categories(group_size=4, num_substances=30)
    p_id_to_category_dict3, p_str_to_category_dict3 = get_functions_categories(group_size=10, num_substances=30)
    
    categories_stacked_data_list = [[],[],[]]
    taxonomy_stacked_data = []
    for folder_path, folder in zip(folder_paths_list, folder_list):
        for i,coarsed_p_str_to_category_dict in enumerate([p_str_to_category_dict1,p_str_to_category_dict2,p_str_to_category_dict3]):
            categories_df, rel_B_df = get_taxonomy_and_functions_data(folder_path, sample_hours, coarsed_p_str_to_category_dict, overwrite=False)
            last_time_row = categories_df[categories_df['t'] == categories_df['t'].max()].drop(columns='t')
            melted = last_time_row.melt(var_name='Category', value_name='Proportion')
            melted['Dataset'] = folder
            categories_stacked_data_list[i].append(melted)
        last_time_row = rel_B_df[rel_B_df['t'] == rel_B_df['t'].max()].drop(columns='t')
        melted = last_time_row.melt(var_name='Identity', value_name='Proportion')
        melted['Dataset'] = folder
        taxonomy_stacked_data.append(melted)
    categories_stacked_df1 = pd.concat(categories_stacked_data_list[0], ignore_index=True)
    categories_stacked_df2 = pd.concat(categories_stacked_data_list[1], ignore_index=True)
    categories_stacked_df3 = pd.concat(categories_stacked_data_list[2], ignore_index=True)
    taxonomy_stacked_df = pd.concat(taxonomy_stacked_data, ignore_index=True)
    
    # Prepare the df's  so that they are plotted with the same color for same identities and the pathways or identities with higher median go first
    dfs = [taxonomy_stacked_df, categories_stacked_df1, categories_stacked_df2, categories_stacked_df3]
    for idx,(wanted_df,ax) in enumerate(zip(dfs,[ax30,ax31,ax32,ax33])):
        if idx == 0:
            name = 'Identity'
            palette_name = 'tab20c'
            
        else:
            name = 'Category'
            palette_name = 'deep'
            
        pathway_medians = wanted_df.groupby(name)['Proportion'].median().sort_values(ascending=False)
        wanted_df[name] = pd.Categorical(wanted_df[name], categories=pathway_medians.index, ordered=True)
        if idx == 0:
            palette = sns.color_palette(palette_name, n_colors=len(pathway_medians))
            color_mapping = {pathway: palette[i] for i, pathway in enumerate(pathway_medians.index)}  
        else:
            color_mapping = color_mappings[idx]
        pivot_df = wanted_df.pivot(index='Dataset', columns=name, values='Proportion').fillna(0)
        pivot_df.plot(ax=ax, kind='bar', stacked=True,  width=1, color=[color_mapping[col] for col in pivot_df.columns], legend=False)
    
    ax00.set_ylabel(r'Biomass fraction', labelpad=2, fontsize=10)
    ax10.set_ylabel(r'Biomass fraction', labelpad=2, fontsize=10)
    ax20.set_ylabel(r'Biomass fraction', labelpad=2, fontsize=10)
    ax30.set_ylabel(r'Biomass fraction', labelpad=2, fontsize=10)
    
    ax01.set_ylabel(r'Enzymes fraction', labelpad=2, fontsize=10)
    ax11.set_ylabel(r'Enzymes fraction', labelpad=2, fontsize=10)
    ax21.set_ylabel(r'Enzymes fraction', labelpad=2, fontsize=10)
    ax31.set_ylabel(r'Enzymes fraction', labelpad=2, fontsize=10)

    for i,ax in enumerate([ax00, ax01, ax02, ax03, ax10, ax11, ax12, ax13, ax20, ax21, ax22, ax23]):  
        ax.tick_params(labelleft=False)  
        ax.set_yticks([])
        ax.set_xticks(scaled_tick_positions)
        if i >= 8 and i<=11:
            ax.set_xticklabels(desired_ticks, rotation=0)   
            ax.set_xlabel('Time', fontsize=10)
        else:
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
        if ax.get_legend():
            ax.legend_.remove()
        ax.set_ylim([0,1])
    for i,ax in enumerate([ax30,ax31,ax32,ax33]):
        ax.set_xlim(-0.5, len(folder_list)-0.5)
        ax.set_ylim([0,1])
        ax.set_yticks([])
        ax.set_xlabel('Individual', fontsize=10)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        if ax.get_legend():
            ax.legend_.remove()
        if i>1:
            ax.set_ylabel('')
    #ax0.set_xlabel('')  # Remove x-axis label
    #ax0.set_xticklabels([])  # Remove x tick labels
    #ax1.set_xlabel('Number of species')

    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    image_name = os.path.join(image_dir, f'functional_redundancy_final.svg')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    plt.subplots_adjust(top=0.97, bottom=0.05, left=0.035, right=0.98, hspace=0.2, wspace=0.15)
    plt.savefig(image_name, format='svg', transparent=False, dpi=300)
    print('SVG image saved')
    image_name = os.path.join(image_dir, f'functional_redundancy_final.png')   
    plt.savefig(image_name, format='png', transparent=False, dpi=2400)
    plt.show()
    '''
    
    fig = plt.figure(figsize=(18.4 * 0.393701, 15 * 0.393701))  # Convert cm to inches
    gs = fig.add_gridspec(5, 5, height_ratios=[0.22, 0.22, 0.22, 0.04, 0.3], 
                           width_ratios=[0.2475, 0.01, 0.2475, 0.2475, 0.2475])
    
    # Define subplots
    ax00, ax10, ax20 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0], sharex=ax00), fig.add_subplot(gs[2,0], sharex=ax00)
    ax01, ax11, ax21 = fig.add_subplot(gs[0,2]), fig.add_subplot(gs[1,2], sharex=ax01), fig.add_subplot(gs[2,2], sharex=ax01)
    ax02, ax12, ax22 = fig.add_subplot(gs[0,3]), fig.add_subplot(gs[1,3], sharex=ax02), fig.add_subplot(gs[2,3], sharex=ax02)
    ax03, ax13, ax23 = fig.add_subplot(gs[0,4]), fig.add_subplot(gs[1,4], sharex=ax03), fig.add_subplot(gs[2,4], sharex=ax03)
    ax30, ax31, ax32, ax33 = fig.add_subplot(gs[4,0]), fig.add_subplot(gs[4,2]), fig.add_subplot(gs[4,3]), fig.add_subplot(gs[4,4])
    
    folder_paths_list = [os.path.join(rootpath, folder) for folder in folder_list]
    
    # Create global color mappings to ensure consistency
    all_identities = set()
    
    for group_size in [1, 4, 10]:
        _, p_str_to_category_dict = get_functions_categories(group_size, num_substances=30)
        for folder_path in folder_paths_list:
            categories_df, _ = get_taxonomy_and_functions_data(folder_path, sample_hours, p_str_to_category_dict, overwrite=False)
            all_identities.update(categories_df.columns.difference(['t']))
    
    palette = sns.color_palette("tab20c", n_colors=len(all_identities))
    color_mapping = {identity: palette[i] for i, identity in enumerate(sorted(all_identities))}
    
    for i, (axes1, axes2, axes3, group_size) in enumerate([(ax01, ax11, ax21, 1), (ax02, ax12, ax22, 4), (ax03, ax13, ax23, 10)]):
        _, p_str_to_category_dict = get_functions_categories(group_size, num_substances=30)
        
        for idx, folder_path in enumerate(folder_paths_list):
            categories_df, _ = get_taxonomy_and_functions_data(folder_path, sample_hours, p_str_to_category_dict, overwrite=False)
            categories_df['t'] = categories_df['t'] / (24 * 365)
            categories_df.plot(x='t', kind='bar', stacked=True, width=1, ax=[axes1, axes2, axes3][idx], legend=False, color=[color_mapping[col] for col in categories_df.columns if col != 't'])
    
    # Process data for stacked bar plots in ax30, ax31, ax32, ax33
    dfs = []
    for group_size in [1, 4, 10]:
        _, p_str_to_category_dict = get_functions_categories(group_size, num_substances=30)
        stacked_data_list = []
        
        for folder_path, folder in zip(folder_paths_list, folder_list):
            categories_df, _ = get_taxonomy_and_functions_data(folder_path, sample_hours, p_str_to_category_dict, overwrite=False)
            last_time_row = categories_df[categories_df['t'] == categories_df['t'].max()].drop(columns='t')
            melted = last_time_row.melt(var_name='Category', value_name='Proportion')
            melted['Dataset'] = folder
            stacked_data_list.append(melted)
        
        dfs.append(pd.concat(stacked_data_list, ignore_index=True))
    
    for idx, (wanted_df, ax) in enumerate(zip(dfs, [ax31, ax32, ax33])):
        name = 'Category'
        pathway_medians = wanted_df.groupby(name)['Proportion'].median().sort_values(ascending=False)
        wanted_df[name] = pd.Categorical(wanted_df[name], categories=pathway_medians.index, ordered=True)
        pivot_df = wanted_df.pivot(index='Dataset', columns=name, values='Proportion').fillna(0)
        pivot_df.plot(ax=ax, kind='bar', stacked=True, width=1, color=[color_mapping[col] for col in pivot_df.columns], legend=False)
    
    # Formatting
    for ax in [ax00, ax10, ax20, ax30]:
        ax.set_ylabel(r'Biomass fraction', labelpad=2, fontsize=10)
    
    for ax in [ax01, ax11, ax21, ax31]:
        ax.set_ylabel(r'Enzymes fraction', labelpad=2, fontsize=10)
    
    for ax in [ax00, ax01, ax02, ax03, ax10, ax11, ax12, ax13, ax20, ax21, ax22, ax23]:  
        ax.tick_params(labelleft=False)
        ax.set_yticks([])
        ax.set_ylim([0, 1])
    
    for ax in [ax30, ax31, ax32, ax33]:
        ax.set_xlim(-0.5, len(folder_list) - 0.5)
        ax.set_ylim([0, 1])
        ax.set_yticks([])
        ax.set_xlabel('Individual', fontsize=10)
    
    # Save images
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    
    plt.subplots_adjust(top=0.97, bottom=0.05, left=0.035, right=0.98, hspace=0.2, wspace=0.15)
    
    image_name_svg = os.path.join(image_dir, 'functional_redundancy_final.svg')
    plt.savefig(image_name_svg, format='svg', transparent=False, dpi=300)
    
    image_name_png = os.path.join(image_dir, 'functional_redundancy_final.png')
    plt.savefig(image_name_png, format='png', transparent=False, dpi=2400)
    
    plt.show()
    print('SVG and PNG images saved.')

    
def coarse_grain_pathways(rel_B_df, p_id_to_category_dict):
    """
    Perform coarse-graining on bacterial pathway identities based on categories and sum corresponding columns.
    
    :param rel_B_df: DataFrame where columns are bacterial pathway identities (e.g., '5.23.45.98') and rows are samples.
    :param p_id_to_category_dict: Dictionary mapping pathway IDs to categories.
    :return: DataFrame with coarse-grained pathway categories as columns.
    """
    def normalize_pathways(pathway_str):
        # Normalize pathway by sorting pathway IDs in ascending order
        pathway_ids = sorted(map(str, pathway_str.split('.')))
        return '.'.join(map(str, pathway_ids))
    
    def coarse_grain(pathway_str):
        # Map normalized pathway IDs to categories
        normalized_ids = normalize_pathways(pathway_str).split('.')
        return '.'.join(sorted({p_id_to_category_dict[p] for p in normalized_ids}))
    
    # Apply normalization and coarse-graining to column names
    coarse_grained_columns = {
        col: coarse_grain(col) for col in rel_B_df.columns if col != 't'
    }
    '''
    # Create a new DataFrame with coarse-grained columns
    rel_B_coarse_df = pd.DataFrame(index=rel_B_df.index)
    rel_B_coarse_df['t'] = rel_B_df['t']
    
    # Create a Series from the coarse-grained mapping
    coarse_grained_series = pd.Series(coarse_grained_columns)
    # Group by the coarse-grained categories and sum the columns
    for coarse_col, original_cols in coarse_grained_series.groupby(coarse_grained_series):
        rel_B_coarse_df[coarse_col] = rel_B_df[original_cols.index].sum(axis=1, skipna=True)

    return rel_B_coarse_df
    '''
    coarse_grained_data = {}

    # Convert dictionary to Series for grouping
    coarse_grained_series = pd.Series(coarse_grained_columns)
    
    # Accumulate results in a dictionary
    coarse_grained_data = {}
    for coarse_col, original_cols in coarse_grained_series.groupby(coarse_grained_series):
        coarse_grained_data[coarse_col] = rel_B_df[original_cols.index].sum(axis=1, skipna=True)
    
    # Combine all results into a single DataFrame
    coarse_grained_df = pd.DataFrame(coarse_grained_data)
    
    # Add the time column back
    coarse_grained_df.insert(0, 't', rel_B_df['t'])
    
    return coarse_grained_df
def compute_bacterial_metrics(rel_B_df):
    """
    Compute various metrics for bacterial populations given their relative proportions over time.
    
    :param rel_B_df: DataFrame with columns 't' (time) and bacterial identities, containing NaN for absent bacteria.
    :return: A dictionary containing richness distribution, mean, variance, prevalence, abundance change distribution,
            and residence time distribution.
    """
    # Extract time column and bacterial data
    times = rel_B_df['t']
    bacteria_data = rel_B_df.drop(columns=['t'])
    
    # Richness distribution (number of bacteria with relative abundance > 0 at each time)
    richness_distribution = (bacteria_data > 0).sum(axis=1).values

    # Mean and variance vectors for each bacterial identity (ignoring NaN)
    mean_ivec = bacteria_data.mean(axis=0, skipna=True).values
    variance_ivec = bacteria_data.var(axis=0, skipna=True).values

    # Prevalence vector (fraction of sampling times bacteria is present)
    prevalence_ivec = (bacteria_data.notna() & (bacteria_data > 0)).sum(axis=0) / len(times)

    # Abundance change distribution
    log_abundance_changes = []
    for i in range(len(times) - 1):
        current_row = bacteria_data.iloc[i]
        next_row = bacteria_data.iloc[i + 1]
        valid_indices = (current_row > 0) & (next_row > 0)  # Ensure both times have valid data
        changes = np.log(next_row[valid_indices] / current_row[valid_indices])
        log_abundance_changes.extend(changes.values)
    abu_change_distribution = np.array(log_abundance_changes)

    # Residence time distribution
    t_res_dist = []
    for col in bacteria_data.columns:
        binary_presence = bacteria_data[col].notna() & (bacteria_data[col] > 0)
        presence_times = times[binary_presence].values
        if len(presence_times) > 1:
            gaps = np.diff(np.where(np.diff(np.r_[False, binary_presence.values, False]) != 0)[0][::2])
            t_res_dist.extend(gaps)
    metrics_dict = {'richness_distribution': richness_distribution, 'mean_ivec': mean_ivec, 'variance_ivec': variance_ivec,
                    'prevalence_ivec': prevalence_ivec.values, 'abu_change_distribution': abu_change_distribution,
                    't_res_dist': np.array(t_res_dist)
    }
    return metrics_dict

def get_needed_data(folder_path, sample_hours, relaxation_time, subsample_model=None, coarsed_size=None):
    t_vec = get_tvec(folder_path, sample_hours, wanted_tvec='t')
    B_types_dict = get_B_types_dict(folder_path, sample_hours)
    rel_B_df = get_rel_B_df(folder_path, sample_hours, t_vec, B_types_dict, overwrite=False)

    if coarsed_size is not None:
        D_mat = DA.get_D_mat(folder_path)
        E_avec = DA.get_E_avec(folder_path)
        pathways_strs = DA.get_pathways_strs(D_mat)
        pathways_ids, _ = DA.get_pathways_id(D_mat, E_avec)
        p_str_to_p_id_dict = dict(zip(pathways_strs, pathways_ids))
        p_id_to_category_dict, _ = get_functions_categories(coarsed_size, 30, p_str_to_p_id_dict)
        rel_B_df = coarse_grain_pathways(rel_B_df, p_id_to_category_dict)

    # Get interaction matrices
    crossfeeding_ijmat_tvec, competition_ijmat_tvec = get_cf_cp_mat_tvecs(folder_path, sample_hours)
    cf_tvec = np.array([np.sum(cf) for cf in crossfeeding_ijmat_tvec])
    cp_tvec = np.array([(np.sum(cp) - np.trace(cp)) for cp in competition_ijmat_tvec])
    rho_tvec = (cf_tvec - cp_tvec) / (cf_tvec + cp_tvec)

    # Add rho to rel_B_df
    rel_B_df['rho'] = rho_tvec
    rel_B_df = rel_B_df[rel_B_df['t'] > relaxation_time]
    if subsample_model is not None:
        rel_B_df = rel_B_df.iloc[::subsample_model]

    # Compute metrics on all, healthy, dysbiotic
    def compute_group(df):
        return compute_bacterial_metrics(df.drop(columns='rho'))

    metrics_all = compute_group(rel_B_df)
    metrics_healthy = compute_group(rel_B_df[rel_B_df['rho'] <= 0])
    metrics_dysbiotic = compute_group(rel_B_df[rel_B_df['rho'] > 0])

    return {'all': metrics_all, 'healthy': metrics_healthy, 'dysbiotic': metrics_dysbiotic}
def get_real_data(gut_dir, family=True, lower_threshold=None):
    def create_host_taxonomy_df(metadata_df, taxonomy_df, OTU_df, host_value, family):
        """
        Create a dataframe for a given host with relative counts per family.
        
        Args:
        - metadata_df: DataFrame with ['X.SampleID', 'days_since_experiment_start', 'host'].
        - taxonomy_df: DataFrame with OTU as index and taxonomical hierarchy as columns.
        - OTU_df: DataFrame with OTUs as index and X.SampleID as columns (counts).
        - host_value: Host value to filter ('F4', 'M3', etc.).
        
        Returns:
        - A dataframe with 't' (time) as a column and one column per family with normalized counts.
        """
        # Step 1: Filter metadata for the given host
        host_metadata = metadata_df[metadata_df['host'] == host_value]
        
        # Step 2: Filter OTU_df for the samples in the host_metadata
        sample_ids = host_metadata['X.SampleID'].values
        filtered_OTU_df = OTU_df[sample_ids]
        df = filtered_OTU_df
        if family:
            # Step 3: Filter taxonomy_df for valid 'Family' and 'Kingdom' == 'Bacteria'
            valid_taxonomy = taxonomy_df[
                (taxonomy_df['Kingdom'] == 'Bacteria') & 
                taxonomy_df['Family'].notna()
            ]
            # Step 4: Map OTUs to families and group counts by family
            otu_to_family = valid_taxonomy['Family']
            family_counts = df.groupby(otu_to_family).sum()

            df = family_counts
        else:  
            # Step 3: Filter taxonomy_df for valid  'Kingdom' == 'Bacteria'
            bacterial_OTUs = taxonomy_df[(taxonomy_df['Kingdom'] == 'Bacteria')].index
            df = df.loc[bacterial_OTUs]
            
        
        # Step 5: Normalize counts column-wise (per sample)
        normalized_df = df.div(df.sum(axis=0), axis=1)
        
        # Step 5.5: Apply extinction threshold (if set)
        if lower_threshold is not None:
            normalized_df = normalized_df.where(normalized_df >= lower_threshold, other=0)

        # Step 6: Add 't' column (time) from metadata_df
        normalized_df = normalized_df.transpose()  # Make samples as rows
        normalized_df = normalized_df.join(
            host_metadata.set_index('X.SampleID')['days_since_experiment_start']
        )
        normalized_df.rename(columns={'days_since_experiment_start': 't'}, inplace=True)
        sorted_df = normalized_df.sort_values(by='t').reset_index(drop=True)
        return sorted_df

    metadata_file = os.path.join(gut_dir, 'real_data', 'MovingPictures', 'metadata.csv')
    metadata_df = pd.read_csv(metadata_file)
    wanted_columns = ['X.SampleID', 'days_since_experiment_start', 'host']
    metadata_df['X.SampleID'] = 'X'+metadata_df['X.SampleID']
    metadata_df = metadata_df[metadata_df['common_sample_site'] == 'feces'][wanted_columns]
    taxonomy_file = os.path.join(gut_dir, 'real_data', 'MovingPictures', 'taxonomy_table.csv')
    taxonomy_df = pd.read_csv(taxonomy_file, index_col=0)
    otu_file = os.path.join(gut_dir, 'real_data', 'MovingPictures', 'otu_table.csv')
    otu_df = pd.read_csv(otu_file, index_col=0)
    # Create the dataframes for each host
    #metadata_df['days_since_experiment_start'] = pd.to_numeric(metadata_df['days_since_experiment_start'], errors='coerce')
    F4_df = create_host_taxonomy_df(metadata_df, taxonomy_df, otu_df, 'F4', family)
    M3_df = create_host_taxonomy_df(metadata_df, taxonomy_df, otu_df, 'M3', family)
    return F4_df, M3_df

def pad_vectors_to_same_length(vectors, pad_value=np.nan):
    """
    Pads all vectors in a 2D list to the same length with np.nan.

    Args:
    - vectors: List of 1D numpy arrays or lists, each of different lengths.

    Returns:
    - padded_array: 2D numpy array where all rows have the same length.
    """
    # Find the maximum length among all vectors
    max_length = max(len(vec) for vec in vectors)
    
    # Pad each vector with np.nan to make them the same length
    padded_vectors = [np.pad(vec, (0, max_length - len(vec)), constant_values=pad_value) for vec in vectors]
    
    # Convert to a 2D numpy array
    padded_array = np.vstack(padded_vectors)
    
    return padded_array
def get_95_CI(x_ivec, y_ivec, bins=50, log=True):
    
    if log:
        valid_indices = x_ivec > 0
        x_ivec = x_ivec[valid_indices]
        y_ivec = y_ivec[valid_indices]
        log_min = np.log10(np.min(x_ivec))
        log_max = np.log10(np.max(x_ivec))
        bin_edges = np.logspace(log_min, log_max, bins + 1)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for bin centers
    else:
        # Bin the mean abundance values
        bin_edges = np.linspace(np.min(x_ivec), np.max(x_ivec), bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    lower_percentiles = []
    upper_percentiles = []
    averages = []

    # Calculate 2.5th, 97.5th percentiles, and averages for each bin
    for i in range(len(bin_edges) - 1):
        in_bin = (x_ivec >= bin_edges[i]) & (x_ivec < bin_edges[i + 1])
        if np.sum(in_bin) > 0:  # If there are points in the bin
            lower_percentiles.append(np.percentile(y_ivec[in_bin], 2.5))
            upper_percentiles.append(np.percentile(y_ivec[in_bin], 97.5))
            averages.append(np.mean(y_ivec[in_bin]))
        else:  # If no data in bin, append NaN
            lower_percentiles.append(np.nan)
            upper_percentiles.append(np.nan)
            averages.append(np.nan)

    # Convert to numpy arrays for plotting
    lower_percentiles = np.array(lower_percentiles)
    upper_percentiles = np.array(upper_percentiles)
    averages = np.array(averages)

    # Return bin centers, lower percentiles, upper percentiles, and averages
    return bin_centers, averages, lower_percentiles, upper_percentiles

def robustness_interaction_fig(gut_dir):

    CRC_bootstrap_path = os.path.join(gut_dir, 'real_data', 'CRC_Yachida_NatMed', 'bootstrap2_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    IBS_bootstrap_path = os.path.join(gut_dir, 'real_data', 'IBS_Mars_Cell', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    IBD_bootstrap_path = os.path.join(gut_dir, 'real_data', 'IBD_MDB', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    CD_bootstrap_path = os.path.join(gut_dir, 'real_data', 'CD_Ferretti_Elife', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    Model_bootstrap_path = os.path.join(gut_dir, 'real_data', 'ModelData', 'invasion', 'E_D_0', 'fraction_nl_6_0_1.1',
                                        'bootstrap_HE-S', 'BacFrac_1.0_FullsetFrac_1.0', 'coarsed_True', 'samples_1000')
    full_color_map = {}
    base_color_map = {'Healthy': '#0078B9', 'H': '#0078B9', 'Unhealthy': '#EA0017', 'U': '#EA0017',
                      'IBD': '#EA0017', 'IBS': '#EA0017', 'CDI': '#EA0017'}

    CRC_color_map = {"Healthy": '#0078B9',
                     "CRC": '#EA0017',
                     "CRC-0": '#F9B3B3',  # Lightest red
                     "CRC-1": "#F48A8A",  # Light red
                     "CRC-2": "#E03D3D",  # Dark red
                     "CRC-3": "#A70000"   # Darkest Red
                     #"HS": "#D4C2E5"  # Purple
                     }
    Model_color_map = {"FS": '#0078B9',
                       #"Healthy_0": "#66AEDD",
                       #"Healthy_1": "#005080",
                       "CS": '#EA0017',
                       "CS-0": '#F9B3B3',
                       "CS-1": "#F48A8A",
                       "CS-2": "#E03D3D",
                       "CS-3": "#A70000"
                      }
    full_color_map.update(Model_color_map)
    full_color_map.update(base_color_map)
    full_color_map.update(CRC_color_map)
    
    
    fig = plt.figure(figsize = (18.4*0.393701, 11*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(5,1, height_ratios=[1/5,1/5,1/5,1/5,1/5], width_ratios=[1])
    
    ax00 = fig.add_subplot(gs[0,0])
    ax10 = fig.add_subplot(gs[1,0])
    ax20 = fig.add_subplot(gs[2,0])
    ax30 = fig.add_subplot(gs[3,0])
    ax40 = fig.add_subplot(gs[4,0])
    '''
    full_Model_df = pd.DataFrame(columns=['subset_fraction', 'Group', 'rho'])
    full_CRC_df = pd.DataFrame(columns=['subset_fraction', 'Group', 'rho'])
    full_IBD_df = pd.DataFrame(columns=['subset_fraction', 'Group', 'rho'])
    full_IBS_df = pd.DataFrame(columns=['subset_fraction', 'Group', 'rho'])
    full_CDI_df = pd.DataFrame(columns=['subset_fraction', 'Group', 'rho'])
    '''
    Model_replacement_dict = dict(zip(['Healthy', 'Unhealthy', 'Unhealthy_0', 'Unhealthy_1', 'Unhealthy_2', 'Unhealthy_3'],
                                       ['FS', 'CS', 'CS-0', 'CS-1', 'CS-2', 'CS-3']))
    CRC_replacement_dict = dict(zip(['Unhealthy', 'MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV'],
                                     ['CRC', 'CRC-0', 'CRC-1', 'CRC-2', 'CRC-3']))
    IBD_replacement_dict = dict(zip(['Unhealthy'], ['IBD']))
    IBS_replacement_dict = dict(zip(['Unhealthy'], ['IBS']))
    CDI_replacement_dict = dict(zip(['Unhealthy'], ['CDI']))
    for subset_fraction in ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:
        Model_df = get_data(Model_bootstrap_path, ['Unhealthy_0', 'Unhealthy_1', 'Unhealthy_2', 'Unhealthy_3'], None, subset_fraction, healthy_normalized=False, overwrite=True)
        CRC_df = get_data(CRC_bootstrap_path,['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV'], None, subset_fraction, healthy_normalized=False, overwrite=True)
        IBD_df = get_data(IBD_bootstrap_path, None, None, subset_fraction, healthy_normalized=False, overwrite=True)
        IBS_df = get_data(IBS_bootstrap_path, None, None, subset_fraction, healthy_normalized=False, overwrite=True)
        CDI_df = get_data(CD_bootstrap_path, None, None, subset_fraction, healthy_normalized=False, overwrite=True)

        Model_df['Group'] = Model_df['Group'].cat.rename_categories(Model_replacement_dict)
        Model_df['Group'] = pd.Categorical(Model_df['Group'], ordered=True, categories=['FS', 'CS', 'CS-0', 'CS-1', 'CS-2', 'CS-3'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group', data=Model_df, palette=full_color_map, ax=ax00, showfliers=False)
        CRC_df['Group'] = CRC_df['Group'].cat.rename_categories(CRC_replacement_dict)
        CRC_df['Group'] = pd.Categorical(CRC_df['Group'], ordered=True, categories=['Healthy', 'CRC', 'CRC-0', 'CRC-1', 'CRC-2', 'CRC-3'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group',  data=CRC_df, palette=full_color_map, ax=ax10, showfliers=False)
        IBD_df['Group'] = IBD_df['Group'].cat.rename_categories(IBD_replacement_dict)
        IBD_df['Group'] = pd.Categorical(IBD_df['Group'], ordered=True, categories=['Healthy', 'IBD'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group', width=0.4, data=IBD_df, palette=full_color_map, ax=ax20, showfliers=False)
        IBS_df['Group'] = IBS_df['Group'].cat.rename_categories(IBS_replacement_dict)
        IBS_df['Group'] = pd.Categorical(IBS_df['Group'], ordered=True, categories=['Healthy', 'IBS'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group', width=0.4, data=IBS_df, palette=full_color_map, ax=ax30, showfliers=False)
        CDI_df['Group'] = CDI_df['Group'].cat.rename_categories(CDI_replacement_dict)
        CDI_df['Group'] = pd.Categorical(CDI_df['Group'], ordered=True, categories=['Healthy', 'CDI'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group', width=0.4, data=CDI_df, palette=full_color_map, ax=ax40, showfliers=False)
    
    j=0
    for i,ax in enumerate([ax00, ax10, ax20, ax30, ax40]):
        # Remove only the horizontal whisker caps
        for line in ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            # Horizontal whisker caps have exactly two points and are OUTSIDE the box limits
            if len(x_data) == 2 and np.isclose(y_data[0], y_data[1]):
                if j in [0,1]:
                    line.set_visible(False)
                j += 1
                if j==3:
                    j=0
        despine(ax)
        if ax.get_legend():
            ax.legend_.remove()
        ax.set_ylabel('')
        if i < 4:
            ax.set_xlabel('')
            ax.set_xticklabels([])
    ax20.set_ylabel('Inferred Net Interaction', labelpad=5)
    
    # Define custom legends for each subplot
    # Custom legend handles with edge color and adjusted font size
    def create_legend_handles(color_dict, edgecolor='black'):
        return [Patch(facecolor=color, edgecolor=edgecolor, linewidth=1.2, label=label) 
                for label, color in color_dict.items()]

    # Create legend handles for each group
    custom_legends = {
        'Model': create_legend_handles(Model_color_map),
        'CRC': create_legend_handles(CRC_color_map),
        'IBD': create_legend_handles({'Healthy': full_color_map['Healthy'], 'IBD': full_color_map['IBD']}),
        'IBS': create_legend_handles({'Healthy': full_color_map['Healthy'], 'IBS': full_color_map['IBS']}),
        'CDI': create_legend_handles({'Healthy': full_color_map['Healthy'], 'CDI': full_color_map['CDI']}),
    }

    # Add legends to the right of each subplot with two columns, edge color, and smaller font size
    legend_fontsize = 8

    ax00.legend(handles=custom_legends['Model'], loc='center left', bbox_to_anchor=(1, 0.5), 
                title='Model', frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0, handletextpad=0.5)
    ax10.legend(handles=custom_legends['CRC'], loc='center left', bbox_to_anchor=(1, 0.5), 
                title='CRC', frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0,handletextpad=0.5)
    ax20.legend(handles=custom_legends['IBD'], loc='center left', bbox_to_anchor=(1, 0.5), 
                title='IBD', frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0, handletextpad=0.5)
    ax30.legend(handles=custom_legends['IBS'], loc='center left', bbox_to_anchor=(1, 0.5), 
                title='IBS', frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0, handletextpad=0.5)
    ax40.legend(handles=custom_legends['CDI'], loc='center left', bbox_to_anchor=(1, 0.5), 
                title='CDI', frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0, handletextpad=0.5)
    
    for ax in fig.get_axes():  # Iterate through all axes in the figure
        ax.xaxis.label.set_size(12)  # Set x-axis label size
        ax.yaxis.label.set_size(12)  # Set y-axis label size
        ax.tick_params(axis='x', which='major', labelsize=10)  # Major xtick size
        ax.tick_params(axis='y', which='major', labelsize=10)  # Major ytick size
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        despine(ax)
    ax10.set_yticks([0.6,0.8,1.0])
    ax30.set_yticks([0.84,0.88,0.92])
    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'robustness_interactions_fig_notnormalized.svg')

    plt.subplots_adjust(top=0.98, bottom=0.112, left=0.09, right=0.79, hspace=0.3, wspace=0.35)
    #plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    plt.show()


def robustness_taxonomic_profiling_interaction_fig(gut_dir):

    CRC_bootstrap2_path = os.path.join(gut_dir, 'real_data', 'CRC_Yachida_NatMed', 'bootstrap2_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    CRC_bootstrap3_path = os.path.join(gut_dir, 'real_data', 'CRC_Yachida_NatMed', 'bootstrap3_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    
    full_color_map = {}
    base_color_map = {'Healthy': '#0078B9', 'H': '#0078B9', 'Unhealthy': '#EA0017', 'U': '#EA0017',
                      'IBD': '#EA0017', 'IBS': '#EA0017', 'CDI': '#EA0017'}

    CRC_color_map = {"Healthy": '#0078B9',
                     "CRC": '#EA0017',
                     "CRC-0": '#F9B3B3',  # Lightest red
                     "CRC-1": "#F48A8A",  # Light red
                     "CRC-2": "#E03D3D",  # Dark red
                     "CRC-3": "#A70000"   # Darkest Red
                     #"HS": "#D4C2E5"  # Purple
                     }
    
    full_color_map.update(base_color_map)
    full_color_map.update(CRC_color_map)
    
    
    fig = plt.figure(figsize = (18.4*0.393701, 11*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(2,1, height_ratios=[1/2,1/2], width_ratios=[1])
    
    ax00 = fig.add_subplot(gs[0,0])
    ax10 = fig.add_subplot(gs[1,0])

    CRC_replacement_dict = dict(zip(['Unhealthy', 'MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV'],
                                     ['CRC', 'CRC-0', 'CRC-1', 'CRC-2', 'CRC-3']))
    
    for subset_fraction in ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:
        CRC2_df = get_data(CRC_bootstrap2_path,['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV'], None, subset_fraction, healthy_normalized=True, overwrite=True)
        CRC3_df = get_data(CRC_bootstrap3_path,['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV'], None, subset_fraction, healthy_normalized=True, overwrite=True)
        
        CRC2_df['Group'] = CRC2_df['Group'].cat.rename_categories(CRC_replacement_dict)
        CRC2_df['Group'] = pd.Categorical(CRC2_df['Group'], ordered=True, categories=['Healthy', 'CRC', 'CRC-0', 'CRC-1', 'CRC-2', 'CRC-3'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group',  data=CRC2_df, palette=full_color_map, ax=ax00, showfliers=False)
        CRC3_df['Group'] = CRC3_df['Group'].cat.rename_categories(CRC_replacement_dict)
        CRC3_df['Group'] = pd.Categorical(CRC3_df['Group'], ordered=True, categories=['Healthy', 'CRC', 'CRC-0', 'CRC-1', 'CRC-2', 'CRC-3'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group',  data=CRC3_df, palette=full_color_map, ax=ax10, showfliers=False)
    
    j=0
    for i,ax in enumerate([ax00,ax10]):
        # Remove only the horizontal whisker caps
        for line in ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            # Horizontal whisker caps have exactly two points and are OUTSIDE the box limits
            if len(x_data) == 2 and np.isclose(y_data[0], y_data[1]):
                if j in [0,1]:
                    line.set_visible(False)
                j += 1
                if j==3:
                    j=0
        despine(ax)
        if ax.get_legend():
            ax.legend_.remove()
        ax.set_ylabel('')
        if i < 1:
            ax.set_xlabel('')
            ax.set_xticklabels([])
    ax00.set_ylabel('Ecological Balance', labelpad=2)
    
    # Define custom legends for each subplot
    # Custom legend handles with edge color and adjusted font size
    def create_legend_handles(color_dict, edgecolor='black'):
        return [Patch(facecolor=color, edgecolor=edgecolor, linewidth=1.2, label=label) 
                for label, color in color_dict.items()]

    # Create legend handles for each group
    custom_legends = {
        'CRC': create_legend_handles(CRC_color_map)
    }

    # Add legends to the right of each subplot with two columns, edge color, and smaller font size
    legend_fontsize = 8

    ax00.legend(handles=custom_legends['CRC'], loc=(1,-1), bbox_to_anchor=(1, 0.5),
                frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0, handletextpad=0.5)
    

    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'robustness_taxonomic_profiling_interactions_fig.svg')

    plt.subplots_adjust(top=0.98, bottom=0.112, left=0.09, right=0.79, hspace=0.3, wspace=0.35)
    plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    #plt.show()

def disease_indicators(gut_dir):
    def get_metric(participant_dict,metric):
        healthy_metric = []
        unhealthy_metric = []
        for participant_id, samples in participant_dict.items():                
            for sample in samples:
                if metric == 'Shannon':
                    metric_value = calculate_shannon_index(sample['Data'])
                elif metric == 'Number':
                    metric_value = calculate_nonzero_number(sample['Data'])
                else:
                    print(f'Metric:{metric} not allowed')
                if sample['Diagnosis'] == 'Healthy':
                    healthy_metric.append(metric_value)
                else:
                    unhealthy_metric.append(metric_value)
        
        return healthy_metric, unhealthy_metric

    fig = plt.figure(figsize = (18.4*0.393701, 11*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(2,2, height_ratios=[1/2,1/2], width_ratios=[1/2,1/2])
    
    ax00 = fig.add_subplot(gs[0,0])
    ax10 = fig.add_subplot(gs[1,0])
    ax01 = fig.add_subplot(gs[0,1])
    ax11 = fig.add_subplot(gs[1,1])

    full_data_dict_IBS_fecal_positive = MET.get_full_data_dict_IBS_Lijuan(gut_dir, fecal=True, positive=True)
    full_data_dict_IBS_fecal_negative = MET.get_full_data_dict_IBS_Lijuan(gut_dir, fecal=True, positive=False)
    full_data_dict_IBS_serum_positive = MET.get_full_data_dict_IBS_Lijuan(gut_dir, fecal=False, positive=True)
    full_data_dict_IBS_serum_negative = MET.get_full_data_dict_IBS_Lijuan(gut_dir, fecal=False, positive=False)
    data_dicts = [full_data_dict_IBS_fecal_positive, full_data_dict_IBS_fecal_negative, full_data_dict_IBS_serum_positive, full_data_dict_IBS_serum_negative]
    axs_list = [ax00,ax10,ax01,ax11]
    diagnosis_color_map = {"Healthy": '#5F60F5',  # Blue 
                           "IBS": "#ED3A32"  # Red
                           }
    category_order2 = ['Healthy', 'IBS']
    for ax, data_dict in zip(axs_list, data_dicts):
        healthy_substances, unhealthy_substances = get_metric(data_dict, metric='Shannon')
        IBS_substances_df = pd.DataFrame({'Metric Value': healthy_substances + unhealthy_substances, 
                                          'Diagnosis': ['Healthy'] * len(healthy_substances) + ['IBS'] * len(unhealthy_substances)})
        sns.boxplot(x='Diagnosis', y='Metric Value', hue='Diagnosis', data=IBS_substances_df, order=category_order2, 
                    palette=diagnosis_color_map, showfliers=False, legend=False, width=0.5, ax=ax)

    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'enzymes_substances.svg')

    plt.subplots_adjust(top=0.98, bottom=0.112, left=0.09, right=0.79, hspace=0.3, wspace=0.35)
    #plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    plt.show()

def plot_robustness_tradeoff_averaged(gut_dir, rootpath, rootpath2, rootpath3):
    fig = plt.figure(figsize = (18.4*0.393701, 14*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(3,3, height_ratios=[1/3, 1/3, 1/3],
                                width_ratios=[1/3, 1/3, 1/3])
    ax00 = fig.add_subplot(gs[0,0])
    ax10 = fig.add_subplot(gs[1,0], sharex=ax00)
    ax20 = fig.add_subplot(gs[2,0], sharex=ax00)
    ax01 = fig.add_subplot(gs[0,1])
    ax11 = fig.add_subplot(gs[1,1], sharex=ax01)
    ax21 = fig.add_subplot(gs[2,1], sharex=ax01)
    ax02 = fig.add_subplot(gs[0,2])
    ax12 = fig.add_subplot(gs[1,2], sharex=ax02)
    ax22 = fig.add_subplot(gs[2,2], sharex=ax02)
    
    diagnosis_color_map = {
        "H": '#1771a2ff',  # Blue
        "U": "#cd1d2eff",  # Red
        "Healthy": '#1771a2ff',
        "IBD": "#cd1d2eff"
    }
    
    category_order = ['H', 'U']
    paths = [rootpath] + [rootpath2] + [rootpath3]
    sizes = [200] + [500] + [500]
    axs = [(ax00, ax10, ax20), (ax01, ax11, ax21), (ax02, ax12, ax22)]

    for path, size, ax in zip(paths, sizes, axs):
        shannon_df, p_value_shannon = get_sampled_metric_df(path, data_type='taxonomy', metric='Shannon', sample_size=size)
        pathways_df, p_value_enzymes = get_sampled_metric_df(path, data_type='pathway', metric='Enzymes', sample_size=size)
        N_substances_df, p_value_substances = get_sampled_metric_df(path, data_type='substances', metric='Substances', sample_size=size)
        sns.boxplot(x='Diagnosis', y='Metric Value', hue='Diagnosis', data=shannon_df, order=category_order, 
                    palette=diagnosis_color_map, showfliers=False, legend=False, width=0.3, ax=ax[0])
        sns.boxplot(x='Diagnosis', y='Metric Value', hue='Diagnosis', data=pathways_df, order=category_order, 
                    palette=diagnosis_color_map, showfliers=False, legend=False, width=0.3, ax=ax[1])
        sns.boxplot(x='Diagnosis', y='Metric Value', hue='Diagnosis', data=N_substances_df, order=category_order, 
                    palette=diagnosis_color_map, showfliers=False, legend=False, width=0.3, ax=ax[2])
        ax[0].text(0,1.1, f'P={p_value_shannon:.3e}', fontsize=6, verticalalignment='center', transform=ax[0].transAxes)
        ax[1].text(0,1.1, f'P={p_value_enzymes:.3e}', fontsize=6, verticalalignment='center', transform=ax[1].transAxes)
        ax[2].text(0,1.1, f'P={p_value_substances:.3e}', fontsize=6, verticalalignment='center', transform=ax[2].transAxes)
    
    axs = [ax00, ax10, ax20, ax01, ax11, ax21, ax02, ax12, ax22]
    for ax in [ax00, ax10, ax20, ax01, ax11, ax21, ax02, ax12, ax22]:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='y', which='major', labelsize=8)
        ax.tick_params(axis='x', which='major', labelsize=8)
    for ax in [ax00, ax10, ax01, ax11, ax02, ax12]:
        ax.set_xticklabels([])

    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'robustness_tradeoff_averaged.svg')

    plt.subplots_adjust(top=0.98, bottom=0.112, left=0.09, right=0.97, hspace=0.25, wspace=0.35)
    plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    #plt.show()
def tradeoff_function(x, tf_type):
    if tf_type==0:
        return x / (1 + 5 * x**1.1), 1.8778618175382 # This value on the right is the x_max of the function
    elif tf_type==1:
        return x * np.exp(-0.5*x), 2
    elif tf_type==2:
        return x * np.exp(-0.6*x), 1.6666666703465
    else:
        print(f'Tradeoff function raises an erro because type:{tf_type} is not valid.')
        raise ValueError
def plot_robustness_tradeoff(gut_dir, realization_path, realization_path2, realization_path3):
    
    fig = plt.figure(figsize = (18.4*0.393701, 17*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(7,3, height_ratios=[0.2, 0.145, 0.145, 0.145, 0.145, 0.02, 0.2],
                               width_ratios=[1/3, 1/3, 1/3])
    ax00 = fig.add_subplot(gs[0,0])
    ax10 = fig.add_subplot(gs[1,0], sharex=ax00)
    ax20 = fig.add_subplot(gs[2,0], sharex=ax00)
    ax30 = fig.add_subplot(gs[3,0], sharex=ax00)
    ax40 = fig.add_subplot(gs[4,0], sharex=ax00)
    ax50 = fig.add_subplot(gs[6,0])
    ax01 = fig.add_subplot(gs[0,1])
    ax11 = fig.add_subplot(gs[1,1], sharex=ax01)
    ax21 = fig.add_subplot(gs[2,1], sharex=ax01)
    ax31 = fig.add_subplot(gs[3,1], sharex=ax01)
    ax41 = fig.add_subplot(gs[4,1], sharex=ax01)
    ax51 = fig.add_subplot(gs[6,1])
    ax02 = fig.add_subplot(gs[0,2])
    ax12 = fig.add_subplot(gs[1,2], sharex=ax02)
    ax22 = fig.add_subplot(gs[2,2], sharex=ax02)
    ax32 = fig.add_subplot(gs[3,2], sharex=ax02)
    ax42 = fig.add_subplot(gs[4,2], sharex=ax02)
    ax52 = fig.add_subplot(gs[6,2])
    
    paths = [realization_path, realization_path2, realization_path3]
    axs = [[ax00,ax10,ax20,ax30,ax40,ax50],[ax01,ax11,ax21,ax31,ax41,ax51],[ax02,ax12,ax22,ax32,ax42,ax52]]
    sample_hours_list = [24*7,12*7,12*7]
    #get data for big realization
    for i, (path, ax_list, sample_hours) in enumerate(zip(paths, axs, sample_hours_list)):
        t_vec, S_tvec, B_types_dict, shannon_tvec, enzymatic_cost_tvec, D_data_dict, rho_tvec = get_realization_data(path, sample_hours, full=True)
        t_vec = t_vec/24/365 #(time in years)
        
        N_substances =  np.count_nonzero(S_tvec, axis=1)
        N_substances_rolling = pd.Series(N_substances).rolling(window=100, min_periods=1).mean()
        Dtotal_tvec = D_data_dict['Dtotal_tvec']
        
        Dtotal_df = pd.DataFrame({'Dtotal': np.concatenate(Dtotal_tvec).ravel()}) # See here for how to flatten a list of numpy arrays https://stackoverflow.com/questions/33711985/flattening-a-list-of-numpy-arrays
        
        palette = sns.color_palette("tab20c", n_colors=len(B_types_dict.keys()))
        color_mapping = {Type: palette[i % len(palette)] for i, Type in enumerate(B_types_dict.keys())}
        # Plot the Biomasses vs time
        for Type, (t_init, t_end, type_functional_identity, B_type_t) in B_types_dict.items():
            wanted_t = t_vec[(t_vec>=t_init/24/365) & (t_vec<t_end/24/365)]
            ax_list[0].plot(wanted_t,B_type_t,label=Type,color=color_mapping[Type])
        
        # Plot Shannon vs t
        ax_list[1].plot(t_vec, shannon_tvec, color='dimgrey')
        # Plot Enzymatic cost vs time
        ax_list[2].plot(t_vec, enzymatic_cost_tvec, color='dimgrey')
        # Plot Number of substances (rolling average)
        ax_list[3].plot(t_vec,N_substances, color='dimgrey', alpha=0.25)
        ax_list[3].plot(t_vec,[int(x) for x in N_substances_rolling], color='dimgrey')
        ax_list[4].plot(t_vec, rho_tvec, color='dimgrey')
        sns.histplot(Dtotal_df, x='Dtotal', binwidth=0.1, ax = ax_list[5], stat='probability', edgecolor='black', linewidth=0.5, color='dimgrey')
        ax_twin = ax_list[5].twinx()
        
        x_vals = np.linspace(Dtotal_df['Dtotal'].min(), Dtotal_df['Dtotal'].max(), 500)
        y_vals, x_max = tradeoff_function(x_vals, tf_type=i)
        y_max, _ = tradeoff_function(x_max, tf_type=i)
        ax_twin.plot(x_vals, y_vals, color='#dc143c', alpha=1, linewidth=1) # label=r"$C_{Total} \text{tradeoff}(C_{Total})$")
        ax_twin.axvline(x=x_max, ymin=0, ymax=y_max/ ax_twin.get_ylim()[1], color='#8b0000', linestyle='--', alpha=1, linewidth=1.25) # (0, (5,5))

        ax_list[4].set_xlabel('Time', fontsize=10)
        ax_list[5].set_xlabel(r'Total Enzymatic cost', labelpad=2, fontsize=10)
        despine(ax_twin)
        #ax_twin.spines['right'].set_visible(False)
        ax_twin.yaxis.set_ticks([])
        ax_twin.set_ylabel(None)
        #ax_twin.legend(loc="upper right")
        
        
        # Plot the second dataset on the right y-axis
        #ax_twin.set_ylabel('', color='dimgrey')
        #ax_twin.tick_params('y', colors='dimgrey')
        
    ax00.set_ylabel(r'Biomass', labelpad=2, fontsize=10)
    ax10.set_ylabel(r'Shannon', labelpad=2, fontsize=10)
    ax20.set_ylabel(r'# Enzymes', labelpad=2, fontsize=10)
    ax30.set_ylabel(r'# Substances', labelpad=2, fontsize=10)
    ax40.set_ylabel(r'Net Interaction', labelpad=2, fontsize=10)
    ax50.set_ylabel(r'Probability', labelpad=3, fontsize=10)
    ax51.set_ylabel(r'')
    ax52.set_ylabel(r'')
    
    
    for ax in [ax00,ax10,ax20,ax30,ax40,ax50,ax01,ax11,ax21,ax31,ax41,ax51,ax02,ax12,ax22,ax32,ax42,ax52]:
        despine(ax)    
        ax.tick_params(axis='both', labelsize=8)  
    for i,ax in enumerate([ax00,ax10,ax20,ax30,ax40]):
    # Set the x-axis limits to show only data from 3 to 63
        ax.set_xlim(3, 40)

        # Manually set the tick positions (corresponding to the original data values)
        tick_positions = [3, 13, 23, 33]  # These are the actual data points in the original data

        # Set the new tick labels to start from 0
        tick_labels = [0, 10, 20, 30]  # These are the custom labels you want to show

        # Update the x-ticks and their corresponding labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        if i not in [4]:
            ax.tick_params(labelbottom=False)
    for i,ax in enumerate([ax01,ax11,ax21,ax31,ax41,ax02,ax12,ax22,ax32,ax42]):
        # Set the x-axis limits to show only data from 3 to 63
        ax.set_xlim(3, 60)

        # Manually set the tick positions (corresponding to the original data values)
        tick_positions = [3, 13, 23, 33, 43, 53]  # These are the actual data points in the original data

        # Set the new tick labels to start from 0
        tick_labels = [0, 10, 20, 30, 40, 50]  # These are the custom labels you want to show

        # Update the x-ticks and their corresponding labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        if i not in [4,9]:
            ax.tick_params(labelbottom=False)
    
    ax00.set_ylim([0,4e12])
    ax00.set_yticks([0,2e12,4e12])
    ax01.set_ylim([0,1e13])
    ax01.set_yticks([0,0.5e13,1e13])
    ax02.set_ylim([0,1.5e13])
    ax02.set_yticks([0,0.75e13,1.5e13])
    ax10.set_ylim([1,3.4])
    ax10.set_yticks([1,2,3])
    ax11.set_ylim([1,3.5])
    ax11.set_yticks([1,2,3])
    ax12.set_ylim([0.5,3.2])
    ax12.set_yticks([1,2,3])
    ax20.set_ylim([27,100])
    ax20.set_yticks([50,75,100])
    ax21.set_ylim([0,220])
    ax21.set_yticks([0,100,200])
    ax22.set_ylim([0,110])
    ax22.set_yticks([0,50,100])
    ax30.set_ylim([18,30])
    ax30.set_yticks([20,25,30])
    ax31.set_ylim([15,30])
    ax31.set_yticks([18,24,30])
    ax32.set_ylim([3,30])
    ax32.set_yticks([10,20,30])
    ax40.set_ylim([-0.5,0.78])
    ax40.set_yticks([-0.5,0,0.5])
    ax41.set_ylim([-0.55,0.5])
    ax41.set_yticks([-0.5,0,0.5])
    ax42.set_ylim([-0.35,1])
    ax42.set_yticks([0,0.5,1])
    ax50.set_ylim([0,0.15])
    ax50.set_yticks([0,0.05,0.1])
    ax50.set_xlim([0.5,3.5])
    ax51.set_xlim([1,3])
    #ax51.set_yticks([-0.5,0,0.5])
    ax52.set_xlim([0.5,2.5])
    #ax52.set_yticks([0,0.5,1])
    
    '''
    ax00.set_yticks([0,1e12,2e12, 3e12])
    ax01.set_ylim([0,2e12])
    ax01.set_yticks([0,1e12,2e12])
    ax21.set_yticks([0,2e12,4e12])
    ax10.set_ylim((0.8,3.4))
    ax10.set_yticks([1,2,3])
    ax20.set_ylim([15,100])
    ax20.set_yticks([25,50,75,100])
    ax30.set_ylim([16,32])
    ax30.set_yticks([18,24,30])
    '''
    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'robustness_tradeoff.svg')

    plt.subplots_adjust(top=0.98, bottom=0.06, left=0.075, right=0.98, hspace=0.25, wspace=0.2)
    plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    #plt.show()

def plot_robustness_tradeoff_extra(gut_dir, sample_sizes=[300,300,500], data_type='taxonomy', metric='rho'):
    def map_diagnosis_labels(df):
        return df.replace({'Diagnosis': {'H': 'Healthy', 'U': 'Dysbiotic'}})
    rootpath1 = os.path.join(gut_dir, 'results', 'invasion', f'E_D_0', 'changing_tradeoff', 'fraction_nl_5_0_1.1')
    rootpath2 = os.path.join(gut_dir, 'results', 'invasion', f'E_D_0', 'changing_tradeoff', 'exponential_0.5_0_1')
    rootpath3 = os.path.join(gut_dir, 'results', 'invasion', f'E_D_0', 'changing_tradeoff', 'exponential_0.6_0_1')
    params_rootpath_dict = {1: rootpath1, 2: rootpath2, 3: rootpath3}
    # === Setup ===
    fig = plt.figure(figsize=(18.4 * 0.393701, 6 * 0.393701))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

    param_items = list(params_rootpath_dict.items())
    for i, (param_name, rootpath) in enumerate(param_items):
        ax = fig.add_subplot(gs[0, i])
        df, p_val = get_sampled_metric_df(rootpath, data_type, metric, sample_sizes[i])
        df = map_diagnosis_labels(df)
        sns.boxplot(data=df, x='Diagnosis', y='Metric Value', hue='Diagnosis',
                    showfliers=False, palette={'Healthy': '#0078B9', 'Dysbiotic': '#EA0017'},
                    ax=ax, legend=False)

        ax.set_title(f'{param_name}\np = {p_val:.2e}', fontsize=9)
        ax.set_xlabel('')
        ax.set_ylabel('Net Interaction', fontsize=9)
        ax.set_xticklabels(['Healthy', 'Dysbiotic'], fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        despine(ax)

    # === Save ===
    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.08, right=0.98, hspace=0.2, wspace=0.6)
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, 'robustness_tradeoff_extra.svg'), format='svg', dpi=1200)
    plt.show()

def efficiency_plots(gut_dir, realization_path1, realization_path2, realization_path3):
    fig = plt.figure(figsize = (18.4*0.393701, 12*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(3,3, height_ratios=[0.4, 0.3, 0.3],
                               width_ratios=[1/3, 1/3, 1/3])
    ax00 = fig.add_subplot(gs[0,0])
    ax10 = fig.add_subplot(gs[1,0], sharex=ax00)
    ax20 = fig.add_subplot(gs[2,0], sharex=ax00)
    ax01 = fig.add_subplot(gs[0,1])
    ax11 = fig.add_subplot(gs[1,1], sharex=ax01)
    ax21 = fig.add_subplot(gs[2,1], sharex=ax01)
    ax02 = fig.add_subplot(gs[0,2])
    ax12 = fig.add_subplot(gs[1,2], sharex=ax02)
    ax22 = fig.add_subplot(gs[2,2], sharex=ax02)

    V_colon = 0.41
    molar_mass = 100
    E_avec1 = DA.get_E_avec(realization_path1)
    E_avec2 = DA.get_E_avec(realization_path2)
    E_avec3 = DA.get_E_avec(realization_path3)
    E_avecs = [E_avec1, E_avec2, E_avec3]
    paths = [realization_path1, realization_path2, realization_path3]
    axs = [[ax00,ax10,ax20],[ax01,ax11,ax21],[ax02,ax12,ax22]]

    for i, (path, ax_list, E_avec) in enumerate(zip(paths, axs, E_avecs)):
        t_vec, S_tvec, B_types_dict, shannon_tvec, enzymatic_cost_tvec, D_data_dict, rho_tvec = get_realization_data(path, sample_hours=12*7, full=True)
        B_tvec = get_tvec(path, sample_hours=12*7, wanted_tvec='B')
        
        t_vec = t_vec/24/365 #(time in years)
        B_T_perspecies_tvec = [np.mean(B_ivec) for B_ivec in B_tvec]
        log_energy_available = np.log10(np.sum(S_tvec * E_avec *V_colon/molar_mass, axis=1))
        log_energy_available_rolling = pd.Series(log_energy_available).rolling(window=100, min_periods=1).mean() # SInce we will represent in log scale we average in log scale!!
        energy_available = np.sum(S_tvec * E_avec *V_colon/molar_mass, axis=1)
        energy_available_rolling = 10**log_energy_available_rolling #pd.Series(energy_available).rolling(window=100, min_periods=1).mean() #np.exp(log_energy_available_rolling)#


        palette = sns.color_palette("tab20c", n_colors=len(B_types_dict.keys()))
        color_mapping = {Type: palette[i % len(palette)] for i, Type in enumerate(B_types_dict.keys())}
        # Plot the Biomasses vs time
        for Type, (t_init, t_end, type_functional_identity, B_type_t) in B_types_dict.items():
            wanted_t = t_vec[(t_vec>=t_init/24/365) & (t_vec<t_end/24/365)]
            ax_list[0].plot(wanted_t,B_type_t,label=Type,color=color_mapping[Type])
        
        # Plot Shannon vs t
        ax_list[1].plot(t_vec, B_T_perspecies_tvec, color='dimgrey')
        ax_list[2].plot(t_vec,energy_available, color='dimgrey', alpha=0.25)
        ax_list[2].plot(t_vec,energy_available_rolling, color='dimgrey')
        ax_list[2].set_xlabel(r'Time', labelpad=2, fontsize=10)
        ax_list[2].set_yscale('log')

    ax00.set_ylabel(r'Biomass', labelpad=2, fontsize=10)
    ax10.set_ylabel(r'Average Biomass', labelpad=2, fontsize=10)
    ax20.set_ylabel(r'Energy available', labelpad=2, fontsize=10)
    
    
    for ax in [ax00,ax10,ax20,ax01,ax11,ax21,ax02,ax12,ax22]:
        despine(ax)    
        ax.tick_params(axis='both', labelsize=8)  
    for i,ax in enumerate([ax00,ax10,ax20,ax01,ax11,ax21,ax02,ax12,ax22]):
        # Set the x-axis limits to show only data from 3 to 63
        ax.set_xlim(5, 60)

        # Manually set the tick positions (corresponding to the original data values)
        tick_positions = [5, 15, 25, 35, 45, 55]  # These are the actual data points in the original data

        # Set the new tick labels to start from 0
        tick_labels = [0, 10, 20, 30, 40, 50]  # These are the custom labels you want to show

        # Update the x-ticks and their corresponding labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        if i not in [2,5,8]:
            ax.tick_params(labelbottom=False)
    
    ax10.set_ylim([0,2.5e11])
    ax10.set_yticks([0,1e11,2e11])
    ax11.set_ylim([0,3.7e11])
    ax11.set_yticks([0,1.5e11,3e11])
    ax12.set_ylim([0,3e11])
    ax12.set_yticks([0,1.5e11,3e11])
    ax20.set_ylim([7e-8,1e-6])
    #ax20.set_yticks([1e-8,1e-7,1e-6])
    ax21.set_ylim([5e-8,1e-6])
    #ax21.set_yticks([1e-8,1e-7,1e-6])
    ax22.set_ylim([6e-8,1e-6])
    #ax21.set_yticks([1e-8,1e-7,1e-6])
    '''
    ax20.set_ylim([1e-8,1e-6])
    ax20.set_yticks([1e-8,1e-7,1e-6])
    ax21.set_ylim([1e-8,1e-6])
    ax21.set_yticks([1e-8,1e-7,1e-6])
    '''
    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'efficiency_plots.svg')

    plt.subplots_adjust(top=0.975, bottom=0.09, left=0.075, right=0.98, hspace=0.25, wspace=0.2)
    plt.savefig(image_name, format='svg', transparent=True, dpi=2400)
    #plt.show()

def macroecological_patterns(gut_dir, rootpath, folder_list, sample_hours, subsample_model=None,  lower_threshold=None):
    def get_rel_B_df(t_vec, B_types_dict):
        B_df = pd.DataFrame({'t': t_vec})
        if len(t_vec) != len(set(t_vec)):
            print("The t_vec has values repeated. This would cause problems in the merging of the dataframes right after in the code. Solve that before proceeding")
            raise
        for Type, (t_init, t_end, type_functional_identity, B_type_t) in B_types_dict.items():
            wanted_t = t_vec[(t_vec>=t_init) & (t_vec<t_end)]
            aux_df = pd.DataFrame({'t': wanted_t, f'B_{Type}': B_type_t})
            B_df = B_df.merge(aux_df, how='left', on='t')
                
        
        excluded_columns = ['t']
        other_cols = B_df.columns.difference(excluded_columns)
        B_df['total_B'] = B_df[other_cols].sum(axis=1, skipna=True) # We sum all columns without including t_vec obviously!
        B_df[other_cols] = B_df[other_cols].div(B_df['total_B'], axis=0) # Remember here other cols does not include 'total_B' because it has been created after
        rel_B_df = B_df.copy()
        rel_B_df = rel_B_df.fillna(0)
        rel_B_df = rel_B_df.set_index('t')
        rel_B_df = rel_B_df.drop('total_B', axis=1).T
        return rel_B_df
    def collect_data(folder_list, sample_hours=12*7, coarsed_size=1, relaxation_time=0):
        # Initialize data containers for each group
        data_store = {
            'all': {'richness': [], 'mean': [], 'var': [], 'prev': [], 'abu': [], 'tres': [], 'mean_ranks': []},
            'healthy': {'richness': [], 'mean': [], 'var': [], 'prev': [], 'abu': [], 'tres': [], 'mean_ranks': []},
            'dysbiotic': {'richness': [], 'mean': [], 'var': [], 'prev': [], 'abu': [], 'tres': [], 'mean_ranks': []},
        }

        for folder in folder_list:
            folder_path = os.path.join(rootpath, folder)
            metrics = get_needed_data(folder_path, sample_hours, relaxation_time, coarsed_size=coarsed_size, subsample_model=subsample_model)

            for key in ['all', 'healthy', 'dysbiotic']:
                m = metrics[key]
                data_store[key]['richness'].extend(m['richness_distribution'])
                data_store[key]['mean'].extend(m['mean_ivec'])
                data_store[key]['var'].extend(m['variance_ivec'])
                data_store[key]['prev'].append(np.array(m['prevalence_ivec']))
                data_store[key]['abu'].extend(m['abu_change_distribution'])
                data_store[key]['tres'].extend(m['t_res_dist'])
                data_store[key]['mean_ranks'].append(m['mean_ivec'])

        # Process each group into output format
        def finalize_metrics(store):
            richness = np.array(store['richness'])
            mean = np.array(store['mean'])
            var = np.array(store['var'])
            abu = np.array(store['abu'])
            tres = np.array(store['tres'])
            mean_ranks = pad_vectors_to_same_length(store['mean_ranks'], pad_value=0)
            return [richness, mean, var, store['prev'], abu, tres, mean_ranks]

        return {'all': finalize_metrics(data_store['all']), 'Healthy': finalize_metrics(data_store['healthy']),'Dysbiotic': finalize_metrics(data_store['dysbiotic'])}
    def bin_data_log(x, y, bins):
        # Step 1: Remove non-positive values (log-space requires positive values)
        valid_mask = x > 0
        x = x[valid_mask]
        y = y[valid_mask]

        # Step 2: Define bin edges in log space
        log_min = np.log10(x.min())
        log_max = np.log10(x.max())
        bin_edges = np.logspace(log_min, log_max, bins + 1)

        # Step 3: Assign values to bins
        bin_centers = 10 ** (0.5 * (np.log10(bin_edges[:-1]) + np.log10(bin_edges[1:])))
        bin_indices = np.digitize(x, bin_edges) - 1

        # Step 4: Calculate mean and variance per bin
        binned_means = []
        binned_variances = []
        for i in range(bins):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                binned_means.append(bin_centers[i])
                binned_variances.append(np.mean(y[bin_mask]))

        return np.array(binned_means), np.array(binned_variances)
    def find_max_species_fixed_samples(df, min_samples):
        """ Finds the maximum number of species (N) such that N species are all present in at least min_samples time points,
        and extends the samples to include all time points where those species are present.
        
        Parameters:
        - df: DataFrame with species as rows and samples as columns (values are abundances).
        - min_samples: Minimum number of samples required for each species.
        
        Returns:
        - best_species: List of species that meet the criterion.
        - best_samples: List of sample IDs where all selected species are present.
        - extended_samples: List of all time points where the selected species are present.
        - filtered_df: Filtered DataFrame with the selected species and extended samples.
        """
        
        # Step 1: Calculate presence sets for each species
        species_presence = {species: set(df.columns[df.loc[species] > 0]) for species in df.index}
        
        # Filter species with fewer than min_samples time points
        species_info = pd.DataFrame({
            'presence': (df > 0).sum(axis=1),
            'median_abundance': df.median(axis=1)
        })
        
        eligible_species = species_info[species_info['presence'] >= min_samples].index.tolist()
        
        # Sort eligible species by presence and then by median abundance
        eligible_species.sort(key=lambda x: (species_info.loc[x, 'presence'], species_info.loc[x, 'median_abundance']), reverse=True)
        
        # Subset the presence dictionary to only eligible species
        filtered_species_presence = {species: samples for species, samples in species_presence.items() if species in eligible_species}
        
        # Step 2: Find the maximum number of species meeting the min_samples criteria
        best_species = []
        best_samples = []
        
        for start_idx, species in enumerate(eligible_species):
            current_species = [species]
            current_samples = filtered_species_presence[species]
            
            for next_species in eligible_species:
                if next_species == species:
                    continue  # Skip the current species
                
                new_samples = current_samples.intersection(filtered_species_presence[next_species])
                if len(new_samples) >= min_samples:
                    current_species.append(next_species)
                    current_samples = new_samples
            
            if len(current_species) > len(best_species):
                best_species = current_species
                best_samples = sorted(current_samples)
        
        # Step 3: Extend the samples to include all time points where all the selected species are present
        extended_samples = sorted(set.intersection(*(filtered_species_presence[species] for species in best_species)))
        
        # Step 4: Build the final filtered DataFrame
        filtered_df = df.loc[best_species, extended_samples]
        
        return best_species, extended_samples, filtered_df
    def get_lognormal_parameters(df_list, model=False):
        """
        Process the species DataFrame:
        - Order species by presence in samples and total abundance.
        - Convert zeros to NaNs.
        - Extract the first `x` rows.

        Parameters:
        - df: DataFrame with species as rows and samples as columns (values are relative abundances).

        Returns:
        - processed_df: Processed DataFrame.
        """
        # Define the equations
        def equations(vars):
            mu, sigma = vars
            # Equation 1: m1
            eq1 = (np.sqrt(2 / np.pi) * sigma * np.exp(-((np.log(c) - mu) ** 2) / (2 * sigma ** 2)) +
                (mu - m1) * erfc((np.log(c) - mu) / (np.sqrt(2) * sigma)))

            # Equation 2: m2
            eq2 = sigma ** 2 + m1 * mu + np.log(c) * m1 - mu * np.log(c) - m2
            return np.array([eq1, eq2])
        # Define the objective function
        def objective(vars):
            residuals = equations(vars)
            return np.sum(residuals**2)  # Minimize the sum of squared residuals
        log_means_list, mus, sigmas, cutoffs = [], [], [], []
        for df in df_list:
            # Step 0 delete rows with all 0 values
            df = df.loc[(df != 0).any(axis=1)]
            # Step 1: Calculate presence (non-zero counts) and total abundance
            mean_abundance = df.mean(axis=1).sort_values(ascending=False)  # Mean abundance for each species ordered
            
            log_means = np.log(mean_abundance)
            #plt.hist(log_means, bins=100)
            #plt.show()
            if model:
                # Filter out rows with the minimum log mean value because in the way the model is built is a clear outlier
                log_means = log_means[log_means > log_means.min()+1.5]
            m1 = np.mean(log_means)
            m2 = np.mean(log_means**2)
            c = np.exp(log_means.iloc[-1])
            
            # Constraints: sigma > 0
            constraints = [{'type': 'ineq', 'fun': lambda vars: vars[1]}]  # sigma > 0
            # Try different initial guesses
            initial_guess = [m1,1]
            result = minimize(objective, initial_guess, constraints=constraints, method='SLSQP', options={'ftol': 1e-18, 'maxiter': 10000})
            mu_solution, sigma_solution = result.x
            if math.isclose(mu_solution, m1, rel_tol=1e-4) or math.isclose(sigma_solution, m2, rel_tol=1e-4):
                print('Error in the mu, sigma estimation for log normal parameters. Change initial guess or solver.')
            log_means_list.append(log_means)
            mus.append(mu_solution)
            sigmas.append(sigma_solution)
            cutoffs.append(c)

        return log_means_list, mus, sigmas, cutoffs
    def get_bin_data_gamma(df, n_bin=10):
        # Step 1: Log-transform relative abundances
        df_log = np.log(df)  # Log-transform the relative abundances

        # Step 2: Calculate mean and standard deviation across samples (per species)
        mean_log = df_log.mean(axis=1)  # Mean of log-abundances (species-wise)
        std_log = df_log.std(axis=1)    # Standard deviation of log-abundances (species-wise)

        # Step 3: Normalize log-transformed values (per species)
        df_normalized = (df_log.sub(mean_log, axis=0)).div(std_log, axis=0)  # (l - ml) / sl

        # Step 4: Bin combined data for scatter plot
        min_val = df_normalized.min().min()  # Minimum value across all species and samples
        max_val = df_normalized.max().max()  # Maximum value across all species and samples
        bin_width = (max_val - min_val) / n_bin  # Bin width

        # Assign bins for combined plot
        df_bins = ((df_normalized - min_val) / bin_width).astype(int)

        # Count the number of species in each bin (for combined scatter plot)
        bin_counts = df_bins.stack().value_counts(normalize=True).sort_index()  # Normalized bin counts
        bin_probabilities = bin_counts / bin_width  # Probability density
        # Scatter plot of combined binned probabilities
        bin_centers = bin_counts.index * bin_width + min_val  # Bin centers
        
        return bin_centers, bin_probabilities
    def get_aggregated_bin_data_gamma(df_list, n_bin=10):
        """
        Aggregates binned probabilities across multiple datasets, calculates mean and 95% confidence intervals,
        and returns data for plotting.

        Parameters:
        - df_list: List of DataFrames, each representing relative abundances (species as rows, samples as columns).
        - n_bin: Number of bins for the histogram.

        Returns:
        - bin_centers: Centers of the bins.
        - avg_probabilities: Average probabilities across datasets.
        - ci_lower: Lower bound of the 95% confidence interval for each bin.
        - ci_upper: Upper bound of the 95% confidence interval for each bin.
        """
        all_normalized_data = []

        # Step 1: Normalize and combine data
        for df in df_list:
            # Log-transform
            df_log = np.log(df.replace(0, np.nan)).dropna(how='all')  # Handle zeros with NaN
            mean_log = df_log.mean(axis=1)
            std_log = df_log.std(axis=1)

            # Normalize
            df_normalized = (df_log.sub(mean_log, axis=0)).div(std_log, axis=0)
            all_normalized_data.extend(df_normalized.stack().values)

        all_normalized_data = pd.Series(all_normalized_data)

        # Step 2: Calculate global min, max, and bin width
        min_val = all_normalized_data.min()
        max_val = all_normalized_data.max()
        bin_width = (max_val - min_val) / n_bin
        bin_edges = np.linspace(min_val, max_val, n_bin + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Step 3: Aggregate counts across all datasets
        bin_probabilities_list = []

        for df in df_list:
            # Log-transform and normalize
            df_log = np.log(df.replace(0, np.nan)).dropna(how='all')
            mean_log = df_log.mean(axis=1)
            std_log = df_log.std(axis=1)
            df_normalized = (df_log.sub(mean_log, axis=0)).div(std_log, axis=0)

            # Bin data
            normalized_data = df_normalized.stack().values
            bin_indices = np.digitize(normalized_data, bin_edges) - 1  # Map data to bins
            bin_indices = bin_indices[(bin_indices >= 0) & (bin_indices < n_bin)]  # Ignore out-of-range data

            # Count values in each bin
            bin_counts = pd.Series(bin_indices).value_counts().sort_index()
            probabilities = bin_counts / (len(normalized_data) * bin_width)

            # Align probabilities with global bin centers
            aligned_probs = pd.Series(0, index=range(n_bin), dtype=float)
            aligned_probs[bin_counts.index] = probabilities
            bin_probabilities_list.append(aligned_probs)
            #print(f'Probabilities: {aligned_probs}')
            #print(f'Bin counts: {bin_counts}')
            #print(f'Total counts, bin_width: {len(normalized_data), bin_width}')

        # Step 4: Calculate average probabilities and 95% confidence intervals
        bin_probabilities_df = pd.DataFrame(bin_probabilities_list)
        avg_probabilities = bin_probabilities_df.mean(axis=0).values
        ci_lower = bin_probabilities_df.quantile(0.025, axis=0).values
        ci_upper = bin_probabilities_df.quantile(0.975, axis=0).values

        return bin_centers, avg_probabilities, ci_lower, ci_upper
    def get_aggregated_bin_data_lognormal(log_means_list, mus, sigmas, cutoffs, n_bin=10, n_threshold=10):
        """
        Aggregates binned probabilities across multiple datasets, calculates mean and 95% confidence intervals,
        and returns data for plotting shaded areas.

        Parameters:
        - log_means_list: List of Series, each representing log-transformed relative abundances.
        - mus: List of mean values for each dataset.
        - sigmas: List of standard deviations for each dataset.
        - cutoffs: List of cutoff values for each dataset.
        - n_bin: Number of bins for the histogram.
        - n_threshold: Minimum number of counts per bin to include.

        Returns:
        - bin_centers: Centers of the bins (rescaled by sqrt(2)).
        - avg_probabilities: Average probabilities across datasets.
        - lower_CI: Lower bound (2.5th percentile) for each bin.
        - upper_CI: Upper bound (97.5th percentile) for each bin.
        """
        all_filtered_data = []

        # Step 1: Normalize and filter each dataset
        for log_means, mu, sigma, c in zip(log_means_list, mus, sigmas, cutoffs):
            filtered_data = log_means[log_means >= np.log(c)]
            normalized_data = (filtered_data - mu) / sigma
            all_filtered_data.extend(normalized_data)

        all_filtered_data = pd.Series(all_filtered_data)

        # Step 2: Calculate global min and max
        min_val = all_filtered_data.min()
        max_val = all_filtered_data.max()
        bin_width = (max_val - min_val) / n_bin

        # Step 3: Aggregate counts across all datasets
        bin_edges = np.linspace(min_val, max_val, n_bin + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_probabilities_list = []

        for log_means, mu, sigma, c in zip(log_means_list, mus, sigmas, cutoffs):
            filtered_data = log_means[log_means >= np.log(c)]
            normalized_data = (filtered_data - mu) / sigma
            bin_indices = ((normalized_data - min_val) / bin_width).astype(int)

            # Count values in each bin
            bin_counts = bin_indices.value_counts().sort_index()
            total_count = 2 * bin_counts.sum() / erfc((np.log(c) - mu) / np.sqrt(2 * sigma**2))

            # Filter bins with sufficient counts
            filtered_bins = bin_counts[bin_counts >= n_threshold]
            probabilities = filtered_bins / (total_count * bin_width)

            # Align probabilities with global bin centers
            aligned_probs = pd.Series(0, index=range(n_bin), dtype=float)
            aligned_probs[filtered_bins.index] = probabilities
            bin_probabilities_list.append(aligned_probs)

        # Step 4: Calculate average and 95% CI for probabilities
        bin_probabilities_df = pd.DataFrame(bin_probabilities_list)
        avg_probabilities = bin_probabilities_df.mean(axis=0).values
        lower_CI = bin_probabilities_df.quantile(0.025, axis=0).values
        upper_CI = bin_probabilities_df.quantile(0.975, axis=0).values


        return bin_centers, avg_probabilities, lower_CI, upper_CI
    def get_bin_data_lognormal(log_means, mu, sigma, c, n_bin=10, n_threshold=10):
        """
        Processes log-transformed and rescaled relative abundances, bins the data, 
        and returns bin centers and probabilities.

        Parameters:
        - df: DataFrame with species as rows and samples as columns (relative abundances).
        - n_bin: Number of bins for the histogram.

        Returns:
        - bin_centers: Centers of the bins.
        - bin_probabilities: Normalized probabilities for each bin.
        - min_val: Minimum value of the normalized data.
        - max_val: Maximum value of the normalized data.
        """
        # Step 1: Filter data above the cutoff
        filtered_data = log_means[log_means >= np.log(c)]
        filtered_data = (filtered_data-mu)/sigma
        #mu = np.mean(log_means)
        #sigma = np.std(log_means) 
        # Step 2: Calculate bin width
        min_val = filtered_data.min()
        max_val = filtered_data.max()
        bin_width = (max_val - min_val) / n_bin

        # Step 3: Assign bins
        bin_indices = ((filtered_data - min_val) / bin_width).astype(int)

        # Step 4: Calculate probabilities
        bin_counts = bin_indices.value_counts().sort_index()
        #total_count = bin_counts.sum()
        total_count = 2*bin_counts.sum()/erfc((np.log(c)-mu)/np.sqrt(2*sigma**2))
        
        # Filter bins with n > n_threshold
        filtered_bins = bin_counts[bin_counts >= n_threshold]

        # Calculate probabilities for filtered bins
        probabilities = filtered_bins / (total_count * bin_width)
        
        # Step 5: Rescale for plotting
        bin_centers = filtered_bins.index * bin_width + min_val
        #rescaled_x = (bin_centers - mu) / sigma
        '''
        print(f'Bin counts: {bin_counts}')
        print(f'Log means: {log_means}')
        print(f'Total count observed: {bin_counts.sum()}')
        print(f'Total count estimated: {total_count}')
        '''
        # Step 6: Adjust probabilities for comparison
        #adjusted_probabilities = np.power(10, np.log(probabilities) -
        #                                np.log(0.5 * erfc((mu - np.log(c)) / (np.sqrt(2) * sigma))) +
        #                                0.5 * np.log(2 * np.pi))
        return bin_centers, probabilities
    def log_gamma_standardized(x, k, a=1):
        """If data for the abundances is gamma distributed, this is the function for a variable Z=(log(x)-mean)/sigma 
        where mean and sigma are the average and std from log(x). The average is the digamma(k)+log(a) and std is sqrt(trigamma)"""
        return k * (np.sqrt(polygamma(1, k)) * x + digamma(k) +np.log(a) - np.exp( np.sqrt(polygamma(1, k)) * x + digamma(k) + np.log(a) ) + np.log(k)) - np.log(gamma(k)) + np.log(np.sqrt(polygamma(1, k)))
    def log_gamma_defined(x,a,b,c):
        return a*x+b*np.exp(x)+c
    def log_gamma_jacopo(x, k):
        return ( k*polygamma(1,k)*x - np.exp( np.sqrt(polygamma(1,k))*x+ digamma(k)) ) - np.log(gamma(k)) + k*digamma(k) + np.log10(np.exp(1))
    def lognormal(x):
        """Log-normal function."""
        return x**2
    def loggamma(z,b,a=1):
        return 1/gamma(b) * (b/a)**b * np.exp(b*z) *np.exp(-b/a*np.exp(z))

    def gamma_plot(ax, F4_df, M3_df, model_df_list):
        best_species, best_samples, filtered_df_M3 = find_max_species_fixed_samples(M3_df, min_samples=200)
        best_species, best_samples, filtered_df_F4 = find_max_species_fixed_samples(F4_df, min_samples=120)
        filtered_df_model_list = []
        for model_df in model_df_list:
            best_species, best_samples, filtered_df_model = find_max_species_fixed_samples(model_df, min_samples=200)
            filtered_df_model_list.append(filtered_df_model)
        #print(f"The gamma dfs shapes: {(filtered_df_model.shape, filtered_df_F4.shape, filtered_df_M3.shape)}")
        bin_centers_model, avg_probabilities_model, ci_lower, ci_upper = get_aggregated_bin_data_gamma(filtered_df_model_list, n_bin=15)       
        bin_centers_M3, bin_probabilities_M3, = get_bin_data_gamma(filtered_df_M3, n_bin=15)
        bin_centers_F4, bin_probabilities_F4 = get_bin_data_gamma(filtered_df_F4, n_bin=15)
        aux = [#(bin_centers_model, avg_probabilities_model, 'dimgrey', 'o'), 
               (bin_centers_M3, bin_probabilities_M3, '#c2a5cf', '^'), 
               (bin_centers_F4, bin_probabilities_F4, '#7b3294', 's')]
        for (bin_centers, bin_probabilities, color, m) in aux:
            ax.scatter(bin_centers, bin_probabilities, color=color, label="Combined Binned Probabilities", marker=m,
                       alpha=0.7, s=30, edgecolor='Black', linewidth=0.5)
        ax.fill_between(bin_centers_model, ci_lower, ci_upper, color='#999999', alpha=0.3, zorder=0)
        # Add gamma distribution curve
        x_vals = np.linspace(-6, 3.5, 100)
        #ax.plot(x_vals, np.exp(log_gamma_standardized(x_vals, k=1.4, a=1)), color="black", label="Gamma Distribution", linewidth=2)
        ax.plot(x_vals, np.exp(log_gamma_defined(x_vals, a=1.4, b=-1.0, c=0.15)), color="black", label="Gamma Distribution", linewidth=2, zorder=0)
        #ax.plot(x_vals, np.exp(log_gamma_jacopo(x_vals, k=1.7)), color="red", label="Gamma Distribution", linewidth=2)
        
        # Configure plot
        ax.set_xlim([-5.5,3.2])
        ax.set_xticks([-4,-2,0,2])
        ax.set_ylim([1e-3,1])
        ax.set_yticks([1e-3,1e-2,1e-1,1e+0])
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlabel("Rescaled log relative abundance")
        ax.set_ylabel("Probability density")
        return ax
    def taylor_plot(ax, F4_metrics_dict, M3_metrics_dict, all_means_model, all_variances_model, bins=50):
        """
        Plots variance vs. mean abundance for real and model data, with binned real data and 95% CI for the model.

        Parameters:
        - ax: Matplotlib axis to plot on.
        - F4_metrics_dict: Dictionary containing F4 mean and variance data.
        - M3_metrics_dict: Dictionary containing M3 mean and variance data.
        - all_means_model: List of mean abundances from the model realizations.
        - all_variances_model: List of variance values from the model realizations.
        - bins: Number of bins for the histogram.
        """
        
        # Step 1: Get 95% CI for model data
        bin_centers_model, averages, lower_percentiles, upper_percentiles = get_95_CI(
            all_means_model, all_variances_model, log=True, bins=bins
        )

        # Step 2: Bin F4 and M3 data
        F4_means, F4_variances = bin_data_log(
            np.array(F4_metrics_dict['mean_ivec']),
            np.array(F4_metrics_dict['variance_ivec']),
            bins
        )
        M3_means, M3_variances = bin_data_log(
            np.array(M3_metrics_dict['mean_ivec']),
            np.array(M3_metrics_dict['variance_ivec']),
            bins
        )

        # Step 3: Plot 95% CI for model
        #ax.scatter(bin_centers_model, averages, color='dimgrey', alpha=0.7, s=30, edgecolor='Black', linewidth=0.5)
        ax.fill_between(bin_centers_model, lower_percentiles, upper_percentiles, color='#999999', alpha=0.3, label='95% CI', zorder=0)

        # Step 4: Plot binned real data
        ax.scatter(F4_means, F4_variances, color='#7b3294', alpha=0.7, label="Binned F4 Data", s=30, edgecolor='Black', linewidth=0.5, marker='s')
        ax.scatter(M3_means, M3_variances, color='#c2a5cf', alpha=0.7, label="Binned M3 Data", s=30, edgecolor='Black', linewidth=0.5, marker='^')
        x_vals = np.logspace(np.log10(bin_centers_model.min()), np.log10(bin_centers_model.max()),100)
        ax.plot(x_vals, 0.1*x_vals**1.5, color='black', alpha=1, label="Theory", linewidth=2, zorder=0)

        # Step 5: Set axis scales and labels
        ax.set_ylim([3e-12,1e-1])
        ax.set_yticks([1e-10,1e-7,1e-4,1e-1])
        ax.set_xlim([1e-7,1e-1])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Mean Abundance")
        ax.set_ylabel("Variance")
        return ax

    def lognormal_plot(ax, F4_df, M3_df, model_df_list):
        log_means_model_list, mu_model_list, sigma_model_list, c_model_list = get_lognormal_parameters(model_df_list, model=True)
        [log_means_M3], [mu_M3], [sigma_M3], [c_M3] = get_lognormal_parameters([M3_df])
        [log_means_F4], [mu_F4], [sigma_F4], [c_F4] = get_lognormal_parameters([F4_df])
        #bin_centers_model, bin_probabilities_model = get_bin_data_lognormal(log_means_model, mu_model, sigma_model, c_model, n_bin=12, n_threshold=2)
        #bin_centers_model, avg_probabilities, avg_minus_std, avg_plus_std = get_aggregated_bin_data_lognormal(log_means_model_list, mu_model_list, 
                                                                                                                 #sigma_model_list, c_model_list, 
                                                                                                                 #n_bin=12, n_threshold=2)
        bin_centers_model, probabilities, ci_lower, ci_upper = get_aggregated_bin_data_lognormal(log_means_model_list, mu_model_list, 
                                                                                                 sigma_model_list, c_model_list, 
                                                                                                 n_bin=12, n_threshold=2)
        bin_centers_M3, bin_probabilities_M3 = get_bin_data_lognormal(log_means_M3, mu_M3, sigma_M3, c_M3, n_bin=15, n_threshold=2)
        bin_centers_F4, bin_probabilities_F4 = get_bin_data_lognormal(log_means_F4, mu_F4, sigma_F4, c_F4, n_bin=15, n_threshold=2)
        
        aux = [#(bin_centers_model, probabilities, 'dimgrey', 'o'), 
               (bin_centers_M3, bin_probabilities_M3, '#c2a5cf', 's'), 
               (bin_centers_F4, bin_probabilities_F4,  '#7b3294', '^')]
        for (bin_centers, bin_probabilities, color, m) in aux:
            x = bin_centers#/np.sqrt(2)
            y = bin_probabilities # already adjusted in get_bin_data_lognormal
            ax.scatter(x, y, color=color, label="Combined Binned Probabilities", marker=m, alpha=0.7, zorder=2, s=30, edgecolor='Black', linewidth=0.5)
        ax.fill_between(bin_centers_model, ci_lower, ci_upper, color='#999999', alpha=0.3, zorder=0)
        #center = (np.log(c_model)-mu_model)/np.sqrt(2*sigma_model**2)
        #ax.axvline(x=center, color='red', linestyle='--')
        # Add gamma distribution curve
        x_vals = np.linspace(-5.1, 5.1, 100)
        #ax.plot(x_vals/np.sqrt(2), 10**(-1/250*(x_vals+1.5)**2)-0.85, color="black", label="Lognormal", linewidth=2)
        ax.plot(x_vals, 10**(-(x_vals/2)**2-0.4), color="black", label="Lognormal", linewidth=2, zorder=0)

        # Configure plot
        ax.set_ylim([1e-3,1])
        ax.set_xlim([-3.5,3.5])
        ax.set_xticks([-2,0,2])
        ax.set_yticks([1e-3,1e-2,1e-1,1e+0])
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlabel("Rescaled log average relative abundance")
        ax.set_ylabel("Probability density")
        return ax
    def prevalence_dist_plot(ax, F4_metrics_dict, M3_metrics_dict, all_prevalences_list, bins=20):
        """
        Plots the prevalence distribution using scatter points for F4 and M3 data
        and overlays the average and 95% CI for the model's realizations.

        Parameters:
        - ax: Matplotlib axis to plot on.
        - F4_metrics_dict: Dictionary containing F4 prevalence data.
        - M3_metrics_dict: Dictionary containing M3 prevalence data.
        - all_prevalences_list: List of prevalence arrays (this arrays has prevalences for each species, ie. ivecs) for model realizations.
        - bins: Number of bins for the histogram.
        """
        # Plot F4 scatter histogram
        F4_prevalences = F4_metrics_dict['prevalence_ivec']
        F4_hist, F4_bin_edges = np.histogram(F4_prevalences, bins=bins, density=True)
        F4_bin_centers = 0.5 * (F4_bin_edges[:-1] + F4_bin_edges[1:])
        ax.scatter(F4_bin_centers, F4_hist, color='#7b3294', alpha=0.7, label="F4 Real Data", s=30, edgecolor='Black', linewidth=0.5, marker='s')

        # Plot M3 scatter histogram
        M3_prevalences = M3_metrics_dict['prevalence_ivec']
        M3_hist, M3_bin_edges = np.histogram(M3_prevalences, bins=bins, density=True)
        M3_bin_centers = 0.5 * (M3_bin_edges[:-1] + M3_bin_edges[1:])
        ax.scatter(M3_bin_centers, M3_hist, color='#c2a5cf', alpha=0.7, label="M3 Real Data", s=30, edgecolor='Black', linewidth=0.5, marker='^')

        # Aggregate model realizations
        all_prevalences = np.concatenate(all_prevalences_list)
        min_val = all_prevalences.min()
        max_val = all_prevalences.max()
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Calculate probabilities and 95% CI for each bin
        binned_probs = []
        for prevalences in all_prevalences_list:
            hist, _ = np.histogram(prevalences, bins=bin_edges, density=True)
            binned_probs.append(hist)

        binned_probs_df = pd.DataFrame(binned_probs)
        avg_probs = binned_probs_df.mean(axis=0).values
        lower_ci = binned_probs_df.quantile(0.025, axis=0).values
        upper_ci = binned_probs_df.quantile(0.975, axis=0).values

        # Plot the average and 95% CI for model data
        #ax.scatter(bin_centers, avg_probs, color='dimgrey', alpha=0.7, label="Model Average", s=30, edgecolor='Black', linewidth=0.5)
        ax.fill_between(bin_centers, lower_ci, upper_ci, color='#999999', alpha=0.3, label="95% CI", zorder=0)

        # Set labels and legend
        #ax.set_ylim([1e-2,25])
        #ax.set_yticks([1e-2,1e-1,1e0,1e1])
        ax.set_xlabel("Prevalence")
        ax.set_ylabel("Probability")
        ax.set_yscale('log')
        return ax
    
    def prevalence_mean_plot(ax, F4_metrics_dict, M3_metrics_dict, all_prevalences_list, all_means, bins=50):
        all_prevalences = np.concatenate(all_prevalences_list)
        bin_centers_model, averages, lower_percentiles, upper_percentiles = get_95_CI(all_means, all_prevalences, bins =bins)

        # Subplot 5: Prevalence vs Mean Abundance
        #ax.scatter(bin_centers_model, averages, color='dimgrey', alpha=0.7, s=30, edgecolor='Black', linewidth=0.5)
        ax.fill_between(bin_centers_model, lower_percentiles, upper_percentiles, color='#999999', alpha=0.3, label='95% CI', zorder=0)

        # Bin the F4 M3 data
        bin_centers_F4, avg_F4 = bin_data_log(np.array(F4_metrics_dict['mean_ivec']), np.array(F4_metrics_dict['prevalence_ivec']), bins)
        bin_centers_M3, avg_M3 = bin_data_log(np.array(M3_metrics_dict['mean_ivec']),np.array(M3_metrics_dict['prevalence_ivec']),bins)

        ax.scatter(bin_centers_F4,avg_F4,color='#7b3294',alpha=0.7,label='F4 Binned Data', s=30, edgecolor='Black', linewidth=0.5, marker='s')

        # Plot binned M3 data
        ax.scatter(bin_centers_M3,avg_M3,color='#c2a5cf',alpha=0.7,label='M3 Binned Data', s=30, edgecolor='Black', linewidth=0.5, marker='^')

        #ax.scatter(F4_metrics_dict['mean_ivec'], F4_metrics_dict['prevalence_ivec'], color='red', alpha=0.7, label="Real Data")
        #ax.scatter(M3_metrics_dict['mean_ivec'], M3_metrics_dict['prevalence_ivec'], color='blue', alpha=0.7, label="Real Data")
        ax.set_xlim([1e-8,3e-1])
        ax.set_xticks([1e-7,1e-5,1e-3,1e-1])
        ax.set_xscale('log')
        ax.set_xlabel("Mean Abundance")
        ax.set_ylabel("Prevalence")
        return ax

    def rank_plot(ax, F4_metrics_dict, M3_metrics_dict, all_means_for_ranks, bins=20):
        # Subplot 7: Ranked distribution for mean
        real_sorted_F4 = np.sort(F4_metrics_dict['mean_ivec'])[::-1]  # Sort in descending order
        real_ranks_F4 = np.arange(1, len(real_sorted_F4) + 1)  # Assign ranks

        real_sorted_M3 = np.sort(M3_metrics_dict['mean_ivec'])[::-1]  # Sort in descending order
        real_ranks_M3 = np.arange(1, len(real_sorted_M3) + 1)  # Assign ranks

        num_ranks_model = all_means_for_ranks.shape[1]
        sorted_all_means_for_ranks = -np.sort(-all_means_for_ranks, axis=1)  # Sort in descending order for model data
        sorted_all_means_for_ranks = np.where(sorted_all_means_for_ranks == 0, np.nan, sorted_all_means_for_ranks)

        # Calculate 95% confidence intervals and averages
        lower_percentiles = np.nanpercentile(sorted_all_means_for_ranks, 2.5, axis=0)
        upper_percentiles = np.nanpercentile(sorted_all_means_for_ranks, 97.5, axis=0)
        average_model = np.nanmean(sorted_all_means_for_ranks, axis=0)

        # Plot real data (F4 and M3)
        ax.scatter(real_ranks_F4, real_sorted_F4, label='Real Data F4', color='#7b3294', alpha=0.7, marker='s', s=30, linewidth=0.5)
        ax.scatter(real_ranks_M3, real_sorted_M3, label='Real Data M3', color='#c2a5cf', alpha=0.7, marker='^', s=30, linewidth=0.5)

        # Plot model data (95% CI and average)
        ax.fill_between(np.arange(1, num_ranks_model + 1), lower_percentiles, upper_percentiles, 
                        color='#999999', alpha=0.3, label='95% CI', zorder=0)
        #ax.scatter(np.arange(1, num_ranks_model + 1), average_model, label='Model Average', 
        #        color='dimgrey', alpha=0.7, s=30, linewidth=0.5)
        ax.set_xlim([3,7.5e3])
        ax.set_xticks([1e0,1e1,1e2,1e3])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('Mean Abundance', fontsize=12)

    groups = ['all', 'Healthy', 'Dysbiotic']

    # Collect model data for both folder lists
    model_data = collect_data(folder_list, sample_hours=sample_hours, coarsed_size=1, relaxation_time=0)

    # Load host data and compute real metrics
    F4_df, M3_df = get_real_data(gut_dir, family=False, lower_threshold=lower_threshold)
    M3_metrics_dict = compute_bacterial_metrics(M3_df)
    F4_metrics_dict = compute_bacterial_metrics(F4_df)
    M3_df = M3_df.set_index('t').T
    F4_df = F4_df.set_index('t').T

    # Load model_df_list only once
    model_df_list = []
    for folder in folder_list:
        realization_path = os.path.join(rootpath, folder)
        t_vec, _, B_types_dict, _, _, _, _ = get_realization_data(realization_path, sample_hours=sample_hours, full=False)
        model_df_list.append(get_rel_B_df(t_vec, B_types_dict))

    # Plot for each group
    for group in groups:
        fig = plt.figure(figsize=(18.4 * 0.393701, 12 * 0.393701))
        gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 0.5], width_ratios=[1 / 3] * 3)
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax02 = fig.add_subplot(gs[0, 2])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
        ax12 = fig.add_subplot(gs[1, 2])

        # Unpack group-specific model metrics
        _, all_means, all_variances, all_prevalences_list, _, _, all_means_for_ranks = model_data[group]

        # Plotting
        gamma_plot(ax00, F4_df, M3_df, model_df_list)
        taylor_plot(ax01, F4_metrics_dict, M3_metrics_dict, all_means, all_variances, bins=15)
        lognormal_plot(ax02, F4_df, M3_df, model_df_list)
        prevalence_dist_plot(ax10, F4_metrics_dict, M3_metrics_dict, all_prevalences_list, bins=20)
        prevalence_mean_plot(ax11, F4_metrics_dict, M3_metrics_dict, all_prevalences_list, all_means, bins=20)
        rank_plot(ax12, F4_metrics_dict, M3_metrics_dict, all_means_for_ranks, bins=20)

        for ax in fig.get_axes():
            ax.xaxis.label.set_size(10)
            ax.yaxis.label.set_size(10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            ax.minorticks_off()

        # Save output
        image_dir = os.path.join(gut_dir, 'images', 'paperplots')
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"{group}_macroecological_patterns_rebuttal_{subsample_model}.svg")

        plt.subplots_adjust(top=0.975, bottom=0.12, left=0.075, right=0.98, hspace=0.35, wspace=0.4)
        plt.savefig(image_path, format='svg', transparent=True, dpi=1200)
        plt.close(fig)
def IBS_CRC_biomarkers(gut_dir):
    # Function to preprocess enzyme data
    def preprocess_enzymes(metadata_path, enzyme_data_path, healthy_label, unhealthy_labels, sample_id_col, cohort_col=None):
        metadata = pd.read_csv(metadata_path)
        metadata[sample_id_col] = metadata[sample_id_col].astype(str)
        if cohort_col:
            metadata = metadata[metadata[cohort_col].isin([healthy_label] + unhealthy_labels)]
        metadata['Group'] = metadata[cohort_col].replace({healthy_label: "Healthy", **{label: "Unhealthy" for label in unhealthy_labels}})
        enzyme_data = pd.read_csv(enzyme_data_path, index_col=0)
        # Use only intersecting sample IDs
        sample_ids = list(set(metadata[sample_id_col]).intersection(enzyme_data.columns))
        enzyme_data = enzyme_data[sample_ids]
        
        # Count the number of enzymes per sample
        enzyme_counts = (enzyme_data > 0).sum(axis=0)
        
        # Merge with metadata for grouping
        metadata_filtered = metadata[metadata[sample_id_col].isin(sample_ids)]
        merged = metadata_filtered.merge(enzyme_counts.rename('Num Enzymes'), left_on=sample_id_col, right_index=True)
        return merged[['Group', 'Num Enzymes']]
    # Function to preprocess metabolite data
    def preprocess_metabolites(metadata_path, data_path, group_col, metabolite_col, patient_col, healthy_label, unhealthy_labels):
        metadata = pd.read_csv(metadata_path)[[patient_col, group_col]]
        metadata[patient_col] = metadata[patient_col].astype(str)
        data = pd.read_csv(data_path)
        data.set_index(metabolite_col, inplace=True)
        data = data.T
        metadata['Group'] = metadata[group_col].replace({healthy_label: "Healthy", **{label: "Unhealthy" for label in unhealthy_labels}})
        merged = metadata.merge(data, left_on=patient_col, right_index=True)
        def calculate_shannon(row):
            proportions = row / row.sum()
            return -np.sum(proportions * np.log(proportions + 1e-9))
        merged['Shannon Index'] = merged.iloc[:, 2:].apply(calculate_shannon, axis=1)
        return merged[['Group', 'Shannon Index']]
    # Function to add p-values to plots
    def add_p_value(ax, x1, x2, y, p_value):
        """
        Add p-value annotations to the plot.
        - ax: axis to annotate.
        - x1, x2: indices of the groups being compared.
        - y: height at which to place the annotation.
        - p_value: the p-value to display.
        """
        ax.plot([x1, x2], [y, y], lw=1.5, color='black')
        ax.text((x1 + x2) / 2, y + 0.02, f'p = {p_value:.3g}', ha='center', va='bottom', fontsize=10)
    real_data_path = os.path.join(gut_dir, 'real_data')
    # Function to relabel groups for plotting
    def relabel_groups(data, unhealthy_label):
        """Relabel 'Unhealthy' to a specific label (e.g., 'IBS' or 'CRC') for plotting."""
        data = data.copy()
        data['Group'] = data['Group'].replace({'Unhealthy': unhealthy_label})
        return data
    # Paths to your local files (adjust these paths)
    paths = {
        'IBS1_metadata': os.path.join(real_data_path, 'IBS_Mars_Cell', 'stool_metadata.csv'),
        'IBS1_data': os.path.join(real_data_path, 'IBS_Mars_Cell', 'enzymes_stool_data.csv'),
        'CRC_metadata': os.path.join(real_data_path, 'CRC_Yachida_NatMed', 'metadata.csv'),
        'CRC_data': os.path.join(real_data_path, 'CRC_Yachida_NatMed', 'enzymes_data.csv'),
        'IBS2_metadata': os.path.join(real_data_path, 'IBS_Jacobs_Microbiome', 'IBS_metadata.csv'),
        'IBS2_data': os.path.join(real_data_path, 'IBS_Jacobs_Microbiome', 'IBS_metabolites.csv'),
        'CRC_metabolite_metadata': os.path.join(real_data_path, 'CRC_Yachida_NatMed', 'metadata.csv'),
        'CRC_metabolite_data': os.path.join(real_data_path, 'CRC_Yachida_NatMed', 'metabolite_data.csv')
    }

    
    # Preprocess the data
    ibs_enzyme_counts = preprocess_enzymes(
        paths['IBS1_metadata'], paths['IBS1_data'], healthy_label='H', unhealthy_labels=['C', 'D'],
        sample_id_col='SampleID', cohort_col='Cohort'
    )

    crc_enzyme_counts = preprocess_enzymes(
        paths['CRC_metadata'], paths['CRC_data'], healthy_label='Healthy', unhealthy_labels=['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV', 'HS'],
        sample_id_col='Subject_ID', cohort_col='Group'
    )
    
    ibs_shannon_values = preprocess_metabolites(
        paths['IBS2_metadata'], paths['IBS2_data'], group_col='Group', metabolite_col='Metabolite',
        healthy_label='Control', unhealthy_labels=['IBS'], patient_col='Patient'
    )
    
    crc_shannon_values = preprocess_metabolites(
        paths['CRC_metabolite_metadata'], paths['CRC_metabolite_data'], group_col='Group', metabolite_col='Metabolite',
        healthy_label='Healthy', unhealthy_labels=['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV', 'HS'], patient_col='Subject_ID'
    )

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18.4*0.393701, 14*0.393701))
    # Custom colors for groups
    colors = {'Healthy': '#0078B9', 'IBS': '#EA0017', 'CRC': '#EA0017'}

    

    # Order the data by categorizing 'Group'
    ibs_enzyme_counts_plot = ibs_enzyme_counts.assign(Group=pd.Categorical(ibs_enzyme_counts['Group'].replace({'Unhealthy': 'IBS'}), categories=['Healthy', 'IBS'], ordered=True))
    crc_enzyme_counts_plot = crc_enzyme_counts.assign(Group=pd.Categorical(crc_enzyme_counts['Group'].replace({'Unhealthy': 'CRC'}), categories=['Healthy', 'CRC'], ordered=True))
    ibs_shannon_values_plot = ibs_shannon_values.assign(Group=pd.Categorical(ibs_shannon_values['Group'].replace({'Unhealthy': 'IBS'}), categories=['Healthy', 'IBS'], ordered=True))
    crc_shannon_values_plot = crc_shannon_values.assign(Group=pd.Categorical(crc_shannon_values['Group'].replace({'Unhealthy': 'CRC'}), categories=['Healthy', 'CRC'], ordered=True))

    # Plot IBS enzyme counts
    sns.boxplot(data=ibs_enzyme_counts_plot, x='Group', y='Num Enzymes', hue='Group', ax=axes[0, 0], palette=[colors['Healthy'], colors['IBS']],
                 showfliers=False, width=0.3, dodge=False)
    
    stat, p_value = mannwhitneyu(
        ibs_enzyme_counts[ibs_enzyme_counts['Group'] == 'Healthy']['Num Enzymes'],
        ibs_enzyme_counts[ibs_enzyme_counts['Group'] == 'Unhealthy']['Num Enzymes']
    )
    add_p_value(axes[0, 0], 0, 1, ibs_enzyme_counts['Num Enzymes'].max() + 1, p_value)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Number of Enzymes")

    # Plot CRC enzyme counts
    sns.boxplot(data=crc_enzyme_counts_plot, x='Group', y='Num Enzymes', hue='Group', ax=axes[0, 1], palette=[colors['Healthy'], colors['CRC']],
                 showfliers=False, width=0.3, dodge=False)
    
    stat, p_value = mannwhitneyu(
        crc_enzyme_counts[crc_enzyme_counts['Group'] == 'Healthy']['Num Enzymes'],
        crc_enzyme_counts[crc_enzyme_counts['Group'] == 'Unhealthy']['Num Enzymes']
    )
    add_p_value(axes[0, 1], 0, 1, crc_enzyme_counts['Num Enzymes'].max() + 1, p_value)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")

    # Plot IBS Shannon Index
    sns.boxplot(data=ibs_shannon_values_plot, x='Group', y='Shannon Index', hue='Group', ax=axes[1, 0], palette=[colors['Healthy'], colors['IBS']],
                 showfliers=False, width=0.3, dodge=False)
    
    stat, p_value = mannwhitneyu(
        ibs_shannon_values[ibs_shannon_values['Group'] == 'Healthy']['Shannon Index'],
        ibs_shannon_values[ibs_shannon_values['Group'] == 'Unhealthy']['Shannon Index']
    )
    add_p_value(axes[1, 0], 0, 1, ibs_shannon_values['Shannon Index'].max() + 0.1, p_value)
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Metabolites Shannon")

    # Plot CRC Shannon Index
    sns.boxplot(data=crc_shannon_values_plot, x='Group', y='Shannon Index', hue='Group', ax=axes[1, 1], palette=[colors['Healthy'], colors['CRC']],
                 showfliers=False, width=0.3, dodge=False)
    
    stat, p_value = mannwhitneyu(
        crc_shannon_values[crc_shannon_values['Group'] == 'Healthy']['Shannon Index'],
        crc_shannon_values[crc_shannon_values['Group'] == 'Unhealthy']['Shannon Index']
    )
    add_p_value(axes[1, 1], 0, 1, crc_shannon_values['Shannon Index'].max() + 0.1, p_value)
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("")


    
    for ax in fig.get_axes():  # Iterate through all axes in the figure
        ax.xaxis.label.set_size(10)  # Set x-axis label size
        ax.yaxis.label.set_size(10)  # Set y-axis label size
        ax.tick_params(axis='x', which='major', labelsize=10)  # Major xtick size
        ax.tick_params(axis='y', which='major', labelsize=10)  # Major ytick size
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        despine(ax)
    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'IBS_CRC_biomarkers.svg')

    plt.subplots_adjust(top=0.975, bottom=0.08, left=0.12, right=0.98, hspace=0.25, wspace=0.25)
    plt.savefig(image_name, format='svg', transparent=True, dpi=1200)
    #plt.show()


def plot_species_and_biomass_ratios(gut_dir, folder_path, sample_hours):
    t_vec = get_tvec(folder_path, sample_hours, wanted_tvec='t')
    B_tvec = get_tvec(folder_path, sample_hours, wanted_tvec='B')
    S_tvec = get_tvec(folder_path, sample_hours, wanted_tvec='S')
    # Compute ratios
    num_species = np.array([len(B) for B in B_tvec])
    num_nonzero_resources = np.array([np.count_nonzero(S) for S in S_tvec]).astype(float)
    total_biomass = np.array([np.sum(B) for B in B_tvec])
    total_resources = np.array([np.sum(S) for S in S_tvec])
    
    # Avoid division by zero
    num_nonzero_resources[num_nonzero_resources == 0] = np.nan  # Prevent division errors
    
    # Compute metrics
    species_per_resource = num_species / num_nonzero_resources
    biomass_per_resource = total_biomass / total_resources
    
    # Normalize between 0 and 1
    species_per_resource_norm = (species_per_resource - np.nanmin(species_per_resource)) / (np.nanmax(species_per_resource) - np.nanmin(species_per_resource))
    biomass_per_resource_norm = (biomass_per_resource - np.nanmin(biomass_per_resource)) / (np.nanmax(biomass_per_resource) - np.nanmin(biomass_per_resource))
    
    # Normalize between -1 and 1
    species_per_resource_std = 2 * species_per_resource_norm - 1
    biomass_per_resource_std = 2 * biomass_per_resource_norm - 1
    
    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    axs[0].plot(t_vec, species_per_resource, label='Species / Non-Zero Resources', color='b')
    axs[0].set_ylabel('Ratio')
    axs[0].legend()
    
    axs[1].plot(t_vec, biomass_per_resource, label='Total Biomass / Total Resources', color='g')
    axs[1].set_ylabel('Ratio')
    axs[1].legend()
    
    axs[2].plot(t_vec, species_per_resource_std, label='Normalized ([-1,1]) Species/Resource', color='r', linestyle='dashed')
    axs[2].plot(t_vec, biomass_per_resource_std, label='Normalized ([-1,1]) Biomass/Resource', color='purple', linestyle='dashed')
    axs[2].set_ylabel('Normalized Ratio')
    axs[2].legend()
    
    plt.xlabel('Time')
    plt.suptitle('System Ratios Over Time')
    plt.tight_layout()
    plt.show()


def diversity_vs_net_interaction_violin(gut_dir, rootpath, folder_list, sample_hours, relaxing_time=5000*24, num_bins=10):
    fig, ax = plt.subplots(1, 1, figsize=(18.4 * 0.393701, 10 * 0.393701))
    all_diversity_values = []
    all_net_interaction_values = []

    # Collect all data
    for folder in folder_list:
        folder_path = os.path.join(rootpath, folder)
        t_vec = np.array(get_tvec(folder_path, sample_hours, wanted_tvec='t'))
        B_tvec = get_tvec(folder_path, sample_hours, wanted_tvec='B')
        cf_ij_tvec, cp_ij_tvec = get_cf_cp_mat_tvecs(folder_path, sample_hours)

        mask = t_vec >= relaxing_time
        B_tvec = [b for i, b in enumerate(B_tvec) if mask[i]]
        cf_ij_tvec = [cf for i, cf in enumerate(cf_ij_tvec) if mask[i]]
        cp_ij_tvec = [cp for i, cp in enumerate(cp_ij_tvec) if mask[i]]

        for B, cf, cp in zip(B_tvec, cf_ij_tvec, cp_ij_tvec):
            proportions = B / np.sum(B)
            diversity = entropy(proportions)
            net_interaction = (np.sum(cf) - (np.sum(cp) - np.trace(cp))) / (np.sum(cf) + (np.sum(cp) - np.trace(cp)))
            all_diversity_values.append(diversity)
            all_net_interaction_values.append(net_interaction)

    # Binning
    bins = np.linspace(min(all_net_interaction_values), max(all_net_interaction_values), num_bins + 1)
    bin_indices = np.digitize(all_net_interaction_values, bins) - 1
    bin_labels = [(round((bins[i] + bins[i+1]) / 2, 3)) for i in range(num_bins)]

    # Create data for seaborn
    data = {
        "Net Interaction Bin": [bin_labels[min(b, num_bins - 1)] for b in bin_indices],
        "Diversity": all_diversity_values
    }
    df = pd.DataFrame(data)

    # Plot
    sns.violinplot(x="Net Interaction Bin", y="Diversity", data=df, ax=ax, palette="muted", inner="box", linewidth=0.9)
    ax.set_xlabel("Net Interaction (binned)", fontsize=10)
    ax.set_ylabel("Shannon Diversity Index", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    image_name = os.path.join(image_dir, 'diversity_vs_net_interaction_violin_good.svg')

    plt.subplots_adjust(top=0.975, bottom=0.18, left=0.12, right=0.98)
    plt.savefig(image_name, format='svg', transparent=False, dpi=1200)
    plt.show()

def beem_static_plots(gut_dir):

    #CRC_bootstrap_path = os.path.join(gut_dir, 'real_data', 'CRC_Yachida_NatMed', 'beem_static', 'bootstrap_alpha_0.5-lambda_1.0', 'BacFrac_1.0_FullsetFrac_1.0')
    CRC_bootstrap_path = os.path.join(gut_dir, 'real_data', 'CRC_Yachida_NatMed', 'beem_static', 'bootstrap_alpha_1.0-lambda_1.0', 'BacFrac_1.0_FullsetFrac_1.0')
    IBS_bootstrap_path = os.path.join(gut_dir, 'real_data', 'IBS_Mars_Cell', 'beem_static', 'bootstrap_alpha_1.0-lambda_1.0', 'BacFrac_1.0_FullsetFrac_1.0')
    IBD_bootstrap_path = os.path.join(gut_dir, 'real_data', 'IBD_MDB', 'beem_static', 'bootstrap_alpha_1.0-lambda_1.0', 'BacFrac_1.0_FullsetFrac_1.0')
    CD_bootstrap_path = os.path.join(gut_dir, 'real_data', 'CD_Ferretti_Elife', 'beem_static', 'bootstrap_alpha_1.0-lambda_1.0', 'BacFrac_1.0_FullsetFrac_1.0')
    
    Model_bootstrap_path = os.path.join(gut_dir, 'real_data', 'ModelData', 'invasion', 'E_D_0', 'fraction_nl_6_0_1.1',
                                        'beem_static', 'bootstrap_alpha_1.0-lambda_1.0', 'BacFrac_1.0_FullsetFrac_1.0', 'coarsed_True', 'samples_1000')
    full_color_map = {}
    base_color_map = {'Healthy': '#0078B9', 'H': '#0078B9', 'Unhealthy': '#EA0017', 'U': '#EA0017',
                      'IBD': '#EA0017', 'IBS': '#EA0017', 'CDI': '#EA0017'}

    CRC_color_map = {"Healthy": '#0078B9',
                     "CRC": '#EA0017',
                     "CRC-0": '#F9B3B3',  # Lightest red
                     "CRC-1": "#F48A8A",  # Light red
                     "CRC-2": "#E03D3D",  # Dark red
                     "CRC-3": "#A70000"   # Darkest Red
                     #"HS": "#D4C2E5"  # Purple
                     }
    Model_color_map = {"FS": '#0078B9',
                       #"Healthy_0": "#66AEDD",
                       #"Healthy_1": "#005080",
                       "CS": '#EA0017',
                       "CS-0": '#F9B3B3',
                       "CS-1": "#F48A8A",
                       "CS-2": "#E03D3D",
                       "CS-3": "#A70000"
                      }
    full_color_map.update(Model_color_map)
    full_color_map.update(base_color_map)
    full_color_map.update(CRC_color_map)
    
    
    fig = plt.figure(figsize = (18.4*0.393701, 12*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(5,1, height_ratios=[1/5,1/5,1/5,1/5,1/5], width_ratios=[1])
    
    ax00 = fig.add_subplot(gs[0,0])
    ax10 = fig.add_subplot(gs[1,0])
    ax20 = fig.add_subplot(gs[2,0])
    ax30 = fig.add_subplot(gs[3,0])
    ax40 = fig.add_subplot(gs[4,0])
    '''
    full_Model_df = pd.DataFrame(columns=['subset_fraction', 'Group', 'rho'])
    full_CRC_df = pd.DataFrame(columns=['subset_fraction', 'Group', 'rho'])
    full_IBD_df = pd.DataFrame(columns=['subset_fraction', 'Group', 'rho'])
    full_IBS_df = pd.DataFrame(columns=['subset_fraction', 'Group', 'rho'])
    full_CDI_df = pd.DataFrame(columns=['subset_fraction', 'Group', 'rho'])
    '''
    Model_replacement_dict = dict(zip(['Healthy', 'Unhealthy', 'Unhealthy_0', 'Unhealthy_1', 'Unhealthy_2', 'Unhealthy_3'],
                                       ['FS', 'CS', 'CS-0', 'CS-1', 'CS-2', 'CS-3']))
    CRC_replacement_dict = dict(zip(['Unhealthy', 'MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV'],
                                     ['CRC', 'CRC-0', 'CRC-1', 'CRC-2', 'CRC-3']))
    IBD_replacement_dict = dict(zip(['Unhealthy'], ['IBD']))
    IBS_replacement_dict = dict(zip(['Unhealthy'], ['IBS']))
    CDI_replacement_dict = dict(zip(['Unhealthy'], ['CDI']))
    for subset_fraction in ['0.4', '0.5', '0.6', '0.7', '0.8']:
        Model_df = get_data(Model_bootstrap_path, None, None, subset_fraction, healthy_normalized=False, overwrite=True)
        CRC_df = get_data(CRC_bootstrap_path,None, None, subset_fraction, healthy_normalized=False, overwrite=True)
        #Model_df2 = get_data(Model_bootstrap_path, ['Unhealthy_0', 'Unhealthy_1', 'Unhealthy_2', 'Unhealthy_3'], None, subset_fraction, healthy_normalized=False, overwrite=True)
        #CRC_df2 = get_data(CRC_bootstrap_path,['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV'], None, subset_fraction, healthy_normalized=False, overwrite=True)
        IBD_df = get_data(IBD_bootstrap_path, None, None, subset_fraction, healthy_normalized=False, overwrite=True)
        IBS_df = get_data(IBS_bootstrap_path, None, None, subset_fraction, healthy_normalized=False, overwrite=True)
        CDI_df = get_data(CD_bootstrap_path, None, None, subset_fraction, healthy_normalized=False, overwrite=True)

        Model_df['Group'] = Model_df['Group'].cat.rename_categories(Model_replacement_dict)
        Model_df['Group'] = pd.Categorical(Model_df['Group'], ordered=True, categories=['FS', 'CS'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group', width=0.4, data=Model_df, palette=full_color_map, ax=ax00, showfliers=False)
        CRC_df['Group'] = CRC_df['Group'].cat.rename_categories(CRC_replacement_dict)
        CRC_df['Group'] = pd.Categorical(CRC_df['Group'], ordered=True, categories=['Healthy', 'CRC'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group', width=0.4, data=CRC_df, palette=full_color_map, ax=ax10, showfliers=False) 
        
        IBD_df['Group'] = IBD_df['Group'].cat.rename_categories(IBD_replacement_dict)
        IBD_df['Group'] = pd.Categorical(IBD_df['Group'], ordered=True, categories=['Healthy', 'IBD'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group', width=0.4, data=IBD_df, palette=full_color_map, ax=ax20, showfliers=False)
        IBS_df['Group'] = IBS_df['Group'].cat.rename_categories(IBS_replacement_dict)
        IBS_df['Group'] = pd.Categorical(IBS_df['Group'], ordered=True, categories=['Healthy', 'IBS'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group', width=0.4, data=IBS_df, palette=full_color_map, ax=ax30, showfliers=False)
        CDI_df['Group'] = CDI_df['Group'].cat.rename_categories(CDI_replacement_dict)
        CDI_df['Group'] = pd.Categorical(CDI_df['Group'], ordered=True, categories=['Healthy', 'CDI'])
        sns.boxplot(x='Subset Fraction', y='rho', hue='Group', width=0.4, data=CDI_df, palette=full_color_map, ax=ax40, showfliers=False)
        
    j=0
    for i,ax in enumerate([ax00, ax10, ax20, ax30, ax40]):
        '''
        # Remove only the horizontal whisker caps
        for line in ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            # Horizontal whisker caps have exactly two points and are OUTSIDE the box limits
            if len(x_data) == 2 and np.isclose(y_data[0], y_data[1]):
                if j in [0,1]:
                    line.set_visible(False)
                j += 1
                if j==3:
                    j=0
        '''
        despine(ax)
        if ax.get_legend():
            ax.legend_.remove()
        ax.set_ylabel('')
        if i < 4:
            ax.set_xlabel('')
            ax.set_xticklabels([])
    ax20.set_ylabel('Inferred Net Interaction', labelpad=5)
    
    # Define custom legends for each subplot
    # Custom legend handles with edge color and adjusted font size
    def create_legend_handles(color_dict, edgecolor='black'):
        return [Patch(facecolor=color, edgecolor=edgecolor, linewidth=1.2, label=label) 
                for label, color in color_dict.items()]

    # Create legend handles for each group
    custom_legends = {
        'Legend': create_legend_handles({'Healthy': full_color_map['Healthy'], 'Unhealthy': full_color_map['IBD']}),
    }

    # Add legends to the right of each subplot with two columns, edge color, and smaller font size
    legend_fontsize = 8
    
    ax00.legend(handles=custom_legends['Legend'], loc='center', bbox_to_anchor=(0.5, 1.2), 
                frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0, handletextpad=0.5)
    '''
    ax00.legend(handles=custom_legends['Model'], loc='center left', bbox_to_anchor=(1, 0.5), 
                title='Model', frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0, handletextpad=0.5)
    ax10.legend(handles=custom_legends['CRC'], loc='center left', bbox_to_anchor=(1, 0.5), 
                title='CRC', frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0,handletextpad=0.5)
    ax20.legend(handles=custom_legends['IBD'], loc='center left', bbox_to_anchor=(1, 0.5), 
                title='IBD', frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0, handletextpad=0.5)
    ax30.legend(handles=custom_legends['IBS'], loc='center left', bbox_to_anchor=(1, 0.5), 
                title='IBS', frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0, handletextpad=0.5)
    ax40.legend(handles=custom_legends['CDI'], loc='center left', bbox_to_anchor=(1, 0.5), 
                title='CDI', frameon=False, ncol=2, fontsize=legend_fontsize, columnspacing=1.0, handletextpad=0.5)
    '''
    for ax in fig.get_axes():  # Iterate through all axes in the figure
        ax.xaxis.label.set_size(12)  # Set x-axis label size
        ax.yaxis.label.set_size(12)  # Set y-axis label size
        ax.tick_params(axis='x', which='major', labelsize=10)  # Major xtick size
        ax.tick_params(axis='y', which='major', labelsize=10)  # Major ytick size
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        despine(ax)
    #ax10.set_yticks([0.6,0.8,1.0])
    #ax30.set_yticks([0.84,0.88,0.92])
    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'beem_static_plot.svg')

    plt.subplots_adjust(top=0.94, bottom=0.112, left=0.09, right=0.96, hspace=0.4, wspace=0.35)
    plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    #plt.show()


def calculate_bray_curtis(species_a, species_b):
    """
    Calculate the Bray-Curtis dissimilarity between two samples.

    :param species_a: dict with species as keys and biomasses at time a as values
    :param species_b: dict with species as keys and biomasses at time b as values
    :return: Bray-Curtis dissimilarity
    """
    sum_min = 0
    sum_total = 0
    
    # Union of species in both samples
    all_species = set(species_a.keys()).union(species_b.keys())
    
    for species in all_species:
        Nij = species_a.get(species, 0)
        Nik = species_b.get(species, 0)
        sum_min += min(Nij, Nik)
        sum_total += Nij + Nik
    
    if sum_total == 0:  # Avoid division by zero if both samples are empty
        return 1
    
    return 1 - (2 * sum_min / sum_total)

def compute_beta_diversity_per_disease(full_data_dicts, model=False):
    """
    Computes Bray-Curtis beta diversity for Healthy and Unhealthy groups within each dataset separately.

    Args:
        full_data_dicts (dict): Dictionary containing disease datasets with participant samples.

    Returns:
        dict: Dictionary of beta diversity data for each disease.
    """
    beta_diversity_results = {}

    healthy_diagnosis_dict = {'IBD': ['nonIBD'], 'CDI': ['Healthy'], 'IBS': ['Healthy'], 'CRC': ['Healthy'], 'Model': ['Healthy'], 'Model_Coarsed': ['Healthy']}#'ModelData': [-0.4, -0.2, 0]}#'ModelData': ['Healthy']}
    unhealthy_diagnosis_dict = {'IBD': ['CD', 'UC'], 'IBS': ['IBS-C', 'IBS-D'], 'CRC': ['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV']} #[0.2, 0.4, 0.6, 0.85]}#}

    for disease, full_data_dict in full_data_dicts.items():
        healthy_samples = []
        unhealthy_samples = []

        for participant_id, samples in full_data_dict.items():
            for sample in samples[:10]+samples[-10:]:
                sample_data = sample['Data'].to_dict()  # Convert Series to dictionary
                sample_diag = sample['Diagnosis']

                if sample_diag in healthy_diagnosis_dict[disease]:
                    healthy_samples.append(sample_data)
                else:
                    unhealthy_samples.append(sample_data)

        # Compute pairwise Bray-Curtis dissimilarity
        healthy_pairs = list(combinations(healthy_samples, 2))
        unhealthy_pairs = list(combinations(unhealthy_samples, 2))
        healthy_sampled_pairs = random.sample(healthy_pairs, min(10000, len(healthy_pairs)))  # Limit to 10,000 comparisons
        unhealthy_sampled_pairs = random.sample(unhealthy_pairs, min(10000, len(unhealthy_pairs)))  # Limit to 10,000 comparisons

        beta_healthy = [calculate_bray_curtis(a, b) for a, b in healthy_sampled_pairs]
        
        beta_unhealthy = [calculate_bray_curtis(a, b) for a, b in unhealthy_sampled_pairs]

        # Store results
        beta_diversity_results[disease] = pd.DataFrame({
            'Beta Diversity': np.concatenate([beta_healthy, beta_unhealthy]) if beta_healthy and beta_unhealthy else [],
            'Group': ['Healthy'] * len(beta_healthy) + ['Unhealthy'] * len(beta_unhealthy)
        })

    return beta_diversity_results

def plot_beta_diversities(gut_dir, percentile=0):
    def filter_top_percent_types(data_df, percentile):
        """
        Filters the top `percentile` most abundant bacterial types per sample.

        Parameters:
            data_df (pd.DataFrame): DataFrame with types as rows and samples as columns.
            percentile (float): Percentile threshold (e.g., 90 for top 10%).

        Returns:
            pd.DataFrame: Filtered DataFrame with only top abundant types retained per sample.
        """
        filtered_df = data_df.copy()
        for sample in data_df.columns:
            sample_values = data_df[sample]
            nonzero_values = sample_values[sample_values > 0]
            if nonzero_values.empty:
                filtered_df[sample] = 0
                continue
            threshold = np.percentile(nonzero_values, percentile)
            filtered_df[sample] = sample_values.where(sample_values >= threshold, other=0)
        filtered_df = filtered_df.loc[(filtered_df != 0).any(axis=1)]
        return filtered_df

    f = 'fraction_nl'
    c = '6'
    m = '0'
    e = '1.1'  
    E_D_type = 0
    tradeoff_data = f'{f}_{c}_{m}_{e}'
    rootpath_data_model = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)


    rootpath_data_IBD = os.path.join(gut_dir,'real_data','IBD_MDB')
    rootpath_data_IBS = os.path.join(gut_dir,'real_data','IBS_Mars_Cell')
    rootpath_data_CDI = os.path.join(gut_dir,'real_data','CD_Ferretti_Elife')
    rootpath_data_CRC = os.path.join(gut_dir,'real_data','CRC_Yachida_NatMed')

    
    metadata_dict_IBD = IBD.get_metadata_dict(rootpath_data_IBD)
    data_df_taxonomy_IBD = filter_top_percent_types(IBD.get_data_df(rootpath_data_IBD, data_type = 'taxonomy'), percentile)
    full_data_dict_IBD = IBD.get_full_data_dict(metadata_dict_IBD, data_df_taxonomy_IBD)
    metadata_dict_IBS = IBS.get_metadata_dict(rootpath_data_IBS)
    data_df_taxonomy_IBS = filter_top_percent_types(IBS.get_data_df(rootpath_data_IBS, data_type = 'taxonomy'), percentile)
    full_data_dict_IBS = IBD.get_full_data_dict(metadata_dict_IBS, data_df_taxonomy_IBS)
    metadata_dict_CDI = CDI.get_metadata_dict(rootpath_data_CDI)
    data_df_taxonomy_CDI = filter_top_percent_types(CDI.get_data_df(rootpath_data_CDI), percentile)
    full_data_dict_CDI = CDI.get_full_data_dict(metadata_dict_CDI, data_df_taxonomy_CDI)
    metadata_dict_CRC = CRC.get_metadata_dict(rootpath_data_CRC)
    data_df_taxonomy_CRC = filter_top_percent_types(CRC.get_data_df(rootpath_data_CRC, data_type = 'taxonomy2'), percentile)
    full_data_dict_CRC = CRC.get_full_data_dict(metadata_dict_CRC, data_df_taxonomy_CRC)
    
    metadata_dict_model = MDA.get_metadata_dict(rootpath_data_model, overwrite=False, sample_size=200)
    data_df_model = filter_top_percent_types(MDA.get_data_df(rootpath_data_model, data_type='taxonomy', overwrite=False, sample_size=200), percentile)
    full_data_dict_model = MDA.get_full_data_dict(metadata_dict_model, data_df_model)
    
    data_df_model_coarsed = filter_top_percent_types(MDA.get_data_df(rootpath_data_model, data_type='coarsed_taxonomy', overwrite=False, wanted_realization='683', sample_size=200), percentile)
    full_data_dict_model_coarsed = MDA.get_full_data_dict(metadata_dict_model, data_df_model_coarsed)

    full_data_dicts = {
        'IBD': full_data_dict_IBD,
        'IBS': full_data_dict_IBS,
        'CDI': full_data_dict_CDI,
        'CRC': full_data_dict_CRC,
        'Model': full_data_dict_model,
        'Model_Coarsed': full_data_dict_model_coarsed
    }
    beta_diversity_dict = compute_beta_diversity_per_disease(full_data_dicts)

    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(18.4*0.393701, 16*0.393701))
    axes = axes.flatten()
    disease_names = list(beta_diversity_dict.keys())

    for i, disease in enumerate(disease_names):
        data = beta_diversity_dict[disease]
        # Perform Mann-Whitney U test
        beta_healthy = data[data["Group"] == "Healthy"]["Beta Diversity"]
        beta_unhealthy = data[data["Group"] == "Unhealthy"]["Beta Diversity"]
        
        if len(beta_healthy) > 0 and len(beta_unhealthy) > 0:
            _, p_value = mannwhitneyu(beta_healthy, beta_unhealthy, alternative='two-sided')
            p_value_text = f"p = {p_value:.3e}"  # Format in scientific notation
        else:
            p_value_text = "p = N/A"  # If no data is available
        ax = axes[i]
        sns.boxplot(x='Group', y='Beta Diversity', data=data, palette={"Healthy": "#0078B9", "Unhealthy": "#EA0017"}, ax=ax, showfliers=False, width=0.5)
        ax.set_title(f'{disease}: {p_value_text}')
        for i in [0, 2, 4]:
            ax.set_ylabel('Bray-Curtis Dissimilarity')
        ax.set_xlabel('')
    
    for ax in fig.get_axes():  # Iterate through all axes in the figure
        ax.xaxis.label.set_size(12)  # Set x-axis label size
        ax.yaxis.label.set_size(12)  # Set y-axis label size
        ax.tick_params(axis='x', which='major', labelsize=10)  # Major xtick size
        ax.tick_params(axis='y', which='major', labelsize=10)  # Major ytick size
        
        despine(ax)
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name_png = os.path.join(image_dir, f'beta_diversities_all_{percentile}.png')
    image_name_svg = os.path.join(image_dir, f'beta_diversities_all_{percentile}.svg')

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.1, right=0.98, hspace=0.4, wspace=0.25)
    plt.savefig(image_name_png, format='png', transparent=False, dpi=1200)
    plt.savefig(image_name_svg, format='svg', transparent=False, dpi=1200)
    #plt.show()

# Define plotting function
def plot_robustness_parameters(params_realizations_dict, sample_hours=24*7):
    fig = plt.figure(figsize=(18.4 * 0.393701, 19 * 0.393701))

    gs = fig.add_gridspec(11, 5,
        height_ratios=[0.1, 0.1, 0.055, 0.1, 0.1,   0.09,# top block
                       0.1, 0.1, 0.055, 0.1, 0.1,   # bottom block
                       ],
        width_ratios=[1, 1, 1, 1, 1]
    )

    sample_hours = 24 * 7
    param_names = list(params_realizations_dict.keys())
    axes_dict = {}

    for param_idx, param in enumerate(param_names):
        high_path, low_path = params_realizations_dict[param]
        col_idx = param_idx % 5
        for row_offset, realization_path in enumerate([high_path, low_path]):
            if param_idx < 5:
                row_base = 0
            else:
                row_base = 5

            if param_idx < 5:
                row_base = 0  # top block starts at row 0
            else:
                row_base = 6  # bottom block starts at row 6

            if row_offset == 0:  # High value → top half of block
                row_B = row_base
                row_rho = row_base + 1
            else:  # Low value → bottom half of block
                row_B = row_base + 3
                row_rho = row_base + 4

            B_ax = fig.add_subplot(gs[row_B, col_idx])
            rho_ax = fig.add_subplot(gs[row_rho, col_idx], sharex=B_ax)

            # Load data
            t_vec, _, B_types_dict, _, _, _, rho_tvec = get_realization_data(realization_path, sample_hours, full=True)
            t_vec = t_vec / 24 / 365  # convert to years

            # Biomass
            palette = sns.color_palette("tab20c", n_colors=len(B_types_dict.keys()))
            color_mapping = {Type: palette[i % len(palette)] for i, Type in enumerate(B_types_dict.keys())}
            for Type, (t_init, t_end, _, B_type_t) in B_types_dict.items():
                wanted_t = t_vec[(t_vec >= t_init / 24 / 365) & (t_vec < t_end / 24 / 365)]
                B_ax.plot(wanted_t, B_type_t, label=Type, color=color_mapping[Type])

            
            rho_ax.plot(t_vec, rho_tvec, color='dimgrey')
            if col_idx == 0:
                B_ax.set_ylabel('B', fontsize=8)
                rho_ax.set_ylabel('Net Interaction', fontsize=8)
            rho_ax.set_xlabel('Time (years)', labelpad=0, fontsize=8)

            B_ax.tick_params(labelsize=7, labelbottom=False)
            rho_ax.tick_params(labelsize=7)
            if row_offset == 0:
                B_ax.set_title(param, fontsize=10)

            #axes_dict[(param, 'high' if row_offset == 0 else 'low')] = (B_ax, rho_ax)

    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Set the x-axis limits to show only data from 3 to 60
        ax.set_xlim(3, 50)
        # Manually set the tick positions (corresponding to the original data values)
        tick_positions = [3, 13, 23, 33, 43]  # These are the actual data points in the original data
        # Set the new tick labels to start from 0
        tick_labels = [0, 10, 20, 30, 40]  # These are the custom labels you want to show
        # Update the x-ticks and their corresponding labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    image_name = os.path.join(image_dir, f'parameter_variation_panels.png')

    plt.subplots_adjust(top=0.96, bottom=0.05, left=0.1, right=0.99, hspace=0.1, wspace=0.3)
    plt.savefig(image_name, format='png', dpi=1200)

    image_name = os.path.join(image_dir, f'parameter_variation_panels.svg')
    plt.savefig(image_name, format='svg')
    # plt.show()

def plot_robustness_parameters_extra(gut_dir, sample_size=300, data_type='taxonomy', metric='rho'):
    """
    Plots boxplots of sampled rho (net interaction) for Healthy vs Dysbiotic samples
    across different parameter configurations.

    Args:
        params_rootpath_dict (dict): Keys are parameter labels, values are rootpaths to the sampled data.
        sample_size (int): Number of samples per group to load.
        data_type (str): The data type to load (e.g., 'taxonomy').
        metric (str): The metric to compute (default is 'rho').
    """


    def map_diagnosis_labels(df):
        return df.replace({'Diagnosis': {'H': 'Healthy', 'U': 'Dysbiotic'}})
    rootpath_changing_params = os.path.join(gut_dir, 'results', 'invasion', 'E_D_0', 'changing_parameters')
    params_rootpath_dict = {'delta': [os.path.join(rootpath_changing_params, 'delta_0.10416666666666667'),
                                      os.path.join(rootpath_changing_params, 'delta_0.016666666666666666')],
                            'h': [os.path.join(rootpath_changing_params, 'h_10'),
                                  os.path.join(rootpath_changing_params, 'h_0.5')],
                            'gamma': [os.path.join(rootpath_changing_params, 'gamma_1000000000000000.0'),
                                      os.path.join(rootpath_changing_params, 'gamma_80000000000000.0')],
                            'r': [os.path.join(rootpath_changing_params, 'maxgrowth_1e-09'),
                                  os.path.join(rootpath_changing_params, 'maxgrowth_1e-11')],
                            'k': [os.path.join(rootpath_changing_params, 'k_0.001'),
                                  os.path.join(rootpath_changing_params, 'k_1e-05')],
                            'emax': [os.path.join(rootpath_changing_params, 'emax_4'),
                                     os.path.join(rootpath_changing_params, 'emax_2')],
                            'R': [os.path.join(rootpath_changing_params, 'Nsubstances_40'),
                                     os.path.join(rootpath_changing_params, 'Nsubstances_20')],
                            'M': [os.path.join(rootpath_changing_params, 'molecularmass_120'),
                                  os.path.join(rootpath_changing_params, 'molecularmass_80')],
                            'U': [os.path.join(rootpath_changing_params, 'invasionperiod_4'),
                                  os.path.join(rootpath_changing_params, 'invasionperiod_1')],
                            'Ex. th': [os.path.join(rootpath_changing_params, 'extinctionth_0.0005'),
                                       os.path.join(rootpath_changing_params, 'extinctionth_5e-05')],        
                            }
    # === Layout Configuration ===
    num_params = len(params_rootpath_dict)
     # === Setup ===
    f = plt.figure(figsize=(18.4 * 0.393701, 13 * 0.393701))
    gs = f.add_gridspec(
        7, 5,
        height_ratios=[0.2, 0.055, 0.2, 0.09, 0.2, 0.055, 0.2],
        width_ratios=[1, 1, 1, 1, 1]
    )

    # Grid placement: 5 columns × 4 rows (top rows = high value, bottom = low)
    gs_indices = [(0, i) for i in range(5)] + [(2, i) for i in range(5)]
    axs_high = [f.add_subplot(gs[r, c]) for r, c in gs_indices]

    gs_indices_low = [(4, i) for i in range(5)] + [(6, i) for i in range(5)]
    axs_low = [f.add_subplot(gs[r, c]) for r, c in gs_indices_low]

    # === Plotting ===
    param_items = list(params_rootpath_dict.items())
    for idx, (param_name, paths) in enumerate(param_items):
        if len(paths) != 2:
            print(f"Parameter {param_name} must have two rootpaths (high and low).")
            continue

        high_path, low_path = paths
        # --- High ---
        df_high, p_high = get_sampled_metric_df(high_path, data_type, metric, sample_size)
        df_high = map_diagnosis_labels(df_high)
        axh = axs_high[idx]
        sns.boxplot(data=df_high, x='Diagnosis', y='Metric Value', hue='Diagnosis', showfliers=False,
                    palette={'Healthy': '#0078B9', 'Dysbiotic': '#EA0017'}, ax=axh, legend=False)
        #axh.set_title(f'{param_name} ↑\np = {p_high:.2e}', fontsize=9)
        axh.set_ylabel('Net Interaction', fontsize=9, labelpad=1)
        axh.set_xlabel('')
        axh.tick_params(axis='x', labelsize=8)
        axh.tick_params(axis='y', labelsize=8)

        # --- Low ---
        df_low, p_low = get_sampled_metric_df(low_path, data_type, metric, sample_size)
        df_low = map_diagnosis_labels(df_low)
        axl = axs_low[idx]
        sns.boxplot(data=df_low, x='Diagnosis', y='Metric Value', hue='Diagnosis', showfliers=False,
                    palette={'Healthy': '#0078B9', 'Dysbiotic': '#EA0017'}, ax=axl, legend=False)
        #axl.set_title(f'{param_name} ↓\np = {p_low:.2e}', fontsize=9)
        axl.set_ylabel('Net Interaction', fontsize=9)
        axl.set_xlabel('')
        axl.tick_params(axis='x', labelsize=8)
        axl.tick_params(axis='y', labelsize=8)
    axs_low[3].set_yticks([-0.3,0,0.3])
    # === Final layout ===
    plt.subplots_adjust(top=0.96, bottom=0.08, left=0.1, right=0.99, hspace=0.1, wspace=0.6)
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, 'robustness_parameters_extra.svg'), format='svg', dpi=2400)
    plt.show()

def get_all_data(folder_path):
    all_data = []
    value_index = 3 # The rho (net interaction) is in stats index 3
    # Step 1: Traverse each subset_fraction subfolder
    for subset_dir in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subset_dir)
        if not os.path.isdir(subfolder_path):
            continue  # Skip non-directories

        # Load morinaga and nibiohn files in this subset
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.json') and "_stats_boostrap_" in filename:
                diagnosis = filename.split("_stats_boostrap")[0]
                file_path = os.path.join(subfolder_path, filename)

                with open(file_path, "r") as f:
                    data = json.load(f)

                vectors = data.get(subset_dir, [])
                for vec in vectors:
                    if isinstance(vec, list) and len(vec) > value_index:
                        all_data.append({
                            "subset_fraction": subset_dir,
                            "diagnosis": diagnosis,
                            "value": vec[value_index]
                        })
    return pd.DataFrame(all_data)
def rank_biserial_effect_size(U, n1, n2):
    return 1 - (2 * U) / (n1 * n2)
def cliffs_delta(x, y):
    """Compute Cliff's delta effect size between two arrays."""
    x = np.array(x)
    y = np.array(y)
    n_x = len(x)
    n_y = len(y)
    count = 0

    for xi in x:
        count += np.sum(xi > y)
        count -= np.sum(xi < y)

    delta = count / (n_x * n_y)
    return delta
def healthy_interaction_robustness(gut_dir):   
    def center_by_nibiohn(group):
        baseline = group[group["diagnosis"] == "nibiohn"]["value"].median()
        group["value"] = group["value"] - baseline
        return group
    fig, ax = plt.subplots(1, 1, figsize=(18.4 * 0.393701, 10 * 0.393701))
    HealthyPark_bootstrap_path = os.path.join(gut_dir, 'real_data', 'Healthy_Park_BMCMicrobiology', 'flashweave', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    df = get_all_data(HealthyPark_bootstrap_path)
    
    df["diagnosis"] = pd.Categorical(df["diagnosis"], categories=["nibiohn", "morinaga"], ordered=True)
    df = (df.groupby("subset_fraction", group_keys=False).apply(center_by_nibiohn).reset_index(drop=True))
    palette = {"morinaga": "#005B8A", "nibiohn": "#61A8D7"}

    #sns.violinplot(data=df, x="subset_fraction", y="value", hue="diagnosis", palette=palette, inner="box", linewidth=0.9, ax=ax)
    sns.boxplot(data=df, x="subset_fraction", y="value", hue="diagnosis", palette=palette, showfliers=False)
    sns.despine(ax=ax, top=True, right=True)
    # Step 3: Annotate p-values per subset_fraction
    subset_fractions = sorted(df["subset_fraction"].unique(), key=lambda x: float(x))
    for i, sf in enumerate(subset_fractions):
        group = df[df["subset_fraction"] == sf]
        m_vals = group[group["diagnosis"] == "morinaga"]["value"]
        n_vals = group[group["diagnosis"] == "nibiohn"]["value"]

        if not m_vals.empty and not n_vals.empty:
            U, pval = mannwhitneyu(n_vals, m_vals, alternative="two-sided")
            n1, n2 = len(n_vals), len(m_vals)
            effect_size = rank_biserial_effect_size(U, n1, n2)
            delta = cliffs_delta(n_vals, m_vals)
            y = max(group["value"]) * 1.05
            ax.text(i, y, f"p={pval:.2e}\nΔ={delta:.2f}", ha="center", fontsize=8, color="black")

    ax.set_ylabel("ENBI", fontsize=12)
    ax.set_xlabel("Subset Fraction", fontsize=12)
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.12, right=0.98)
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    image_name = os.path.join(image_dir, 'healthy_interaction_robustness.svg')
    plt.savefig(image_name, format='svg', transparent=False, dpi=1200)
    plt.show()
def diet_interaction_robustness(gut_dir):   
    def center_by_control(group):
        baseline = group[group["diagnosis"] == "control"]["value"].median()
        group["value"] = group["value"] - baseline
        return group
    fig, ax = plt.subplots(1, 1, figsize=(18.4 * 0.393701, 10 * 0.393701))
    Barley_Goto_bootstrap_path = os.path.join(gut_dir, 'real_data', 'Barley_Goto_Nutrients', 'flashweave', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    df = get_all_data(Barley_Goto_bootstrap_path)
    df["diagnosis"] = pd.Categorical(df["diagnosis"], categories=["control", "test"], ordered=True)

    df = (df.groupby("subset_fraction", group_keys=False).apply(center_by_control).reset_index(drop=True))
    palette = {"test": "#005B8A", "control": "#61A8D7"}

    #sns.violinplot(data=df, x="subset_fraction", y="value", hue="diagnosis", palette=palette, inner="box", linewidth=0.9, ax=ax)
    sns.boxplot(data=df, x="subset_fraction", y="value", hue="diagnosis", palette=palette, showfliers=False)
    sns.despine(ax=ax, top=True, right=True)
    # Step 3: Annotate p-values per subset_fraction
    subset_fractions = sorted(df["subset_fraction"].unique(), key=lambda x: float(x))
    for i, sf in enumerate(subset_fractions):
        group = df[df["subset_fraction"] == sf]
        m_vals = group[group["diagnosis"] == "test"]["value"]
        n_vals = group[group["diagnosis"] == "control"]["value"]

        if not m_vals.empty and not n_vals.empty:
            U, pval = mannwhitneyu(n_vals, m_vals, alternative="two-sided")
            n1, n2 = len(n_vals), len(m_vals)
            effect_size = rank_biserial_effect_size(U, n1, n2)
            delta = cliffs_delta(n_vals, m_vals)
            y = max(group["value"]) * 1.05
            ax.text(i, y, f"p={pval:.2e}\nΔ={delta:.2f}", ha="center", fontsize=8, color="black")

    ax.set_ylabel("ENBI", fontsize=12)
    ax.set_xlabel("Subset Fraction", fontsize=12)
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.12, right=0.98)
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    image_name = os.path.join(image_dir, 'diet_interaction_robustness.svg')
    plt.savefig(image_name, format='svg', transparent=False, dpi=1200)
    plt.show()

def CDI_study_robustness(gut_dir):

    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18.4 * 0.393701, 12 * 0.393701), sharex=True)  
    CDI_study_bootstrap_path = os.path.join(
        gut_dir, 'real_data', 'CD_Ferretti_Elife', 'Study', 'flashweave',
        'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0'
    )
    df = get_all_data(CDI_study_bootstrap_path)
    df_india = df[df['diagnosis'].str.contains('India')].copy()
    df_usa = df[df['diagnosis'].str.contains('Langdon')].copy()

    for ax, df_subset, title in zip([ax1, ax2], [df_usa, df_india], ['USA', 'India']):
        df_subset['group'] = df_subset['diagnosis'].apply(lambda x: 'Healthy' if 'Healthy' in x else 'CDI')

        # Center values by median of Healthy group in each subset_fraction
        def center_by_healthy(group):
            healthy_median = group.loc[group['group'] == 'Healthy', 'value'].median()
            group['value_centered'] = group['value'] - healthy_median
            return group

        df_subset = (df_subset.groupby("subset_fraction", group_keys=False).apply(center_by_healthy).reset_index(drop=True))
        palette = {"CDI": '#EA0017', "Healthy": '#0078B9'}

        sns.boxplot(data=df_subset, x="subset_fraction", y="value_centered", hue="group", hue_order=["Healthy", "CDI"],
                     palette=palette, showfliers=False, ax=ax)
        ax.legend_.remove()
        # Annotate stats
        subset_fractions = sorted(df_subset["subset_fraction"].unique(), key=lambda x: float(x))
        for i, sf in enumerate(subset_fractions):
            group = df_subset[df_subset["subset_fraction"] == sf]
            m_vals = group[group["group"] == "CDI"]["value_centered"]
            n_vals = group[group["group"] == "Healthy"]["value_centered"]
            if not m_vals.empty and not n_vals.empty:
                U, pval = mannwhitneyu(m_vals, n_vals, alternative="two-sided")
                n1, n2 = len(m_vals), len(n_vals)
                delta = cliffs_delta(n_vals, m_vals)
                y = group["value_centered"].max() * 1.05
                ax.text(i, y, f"p={pval:.2e}\nΔ={delta:.2f}", ha="center", fontsize=8, color="black")

        #ax.set_title(title)
        ax.set_ylabel("ENBI")
        ax.set_xlabel("")
        sns.despine(ax=ax, top=True, right=True)
        ax.text(0.99, 0.02, title, transform=ax.transAxes, ha='right', va='bottom', fontsize=14, fontweight='bold')

    ax2.set_xlabel("Subset Fraction")
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    image_name = os.path.join(image_dir, 'CDI_study_robustness.svg')
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.98, hspace=0.4)
    plt.savefig(image_name, format='svg', dpi=1200)
    plt.show()


def create_full_pathway_structure_figure(gut_dir, healthy_sample_id, dysbiotic_sample_id, p=99, sort_by="abundance"):
    # === Prepare sample-specific pathway matrices and biomass barplots ===
    def process_sample(sample_id, type_df):
        present_types = type_df[sample_id][type_df[sample_id] > 0].index
        type_to_pathways = {functional_id: functional_id.split('.') for functional_id in present_types}
        biomass = type_df[sample_id][present_types].to_dict()
        edge_counter = {}
        for p_list in type_to_pathways.values():
            for pid in p_list:
                if pid in p_id_to_str:
                    src, tgt = p_id_to_str[pid].split('->')
                    edge = (src.strip(), tgt.strip())
                    edge_counter[edge] = edge_counter.get(edge, 0) + 1
        return edge_counter, biomass
    def build_matrix(edge_counter):
        mat = np.zeros((matrix_size, matrix_size))
        mat_color = [[None]*matrix_size for _ in range(matrix_size)]
        for (src, tgt), count in edge_counter.items():
            i, j = sub_idx[src], sub_idx[tgt]
            mat[i, j] = count
            pid = str_to_pid.get(f"{src}->{tgt}", None)
            cat = int(p_id_to_cat.get(pid, 10)) if pid else 10
            mat_color[i][j] = custom_colors[cat]
        return mat, mat_color
    def plot_matrix(ax, mat, mat_color, all_substances):
        
        matrix_size = len(all_substances)
        for (i, j), val in np.ndenumerate(mat):
            if val > 0:
                ax.scatter(j, matrix_size - 1 - i, s=8+(val-1)*10, color=mat_color[i][j], alpha=1)
        ax.set_xticks(range(matrix_size))
        ax.set_xticklabels(all_substances, rotation=90, fontsize=7)
        ax.set_yticks(range(matrix_size))
        ax.set_yticklabels(all_substances[::-1], fontsize=7)
        ax.set_xlabel("Product", fontsize=9)
        ax.set_ylabel("Substrate", fontsize=9)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        max_val = int(np.max(mat))
        for i in range(1, max_val + 1):
            ax.scatter([], [], s=8+(i-1)*10, c='black', alpha=0.5, label=f'{i} type' if i==1 else f'{i} types')
        ax.legend(title="# Types", loc='lower left', frameon=False, fontsize=7, title_fontsize=9, labelspacing = 0.3)

    def plot_biomass_barplot(ax, sample_id, type_df, p_id_to_str_dict, p):
        present_types = type_df[sample_id][type_df[sample_id] > 0]
        sorted_types = present_types.sort_values(ascending=False).head(20)
        formatted_labels = []
        for type_id in sorted_types.index:
            pids = type_id.split('.')
            formatted_pathways = [p_id_to_str_dict.get(pid, pid) for pid in pids]
            label = ' '.join(formatted_pathways)
            formatted_labels.append(label)
        ax.bar(range(len(sorted_types)), sorted_types.values, color="dimgray")
        ax.set_xticks(range(len(sorted_types)))
        ax.set_xticklabels(formatted_labels, fontsize=7, rotation=90)
        ax.set_ylabel("Biomass", fontsize=10)
        ax.set_xlabel("Type Pathways", fontsize=10)
        ax.tick_params(axis='y', labelsize=8)
        despine(ax)
        threshold = present_types.quantile(p/100)
        idx_cutoff = np.argmax(sorted_types.values <= threshold)
        if idx_cutoff > 0:
            ax.axvline(x=idx_cutoff - 0.5, color='red', linestyle='--', linewidth=1)
    def plot_biomass_barplot_h(ax, sample_id, type_df, p_id_to_str_dict, p):
        present_types = type_df[sample_id][type_df[sample_id] > 0]
        sorted_types = present_types.sort_values(ascending=False).head(18)
        
        formatted_labels = []
        for type_id in sorted_types.index:
            pids = type_id.split('.')
            formatted_pathways = [p_id_to_str_dict.get(pid, pid) for pid in pids]
            label = ' '.join(formatted_pathways)
            formatted_labels.append(label)

        ax.barh(range(len(sorted_types)), sorted_types.values, color="dimgray")
        ax.set_yticks(range(len(sorted_types)))
        ax.set_yticklabels(formatted_labels, fontsize=7)
        ax.invert_yaxis()  # most abundant at top
        ax.set_xlabel("Biomass", fontsize=9, labelpad=1)
        ax.set_ylabel("Type Pathways", fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        despine(ax)

        threshold = present_types.quantile(p / 100)
        idx_cutoff = np.argmax(sorted_types.values <= threshold)
        if idx_cutoff > 0:
            ax.axhline(y=idx_cutoff - 0.5, color='red', linestyle='--', linewidth=1)
    def split_data_df(data_df, p):
        """
        Splits the data_df into two DataFrames:
        - One with bacteria whose abundance is >= percentile threshold (most abundant)
        - One with bacteria whose abundance is < percentile threshold (least abundant)

        Parameters:
            data_df (pd.DataFrame): DataFrame with types as rows and samples as columns
            p (float): Percentile to split (e.g., 50 = median, 75 = top 25%)

        Returns:
            most_abundant_df, least_abundant_df: Filtered DataFrames (rows with all 0 are dropped)
        """
        most_abundant_df = data_df.copy()
        least_abundant_df = data_df.copy()
        for sample in data_df.columns:
            nonzero_values = data_df[sample][data_df[sample] > 0]
            threshold = np.percentile(nonzero_values, p)

            most_abundant_df[sample] = data_df[sample].where(data_df[sample] >= threshold, other=0.0)
            least_abundant_df[sample] = data_df[sample].where(data_df[sample] < threshold, other=0.0)

        most_abundant_df = most_abundant_df.loc[(most_abundant_df != 0).any(axis=1)]
        least_abundant_df = least_abundant_df.loc[(least_abundant_df != 0).any(axis=1)]

        return most_abundant_df, least_abundant_df
    def map_colors(pathway_index, cat_dict, color_palette):
        return [color_palette[int(cat_dict.get(p, 10))] for p in pathway_index]
    
    def add_category_inset(ax, abundances, color_map, category_dict, custom_colors):
        # Aggregate by category
        category_abundance = {}
        for pid, val in abundances.items():
            category = int(category_dict.get(pid, 10))  # fallback to last color
            category_abundance[category] = category_abundance.get(category, 0) + val

        # Normalize to sum to 1
        total = sum(category_abundance.values())
        proportions = [category_abundance.get(i, 0) / total for i in range(len(custom_colors))]

        # Create inset axis (To understand behaviour see: https://matplotlib.org/3.1.1/gallery/axes_grid1/inset_locator_demo.html)
        inset_ax = inset_axes(ax, width="100%", height="100%", loc='upper right', bbox_to_anchor=(0.1, 1.03, 0.8, 0.1), 
                              bbox_transform=ax.transAxes, borderpad=0)

        # Plot horizontal stacked bar
        left = 0
        for i, prop in enumerate(proportions):
            if prop > 0:
                inset_ax.barh(y=0, width=prop, left=left, height=1.01, color=custom_colors[i])
                left += prop

        inset_ax.set_xlim(0, 1)
        inset_ax.set_ylim(0, 1)
        inset_ax.set_xticks([0, 0.5, 1])
        inset_ax.set_yticks([])
        inset_ax.tick_params(axis='x', labelsize=7, pad=1, bottom='True', top=False)
        inset_ax.set_frame_on(False)

    def add_type_cost_inset(ax, data_df, E_D_type):
        # Step 1: Compute average abundance per type
        avg_abundance = data_df.mean(axis=1)

        # Step 2: For each type, compute total cost
        type_total_costs = []
        type_abundances = []

        for type_id, abundance in avg_abundance.items():
            try:
                path_ids = [int(pid) for pid in type_id.strip().split('.')]
                costs = [DA.calculate_enzymatic_cost_from_pathway_id(str(pid), E_D_type) for pid in path_ids]
                total_cost = sum(costs)
                type_total_costs.append(total_cost)
                type_abundances.append(abundance)
            except:
                continue  # Skip malformed types

        # Step 3: Bin the data
        bins = np.linspace(0.1, 3, 30)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        bin_sums = np.zeros(len(bin_centers))

        for cost, abundance in zip(type_total_costs, type_abundances):
            bin_idx = np.digitize(cost, bins) - 1
            if 0 <= bin_idx < len(bin_sums):
                bin_sums[bin_idx] += abundance

        # Normalize to sum to 1
        total = bin_sums.sum()
        proportions = bin_sums / total if total > 0 else bin_sums

        # Step 4: Create inset axis
        inset_ax = inset_axes(ax, width="100%", height="100%", loc='upper right',
                            bbox_to_anchor=(0.5, 0.45, 0.45, 0.4),
                            bbox_transform=ax.transAxes, borderpad=0)

        # Step 5: Plot
        inset_ax.bar(x=bin_centers, height=proportions, width=bin_width * 0.98,
                    color='dimgrey', edgecolor='black', linewidth=0.2)
        x_vals = np.linspace(0.1, 3, 500)
        y_vals, x_max = tradeoff_function(x_vals, tf_type=0)
        y_max, _ = tradeoff_function(x_max, tf_type=0)
        inset_ax.axvline(x=x_max, ymin=0, ymax=1, color='#8b0000', linestyle='--', alpha=1, linewidth=1.5) # y_max/inset_ax.get_ylim()[1]
        #inset_ax.axvline(x=3-x_max, ymin=0, ymax=1, color="#4AB3B7", linestyle='--', alpha=1, linewidth=1.25)

        # Step 6: Format
        inset_ax.set_xlim(0, 3)
        inset_ax.set_xticks([0, 1.5, 3])
        inset_ax.set_xlabel('Total enzymatic cost \n per type', loc='center', fontsize=8, labelpad=1)
        inset_ax.set_yticks([])
        inset_ax.tick_params(axis='x', labelsize=7, bottom=True)
        inset_ax.set_frame_on(False)
    def add_type_cost_inset(ax, data_df, E_D_type):
        # Step 1: Compute average abundance per type
        avg_abundance = data_df.mean(axis=1)

        # Step 2: Get the optimal cost from the tradeoff function
        x_vals = np.linspace(0.1, 3, 500)
        _, x_max = tradeoff_function(x_vals, tf_type=0)  # Optimal enzymatic cost

        # Step 3: For each type, compute total cost and distance to x_max
        type_cost_distances = []
        type_abundances = []

        for type_id, abundance in avg_abundance.items():
            try:
                path_ids = [int(pid) for pid in type_id.strip().split('.')]
                costs = [DA.calculate_enzymatic_cost_from_pathway_id(str(pid), E_D_type) for pid in path_ids]
                total_cost = sum(costs)
                distance = abs(total_cost - x_max)
                type_cost_distances.append(distance)
                type_abundances.append(abundance)
            except:
                continue  # Skip malformed types

        # Step 4: Bin the distances
        bins = np.linspace(0, 3, 30)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        bin_sums = np.zeros(len(bin_centers))

        for distance, abundance in zip(type_cost_distances, type_abundances):
            bin_idx = np.digitize(distance, bins) - 1
            if 0 <= bin_idx < len(bin_sums):
                bin_sums[bin_idx] += abundance

        # Step 5: Normalize to sum to 1
        total = bin_sums.sum()
        proportions = bin_sums / total if total > 0 else bin_sums

        # Step 6: Create inset axis
        inset_ax = inset_axes(ax, width="100%", height="100%", loc='upper right',
                            bbox_to_anchor=(0.5, 0.45, 0.45, 0.4),
                            bbox_transform=ax.transAxes, borderpad=0)

        # Step 7: Plot
        inset_ax.bar(x=bin_centers, height=proportions, width=bin_width * 0.8,
                    color='dimgrey', edgecolor='black', linewidth=0.8)

        # Step 8: Format
        inset_ax.set_xlim(0, 0.9)
        inset_ax.set_xticks([0,0.5])
        inset_ax.set_xlabel('Distance to optimal cost', loc='center', fontsize=8, labelpad=1)
        inset_ax.set_yticks([])
        inset_ax.tick_params(axis='x', labelsize=7, bottom=True, top=False)
        inset_ax.set_frame_on(False)
    
    def compute_mean_binary_similarities_across_samples(type_df):
        """
        Compute mean and std of several similarity metrics (Jaccard, Sørensen–Dice, Hamming, Overlap)
        between all types within each sample.

        Returns:
            A dictionary with the global mean and std for each metric.
        """
        from itertools import combinations

        metrics = {
            "Jaccard": [],
            "Dice": [],
            "Hamming": [],
            "Overlap": []
        }

        for sample_id in type_df.columns:
            present_types = type_df[sample_id][type_df[sample_id] > 0].index
            if len(present_types) < 2:
                continue

            type_to_pathways = {t: set(t.split('.')) for t in present_types}
            pairs = combinations(present_types, 2)

            jaccard_vals, dice_vals, hamming_vals, overlap_vals = [], [], [], []

            for a, b in pairs:
                a_set = type_to_pathways[a]
                b_set = type_to_pathways[b]

                inter = len(a_set & b_set)
                union = len(a_set | b_set)
                a_only = len(a_set)
                b_only = len(b_set)
                total = len(a_set ^ b_set) + inter  # total bits (union in hamming sense)

                if union > 0:
                    jaccard_vals.append(inter / union)
                if (a_only + b_only) > 0:
                    dice_vals.append(2 * inter / (a_only + b_only))
                if total > 0:
                    hamming_vals.append(1 - (len(a_set ^ b_set) / total))
                if min(a_only, b_only) > 0:
                    overlap_vals.append(inter / min(a_only, b_only))

            if jaccard_vals:
                metrics["Jaccard"].append(np.mean(jaccard_vals))
            if dice_vals:
                metrics["Dice"].append(np.mean(dice_vals))
            if hamming_vals:
                metrics["Hamming"].append(np.mean(hamming_vals))
            if overlap_vals:
                metrics["Overlap"].append(np.mean(overlap_vals))

        results = {}
        for name, values in metrics.items():
            if values:
                mean = np.mean(values)
                std = np.std(values)
                print(f"{name} similarity across samples: {mean:.3f} ± {std:.3f}")
                results[name] = (mean, std)

        return results
    # === Load data and helper mappings ===
    f, c, m, e, E_D_type = 'fraction_nl', '6', '0', '1.1', 0
    tradeoff_data = f'{f}_{c}_{m}_{e}'
    rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)
    type_df = MDA.get_data_df(rootpath, data_type='taxonomy', sample_size=200)
    metadata = MDA.get_metadata_dict(rootpath, raw=True, sample_size=200)
    folder_path = os.path.join(rootpath, '683')
    D_mat = DA.get_D_mat(folder_path)
    E_avec = DA.get_E_avec(folder_path)
    p_strs = DA.get_pathways_strs(D_mat)
    p_ids, _ = DA.get_pathways_id(D_mat, E_avec)
    p_str_to_pid = dict(zip(p_strs, p_ids))
    p_id_to_str = dict(zip(p_ids, p_strs))
    str_to_pid = {v: k for k, v in p_id_to_str.items()}
    custom_colors = ["#2461AA", "#EC9627", "#58BF91", "#D84752", "#F8CB17", "#90429A", "#2E5C3F", "#B36F41", "#DC79B3", "#4AB3B7", "#000000"]
    p_id_to_cat, _ = get_functions_categories(group_size=10, num_substances=30, p_str_to_p_id_dict=p_str_to_pid)   

    # Build matrices
    ec_h, biomass_h = process_sample(healthy_sample_id, type_df)
    ec_d, biomass_d = process_sample(dysbiotic_sample_id, type_df)
    all_substances = sorted(set(x for e in ec_h | ec_d for x in e), key=lambda s: float(s) if s.replace('.', '', 1).isdigit() else s)
    sub_idx = {s: i for i, s in enumerate(all_substances)}
    matrix_size = len(all_substances)

    mat_h, color_h = build_matrix(ec_h)
    mat_d, color_d = build_matrix(ec_d)

    # === Setup figure ===
    fig = plt.figure(figsize=(18.4 * 0.393701, 19 * 0.393701))
    gs = fig.add_gridspec(4, 4, width_ratios=[0.15, 0.35, 0.15, 0.35],height_ratios=[0.4, 0.3, 0.06, 0.24])
    ax00 = fig.add_subplot(gs[0,:2])
    ax01 = fig.add_subplot(gs[0,2:])
    ax10 = fig.add_subplot(gs[1,1])
    ax11 = fig.add_subplot(gs[1,3], sharex=ax10)
    ax20 = fig.add_subplot(gs[3,:2])
    ax21 = fig.add_subplot(gs[3,2:], sharey=ax20)

    #sns.set(style="whitegrid")

    # === Top row: Matrix plots ===
    with sns.axes_style("whitegrid"):
        plot_matrix(ax00, mat_h, color_h, all_substances)
        plot_matrix(ax01, mat_d, color_d, all_substances)

    # === Second row: Biomass barplots ===
    plot_biomass_barplot_h(ax10, healthy_sample_id, type_df, p_id_to_str, p)
    plot_biomass_barplot_h(ax11, dysbiotic_sample_id, type_df, p_id_to_str, p)

    # === Bottom row: Most abundant pathway barplots ===
    grouped_data = {
        "Healthy": type_df[[s for s in type_df.columns if s in metadata and metadata[s]["rho"] >= 0]],
        "Dysbiotic": type_df[[s for s in type_df.columns if s in metadata and metadata[s]["rho"] < 0]],
    }

    for ax, (label, df_group) in zip([ax20, ax21], grouped_data.items()):
        most_df, _ = split_data_df(df_group, p)
        most_pathways = MDA.calculate_pathways_df(most_df, E_D_type=0, include_cost=False, include_presence=True)
        avg_most = most_pathways.mean(axis=1)
        if sort_by == "abundance":
            #avg_most = avg_most[avg_most > 0.01].sort_values(ascending=False)
            avg_most = avg_most.sort_values(ascending=False).head(20)
            avg_most = avg_most / avg_most.sum()  # Normalize to fractions
        else:
            avg_most = avg_most.loc[sorted(avg_most.index, key=lambda x: int(x))]
        colors = map_colors(avg_most.index, p_id_to_cat, custom_colors)
        ax.bar(x=avg_most.index, height=avg_most.values, color=colors, width=1)
        ax.set_xticks(range(len(avg_most.index)))
        ax.set_xticklabels([p_id_to_str.get(pid, pid) for pid in avg_most.index], rotation=90, fontsize=7)
        ax.set_xlim(-0.5, len(avg_most.index) - 0.5)
        ax.set_ylabel("Normalized Avg Abundance", fontsize=9)
        ax.set_yticks([0,0.1,0.2])
        ax.set_xlabel("Pathways", fontsize=9)
        ax.tick_params(axis='y', labelsize=8, pad=1)
        #axs[2, i].set_title(f"{label} - Most Abundant", fontsize=10)
        despine(ax)
        add_category_inset(ax, avg_most, custom_colors, p_id_to_cat, custom_colors)
        add_type_cost_inset(ax, most_df, E_D_type)
        compute_mean_binary_similarities_across_samples(most_df)

    # === Final layout ===
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    image_name = os.path.join(image_dir, f'pathway_structure_6panel_{p}.svg')
    plt.subplots_adjust(top=0.94, bottom=0.09, left=0.075, right=0.99, hspace=0.07, wspace=0.4)
    plt.savefig(image_name, format='svg', dpi=1200)
    plt.show()



def plot_diversity_panel(gut_dir):

    def compute_diversity_metrics(abundances):
        values = abundances[abundances > 0].values
        proportions = values / values.sum()
        richness = len(values)
        simpson = 1 - np.sum(proportions**2)
        dominance = np.max(proportions)
        evenness = entropy(proportions) / np.log(len(values)) if len(values) > 1 else 0
        return richness, dominance, simpson, evenness

    def extract_diversity_df(full_data_dict, dataset_name, healthy_diag, unhealthy_diag):
        records = []
        for patient_id, samples in full_data_dict.items():
            for sample in samples:
                diagnosis = sample['Diagnosis']
                sample_id = sample['Sample ID']
                if diagnosis in healthy_diag:
                    status = 'Healthy'
                elif diagnosis in unhealthy_diag:
                    status = 'Unhealthy'
                else:
                    continue  # Skip others
                abundances = sample['Data']
                r, d, s, e = compute_diversity_metrics(abundances)
                records.append({
                    'Dataset': dataset_name,
                    'Diagnosis': status,
                    'Sample ID': sample_id,
                    'Richness': r,
                    'Dominance': d,
                    'Simpson': s,
                    'Evenness': e
                })
        return pd.DataFrame(records)

    # Diagnosis mapping
    healthy_diagnosis_dict = {
        'IBD': ['nonIBD'],
        'CDI': ['Healthy'],
        'IBS': ['Healthy'],
        'CRC': ['Healthy'],
        'Model': ['Healthy']
    }
    unhealthy_diagnosis_dict = {
        'IBD': ['CD', 'UC'],
        'IBS': ['IBS-C', 'IBS-D'],
        'CRC': ['MP', 'Stage_0', 'Stage_I_II', 'Stage_III_IV'],
        'CDI': ['Unhealthy'],
        'Model': ['Unhealthy']
    }

    # Data loading
    f, c, m, e, E_D_type = 'fraction_nl', '6', '0', '1.1', 0
    tradeoff_data = f'{f}_{c}_{m}_{e}'
    rootpath_data_model = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)

    metadata_dict_model = MDA.get_metadata_dict(rootpath_data_model, overwrite=False, sample_size=200)
    data_df_model = MDA.get_data_df(rootpath_data_model, data_type='taxonomy', overwrite=False, sample_size=200)
    full_data_dict_model = MDA.get_full_data_dict(metadata_dict_model, data_df_model)

    rootpath_data_IBD = os.path.join(gut_dir,'real_data','IBD_MDB')
    rootpath_data_IBS = os.path.join(gut_dir,'real_data','IBS_Mars_Cell')
    rootpath_data_CDI = os.path.join(gut_dir,'real_data','CD_Ferretti_Elife')
    rootpath_data_CRC = os.path.join(gut_dir,'real_data','CRC_Yachida_NatMed')

    full_data_dict_IBD = IBD.get_full_data_dict(IBD.get_metadata_dict(rootpath_data_IBD), IBD.get_data_df(rootpath_data_IBD, data_type='taxonomy'))
    full_data_dict_IBS = IBS.get_full_data_dict(IBS.get_metadata_dict(rootpath_data_IBS), IBS.get_data_df(rootpath_data_IBS, data_type='taxonomy'))
    full_data_dict_CDI = CDI.get_full_data_dict(CDI.get_metadata_dict(rootpath_data_CDI), CDI.get_data_df(rootpath_data_CDI))
    full_data_dict_CRC = CRC.get_full_data_dict(CRC.get_metadata_dict(rootpath_data_CRC), CRC.get_data_df(rootpath_data_CRC, data_type='taxonomy2'))

    df_all = pd.concat([
        extract_diversity_df(full_data_dict_model, 'Model', healthy_diagnosis_dict['Model'], unhealthy_diagnosis_dict['Model']),
        extract_diversity_df(full_data_dict_IBD, 'IBD', healthy_diagnosis_dict['IBD'], unhealthy_diagnosis_dict['IBD']),
        extract_diversity_df(full_data_dict_IBS, 'IBS', healthy_diagnosis_dict['IBS'], unhealthy_diagnosis_dict['IBS']),
        extract_diversity_df(full_data_dict_CRC, 'CRC', healthy_diagnosis_dict['CRC'], unhealthy_diagnosis_dict['CRC']),
        extract_diversity_df(full_data_dict_CDI, 'CDI', healthy_diagnosis_dict['CDI'], unhealthy_diagnosis_dict['CDI']),
    ], ignore_index=True)

    metrics = ['Richness', 'Dominance', 'Simpson', 'Evenness']
    datasets = ['Model', 'IBD', 'CDI', 'IBS', 'CRC']

    fig, axs = plt.subplots(len(datasets), len(metrics), figsize=(18.4 * 0.393701, 19 * 0.393701))

    palette = {'Healthy': "#0078B9", 'Unhealthy': "#EA0017"}

    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            ax = axs[i, j]
            sub_df = df_all[df_all['Dataset'] == dataset]
            sub_df = sub_df[sub_df['Diagnosis'].isin(['Healthy', 'Unhealthy'])]
            sub_df['Diagnosis'] = pd.Categorical(sub_df['Diagnosis'], categories=['Healthy', 'Unhealthy'], ordered=True)
            sns.boxplot(data=sub_df, x='Diagnosis', y=metric, hue='Diagnosis', ax=ax, palette=palette, showfliers=False, legend=False)
            if i == 0:
                ax.set_title(metric, fontsize=12)
            if j == 0:
                ax.set_ylabel(dataset, fontsize=11)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(labelsize=9)

    plt.subplots_adjust(top=0.94, bottom=0.09, left=0.075, right=0.99, hspace=0.07, wspace=0.4)
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    image_name = os.path.join(image_dir, f'diversity_measures.svg')
    plt.savefig(image_name, format='svg', dpi=600)
    plt.show()


def plot_cf_cp_model(gut_dir, realization_list, sample_hours=7 * 12):
    # === Load data ===
    f, c, m, e, E_D_type = 'fraction_nl', '6', '0', '1.1', 0
    tradeoff_data = f'{f}_{c}_{m}_{e}'
    rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)
    metadata_dict = MDA.get_metadata_dict(rootpath, raw=True, sample_size=200)

    # === PRELOAD CF/CP MATRICES PER REALIZATION ===
    cf_cp_cache = {}
    for realization in realization_list:
        folder_path = os.path.join(rootpath, realization)
        t_vec = np.array(get_tvec(folder_path, sample_hours, wanted_tvec='t'))
        cf_ij_tvec, cp_ij_tvec = get_cf_cp_mat_tvecs(folder_path, sample_hours)
        cf_cp_cache[realization] = (t_vec, cf_ij_tvec, cp_ij_tvec)

    # === STORAGE ===
    cf_values = defaultdict(list)
    cp_values = defaultdict(list)
    interaction_classes = defaultdict(lambda: defaultdict(int))  # diagnosis -> class -> count
    interaction_totals = defaultdict(int)
    interaction_weights = defaultdict(lambda: defaultdict(float))  # diagnosis -> class -> sum(|a|+|b|)
    total_weight = defaultdict(float)

    # === CLASS LABELS ===
    class_labels = {
        'Mutualism': {('+', '+')},
        'Commensalism': {('+', '0'), ('0', '+')},
        'Exploitation': {('+', '-'), ('-', '+')},
        'Competition': {('-', '-')},
    }

    label_to_class = {}
    for cls, pairs in class_labels.items():
        for a, b in pairs:
            label_to_class[f"{a}/{b}"] = cls

    color_map = {"Healthy": "#0078B9", "Unhealthy": "#EA0017"}
    display_labels = {"Healthy": "Healthy", "Unhealthy": "Dysbiotic"}

    # === PROCESS EACH SAMPLE ===
    for sample_id, meta in metadata_dict.items():
        diagnosis = meta["diagnosis"]
        realization = meta["realization"]
        time = meta["time"]

        t_vec, cf_ij_tvec, cp_ij_tvec = cf_cp_cache[realization]
        time_idx_check = np.where(t_vec == time)[0][0]

        cf_ij = cf_ij_tvec[time_idx_check]
        cp_ij = cp_ij_tvec[time_idx_check]

        cf_values[diagnosis].append(np.sum(cf_ij))
        cp_values[diagnosis].append(np.sum(cp_ij) - np.trace(cp_ij))

        net_interaction = cf_ij - cp_ij
        n = net_interaction.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                a, b = net_interaction[i, j], net_interaction[j, i]
                sign_a = '+' if a > 0 else '-' if a < 0 else '0'
                sign_b = '+' if b > 0 else '-' if b < 0 else '0'
                if sign_a == '0' and sign_b == '0':
                    continue
                label = f"{sign_a}/{sign_b}"
                interaction_class = label_to_class.get(label)
                if interaction_class:
                    interaction_classes[diagnosis][interaction_class] += 1
                    interaction_totals[diagnosis] += 1

                    weight = abs(a) + abs(b)
                    interaction_weights[diagnosis][interaction_class] += weight
                    total_weight[diagnosis] += weight

    all_classes = ['Mutualism', 'Commensalism', 'Exploitation', 'Competition']
    all_classes_labels = ['Mutualism \n (+/+)', 'Commensalism \n (+/0)', 'Exploitation \n (+/-)', 'Competition \n (-/-)']

    # === COUNT-BASED %
    interaction_df = pd.DataFrame(index=all_classes, columns=["Healthy", "Unhealthy"]).fillna(0.0)
    for diag in ["Healthy", "Unhealthy"]:
        total = interaction_totals[diag]
        for label in all_classes:
            count = interaction_classes[diag].get(label, 0)
            interaction_df.loc[label, diag] = 100.0 * count / total if total > 0 else 0.0

    # === WEIGHTED %
    interaction_weight_df = pd.DataFrame(index=all_classes, columns=["Healthy", "Unhealthy"]).fillna(0.0)
    for diag in ["Healthy", "Unhealthy"]:
        total = total_weight[diag]
        for label in all_classes:
            w = interaction_weights[diag].get(label, 0.0)
            interaction_weight_df.loc[label, diag] = 100.0 * w / total if total > 0 else 0.0

    # === PLOTTING ===
    fig = plt.figure(figsize=(18.4 * 0.393701, 12 * 0.393701))  # taller now
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :])

    # Panel 1 – Cross-feeding
    sns.boxplot(data=[cf_values["Healthy"], cf_values["Unhealthy"]],
                ax=ax1,
                palette=[color_map["Healthy"], color_map["Unhealthy"]],
                showfliers=False)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([display_labels["Healthy"], display_labels["Unhealthy"]])
    ax1.set_ylabel("Total Cross-Feeding")
    ax1.set_title("")
    despine(ax1)

    # Panel 2 – Competition
    sns.boxplot(data=[cp_values["Healthy"], cp_values["Unhealthy"]],
                ax=ax2,
                palette=[color_map["Healthy"], color_map["Unhealthy"]],
                showfliers=False)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([display_labels["Healthy"], display_labels["Unhealthy"]])
    ax2.set_ylabel("Total Competition")
    ax2.set_title("")
    despine(ax2)

    # Panel 3 – Count-based interactions
    x = np.arange(len(all_classes))
    width = 0.35
    ax3.bar(x - width / 2, interaction_df["Healthy"], width, edgecolor='black', linewidth=1.5,
            label="Healthy", color=color_map["Healthy"])
    ax3.bar(x + width / 2, interaction_df["Unhealthy"], width, edgecolor='black', linewidth=1.5,
            label="Dysbiotic", color=color_map["Unhealthy"])
    ax3.set_xticks(x)
    ax3.set_xticklabels(all_classes_labels)
    ax3.set_ylabel("Percentage")
    ax3.set_xlabel("Types of Interactions")
    ax3.set_title("")
    ax3.legend()
    despine(ax3)

    # Panel 4 – Weighted interactions
    ax4.bar(x - width / 2, interaction_weight_df["Healthy"], width, edgecolor='black', linewidth=1.5,
            label="Healthy", color=color_map["Healthy"])
    ax4.bar(x + width / 2, interaction_weight_df["Unhealthy"], width, edgecolor='black', linewidth=1.5,
            label="Dysbiotic", color=color_map["Unhealthy"])
    ax4.set_xticks(x)
    ax4.set_xticklabels(all_classes_labels)
    ax4.set_ylabel("Weighted Percentage")
    ax4.set_xlabel("Types of Interactions")
    ax4.set_title("")
    ax4.legend()
    despine(ax4)

    plt.subplots_adjust(top=0.97, bottom=0.06, left=0.09, right=0.98, hspace=0.35, wspace=0.25)
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    image_name = os.path.join(image_dir, f'cf_cp_comparison_model.svg')
    plt.savefig(image_name, format='svg', dpi=600)
    plt.show()

def robustness_taxonomic_level_interaction_fig_IBD(gut_dir):


    IBD_species_path = os.path.join(gut_dir, 'real_data', 'IBD_MDB', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    IBD_genus_path = os.path.join(gut_dir, 'real_data', 'IBD_MDB', 'taxonomy_level_genus', 'flashweave', 'bootstrap_NHE-S', 
                                  'BacFrac_1.0_FullsetFrac_1.0')
    IBD_family_path = os.path.join(gut_dir, 'real_data', 'IBD_MDB', 'taxonomy_level_family', 'flashweave', 'bootstrap_NHE-S', 
                                   'BacFrac_1.0_FullsetFrac_1.0')

    color_map = {'Healthy': '#0078B9', 'IBD': '#EA0017'}
    
    
    fig = plt.figure(figsize = (18.4*0.393701, 7*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(3,1, height_ratios=[1/3, 1/3, 1/3])
    
    ax00 = fig.add_subplot(gs[0,0])
    ax10 = fig.add_subplot(gs[1,0])
    ax20 = fig.add_subplot(gs[2,0])

    IBD_replacement_dict = dict(zip(['Unhealthy'],
                                     ['IBD']))
    
    for subset_fraction in ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:
        IBD_species_df = get_data(IBD_species_path, None, None, subset_fraction, healthy_normalized=True, overwrite=True)
        IBD_genus_df = get_data(IBD_genus_path,None, None, subset_fraction, healthy_normalized=True, overwrite=True)
        IBD_family_df = get_data(IBD_family_path,None, None, subset_fraction, healthy_normalized=True, overwrite=True)
        for IBD_df, ax in zip([IBD_species_df, IBD_genus_df, IBD_family_df], [ax00,ax10,ax20]):
            IBD_df['Group'] = IBD_df['Group'].cat.rename_categories(IBD_replacement_dict)
            IBD_df['Group'] = pd.Categorical(IBD_df['Group'], ordered=True, categories=['Healthy', 'IBD'])
            sns.boxplot(x='Subset Fraction', y='rho', hue='Group',  data=IBD_df, palette=color_map, ax=ax, showfliers=False, width=0.6)
        
    
    j=0
    for i,ax in enumerate([ax00,ax10, ax20]):
        ax.set_ylabel('ENBI', labelpad=1)
        despine(ax)
        if ax.get_legend():
            ax.legend_.remove()
        if i < 2:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        # Remove only the horizontal whisker caps
        for line in ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            # Horizontal whisker caps have exactly two points and are OUTSIDE the box limits
            if len(x_data) == 2 and np.isclose(y_data[0], y_data[1]):
                if j in [0,1]:
                    line.set_visible(False)
                j += 1
                if j==3:
                    j=0
        
    

    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'robustness_taxonomic_level_IBD_ENBI.svg')

    plt.subplots_adjust(top=0.98, bottom=0.17, left=0.1, right=0.98, hspace=0.3, wspace=0.35)
    plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    plt.show()

def robustness_taxonomic_level_interaction_fig_IBS(gut_dir):


    IBD_species_path = os.path.join(gut_dir, 'real_data', 'IBS_Mars_Cell', 'bootstrap_NHE-S', 'BacFrac_1.0_FullsetFrac_1.0')
    IBD_genus_path = os.path.join(gut_dir, 'real_data', 'IBS_Mars_Cell', 'taxonomy_level_genus', 'flashweave', 'bootstrap_NHE-S', 
                                  'BacFrac_1.0_FullsetFrac_1.0')
    IBD_family_path = os.path.join(gut_dir, 'real_data', 'IBS_Mars_Cell', 'taxonomy_level_family', 'flashweave', 'bootstrap_NHE-S', 
                                   'BacFrac_1.0_FullsetFrac_1.0')

    color_map = {'Healthy': '#0078B9', 'IBS': '#EA0017'}
    
    
    fig = plt.figure(figsize = (18.4*0.393701, 7*0.393701)) #20 cm to inches width
    gs = fig.add_gridspec(3,1, height_ratios=[1/3, 1/3, 1/3])
    
    ax00 = fig.add_subplot(gs[0,0])
    ax10 = fig.add_subplot(gs[1,0])
    ax20 = fig.add_subplot(gs[2,0])

    IBD_replacement_dict = dict(zip(['Unhealthy'], ['IBS']))
    
    for subset_fraction in ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:
        IBD_species_df = get_data(IBD_species_path, None, None, subset_fraction, healthy_normalized=True, overwrite=True)
        IBD_genus_df = get_data(IBD_genus_path,None, None, subset_fraction, healthy_normalized=True, overwrite=True)
        IBD_family_df = get_data(IBD_family_path,None, None, subset_fraction, healthy_normalized=True, overwrite=True)
        for IBD_df, ax in zip([IBD_species_df, IBD_genus_df, IBD_family_df], [ax00,ax10,ax20]):
            IBD_df['Group'] = IBD_df['Group'].cat.rename_categories(IBD_replacement_dict)
            IBD_df['Group'] = pd.Categorical(IBD_df['Group'], ordered=True, categories=['Healthy', 'IBS'])
            sns.boxplot(x='Subset Fraction', y='rho', hue='Group',  data=IBD_df, palette=color_map, ax=ax, showfliers=False, width=0.6)
        
    
    j=0
    for i,ax in enumerate([ax00,ax10, ax20]):
        ax.set_ylabel('ENBI', labelpad=1)
        despine(ax)
        if ax.get_legend():
            ax.legend_.remove()
        if i < 2:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        # Remove only the horizontal whisker caps
        for line in ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            # Horizontal whisker caps have exactly two points and are OUTSIDE the box limits
            if len(x_data) == 2 and np.isclose(y_data[0], y_data[1]):
                if j in [0,1]:
                    line.set_visible(False)
                j += 1
                if j==3:
                    j=0
        
    

    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'robustness_taxonomic_level_IBS_ENBI.svg')

    plt.subplots_adjust(top=0.98, bottom=0.17, left=0.1, right=0.98, hspace=0.3, wspace=0.35)
    plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    plt.show()


def analyze_realizations_three_panel(realization_list, sample_hours=12*7, percentile_threshold=80):
    def compute_distance_to_optimal(data_df, E_D_type):
        avg_abundance = data_df.mean(axis=0)
        x_vals = np.linspace(0.1, 3, 500)
        _, x_max = tradeoff_function(x_vals, tf_type=0)
        distances, abundances = [], []
        for type_id, abundance in avg_abundance.items():
            try:
                path_ids = [int(pid) for pid in type_id.strip().split('.')]
                costs = [DA.calculate_enzymatic_cost_from_pathway_id(str(pid), E_D_type) for pid in path_ids]
                total_cost = sum(costs)
                distances.append(abs(total_cost - x_max))
                abundances.append(abundance)
            except:
                continue
        distances = np.array(distances)
        abundances = np.array(abundances)
        return (distances * abundances).sum() / abundances.sum() if abundances.sum() > 0 else 0
    
    def split_data_df(data_df, p):
        most_df = data_df.copy()
        for sample in data_df.index:
            nonzero = data_df.loc[sample][data_df.loc[sample] > 0]
            threshold = np.percentile(nonzero, p)
            most_df.loc[sample] = data_df.loc[sample].where(data_df.loc[sample] >= threshold, 0.0)
        return most_df.loc[:, (most_df != 0).any(axis=0)]

    # === Load data ===
    f, c, m, e, E_D_type = 'fraction_nl', '6', '0', '1.1', 0
    tradeoff_data = f'{f}_{c}_{m}_{e}'
    rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)

    dysbiotic_durations = []
    dist_opt_all = []
    dist_opt_80 = []
    mean_cf_dys = []

    for realization in realization_list:
        folder_path = os.path.join(rootpath, realization)
        t_vec = np.array(get_tvec(folder_path, sample_hours, wanted_tvec="t"))
        cf_tvec, cp_tvec = get_cf_cp_mat_tvecs(folder_path, sample_hours)
        
        B_types_dict = get_B_types_dict(folder_path, sample_hours)
        rel_B_df = get_rel_B_df(folder_path, sample_hours, t_vec, B_types_dict, overwrite=False)

        rho_tvec = np.array([(np.sum(cf)-(np.sum(cp)-np.trace(cp)))/(np.sum(cf)+(np.sum(cp)-np.trace(cp))) for cf,cp in zip(cf_tvec, cp_tvec)])
        dys_indices = np.where(rho_tvec < 0)[0]
        if len(dys_indices) == 0:
            continue
        
        longest = max(np.split(dys_indices, np.where(np.diff(dys_indices) != 1)[0] + 1), key=len)
        t_dys = t_vec[longest]
        mask = rel_B_df["t"].isin(t_dys)
        dys_B_df = rel_B_df[mask].drop(columns=["t"])

        dysbiotic_durations.append(len(t_dys))
        dist_opt_all.append(compute_distance_to_optimal(dys_B_df, E_D_type))
        most_df = split_data_df(dys_B_df, percentile_threshold)
        dist_opt_80.append(compute_distance_to_optimal(most_df, E_D_type))
        mean_cf_dys.append(np.mean([cf_tvec[i].sum() for i in longest]))

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18.4*0.393701, 7*0.393701))
    axs[0].scatter(dysbiotic_durations, dist_opt_all)
    axs[0].set_title("All types")
    axs[0].set_xlabel("Longest Dysbiotic Duration")
    axs[0].set_ylabel("Avg Distance to Optimum")

    axs[1].scatter(dysbiotic_durations, dist_opt_80)
    axs[1].set_title("Top 20% Abundant")
    axs[1].set_xlabel("Longest Dysbiotic Duration")
    axs[1].set_ylabel("Avg Distance to Optimum")

    axs[2].scatter(dysbiotic_durations, mean_cf_dys)
    axs[2].set_title("Mean Cross-Feeding")
    axs[2].set_xlabel("Longest Dysbiotic Duration")
    axs[2].set_ylabel("Cross-Feeding")

    image_dir = os.path.join(gut_dir, 'images')
    image_dir = os.path.join(image_dir, 'paperplots')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'understanding_time_dysbiotic.svg')

    plt.subplots_adjust(top=0.98, bottom=0.17, left=0.12, right=0.98, hspace=0.3, wspace=0.3)
    plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    plt.show()


def understanding_time_durations(gut_dir, realization_list, sample_hours=12*7):
    def compute_distance_to_optimal(data_df, E_D_type):
        avg_abundance = data_df.mean(axis=0)
        x_vals = np.linspace(0.1, 3, 500)
        _, x_max = tradeoff_function(x_vals, tf_type=0)
        distances, abundances = [], []
        for type_id, abundance in avg_abundance.items():
            try:
                path_ids = [int(pid) for pid in type_id.strip().split('.')]
                costs = [DA.calculate_enzymatic_cost_from_pathway_id(str(pid), E_D_type) for pid in path_ids]
                total_cost = sum(costs)
                distances.append(abs(total_cost - x_max))
                abundances.append(abundance)
            except:
                continue
        distances = np.array(distances)
        abundances = np.array(abundances)
        return (distances * abundances).sum() / abundances.sum() if abundances.sum() > 0 else 0

    # === Load data ===
    f, c, m, e, E_D_type = 'fraction_nl', '6', '0', '1.1', 0
    tradeoff_data = f'{f}_{c}_{m}_{e}'
    rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)

    dysbiotic_durations, dist_opt_dys, rho_dys = [], [], []
    healthy_durations, dist_opt_healthy, rho_healthy = [], [], []

    for realization in realization_list:
        folder_path = os.path.join(rootpath, realization)
        t_vec = np.array(get_tvec(folder_path, sample_hours, wanted_tvec="t"))
        cf_tvec, cp_tvec = get_cf_cp_mat_tvecs(folder_path, sample_hours)
        B_types_dict = get_B_types_dict(folder_path, sample_hours)
        rel_B_df = get_rel_B_df(folder_path, sample_hours, t_vec, B_types_dict, overwrite=False)

        rho_tvec = np.array([
            (np.sum(cf) - (np.sum(cp) - np.trace(cp))) / (np.sum(cf) + (np.sum(cp) - np.trace(cp)))
            for cf, cp in zip(cf_tvec, cp_tvec)
        ])
        healthy_indices = np.where(rho_tvec <= 0)[0]
        dys_indices = np.where(rho_tvec > 0)[0]

        if len(dys_indices) > 0:
            longest_dys = max(np.split(dys_indices, np.where(np.diff(dys_indices) != 1)[0] + 1), key=len)
            t_dys = t_vec[longest_dys]
            mask = rel_B_df["t"].isin(t_dys)
            dys_B_df = rel_B_df[mask].drop(columns=["t"])
            dysbiotic_durations.append(len(t_dys))
            dist_opt_dys.append(compute_distance_to_optimal(dys_B_df, E_D_type))
            rho_dys.append(np.mean(rho_tvec[longest_dys]))

        if len(healthy_indices) > 0:
            longest_healthy = max(np.split(healthy_indices, np.where(np.diff(healthy_indices) != 1)[0] + 1), key=len)
            t_healthy = t_vec[longest_healthy]
            mask = rel_B_df["t"].isin(t_healthy)
            healthy_B_df = rel_B_df[mask].drop(columns=["t"])
            healthy_durations.append(len(t_healthy))
            dist_opt_healthy.append(compute_distance_to_optimal(healthy_B_df, E_D_type))
            rho_healthy.append(np.mean(rho_tvec[longest_healthy]))

    # === Plot ===
    fig, axs = plt.subplots(2, 2, figsize=(18.4*0.393701, 10*0.393701))
    color_map = {'H': '#0078B9', 'D': '#EA0017'}
    axs[0, 0].scatter(healthy_durations, dist_opt_healthy, c=color_map['H'], alpha=0.65, edgecolors='black', linewidths=1)
    axs[0, 0].set_xlabel("Time Duration", fontsize=12)
    axs[0, 0].set_ylabel("Weighted Avg Distance \n to Optimum", fontsize=12)
    axs[0, 0].set_yticks([0.3,0.6,0.9])

    axs[1, 0].scatter(healthy_durations, rho_healthy, c=color_map['H'], alpha=0.65, edgecolors='black', linewidths=1)
    axs[1, 0].set_xlabel("Time Duration", fontsize=12)
    axs[1, 0].set_ylabel("Net Interaction", fontsize=12)
    axs[1, 0].set_yticks([-0.3,-0.2,-0.1])

    axs[0, 1].scatter(dysbiotic_durations, dist_opt_dys, c=color_map['D'], alpha=0.65, edgecolors='black', linewidths=1)
    axs[0, 1].set_xlabel("Time Duration", fontsize=12)
    axs[0, 1].set_ylabel("Weighted Avg Distance \n to Optimum", fontsize=12)
    axs[0, 1].set_yticks([0.3,0.6,0.9])

    axs[1, 1].scatter(dysbiotic_durations, rho_dys, c=color_map['D'], alpha=0.65, edgecolors='black', linewidths=1)
    axs[1, 1].set_xlabel("Time Duration", fontsize=12)
    axs[1, 1].set_ylabel("Net Interaction", fontsize=12)
    axs[1, 1].set_yticks([0, 0.5, 1])

    

    for ax in axs.ravel():
        ax.tick_params(axis='both', labelsize=10, pad=2)
        despine(ax)

    plt.tight_layout()
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, 'understanding_time_durations.svg'), format='svg', dpi=1200)
    plt.show()

def plot_biomass_barplot_6_vertical(gut_dir,realization,time_indices,sample_hours=12*7):
    def determine_bar_colors(biomass_series, stage):
        colors, bold_flags = [], []
        for type_id in biomass_series.index:
            if stage == "Dys":
                colors.append('#EA0017')  # red
                bold_flags.append(False)
            elif stage == "PreHealthy":
                if type_id not in ref_types_dys:
                    colors.append('#F4C900')  # yellow
                    bold_flags.append(True)
                else:
                    colors.append('#EA0017')  # red
                    bold_flags.append(False)
            elif stage in ["Healthy1", "PreDys"]:
                if type_id in ref_types_dys:
                    colors.append('#F4C900')  # yellow
                    bold_flags.append(True)
                else:
                    colors.append('#0078B9')  # blue
                    bold_flags.append(False)
            elif stage == "Healthy2":
                if type_id in ref_types_dys:
                    colors.append('#F4C900')  # yellow
                    bold_flags.append(True)
                else:
                    colors.append('#0078B9')  # blue
                    bold_flags.append(False)
        return colors, bold_flags
    def determine_bar_colors(biomass_series, stage):
        colors, bold_flags = [], []

        for i, type_id in enumerate(biomass_series.index):
            if stage == "Healthy1":
                colors.append('#0078B9')  # solid blue
            elif stage == "PreDys":
                if i in [0, 1]:
                    colors.append('#F4C900')  # yellow
                else:
                    colors.append('#0078B9')  # solid blue
            elif stage == "Dys":
                colors.append('#EA0017')  # solid red
            elif stage == "PreHealthy":
                if i >= 2:
                    colors.append('#F4C900')  # yellow
                else:
                    colors.append('#EA0017')  # solid red
            elif stage == "Healthy2":
                colors.append('#0078B9')  # solid blue
            bold_flags.append(False)  # no bolding

        return colors, bold_flags
    # === Load pathway info ===
    f, c, m, e, E_D_type = 'fraction_nl', '6', '0', '1.1', 0
    tradeoff_data = f'{f}_{c}_{m}_{e}'
    rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)
    folder_path = os.path.join(rootpath, realization)

    D_mat = DA.get_D_mat(folder_path)
    E_avec = DA.get_E_avec(folder_path)
    pathways_strs = DA.get_pathways_strs(D_mat)
    pathways_ids, _ = DA.get_pathways_id(D_mat, E_avec)
    p_id_to_str_dict = dict(zip(pathways_ids, pathways_strs))

    t_vec = np.array(get_tvec(folder_path, sample_hours))
    cf_tvec, cp_tvec = get_cf_cp_mat_tvecs(folder_path, sample_hours)
    B_types_dict = get_B_types_dict(folder_path, sample_hours)
    rel_B_df = get_rel_B_df(folder_path, sample_hours, t_vec, B_types_dict)

    rho_tvec = np.array([
        (np.sum(cf) - (np.sum(cp) - np.trace(cp))) /
        (np.sum(cf) + (np.sum(cp) - np.trace(cp)))
        for cf, cp in zip(cf_tvec, cp_tvec)
    ])

    labels = ['Healthy1', 'PreDys', 'Dys', 'PreHealthy', 'Healthy2']
    selected_labels = labels[:len(time_indices)]

    all_top_types = {}
    for idx in time_indices:
        t = t_vec[idx]
        row = rel_B_df[rel_B_df['t'] == t]
        if not row.empty:
            biomass = row.drop(columns=["t"]).iloc[0]
            biomass = biomass[biomass > 0].sort_values(ascending=False).head(12)
            all_top_types[idx] = set(biomass.index)

    ref_types_healthy = set()
    if time_indices[0] in all_top_types:
        ref_types_healthy |= all_top_types[time_indices[0]]
    if time_indices[1] in all_top_types:
        ref_types_healthy |= all_top_types[time_indices[1]]

    ref_types_dys = set()
    if time_indices[2] in all_top_types:
        ref_types_dys |= all_top_types[time_indices[2]]
    if time_indices[3] in all_top_types:
        ref_types_dys |= all_top_types[time_indices[3]]


    fig, axs = plt.subplots(len(time_indices), 1, figsize=(18.4*0.393701, 18*0.393701), sharex=True)

    for i, (idx, stage) in enumerate(zip(time_indices, selected_labels)):
        t = t_vec[idx]
        row = rel_B_df[rel_B_df['t'] == t]
        ax = axs[i]
        if row.empty:
            ax.axis('off')
            continue

        biomass = row.drop(columns=["t"]).iloc[0]
        biomass = biomass[biomass > 0].sort_values(ascending=False).head(12)

        formatted_labels = []
        for type_id in biomass.index:
            pids = type_id.split('.')
            formatted_pathways = [p_id_to_str_dict.get(pid, pid) for pid in pids]
            label = ' '.join(formatted_pathways)
            formatted_labels.append(label)

        bar_colors, bold_flags = determine_bar_colors(biomass, stage)
        ax.barh(range(len(biomass)), biomass.values, color=bar_colors)
        '''
        labels_final = [
            f"$\\bf{{{label}}}$" if bold else label
            for label, bold in zip(formatted_labels, bold_flags)
        ]'''
        #ax.set_yticks(range(len(biomass)))
        #ax.set_yticklabels(labels_final, fontsize=7)
        ax.set_yticks(range(len(biomass)))
        ax.set_yticklabels(formatted_labels, fontsize=7)

        ax.invert_yaxis()
        if i == len(time_indices) - 1:
            ax.set_xlabel("Relative Biomass", fontsize=12)
        ax.set_xlim(0, 0.47)

    fig.subplots_adjust(top=0.98, bottom=0.07, left=0.17, right=0.98, hspace=0.15)
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    image_name = os.path.join(image_dir, f'type_transitions_vertical_cleaned_{realization}.svg')
    plt.savefig(image_name, format='svg', transparent=False, dpi=2400)
    plt.show()

def plot_linear_realizations(gut_dir, realization_paths, sample_hours):
    fig, axs = plt.subplots(1, 3, figsize=(18.4 * 0.393701, 5 * 0.393701), sharex=True)

    for ax, realization_path in zip(axs, realization_paths):
        # Load realization data
        t_vec, _, B_types_dict, _, _, _, rho_tvec = get_realization_data(realization_path, sample_hours, full=True)
        t_vec = t_vec / 24 / 365  # Convert time to years

        # Color mapping
        palette = sns.color_palette("tab20c", n_colors=len(B_types_dict.keys()))
        color_mapping = {Type: palette[i % len(palette)] for i, Type in enumerate(B_types_dict.keys())}

        # Plot each type’s biomass over its valid time range
        for Type, (t_init, t_end, _, B_type_t) in B_types_dict.items():
            t_init_yr = t_init / 24 / 365
            t_end_yr = t_end / 24 / 365
            mask = (t_vec >= t_init_yr) & (t_vec < t_end_yr)
            ax.plot(t_vec[mask], B_type_t, label=Type, color=color_mapping[Type])

        # Beautify plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(3, 50)
        tick_positions = [3, 13, 23, 33, 43]
        tick_labels = [0, 10, 20, 30, 40]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_xlabel("Time", fontsize=10)
        ax.tick_params(axis='y', labelsize=9)
    axs[0].set_ylabel("Biomass", fontsize=10)

    plt.tight_layout()
    #plt.subplots_adjust(left=0.06, bottom=0.12, right=0.98, top=0.94, wspace=0.3)

    # Save
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, 'linear_tradeoff_realizations.svg')
    plt.savefig(output_path, format='svg', dpi=2400)
    plt.show()


def plot_biomass_and_normalized_2panel(gut_dir, sample_hours=12*7, sample_size=200):
    # === Load metadata ===
    f, c, m, e, E_D_type = 'fraction_nl', '6', '0', '1.1', 0
    tradeoff_data = f'{f}_{c}_{m}_{e}'
    rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)

    metadata_dict = MDA.get_metadata_dict(rootpath, raw=True, sample_size=sample_size)

    # === Preload B_tvec for each realization ===
    realization_list = sorted(set(meta['realization'] for meta in metadata_dict.values()))
    B_tvec_cache = {}
    for realization in realization_list:
        folder_path = os.path.join(rootpath, realization)
        B_tvec_cache[realization] = np.array(get_tvec(folder_path, sample_hours, wanted_tvec='B'))

    # === Collect metrics per sample ===
    records = []
    for sample_id, meta in metadata_dict.items():
        diagnosis = meta["diagnosis"]
        realization = meta["realization"]
        time_idx = meta["time_idx"]

        try:
            B_vec = B_tvec_cache[realization][time_idx]
        except IndexError:
            print(f"Missing B_vec for {realization} at time_idx {time_idx}")
            continue

        B_vec = np.array(B_vec)
        biomass = B_vec.sum()
        richness = np.count_nonzero(B_vec)
        norm_biomass = biomass / richness if richness > 0 else 0

        records.append({
            'Diagnosis': 'Healthy' if diagnosis == 'Healthy' else 'Dysbiotic',
            'Total Biomass': biomass,
            'Biomass per Richness': norm_biomass
        })

    df = pd.DataFrame(records)

    # === Setup figure ===
    fig = plt.figure(figsize=(18.4 * 0.393701, 5.5 * 0.393701))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.5, 0.5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    palette = {'Healthy': '#0078B9', 'Dysbiotic': '#EA0017'}

    # === Panel 1: Total Biomass ===
    sns.boxplot(data=df, x='Diagnosis', y='Total Biomass', hue='Diagnosis',
                palette=palette, ax=ax1, legend=False, showfliers=False)
    ax1.set_xlabel('')
    ax1.set_ylabel('Total Biomass', fontsize=10)
    ax1.tick_params(axis='x', labelsize=9)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # === Panel 2: Biomass / Richness ===
    sns.boxplot(data=df, x='Diagnosis', y='Biomass per Richness', hue='Diagnosis',
                palette=palette, ax=ax2, legend=False, showfliers=False)
    ax2.set_xlabel('')
    ax2.set_ylabel('Biomass per Type', fontsize=10)
    ax2.tick_params(axis='x', labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # === Final layout ===
    image_dir = os.path.join(gut_dir, 'images', 'paperplots')
    os.makedirs(image_dir, exist_ok=True)
    image_name = os.path.join(image_dir, 'biomass_vs_richness_2panel.svg')
    plt.subplots_adjust(top=0.94, bottom=0.12, left=0.08, right=0.98, wspace=0.25)
    plt.savefig(image_name, format='svg', dpi=1200)
    plt.show()

#p = float(sys.argv[1])
#healthy_index = int(sys.argv[1])
#IBD_index = int(sys.argv[2])
gut_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

f, f1, f2, f3, f4  = 'fraction_nl', 'fraction_nl', 'exponential', 'exponential', 'linear'
c, c1, c2, c3, c4 = '6', '5', '0.5', '0.6', '1'
m, m1, m2, m3, m4 = '0', '0', '0', '0', '0'
e, e1, e2, e3, e4 = '1.1', '1.1', '1', '1' , '1'
E_D_type = 0
tradeoff_data = f'{f}_{c}_{m}_{e}'
tradeoff_data1 = f'{f1}_{c1}_{m1}_{e1}'
tradeoff_data2 = f'{f2}_{c2}_{m2}_{e2}'
tradeoff_data3 = f'{f3}_{c3}_{m3}_{e3}'
tradeoff_data4 = f'{f4}_{c4}_{m4}_{e4}'
big_realization_number = '688'
small_realization_numbers = ['683'] #['696', '683']#['690', '696', '688']
rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)
rootpath1 = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data1)
rootpath2 = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data2)
rootpath3 = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data3)
rootpath4 = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data4)
rootpath_img = os.path.join(gut_dir, 'images', 'paperplots')
rootpath_changing_params = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_parameters')
params_realizations_dict = {'delta': [os.path.join(rootpath_changing_params, 'delta_0.10416666666666667', '3012'),
                                      os.path.join(rootpath_changing_params, 'delta_0.016666666666666666', '5113')],
                            'h': [os.path.join(rootpath_changing_params, 'h_10', '5334'),
                                  os.path.join(rootpath_changing_params, 'h_0.5', '5303')],
                            'gamma': [os.path.join(rootpath_changing_params, 'gamma_1000000000000000.0', '3633'),
                                      os.path.join(rootpath_changing_params, 'gamma_80000000000000.0', '3743')],
                            'r': [os.path.join(rootpath_changing_params, 'maxgrowth_1e-09', '3114'),
                                  os.path.join(rootpath_changing_params, 'maxgrowth_1e-11', '3127')],
                            'k': [os.path.join(rootpath_changing_params, 'k_0.001', '3218'),
                                  os.path.join(rootpath_changing_params, 'k_1e-05', '3227')],
                            'emax': [os.path.join(rootpath_changing_params, 'emax_4', '5028'),
                                     os.path.join(rootpath_changing_params, 'emax_2', '5019')],
                            'R': [os.path.join(rootpath_changing_params, 'Nsubstances_40', '3828'),
                                     os.path.join(rootpath_changing_params, 'Nsubstances_20', '3812')],
                            'M': [os.path.join(rootpath_changing_params, 'molecularmass_120', '3325'),
                                  os.path.join(rootpath_changing_params, 'molecularmass_80', '3311')],
                            'U': [os.path.join(rootpath_changing_params, 'invasionperiod_4', '3526'),
                                  os.path.join(rootpath_changing_params, 'invasionperiod_1', '3504')],
                            'Ex. th': [os.path.join(rootpath_changing_params, 'extinctionth_0.0005', '5203'),
                                       os.path.join(rootpath_changing_params, 'extinctionth_5e-05', '5226')],        
                            }

#plot_model3(gut_dir, rootpath, rootpath_img, big_realization_number, small_realization_numbers, sample_hours=12*7, 
#            healthy_index=healthy_index, IBD_index=IBD_index)
#plot_model2(rootpath, rootpath_img, big_realization_number, small_realization_numbers, sample_hours=12*7)
#plot_interaction_fig(gut_dir)
#robustness_interaction_fig(gut_dir)
#robustness_taxonomic_profiling_interaction_fig(gut_dir)
#plot_rho_LIMITS(gut_dir, E_D_type, tradeoff_data, avg=True)
#plot_rho_LIMITS(gut_dir, E_D_type, tradeoff_data, avg=False)
#plot_robustness_tradeoff_averaged(gut_dir, rootpath, rootpath2, rootpath3)
#plot_robustness_tradeoff(gut_dir, os.path.join(rootpath1, '617'), os.path.join(rootpath2, '805'), os.path.join(rootpath3, '856'))
#efficiency_plots(gut_dir, os.path.join(rootpath, '688'), os.path.join(rootpath, '683'), os.path.join(rootpath, '699'))
#IBS_CRC_biomarkers(gut_dir)
#disease_indicators(gut_dir)
#beem_static_plots(gut_dir)
#plot_interaction_fig2(gut_dir)
#plot_beta_diversities(gut_dir, percentile = p)
#plot_robustness_parameters(params_realizations_dict, sample_hours=24*7)
#healthy_interaction_robustness(gut_dir)
#diet_interaction_robustness(gut_dir)
#CDI_study_robustness(gut_dir)
#pathway_structure(gut_dir, p=p, sort_by='abundance')
#create_full_pathway_structure_figure(gut_dir, 'Healthy-729-92', 'Unhealthy-729-138', p=p, sort_by="abundance")
#plot_diversity_panel(gut_dir)
#robustness_taxonomic_level_interaction_fig_IBD(gut_dir)
#robustness_taxonomic_level_interaction_fig_IBS(gut_dir)
#plot_biomass_barplot_6_vertical(gut_dir,realization='688',time_indices=[1550,1650,3200,4950,5150],sample_hours=12*7) #2429,4143 5150
#plot_robustness_parameters_extra(gut_dir, sample_size=300, data_type='taxonomy', metric='rho')
#plot_robustness_tradeoff_extra(gut_dir, sample_sizes=[300, 300, 500], data_type='taxonomy', metric='rho')
#realization_paths = [os.path.join(rootpath4, str(f)) for f in [6001, 6005, 6007]]
#plot_linear_realizations(gut_dir, realization_paths= realization_paths, sample_hours=int(24*7))
#plot_biomass_and_normalized_2panel(gut_dir)


#plot_species_and_biomass_ratios(gut_dir, os.path.join(rootpath, '683'), sample_hours=int(12*7))
    
folder_list = [x for x in range(681,749)]
#folder_list = ['683', '688', '690', '696', '699']
folder_list = [str(f) for f in folder_list if f not in [684,685,694]]

#diversity_vs_net_interaction_violin(gut_dir, rootpath, folder_list, sample_hours=int(12*7), relaxing_time=5000*24, num_bins=7)
#diversity_vs_net_interaction(gut_dir, rootpath, folder_list, sample_hours=int(12*7), relaxing_time=5000*24)
#plot_cf_cp_model(gut_dir, folder_list, sample_hours=7*12)
#understanding_time_durations(gut_dir, folder_list, sample_hours=12*7)

#macroecological_patterns(gut_dir, rootpath, folder_list, sample_hours=int(12*7), lower_threshold=1e-4, 
#                         subsample_model=None)
'''
folder_list = [681, 682, 683] + [x for x in range(686,694)] + [x for x in range(695,702)] \
                    + [704,705,706,708,711,712,713,715,716,718,722,724,729,730,731,734,735,741,742,743]
folder_list = [x for x in range(681,749)]
folder_list = [str(f) for f in folder_list if f not in [684,685,694]]
plot_functional_redundancy(rootpath, folder1='688', folder2='696', folder3='700', folder_list=folder_list, sample_hours=12*7)
'''
