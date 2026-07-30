import os
import numpy as np
import json 
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import seaborn as sns
import traceback
import sys
from dataanalyst import *

def change_base_directory(current_path, old_base, new_base):
    # Normalize the path to ensure there are no platform-specific path separators
    normalized_path = os.path.normpath(current_path)
    new_path = normalized_path.replace(old_base, new_base)
    return new_path

def plot_B_and_S(image_dir, t_vec, B_types_dict, S_tvec, survival_thresholds):
    
    width_col, height_col = two_col_fig()
    #fig = plt.figure(figsize = (20*0.3937, height_col/1.2)) #20 cm to inches width
    fig = plt.figure(figsize = (20, 12)) #20 cm to inches width
    gs = fig.add_gridspec(2,3)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])
    ax4 = fig.add_subplot(gs[0,2])
    ax5 = fig.add_subplot(gs[1,2])

    
    # Plot S_a vs t in axis 1 and Relative S abundance in axis 3
    S_df = pd.DataFrame({'t': t_vec[30:]})
    s_cols = []
    n_subs = S_tvec.shape[1]-1
    for s in range(n_subs): # so that last substance is not included
        ax1.plot(t_vec[30:], S_tvec[30:,s])
        S_df[f'S_{s}'] = S_tvec[30:,s]
        #s_cols.append(f'S_{s}')
    #shannon_entropy_S_t = S_df[s_cols].map(lambda x: -x*np.log(x) if x != 0 else 0).sum(axis=1, skipna=True).values
    #ax3.plot(t_vec[30:], shannon_entropy_S_t)
    #ax1.set_ylim(np.min(S_tvec[:,:n_subs-1]), np.max(S_tvec[:,:n_subs-1]))
    excluded_columns = ['t']
    other_cols = S_df.columns.difference(excluded_columns)
    S_df['total_S'] = S_df[other_cols].sum(axis=1, skipna=True) # We sum all columns without including t_vec obviously!
    S_df[other_cols] = S_df[other_cols].div(S_df['total_S'], axis=0) # Remember here other cols does not include 'total_B' because it has been created after
    rel_S_df = S_df.copy()
    rel_S_df = rel_S_df.drop('total_S', axis=1)
    idx = np.round(np.linspace(0, len(t_vec) - 1, 6)).astype(int) # https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last
    #rel_S_df.plot(x='t', kind='bar', stacked=True, ax=ax3, legend=False)#.set_xticks(t_vec[idx])

    # Plot B_i vs t in axis 0 and Relative B abundance in axis 2
    B_df = pd.DataFrame({'t': t_vec})
    if len(t_vec) != len(set(t_vec)):
        print("The t_vec has values repeated. This would cause problems in the merging of the dataframes right after in the code. Solve that before proceeding")
        raise
    for Type, (t_init, t_end, type_functional_identity, B_type_t) in B_types_dict.items():
        wanted_t = t_vec[(t_vec>=t_init) & (t_vec<t_end)]
        ax0.plot(wanted_t,B_type_t,label=Type)
        ax2.plot(wanted_t,B_type_t,label=Type)
        aux_df = pd.DataFrame({'t': wanted_t, f'B_{Type}': B_type_t})
        B_df = B_df.merge(aux_df, how='left', on='t')
            
    
    excluded_columns = ['t']
    other_cols = B_df.columns.difference(excluded_columns)
    N_types_t = B_df[other_cols].count(axis=1).values # COunts the number of not nan values in each row, excluding the column t See https://stackoverflow.com/questions/29971075/count-number-of-non-nan-entries-in-every-column-of-dataframe

    B_df['total_B'] = B_df[other_cols].sum(axis=1, skipna=True) # We sum all columns without including t_vec obviously!
    B_df[other_cols] = B_df[other_cols].div(B_df['total_B'], axis=0) # Remember here other cols does not include 'total_B' because it has been created after
    rel_B_df = B_df.copy()
    rel_B_df = rel_B_df.fillna(0)
    total_B_tvec = B_df['total_B'].values
    rel_B_df = rel_B_df.drop('total_B', axis=1)
    shannon_entropy_t = rel_B_df[other_cols].map(lambda x: -x*np.log(x) if x != 0 else 0).sum(axis=1, skipna=True).values
    ax5.plot(t_vec, shannon_entropy_t, label='Sh_0')
    '''
    for s_t in survival_thresholds:
        survival_shannon_tvec, t_res_shannon_tvec = calculate_survival_shannon_index(B_types_dict, t_vec, s_t)
        ax5.plot(t_vec, survival_shannon_tvec, label=f'Sh_{s_t}')
    
    axtwin5 = ax5.twinx()
    axtwin5.plot(t_vec, t_res_shannon_tvec, label=f'Sh_tres')
    axtwin5.set_ylabel('Shannn Index t res')
    '''
    ax3.plot(t_vec, total_B_tvec)
    #rel_B_df.plot(x='t', kind='bar', stacked=True, ax=ax2, legend=False)#.set_xticks(t_vec[idx])
    
    ax4.plot(t_vec, N_types_t)

    # Creating a twin of the primary axis to plot the second set of data
    '''axtwin1 = ax1.twinx()
    axtwin1.plot(t_vec[30:], N_types_t[30:])  # 'b-' sets the color and line style
    axtwin1.set_ylabel('nº types')
    '''
    
    ax1.set_xlim([t_vec[30],t_vec[-1]])
    ax0.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax0.set_ylabel(r'Type Biomass [Cells]', labelpad=1, fontsize=11)
    ax1.set_xlabel(r't [h]', labelpad=1, fontsize=11)
    ax1.set_ylabel(r'Resource Concentration [g/L]', labelpad=1, fontsize=11)
    ax2.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax2.set_ylabel(r'Type Biomass [Cells]', labelpad=1, fontsize=11)
    ax2.set_yscale('log')
    #ax1.set_yscale('log')
    ax3.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax3.set_ylabel(r'Total Biomass [Cells]', labelpad=1, fontsize=11)
    ax4.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax4.set_ylabel(r'nº types', labelpad=1, fontsize=11)
    ax5.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax5.set_ylabel(r'Shannon Index', labelpad=1, fontsize=11)
    n_ticks = 8
    '''
    for ax in [ax0,ax1,ax2,ax3,ax4,ax5]:
        xticks = ax2.xaxis.get_ticklocs()
        xticklabels = ax2.xaxis.get_ticklabels()
        idx = np.round(np.linspace(0, len(xticks) - 1, n_ticks)).astype(int)
        ax.xaxis.set_ticks(np.array(xticks)[idx])
        ax.xaxis.set_ticklabels(np.array([int(float(t.get_text())/24) for t in xticklabels])[idx], rotation=0)
    '''
    #ax2.set_ylim([0,1])
    
    ax0.set_xticks(np.linspace(t_vec[0], t_vec[-1], n_ticks))
    ax0.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    ax1.set_xticks(np.linspace(t_vec[30], t_vec[-1], n_ticks))
    ax1.set_xticklabels([int(t/24) for t in np.linspace(2100*24, t_vec[-1], n_ticks)])
    ax2.set_xticks(np.linspace(t_vec[0], t_vec[-1], n_ticks))
    ax2.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    ax3.set_xticks(np.linspace(t_vec[0], t_vec[-1], n_ticks))
    ax3.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    ax4.set_xticks(np.linspace(0, t_vec[-1], n_ticks))
    ax4.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    ax5.set_xticks(np.linspace(0, t_vec[-1], n_ticks))
    ax5.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    
    '''
    xticks = ax2.xaxis.get_ticklocs()
    xticklabels = ax2.xaxis.get_ticklabels()
    idx = np.round(np.linspace(0, len(xticks) - 1, n_ticks)).astype(int)
    ax2.xaxis.set_ticks(np.array(xticks)[idx])
    ax2.xaxis.set_ticklabels(np.array([int(float(t.get_text())/24) for t in xticklabels])[idx], rotation=0)
    ax3.xaxis.set_ticks(np.array(xticks)[idx])
    ax3.xaxis.set_ticklabels(np.array([int(float(t.get_text())/24) for t in xticklabels])[idx], rotation=0)
    '''
    

    #image_dir = os.path.join(gut_dir, 'images', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', f'{tradeoff_data}', f'{folder_number}')
    #if not os.path.exists(image_dir):
    #    os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'B_and_S_BCN.png')
    plt.subplots_adjust(top=0.96, bottom=0.07, left=0.06, right=0.96, hspace=0.16, wspace=0.25)
    plt.savefig(image_name, format='png', transparent=False, dpi=600)
    #plt.show()

def subtances_plots(image_dir, t_vec, S_tvec, E_avec):
    def compute_substance_metrics(S_tvec, E_avec, window=10):
        """
        Compute metrics for substance concentrations over time and their rolling averages.

        Parameters:
        - S_tvec: List of arrays or 2D numpy array where each row represents a temporal instance, 
                and each column represents a substance concentration.
        - E_avec: 1D numpy array representing the energy associated with each substance.
        - window: Window size for calculating the rolling average.

        Returns:
        - metrics: Dictionary with the original metrics and their rolling averages.
        """
        S_tvec = np.array(S_tvec)  # Ensure S_tvec is a NumPy array for efficient processing
        
        '''
        min_concentration = np.min(S_tvec[S_tvec > 0])

        # Identify concentrations below the threshold (10^(-14)) but above 0 and their time indices
        threshold = 1e-14
        non_zero_indices = np.where((S_tvec < threshold) & (S_tvec > 0))

        # Extract the corresponding concentrations and time indices
        below_threshold_concentrations = S_tvec[non_zero_indices]
        below_threshold_times = non_zero_indices[0]  # Time indices

        # Prepare the results
        result = {
            "minimum_concentration": min_concentration,
            "below_threshold_concentrations": below_threshold_concentrations.tolist(),
            "below_threshold_times": below_threshold_times.tolist(),
        }

        print(result)
        raise
        '''
        # Exclude the last substance for computations where needed
        S_excluding_last = S_tvec[:, :-1]

        # 1. Compute the number of non-zero concentration substances for each time
        num_non_zero = np.count_nonzero(S_excluding_last, axis=1)

        # 2. Compute the Shannon index
        row_sums = S_excluding_last.sum(axis=1, keepdims=True)  # Normalize concentrations row-wise
        relative_abundances = np.divide(
            S_excluding_last, 
            row_sums, 
            out=np.zeros_like(S_excluding_last), 
            where=row_sums != 0
        )
        shannon_indices = -np.sum(relative_abundances * np.log(relative_abundances + 1e-10), axis=1)

        # 3. Compute the total concentration of available substances (ignoring the last substance)
        total_concentration = np.log(S_excluding_last.sum(axis=1))

        # 4. Compute the total energy available for each time
        energy_available = np.log(np.sum(S_tvec * E_avec, axis=1))

        # Convert metrics to Pandas Series for rolling average
        metrics = {
            'N_substances_tvec': pd.Series(num_non_zero),
            'shannon_substances_tvec': pd.Series(shannon_indices),
            'log_total_concentration_tvec': pd.Series(total_concentration),
            'log_energy_available_tvec': pd.Series(energy_available),
        }
        # Compute rolling averages for each metric
        rolling_metrics = {key: value.rolling(window=window, min_periods=1).mean() for key, value in metrics.items()}

        return metrics, rolling_metrics
    fig = plt.figure(figsize = (20, 12)) #20 cm to inches width
    gs = fig.add_gridspec(2,2)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])
    
    metrics, rolling_metrics = compute_substance_metrics(S_tvec, E_avec, window=100)

    ax0.plot(t_vec/24, metrics['N_substances_tvec'])
    ax1.plot(t_vec/24, metrics['shannon_substances_tvec'])
    ax2.plot(t_vec[100:]/24, metrics['log_total_concentration_tvec'][100:])
    ax3.plot(t_vec[100:]/24, metrics['log_energy_available_tvec'][100:])
    #ax0.plot(t_vec/24, [int(x) for x in rolling_metrics['N_substances_tvec']])
    #ax1.plot(t_vec/24, rolling_metrics['shannon_substances_tvec'])
    #ax2.plot(t_vec[100:]/24, rolling_metrics['log_total_concentration_tvec'][100:])
    #ax3.plot(t_vec[100:]/24, rolling_metrics['log_energy_available_tvec'][100:])

    x_labels = ['time [years]']*4
    y_labels = ['Nº Substances', 'Shannon Substances', 'Log Total Concentration', 'Log Available Energy']
    axes = [ax0,ax1,ax2,ax3]
    for xl,yl,ax in zip(x_labels, y_labels, axes):
        ax.set_xlabel(xl, labelpad=1, fontsize=11)
        ax.set_ylabel(yl, labelpad=1, fontsize=11)
    
    #image_dir = os.path.join(gut_dir, 'images', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', f'{tradeoff_data}', f'{folder_number}')
    #if not os.path.exists(image_dir):
    #    os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'S_plots.png')
    plt.subplots_adjust(top=0.96, bottom=0.07, left=0.06, right=0.96, hspace=0.16, wspace=0.25)
    plt.savefig(image_name, format='png', transparent=False, dpi=600)


def functions_plots(image_dir, t_vec, N_pathways_tvec, N_enzymes_tvec, shannon_tvec, rel_D_B_tvec, rel_D_tvec, N_types_tvec, pathways_strs, 
                    pathways_ids, pathways_id_strs): #, E_values_per_level, D_max, D_width):
    '''
    - gut_dir: the directory where GutMicrobiome folder is
    - N_pathways_tvec, N_enzymes_tvec, rel_D_B_tvec, rel_D_tvec: see get_functions function for the info
    - pathways_strs: "i->j" for all the valid pathways (those for which D_mat=D_mat[~np.isnan(D_mat)]) where i are rows and j are columns
    - pathways_ids: the unique id's for the pathways
    - pathways_id_strs: the str of those id's which is "Elevel_i->Elevel_j" where i are rows and j are columns
    - folder_number, tradeoff_data, E_D_type: needed to identify the specific image folder where we want the result to go
    '''

    fig = plt.figure(figsize = (10, 6.8)) #20 cm to inches width
    gs = fig.add_gridspec(2,2)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])
 
    
    aux_dict_1 = {'t': t_vec}
    aux_dict_2 = {'t': t_vec}
    aux_dict_3 = {'t': t_vec}
    aux_dict_4 = {'t': t_vec}
    aux_dict_5 = {'t': t_vec}
    

    p_categories_to_str_dict, p_str_to_category_dict, category_to_label = get_functions_categories()#E_values_per_level, D_max, D_width, N_substances_entering=1)
    
    #grouped_pathways_id_to_idstr_dict = {}
    for i, (p_str, p_id, p_id_str) in enumerate(zip(pathways_strs, pathways_ids, pathways_id_strs)):
        # .copy() was needed fro aux_dict_1 at least because since i was using same rel_D_tvec after in aux_dict_3 and summing, I was getting weird values for rel_D_per_pathway_df later (basically not summing up)
        aux_dict_1.update({p_str: rel_D_tvec[:,i].copy()})
        aux_dict_2.update({p_str: rel_D_B_tvec[:,i].copy()})
        p_category = p_str_to_category_dict[p_str][0]
        if p_id not in aux_dict_3.keys():
            aux_dict_3[p_str] = rel_D_tvec[:,i].copy()
            aux_dict_4[p_str] = rel_D_B_tvec[:,i].copy()
            #grouped_pathways_id_to_idstr_dict[p_id] = p_id_str
        else:
            aux_dict_3[p_str] += rel_D_tvec[:,i].copy()
            aux_dict_4[p_str] += rel_D_B_tvec[:,i].copy()
        if p_category not in aux_dict_5.keys():
            aux_dict_5[p_category] = rel_D_B_tvec[:,i].copy()
        else:
            aux_dict_5[p_category] += rel_D_B_tvec[:,i].copy()

        

    rel_D_per_pathway_df = pd.DataFrame(aux_dict_1)
    rel_D_B_per_pathway_df = pd.DataFrame(aux_dict_2)
    rel_D_per_real_pathway_df = pd.DataFrame(aux_dict_3)
    rel_D_B_per_real_pathway_df = pd.DataFrame(aux_dict_4)
    rel_D_B_per_category_df = pd.DataFrame(aux_dict_5)
    N_real_pathways_tvec = np.count_nonzero(rel_D_per_real_pathway_df, axis=1) # we count the number of real functions that we have for each time
    
    categories_colors = [[r,g,b] for r,g,b,a in plt.cm.tab10(np.linspace(0,1,len(p_categories_to_str_dict)))] #plt.cm.Set3
    categories_color_dict = dict(zip(p_categories_to_str_dict.keys(), categories_colors))
    '''
    total_pathways_color_dict = {}
    for p_category, p_strs in p_categories_to_str_dict.items():
        n_colors = len(p_strs)
        base_color = categories_color_dict[p_category]
        aux_cmap = get_unicolormap(base_color, n_colors, min_l=0.2, max_l=0.8)
        total_pathways_color_dict.update({p_str: c for p_str, c in dict(zip(p_strs, aux_cmap)).items()})
    
    
    grouped_pathways_ids = [k for k in aux_dict_3.keys() if k!='t'] # is like np.unique(pathways_ids) PERO así preservamos el orden del database (aunque eso da igual, porque los colores da igual por cuál empecemos)
    
    #grouped_pathways_colors = get_qualitatively_different_colors(len(groped_pathways_ids), shuffle_state=2, l_default=0.4, init_h=0.05)
    # FORTUNATELY ENOUGH WE HAVE 10 DIFFERENT GROUPED PATHWAYS SO WE WILL USE TAB10
    grouped_pathways_colors = [[r,g,b] for r,g,b,a in plt.cm.Set3(np.linspace(0,1,len(grouped_pathways_ids)))]
    grouped_pathways_color_dict = dict(zip(grouped_pathways_ids, grouped_pathways_colors))
    pathway_str_to_id = dict(zip(pathways_strs, pathways_ids)) # different str have same id
    pathway_id_to_strs = get_inverse_dict(pathway_str_to_id) # since different str have same id. the values here are list of the different str

    total_pathways_color_dict = {}
    for p_id, p_strs in pathway_id_to_strs.items():
        n_colors = len(p_strs)
        base_color = grouped_pathways_color_dict[p_id]
        aux_cmap = get_unicolormap(base_color, n_colors, min_l=0.3, max_l=0.8)
        total_pathways_color_dict.update({p_str: c for p_str, c in dict(zip(p_strs, aux_cmap)).items()})
    '''
    #idx = np.round(np.linspace(0, len(t_vec) - 1, 6)).astype(int) # https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last
    

    #N_path_per_type_tvec = N_pathways_tvec/N_types_tvec
    #N_enz_per_type_tvec = N_enzymes_tvec/N_types_tvec
    N_path_per_shannon_tvec = N_pathways_tvec/shannon_tvec
    N_enz_per_shannon_tvec = N_enzymes_tvec/shannon_tvec
    #ax0.plot(t_vec, N_pathways_tvec, label='All Pathways')
    #ax2.plot(t_vec, N_path_per_shannon_tvec, label='Functional Groups')
    #ax1.plot(t_vec, N_enzymes_tvec, label='All Enzymes')
    #ax3.plot(t_vec, N_enz_per_shannon_tvec, label='Functional Groups')
    #ax0.legend(loc='upper right')
    #rel_D_per_real_pathway_df = rel_D_per_real_pathway_df.rename(columns=grouped_pathways_id_to_idstr_dict) # we rename the columns to know what type of pathway the ids represent
    #rel_D_B_per_real_pathway_df.plot(x='t', kind='bar', stacked=True, ax=ax2, legend=False, color=total_pathways_color_dict)
    rel_D_B_per_category_df.plot(x='t', kind='bar', stacked=True, ax=ax1, legend=False, color=categories_color_dict)
    #rel_D_per_pathway_df.plot(x='t', kind='bar', stacked=True, ax=ax2, legend=False, color=total_pathways_color_dict)
    #rel_D_per_real_pathway_df.plot(x='t', kind='bar', stacked=True, ax=ax3, legend=True, color=grouped_pathways_color_dict)
    
    ax0.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax0.set_ylabel(r'nº pathways', labelpad=1, fontsize=11)
    ax1.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    #ax1.set_ylabel(r'Relative weighted difficulty for category', labelpad=1, fontsize=11)
    ax1.set_ylabel(r'Enzymatic cost', labelpad=1, fontsize=11)
    ax2.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax2.set_ylabel(r'nº pathways per shannon', labelpad=1, fontsize=11)
    ax3.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax3.set_ylabel(r'Enzymatic cost per shannon', labelpad=1, fontsize=11)
    #ax3.set_ylabel(r'Relative difficulty for function', labelpad=1, fontsize=11)
    
    #ax2.set_ylim([0,N_path_per_shannon_tvec[-1]+1])
    #ax3.set_ylim([0,N_enz_per_shannon_tvec[-1]+2])

    
    #handles, previous_labels = ax1.get_legend_handles_labels() # last comment in https://stackoverflow.com/questions/23037548/change-main-plot-legend-label-text
    #new_labels = [category_to_label[category] for category in previous_labels]
    #ax1.legend(labels=new_labels, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    

    n_ticks = 8
    
    ax0.set_xticks(np.linspace(t_vec[0], t_vec[-1], n_ticks))
    ax0.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    ax1.set_xticks(np.linspace(t_vec[0], t_vec[-1], n_ticks))
    ax1.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    ax2.set_xticks(np.linspace(t_vec[0], t_vec[-1], n_ticks))
    ax2.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    ax3.set_xticks(np.linspace(t_vec[0], t_vec[-1], n_ticks))
    ax3.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    
    #xticks = ax1.xaxis.get_ticklocs()
    #xticklabels = ax1.xaxis.get_ticklabels()
    #idx = np.round(np.linspace(0, len(xticks) - 1, n_ticks)).astype(int)
    #ax1.xaxis.set_ticks(np.array(xticks)[idx])
    #ax1.xaxis.set_ticklabels(np.array([int(float(t.get_text())/24) for t in xticklabels])[idx], rotation=0)
    #ax2.xaxis.set_ticks(np.array(xticks)[idx])
    #ax2.xaxis.set_ticklabels(np.array([int(float(t.get_text())/24) for t in xticklabels])[idx], rotation=0)
    #ax3.xaxis.set_ticks(np.array(xticks)[idx])
    #ax3.xaxis.set_ticklabels(np.array([int(float(t.get_text())/24) for t in xticklabels])[idx], rotation=0)
    #ax3.set_xticklabels([int(float(t.get_text())//24) for t in ax3.get_xticklabels()[::5]], rotation=0)
    # get_text() needed because the get_xticklabels gives a 'Text' type object
    # How to choose xticks and xticklabels: https://stackoverflow.com/questions/51971491/plot-bar-chart-number-of-ticks-on-xaxis
    
    #image_dir = os.path.join(gut_dir, 'images', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', f'{tradeoff_data}', f'{folder_number}')
    #if not os.path.exists(image_dir):
    #    os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'functions_plot_redundancy.png')
    plt.subplots_adjust(top=0.99, bottom=0.07, left=0.06, right=0.85, hspace=0.16, wspace=0.25)
    plt.savefig(image_name, format='png', transparent=False, dpi=600)
    #plt.show()

def D_data_plots(image_dir, t_vec, D_data_dict):
    '''
    D_data_dict = {'Dtotal_tvec': Dtotal_tvec, 'Dtotal_Bmasked_tvec': Dtotal_Bmasked_tvec, 
                    'Dperpway_tvec': Dperpway_tvec, 'Dperpway_Bmasked_tvec': Dperpway_Bmasked_tvec,
                    'Dmaxpway_tvec': Dmaxpway_tvec, 'Dmaxpway_Bmasked_tvec': Dmaxpway_Bmasked_tvec,
                    'Npwaytype_tvec': Npwaytype_tvec, 'Npwaytype_Bmasked_tvec': Npwaytype_Bmasked_tvec}
    D_total_tvec: temporal vector with the difficulties (enzymatic cost) for each type. So dim=(t,types_at_t) (dynamic dim)
    Dperpway_tvec: same but with the average difficulty per pathway
    Dmaxpway_tvec: same with difficulty of the most costly pathway 
    Npwaytype_tvec: same but with number of pathways instead of difficulties. Thus, gives number of pathways for each type
    The B_masked are the same but taking into account the types that have a re_B>2*10^(-4) basically to avoid counting bacteria that dissapear.
    '''

    fig = plt.figure(figsize = (10, 6.8)) #20 cm to inches width
    gs = fig.add_gridspec(2,4)
    ax00 = fig.add_subplot(gs[0,0])
    ax01 = fig.add_subplot(gs[0,1])
    ax02 = fig.add_subplot(gs[0,2])
    ax03 = fig.add_subplot(gs[0,3])
    ax10 = fig.add_subplot(gs[1,0])
    ax11 = fig.add_subplot(gs[1,1])
    ax12 = fig.add_subplot(gs[1,2])
    ax13 = fig.add_subplot(gs[1,3])

    Dtotal_tvec = D_data_dict['Dtotal_tvec']
    Dtotal_Bmasked_tvec = D_data_dict['Dtotal_Bmasked_tvec']
    Dmaxpway_tvec = D_data_dict['Dmaxpway_tvec']
    Dmaxpway_Bmasked_tvec = D_data_dict['Dmaxpway_Bmasked_tvec']
    Dperpway_tvec = D_data_dict['Dperpway_tvec']
    Dperpway_Bmasked_tvec = D_data_dict['Dperpway_Bmasked_tvec']
    Npwaytype_tvec = D_data_dict['Npwaytype_tvec']
    Npwaytype_Bmasked_tvec = D_data_dict['Npwaytype_Bmasked_tvec']
    
    Dtotal_df = pd.DataFrame({'Dtotal': np.concatenate(Dtotal_tvec).ravel()}) # See here for how to flatten a list of numpy arrays https://stackoverflow.com/questions/33711985/flattening-a-list-of-numpy-arrays
    Dperpway_df = pd.DataFrame({'Dperpway': np.concatenate(Dperpway_tvec).ravel()}) 
    Dmaxpway_df = pd.DataFrame({'Dmaxpway': np.concatenate(Dmaxpway_tvec).ravel()}) 
    Npwaytype_df = pd.DataFrame({'Npwaytype': np.concatenate(Npwaytype_tvec).ravel()}) 

    Dtotal_Bmasked_df = pd.DataFrame({'Dtotal_Bmasked': np.concatenate(Dtotal_Bmasked_tvec).ravel()}) # See here for how to flatten a list of numpy arrays https://stackoverflow.com/questions/33711985/flattening-a-list-of-numpy-arrays
    Dperpway_Bmasked_df = pd.DataFrame({'Dperpway_Bmasked': np.concatenate(Dperpway_Bmasked_tvec).ravel()}) 
    Dmaxpway_Bmasked_df = pd.DataFrame({'Dmaxpway_Bmasked': np.concatenate(Dmaxpway_Bmasked_tvec).ravel()}) 
    Npwaytype_Bmasked_df = pd.DataFrame({'Npwaytype_Bmasked': np.concatenate(Npwaytype_Bmasked_tvec).ravel()}) 
    
    sns.histplot(Dtotal_df, x='Dtotal', binwidth=0.1, ax = ax00, stat='probability')
    sns.histplot(Dperpway_df, x='Dperpway', binwidth=0.1, ax = ax01, stat='probability')
    sns.histplot(Dmaxpway_df, x='Dmaxpway', binwidth=0.1, ax = ax02, stat='probability')
    sns.histplot(Npwaytype_df, x='Npwaytype', binwidth=0.1, ax = ax03, stat='probability')

    sns.histplot(Dtotal_Bmasked_df, x='Dtotal_Bmasked', binwidth=0.1, ax = ax10, stat='probability')
    sns.histplot(Dperpway_Bmasked_df, x='Dperpway_Bmasked', binwidth=0.1, ax = ax11, stat='probability')
    sns.histplot(Dmaxpway_Bmasked_df, x='Dmaxpway_Bmasked', binwidth=0.1, ax = ax12, stat='probability')
    sns.histplot(Npwaytype_Bmasked_df, x='Npwaytype_Bmasked', binwidth=0.1, ax = ax13, stat='probability')

    '''
    ax00.plot(t_vec, Dtotal_avg_tvec, label='normal')
    ax10.plot(t_vec, Dtotal_Bweighted_avg_tvec, label='B weighted')
    ax01.plot(t_vec, Dtotal_std_tvec, label='normal')
    ax11.plot(t_vec, Dtotal_Bweighted_std_tvec, label='B weighted')
    ax02.plot(t_vec, Dperpway_avg_tvec, label='normal')
    ax12.plot(t_vec, Dperpway_Bweighted_avg_tvec, label='B weighted', zorder=2)
    ax03.plot(t_vec, Dperpway_std_tvec, label='normal')
    ax13.plot(t_vec, Dperpway_Bweighted_std_tvec, label='B weighted', zorder=2)
    #print(f'Average std of max D non weighted {np.mean(Dperpway_std_tvec)}')
    #print(f'Average std of max D weighted {np.mean(Dperpway_Bweighted_std_tvec)}')
    #print(Dperpway_Bweighted_avg_tvec)
    #print(Dperpway_std_tvec)
    #print(Dperpway_Bweighted_std_tvec)
    #print(f'Equal weighted non weighted: {np.array_equal(Dperpway_Bweighted_std_tvec,Dperpway_std_tvec)}')
    #raise  
    
    
    
    ax0.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax0.set_ylabel(r'Avg Total Enzimatic Cost', labelpad=1, fontsize=11)
    ax2.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax2.set_ylabel(r'Avg Enzimatic Cost per Pathway', labelpad=1, fontsize=11)
    ax1.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax1.set_ylabel(r'Std Total Enzimatic Cost', labelpad=1, fontsize=11)
    ax3.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax3.set_ylabel(r'Std Enzimatic Cost per Pathway', labelpad=1, fontsize=11)

    #n_ticks = 6
    #ax4.set_xticks(np.round(np.linspace(Dtotal_df.min(), Dtotal_df.max(), n_ticks), 1))
    #ax5.set_xticks(np.round(np.linspace(Dperpway_df.min(), Dperpway_df.max(), n_ticks),1))
    '''
    '''
    n_ticks = 8
    ax0.set_xticks(np.linspace(0, t_vec[-1], n_ticks))
    ax0.set_xticklabels([int(t/24) for t in np.linspace(0, t_vec[-1], n_ticks)])

    xticks = ax1.xaxis.get_ticklocs()
    xticklabels = ax1.xaxis.get_ticklabels()
    idx = np.round(np.linspace(0, len(xticks) - 1, n_ticks)).astype(int)
    ax1.xaxis.set_ticks(np.array(xticks)[idx])
    ax1.xaxis.set_ticklabels(np.array([int(float(t.get_text())/24) for t in xticklabels])[idx], rotation=0)
    ax2.xaxis.set_ticks(np.array(xticks)[idx])
    ax2.xaxis.set_ticklabels(np.array([int(float(t.get_text())/24) for t in xticklabels])[idx], rotation=0)
    ax3.xaxis.set_ticks(np.array(xticks)[idx])
    ax3.xaxis.set_ticklabels(np.array([int(float(t.get_text())/24) for t in xticklabels])[idx], rotation=0)
    # get_text() needed because the get_xticklabels gives a 'Text' type object
    # How to choose xticks and xticklabels: https://stackoverflow.com/questions/51971491/plot-bar-chart-number-of-ticks-on-xaxis
    '''
    #image_dir = os.path.join(gut_dir, 'images', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', f'{tradeoff_data}', f'{folder_number}')
    #if not os.path.exists(image_dir):
    #    os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'D_data_plot_new.png')
    plt.subplots_adjust(top=0.99, bottom=0.07, left=0.06, right=0.96, hspace=0.16, wspace=0.25)
    plt.savefig(image_name, format='png', transparent=False, dpi=600)
    #plt.show()


def crossfeeding_competition_plots(image_dir, t_vec, B_tvec, crossfeeding_ijmat_tvec, competition_ijmat_tvec):

    width_col, height_col = two_col_fig()
    #fig = plt.figure(figsize = (20*0.3937, height_col/1.2)) #20 cm to inches width
    fig = plt.figure(figsize = (15, 10)) #20 cm to inches width
    gs = fig.add_gridspec(2,3)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])
    ax4 = fig.add_subplot(gs[0,2])
    ax5 = fig.add_subplot(gs[1,2])
    t_vec = t_vec[10:]
    crossfeeding_ijmat_tvec = crossfeeding_ijmat_tvec[10:] 
    competition_ijmat_tvec = competition_ijmat_tvec[10:]
    #ax0.plot(t_vec, [np.sum(cf) for cf in crossfeeding_ijmat_tvec])
    #ax1.plot(t_vec, [(np.sum(cp)-np.trace(cp)) for cp in competition_ijmat_tvec])
    #ax2.plot(t_vec, [np.sum(cf)/np.sum(B_ivec) for B_ivec,cf in zip(B_tvec, crossfeeding_ijmat_tvec)])
    #ax3.plot(t_vec, [(np.sum(cp)-np.trace(cp))/np.sum(B_ivec) for B_ivec,cp in zip(B_tvec,competition_ijmat_tvec)])
    #ax4.plot(t_vec, [np.sum(cf)-(np.sum(cp)-np.trace(cp)) for cf,cp in zip(crossfeeding_ijmat_tvec,competition_ijmat_tvec)])
    #ax5.plot(t_vec, [(np.sum(cf)-(np.sum(cp)-np.trace(cp)))/np.sum(B_ivec) for B_ivec,cf,cp in zip(B_tvec,crossfeeding_ijmat_tvec,competition_ijmat_tvec)])
    
    rho_tvec = []
    rho_new_tvec = []
    R_new_tvec = []
    R_tvec = []
    for cf, cp in zip(crossfeeding_ijmat_tvec,competition_ijmat_tvec):
        rho_tvec.append((np.sum(cf)-(np.sum(cp)-np.trace(cp)))/(np.sum(cf)+(np.sum(cp)-np.trace(cp))))
        rho_new_tvec.append((np.sum(cf)-np.trace(cf)-(np.sum(cp)-np.trace(cp)))/(np.sum(cf)-np.trace(cf)+(np.sum(cp)-np.trace(cp))))
        R_new_tvec.append(np.sum(cf)-np.trace(cf)-(np.sum(cp)-np.trace(cp))) 
        R_tvec.append(np.sum(cf)-(np.sum(cp)-np.trace(cp)))

    ax2.plot(t_vec, R_tvec)
    ax3.plot(t_vec, R_new_tvec)
    ax4.plot(t_vec, rho_tvec)
    ax5.plot(t_vec, rho_new_tvec)
    '''
    mean_competition_tvec = []
    for x in competition_ijmat_tvec:
        np.fill_diagonal(x, np.nan)
        print(x)
        mean_competition_tvec.append(np.nanmean(x))
    ax3.plot(t_vec, mean_competition_tvec)
    '''

    ax0.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax0.set_ylabel(r'Total Crossfeeding', labelpad=1, fontsize=11)
    ax1.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax1.set_ylabel(r'Total Competition', labelpad=1, fontsize=11)
    ax2.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax2.set_ylabel(r'CF - CP', labelpad=1, fontsize=11)
    ax3.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax3.set_ylabel(r'CF - CP new', labelpad=1, fontsize=11)
    ax4.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax4.set_ylabel(r'Rho', labelpad=1, fontsize=11)
    ax5.set_xlabel(r't [days]', labelpad=1, fontsize=11)
    ax5.set_ylabel(r'Rho new', labelpad=1, fontsize=11)
    
    n_ticks = 8
    for ax in [ax0,ax1,ax2,ax3,ax4,ax5]:
        ax.set_xticks(np.linspace(t_vec[0], t_vec[-1], n_ticks))
        ax.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    '''
    ax1.set_xticks(np.linspace(t_vec[0], t_vec[-1], n_ticks))
    ax1.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    ax2.set_xticks(np.linspace(t_vec[0], t_vec[-1], n_ticks))
    ax2.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    ax3.set_xticks(np.linspace(t_vec[0], t_vec[-1], n_ticks))
    ax3.set_xticklabels([int(t/24) for t in np.linspace(t_vec[0], t_vec[-1], n_ticks)])
    '''

    #image_dir = os.path.join(gut_dir, 'images', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', f'{tradeoff_data}', f'{folder_number}')
    #if not os.path.exists(image_dir):
    #    os.makedirs(image_dir)
    image_name = os.path.join(image_dir, f'cf_cp_new.png')
    plt.subplots_adjust(top=0.96, bottom=0.05, left=0.04, right=0.98, hspace=0.16, wspace=0.19)
    plt.savefig(image_name, format='png', transparent=False, dpi=600)
    #plt.show()



def plot_merged_boxplots(folder_path, subset_fraction="0.8", value_index=3, figsize=(12, 6)):
    all_data = []

    for filename in os.listdir(folder_path):
        print(filename)
        if filename.endswith(".json") and f"_stats_boostrap_{subset_fraction}_" in filename:
            diagnosis = filename.split("_stats_boostrap")[0]
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as f:
                data = json.load(f)
            vectors = data.get(str(subset_fraction), [])
            for vec in vectors:
                if isinstance(vec, list) and len(vec) > value_index:
                    all_data.append({
                        "diagnosis": diagnosis,
                        "value": vec[value_index]
                    })

    df = pd.DataFrame(all_data)
    # Plot
    plt.figure(figsize=figsize)
    #sns.violinplot(data=df, x="diagnosis", y="value", inner="box")
    sns.boxplot(data=df, x="diagnosis", y="value", showfliers=False)
    plt.title(f"Distribution of Value #{value_index+1} Across Diagnoses (subset {subset_fraction})")
    plt.ylabel(f"Metric at Index {value_index}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main_plots(path, sample_hours):#rootpath, folder_number, sample_hours):
    #path = os.path.join(rootpath, f'{folder_number}')
    print(path)
    image_dir = change_base_directory(path, 'results', 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    else:
        return
    t_vec, B_tvec, S_tvec, D_tvec = get_tvecs_together(path, sample_hours, overwrite=False)
    
    types_identity_dict = get_types_identity_dict(path)
    D_mat = get_D_mat(path)
    E_avec = get_E_avec(path)
    max_growth_data, k_data = get_monod_data(path)
    pathways_ids, pathways_id_strs = get_pathways_id(D_mat, E_avec)
    #pathways_strs = get_pathways_strs(D_mat)
    #pathways_id_strs = pathways_strs because we have 1 substance per energy level!
    
    #pathway_id_to_category_dict = get_pathway_id_to_category_dict(pathways_id_strs, pathways_ids)
    B_types_dict = get_B_types_dict(path, t_vec, B_tvec, types_identity_dict, D_tvec, D_mat, pathways_ids, sample_hours, overwrite=False)

    N_pathways_tvec, N_enzymes_tvec, shannon_tvec, rel_D_B_tvec, rel_D_tvec, D_data_dict, N_types_tvec = get_functions_data(path, D_tvec, D_mat, B_tvec, sample_hours, overwrite=True)
    crossfeeding_ijmat_tvec, competition_ijmat_tvec = get_crossfeeding_and_competition_matrices(path, t_vec, B_tvec, S_tvec, D_tvec, D_mat, 
                                                                                                max_growth_data, k_data, sample_hours, overwrite=False)
    #D_max = get_D_max(tradeoff_data)
    #E_diff = np.abs(E_avec[0]-E_avec[1])
    #D_width = 2.5*E_diff
    #coarsed_functional_identities_dict = get_coarsed_functional_identities_dict(t_vec, B_types_dict, pathway_id_to_category_dict, overwrite=False)
    
    
    
    subtances_plots(image_dir, t_vec, S_tvec, E_avec)
    plot_B_and_S(image_dir, t_vec, B_types_dict, S_tvec, survival_thresholds=[0])
    #functions_plots(image_dir, t_vec, N_pathways_tvec, N_enzymes_tvec, shannon_tvec, rel_D_B_tvec, rel_D_tvec, N_types_tvec, pathways_strs, 
    #                pathways_ids, pathways_id_strs) #E_avec, D_max, D_width)
    D_data_plots(image_dir, t_vec, D_data_dict)
    crossfeeding_competition_plots(image_dir, t_vec, B_tvec, crossfeeding_ijmat_tvec, competition_ijmat_tvec)

def main_stationary(gut_dir, wanted_folder_numbers, excluded_folders=[]): 
    rootpath = os.path.join(gut_dir, 'results', 'pruebas', f'{wanted_folder_numbers[0]}') # OJO! All D_mat and E_avec must be the same for the wanted_folders!!
    D_mat = get_D_mat(rootpath)
    E_avec = get_E_avec(rootpath)
    pathways_ids, pathways_id_strs = get_pathways_id(D_mat, E_avec)
    pathways_strs = get_pathways_strs(D_mat)
    stationary_dict = get_stationary_distributions(gut_dir, D_mat, wanted_folder_numbers=wanted_folder_numbers, excluded_folders=excluded_folders)
    stationary_plots(gut_dir, stationary_dict, pathways_strs, pathways_ids, pathways_id_strs) 
    

gut_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


f = str(sys.argv[1])
try:
    c = int(sys.argv[2])
except ValueError:
    c = float(sys.argv[2])
try:
    m = int(sys.argv[3])
except ValueError:
    m = float(sys.argv[3])
try:
    e = int(sys.argv[4])
except ValueError:
    e = float(sys.argv[4])
delta = float(sys.argv[5])    
E_D_type = int(sys.argv[6])
low_lim = int(sys.argv[7])
high_lim = int(sys.argv[8])
tradeoff_data = f'{f}_{c}_{m}_{e}'#_delta{delta:.4f}'

rootpath = os.path.join(gut_dir, 'results', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', tradeoff_data)

wanted_folders = []
for f in os.listdir(rootpath): # This lists the names of all the things inside rootpath (just name, not full path)
    full_path = os.path.join(rootpath, f)
    if os.path.isdir(full_path): # we check that it is a dir and name is a number (below)
        try:
            # Try to convert the directory name to an integer
            int(f)
            wanted_folders.append(f)
        except ValueError:
            # If ValueError is raised, the name is not a number
            continue

print((low_lim, high_lim))        
print(wanted_folders)

for folder_number in wanted_folders:
    path = os.path.join(rootpath, folder_number)
    if int(folder_number) < low_lim or int(folder_number) >= high_lim:
        continue
    print(f'TRYING FOLDER: {folder_number}')
    image_dir = os.path.join(gut_dir, 'images', 'invasion', f'E_D_{E_D_type}', 'changing_tradeoff', f'{tradeoff_data}', f'{folder_number}')
    #if os.path.exists(image_dir):
    #    continue
    try:
        main_plots(path, sample_hours=int(24*7))
    except Exception as exc:
        print(f'Folder: {folder_number} failed, here is the exception info')
        print(traceback.format_exc())
        print(exc)
