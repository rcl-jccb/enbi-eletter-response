import os
import numpy as np
from string import ascii_uppercase

def get_D_mat(rootpath):
    D_mat = np.loadtxt(os.path.join(rootpath, 'D_mat.txt'))
    return D_mat

def get_E_avec(rootpath):
    E_avec = np.loadtxt(os.path.join(rootpath, 'E_avec.txt'))
    return E_avec

def get_pathways_strs(D_mat):
    '''
    Here I obtain the pathways strs, OJO! different from pathways_id_strs. Here we obtain Subs_i->Subs_j for all possible pathways. 
    (pathways_id_strs Elevel_a->Elevel_b). Both are the same if and only if we have 1 substance per energy level.
    '''
    D_mat_str = []
    D_mat_str = np.array([[f"{i}->{j}" for j in range(D_mat.shape[1])] for i in range(D_mat.shape[0])])
    pathways_strs = D_mat_str[~np.isnan(D_mat)]
    return pathways_strs


def get_pathways_tuples(D_mat):
    '''
    Same as pathways_strs but with a tuple (i,j) instead of a string i->j
    '''
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
    - pathways_id_strs: is a vector that gives me each possible pathway. SO 1->2 means the pathway from E_level 1 to E_level 2
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
    return np.array(pathways_ids), pathways_id_strs
