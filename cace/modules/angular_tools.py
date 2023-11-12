import itertools
from .angular import lxlylz_factorial_coef, make_lxlylz

__all__ = ['find_combo_vectors_nu1', 'find_combo_vectors_nu2', 'find_combo_vectors_nu3', 'find_combo_vectors_nu4']

"""
We store the values

vec_dict_allnu = {}
vec_dict_allnu[2], _, _  = find_combo_vectors_nu2(l_max)
vec_dict_allnu[3], _, _  = find_combo_vectors_nu3(l_max)
vec_dict_allnu[4], _, _  = find_combo_vectors_nu4(l_max)

import pickle

# Save dictionary using pickle
with open('symmetrize_angular_l_list.pickle', 'wb') as handle:
    pickle.dump(vec_dict_allnu, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Read dictionary back
with open('symmetrize_angular_l_list.pickle', 'rb') as handle:
    read_dict = pickle.load(handle)
"""

def find_combo_vectors_nu1():
    vector_groups = [0, 0, 0]
    prefactors = 1
    vec_dict = {0: [([0, 0, 1], 1)]}
    return vec_dict, vector_groups, prefactors

def find_combo_vectors_nu2(l_max):
    vector_groups = []
    prefactors = []
    vec_dict = {}
    
    L_list = range(1, l_max+1)
    for i, L in enumerate(L_list):
        for lxlylz_now in make_lxlylz(L):
            lx, ly, lz = lxlylz_now
            prefactor = lxlylz_factorial_coef(lxlylz_now)
            #print(prefactor)
            vector_groups.append(lxlylz_now)
            prefactors.append(prefactor)
            key = L
            vec_dict[key] = vec_dict.get(key, []) + [(lxlylz_now, prefactor)]
    return vec_dict, vector_groups, prefactors

def find_combo_vectors_nu3(l_max):
    vector_groups = []
    prefactors = []
    vec_dict = {}
    
    for lx1, ly1, lz1 in itertools.product(range(l_max+1), repeat=3):
        l1 = lx1 + ly1 + lz1
        if 0 < (lx1 + ly1 + lz1) <= l_max:
            for lx2, ly2, lz2 in itertools.product(range(l_max+1), repeat=3):
                l2 = lx2 + ly2 + lz2
                if (lx1 + ly1 + lz1) <= (lx2 + ly2 + lz2) <= l_max:
                    lx3, ly3, lz3 = lx1 + lx2, ly1 + ly2, lz1 + lz2
                    if (lx3 + ly3 + lz3) <= l_max:
                        if ([lx2, ly2, lz2], [lx1, ly1, lz1], [lx3, ly3, lz3]) not in vector_groups:
                            vector_groups.append(([lx1, ly1, lz1], [lx2, ly2, lz2], [lx3, ly3, lz3]))
                            prefactor = lxlylz_factorial_coef([lx1, ly1, lz1])*lxlylz_factorial_coef([lx2, ly2, lz2])
                            prefactors.append(prefactor)
                            
                            key = (l1 ,l2)
                            vec_dict[key] = vec_dict.get(key, []) + [([lx1, ly1, lz1], [lx2, ly2, lz2], [lx3, ly3, lz3], prefactor)]
    return vec_dict, vector_groups, prefactors

def find_combo_vectors_nu4(l_max):
    vector_groups = []
    vec_dict = {}
    prefactors = []
    for lx1, ly1, lz1 in itertools.product(range(l_max + 1), repeat=3):
        l1 = lx1 + ly1 + lz1
        if 0 < l1 <= l_max:
            for lx2, ly2, lz2 in itertools.product(range(l_max + 1), repeat=3):
                l2 = lx2 + ly2 + lz2
                if l1 < l2 <= l_max:  # Ensuring l2 is strictly greater than l1
                    for dx, dy, dz in itertools.product(range(l_max + 1), repeat=3):
                        dl = dx + dy + dz
                        if dl >= 1:
                            lx3, ly3, lz3 = lx1 + dx, ly1 + dy, lz1 + dz
                            lx4, ly4, lz4 = lx2 + dx, ly2 + dy, lz2 + dz
                            if (lx3 + ly3 + lz3) <= l_max and (lx4 + ly4 + lz4) <= l_max:
                                vector_groups.append(([lx1, ly1, lz1], [lx2, ly2, lz2], 
                                                      [lx3, ly3, lz3], [lx4, ly4, lz4]))
                                prefactor = lxlylz_factorial_coef([lx1, ly1, lz1]) \
                                    *lxlylz_factorial_coef([lx2, ly2, lz2]) \
                                    *lxlylz_factorial_coef([dx, dy, dz])
                                prefactors.append(prefactor)
                                
                                key = (l1 ,l2, dl)
                                vec_dict[key] = vec_dict.get(key, []) + \
                                [([lx1, ly1, lz1], [lx2, ly2, lz2], [lx3, ly3, lz3], [lx4, ly4, lz4], prefactor)]
    return vec_dict, vector_groups, prefactors