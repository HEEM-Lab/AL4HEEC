# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:24:50 2024

@author: ZHANGJUN
"""


import os
import numpy as np
import pandas as pd
from ase.io import read
from agat.lib.high_throughput_lib import get_concentration_from_ase_formula
# from acatal.tool import formula2concentration

def formula2concentration(formulas):
    results = {'Ni': [],
            'Co': [],
            'Fe': [],
            'Pd': [],
            'Pt': [],
            }

    for f in formulas:
        con = get_concentration_from_ase_formula(f)
        con = {k:(con[k] if con.__contains__(k) else 0.0) for k in results}
        results['Ni'].append(con['Ni'])
        results['Co'].append(con['Co'])
        results['Fe'].append(con['Fe'])
        results['Pd'].append(con['Pd'])
        results['Pt'].append(con['Pt'])
    return results

def dict2nparray(data_dict: dict):
    return np.vstack([v for k,v in data_dict.items()])

# working_dir = r'C:\Users\junzh\Desktop\tmp files'
comps = np.loadtxt('all_formulas.txt', dtype=str) # [x for x in os.listdir(working_dir) if os.path.isdir(os.path.join(working_dir, x))]

cons = formula2concentration(comps)
cons = pd.DataFrame(cons)
cons.to_csv('all_cons.csv')

###############################################################################
dirs = [x for x in os.listdir('.') if os.path.isdir(x)]
formulas = []
for d in dirs:
    ase_atoms = read(os.path.join(d, 'POSCAR'))
    formula = ase_atoms.get_chemical_formula(ase_atoms)
    formulas.append(formula)
np.savetxt('formulas.txt', formulas, fmt='%s')
