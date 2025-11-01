# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:23:30 2024

@author: ZHANGJUN

Get results of high-throughput predictions.
"""


raise ValueError('This script is deprecated.')

import os
import numpy as np
import pandas as pd
from acatal.lib import get_d_G_H, increamental_mean, pad_dict_list
from agat.lib.high_throughput_lib import get_concentration_from_ase_formula

formulas = np.loadtxt('formulas.txt', dtype=str)
mean_results = {k:[] for k in formulas}
std_results = {k:[] for k in formulas}

for f in formulas:
    f = str(f)
    surfs = [x for x in os.listdir(f) if x.split('.')[-1] == 'txt']

    for s in surfs:
        data_tmp = np.loadtxt(os.path.join(f, s))

        # remove ill-converged results.
        accept_1 = data_tmp[:,2] == 1.0 # ase exit with poor force convergence.
        surf_ave = np.average(data_tmp[:,1])
        ads_hi = surf_ave + 50 # Generally, the total energy of adsorption structure is close to the clean surface, if you placed a small specie on the surface.
        accept_2 = data_tmp[:,0] < ads_hi
        ads_lo = surf_ave - 50
        accept_3 = data_tmp[:,0] > ads_lo
        accept = np.logical_and(accept_1, accept_2)
        accept = np.logical_and(accept, accept_3)

        data_tmp = data_tmp[accept]

        reject = np.where(np.logical_not(accept))
        if len(reject[0]) > 1:
            print(f'WARNING!!! Some results are rejected in {f}-{s}. {reject[0]} are discarded.')

        d_G = get_d_G_H(data_tmp[:,0], data_tmp[:,1])
        mean, std = np.mean(d_G), np.std(d_G)
        mean_results[f].append(mean)
        std_results[f].append(std)

# analyse results
## mean delta G of all results.
final_mean = {'formula': [],
              'Ni': [],
              'Co': [],
              'Fe': [],
              'NiCoFe': [],
              'Pd': [],
              'Pt': [],
              'mean_d_G': [],}

for f in formulas:
    final_mean['formula'].append(f)

    cons = get_concentration_from_ase_formula(f)
    cons = {k:(0.0 if k not in cons else cons[k]) for k in ['Ni', 'Co', 'Fe', 'Pd', 'Pt']}
    final_mean['NiCoFe'].append(cons['Ni']+cons['Co']+cons['Fe'])
    final_mean['Ni'].append(cons['Ni'])
    final_mean['Co'].append(cons['Co'])
    final_mean['Fe'].append(cons['Fe'])
    final_mean['Pd'].append(cons['Pd'])
    final_mean['Pt'].append(cons['Pt'])
    final_mean['mean_d_G'].append(np.mean(mean_results[f]))

df = pd.DataFrame(final_mean)
df.to_csv('average_delta_G.csv')

## convergence
mean_conv = pad_dict_list({k:increamental_mean(mean_results[k]) for k in formulas}, np.nan)
std_conv = pad_dict_list({k:increamental_mean(std_results[k]) for k in formulas}, np.nan)
pd.DataFrame(mean_conv).to_csv(os.path.join('mean_conv.csv'))
pd.DataFrame(std_conv).to_csv(os.path.join('std_conv.csv'))
