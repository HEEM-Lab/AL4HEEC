# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:15:07 2023

@author: ZHANG Jun
"""

import os
import numpy as np
import pandas as pd

from agat.lib import file_exit
from agat.lib.high_throughput_lib import get_concentration_from_ase_formula
from acatal.module.high_throughput import HtAds
from acatal.default_parameters import default_high_throughput_config
from acatal.lib import get_d_G_H, increamental_mean, pad_dict_list
from acatal.bin.high_throughput_analysis import HtAdsAnalysis

def pred(**kwargs):
    # raise ValueError('Do not use this script. Post analysis may have issues.')
    curdir = os.getcwd()
    if not os.path.exists('high_throughput_prediction'):
        os.mkdir('high_throughput_prediction')

    high_throughput_config = {**default_high_throughput_config,
                              # **high_throughput_config,
                              **kwargs}
    high_throughput_config['opt_config']['device']=high_throughput_config['device']
    if kwargs.__contains__('restart_steps'):
        high_throughput_config['opt_config']['restart_steps'] = high_throughput_config['restart_steps']

    formulas = np.loadtxt(high_throughput_config['formula_file'], dtype=str)
    if not len(formulas.shape):
        formulas = [formulas.tolist()]
    formulas = list(set(list(formulas)))

    HA = HtAds(**high_throughput_config)

    num_calc = 50

    for j in range(0, num_calc):
        for f in formulas:
            path_tmp = os.path.join('high_throughput_prediction', f)
            if not os.path.exists(path_tmp):
                os.mkdir(path_tmp)
            os.chdir(path_tmp)
            HA.run(formula=f, calculation_index=str(j))
            os.chdir(curdir)
            file_exit()

    haa = HtAdsAnalysis(working_dir='high_throughput_prediction')
    haa.get_all_ads_data()
    # # post processing
    # mean_results = {k:[] for k in formulas}
    # std_results = {k:[] for k in formulas}

    # for f in formulas:
    #     surfs = [x for x in os.listdir(os.path.join('high_throughput_prediction',
    #                                                 f)) if x.split('.')[-1] == 'txt']

    #     for s in surfs:
    #         data_tmp = np.loadtxt(os.path.join('high_throughput_prediction',
    #                                            f, s))
    #         # remove ill-converged results.
    #         accept_1 = data_tmp[:,2] == 1.0 # ase exit with poor force convergence.
    #         surf_ave = np.average(data_tmp[:,1])
    #         ads_hi = surf_ave + 50 # Generally, the total energy of adsorption structure is close to the clean surface, if you placed a small specie on the surface.
    #         accept_2 = data_tmp[:,0] < ads_hi
    #         ads_lo = surf_ave - 50
    #         accept_3 = data_tmp[:,0] > ads_lo
    #         accept = np.logical_and(accept_1, accept_2)
    #         accept = np.logical_and(accept, accept_3)
    #         reject = np.where(np.logical_not(accept))
    #         if len(reject[0]) > 0:
    #             print(f'WARNING!!! Some results are rejected in {f}-{s}. {reject[0]} are discarded.')
    #         data_tmp = data_tmp[accept]

    #         d_G = get_d_G_H(data_tmp[:,0], data_tmp[:,1])
    #         mean, std = np.mean(d_G), np.std(d_G)
    #         mean_results[f].append(mean)
    #         std_results[f].append(std)

    # # analyse results
    # ## mean delta G of all results.
    # final_mean = {'formula': [],
    #               'Ni': [],
    #               'Co': [],
    #               'Fe': [],
    #               'NiCoFe': [],
    #               'Pd': [],
    #               'Pt': [],
    #               'mean_d_G': [],}

    # for f in formulas:
    #     final_mean['formula'].append(f)

    #     cons = get_concentration_from_ase_formula(f)
    #     cons = {k:(0.0 if k not in cons else cons[k]) for k in ['Ni', 'Co', 'Fe', 'Pd', 'Pt']}
    #     final_mean['NiCoFe'].append(cons['Ni']+cons['Co']+cons['Fe'])
    #     final_mean['Ni'].append(cons['Ni'])
    #     final_mean['Co'].append(cons['Co'])
    #     final_mean['Fe'].append(cons['Fe'])
    #     final_mean['Pd'].append(cons['Pd'])
    #     final_mean['Pt'].append(cons['Pt'])
    #     final_mean['mean_d_G'].append(np.mean(mean_results[f]))

    # df = pd.DataFrame(final_mean)
    # df.to_csv(os.path.join('high_throughput_prediction', 'average_delta_G.csv'))

    # ## convergence
    # mean_conv = pad_dict_list({k:increamental_mean(mean_results[k]) for k in formulas}, np.nan)
    # std_conv = pad_dict_list({k:increamental_mean(std_results[k]) for k in formulas}, np.nan)
    # pd.DataFrame(mean_conv).to_csv(os.path.join('high_throughput_prediction',
    #                                             'mean_conv.csv'))
    # pd.DataFrame(std_conv).to_csv(os.path.join('high_throughput_prediction',
    #                                            'std_conv.csv'))

if __name__ == '__main__':
    pred(formula_file='formulas.txt', model_save_dir='agat_model',
         graph_build_scheme_dir='.', restart_steps=3,
         fix_H_x_y=False, save_trajectory=True, adsorbates=['H'],
         sites=['bridge'], device='cuda')
