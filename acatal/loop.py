# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:33:11 2023

@author: 18326
"""

import numpy as np
import os
from datetime import datetime
from shutil import copyfile, move, copytree, rmtree
import json

import torch.nn as nn
import pandas as pd
from matplotlib import pyplot  as plt
from matplotlib.ticker import MultipleLocator
import mpltern
# import ternary
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import joblib

import torch

from agat.data import concat_graphs
from agat.lib.model_lib import config_parser
from agat.data import BuildDatabase
from agat.model import Fit
from agat.lib import file_exit
from agat.lib.high_throughput_lib import get_concentration_from_ase_formula
from agat.app.cata import HtDftAds

import acatal
from acatal.lib import get_d_G_H, increamental_mean, pad_dict_list
from acatal.module.cgan import param_opt
from acatal.tool import concentration2formula, formula2concentration, con_dict2con_list
from acatal.module.high_throughput import HtAds
from acatal.module.compare_to_dft import CompareToDFT
from acatal.default_parameters import default_train_config, default_active_learing_config, default_high_throughput_config

debug_mode = False

if debug_mode:
    default_active_learing_config['binary_graphs_dir'] = '/home/jzhang/work/AGAT_HER/9_single_loop_test/graphs_test' # for debug only
    package_dir = '/home/jzhang/work/AGAT_HER/10_active_loops/acatal' # for debug only
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('WARNING!!! You are currently in the debugging mode!!!')
    print('WARNING!!! You are currently in the debugging mode!!!')
    print('WARNING!!! You are currently in the debugging mode!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
else:
    # package_dir = os.path.dirname(acatal.__file__)
    package_dir = acatal.__path__[0]

class SingleLoop(object):
    def __init__(self,
                 device=torch.device('cuda'),
                 **config):
        self.device = device
        self.config = {**default_active_learing_config, **config_parser(config)}
        self.loop_num = self.config['loop_num']
        self.root_dir = os.getcwd()
        self.wd = os.path.join(self.root_dir, f'loop_{self.loop_num}') # current working directory

        with open(os.path.join(self.root_dir, 'active_learning.json'), 'w') as f:
            out_config = {'active_learning': True,
                          'root_dir': self.root_dir,
                          'working_dir': self.wd,
                          'device': self.device}

            out_config = {**self.config, **out_config}
            json.dump(out_config, f, indent=4)

        # check inputs and dependencies
        if not os.path.exists(self.wd):
            print(f'{self.wd} not found, create this file ...')
            os.mkdir(self.wd)
        else:
            print(f'{self.wd} exists, please use another loop_num.')
            raise ValueError(f'{self.wd} exists, please use another loop_num.')
            # if debug_mode:
            #     print(f'For debugging, the folder {self.wd} will be overwritten.') # debug only
            # else:
            #     # in_str = input('Type <y> if you want overwritten this folder.')
            #     in_str = 'y'
            #     if in_str == 'y':
            #         ...
            #     else:
            #         raise ValueError(f'{self.wd} exists, please use another loop_num.')
        _log_fname = os.path.join(self.wd, f'active_cata_{self.loop_num}.log')
        self.logIO = open(_log_fname, 'a+', buffering=1)
        print('Starting time:', self.time, file=self.logIO)
        if os.path.exists(os.path.join(self.config['binary_graphs_dir'],
                                       f'all_graphs_{self.loop_num}.bin')):
            print(f'WARNING!!! all_graphs_{self.loop_num}.bin already exists, and this file is overwritten when building graphs.',
                  file=self.logIO)
        if not os.path.exists(os.path.join(self.config['binary_graphs_dir'],
                                           'graph_build_scheme.json')):
            raise FileNotFoundError(f"graph_build_scheme.json file not found under {self.config['binary_graphs_dir']}")
    @property
    def time(self):
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    def build_graphs(self, raw_dataset_dir=None, binary_graphs_dir=None, **kwargs):
        if isinstance(raw_dataset_dir, str):
            self.config['raw_dataset_dir'] = raw_dataset_dir
        if isinstance(binary_graphs_dir, str):
            self.config['binary_graphs_dir'] = binary_graphs_dir

        # load graph build method
        data_config = config_parser(os.path.join(self.config['binary_graphs_dir'],
                                                 'graph_build_scheme.json'))
        data_config['path_file'] = os.path.join(self.wd, '0_build_graphs', 'paths.log')
        data_config['dataset_path'] = os.path.join(self.wd, '0_build_graphs', 'dataset')
        data_config = {**data_config, **kwargs}

        print('Starting time for building graphs:', self.time, file=self.logIO)
        if not os.path.exists(os.path.join(self.wd, '0_build_graphs')):
            os.mkdir(os.path.join(self.wd, '0_build_graphs'))

        # create paths.log file
        # os.chmod(os.path.join(package_dir, 'bin', 'get_paths.sh'), 755)
        os.chdir(self.config['raw_dataset_dir'])
        os.system(f"bash {os.path.join(package_dir, 'bin', 'get_paths.sh')}")
        copyfile('paths.log', os.path.join(self.wd, '0_build_graphs', 'paths.log'))
        os.chdir(self.root_dir)

        ad = BuildDatabase(**data_config)
        ad.build()
        move(os.path.join(self.wd, '0_build_graphs', 'dataset', 'all_graphs.bin'),
                 os.path.join(self.config['binary_graphs_dir'],
                              f'all_graphs_{self.loop_num}.bin'))

        graph_fnames = [x for x in os.listdir(self.config['binary_graphs_dir']) if x.split('.')[-1] == 'bin']
        concat_graphs(*[os.path.join(self.config['binary_graphs_dir'], x) for x in graph_fnames])
        move('concated_graphs.bin',
              os.path.join(self.wd, '0_build_graphs', 'concated_graphs.bin'))
        print('Complete time for building graphs:', self.time, file=self.logIO)

    def train_agat(self, **kwargs):
        print('Starting time for training AGAT model:', self.time, file=self.logIO)
        if not os.path.exists(os.path.join(self.wd, '1_train_agat_model')):
            os.mkdir(os.path.join(self.wd, '1_train_agat_model'))
        train_config = {
            'dataset_path': os.path.join(self.wd, '0_build_graphs',
                                         'concated_graphs.bin'),
            'model_save_dir': os.path.join(self.wd, '1_train_agat_model',
                                           'agat_model'),
            'output_files': os.path.join(self.wd, '1_train_agat_model',
                                         'agat_train_out_file'),
            'device': self.device}
        train_config = {**default_train_config, **train_config, **kwargs}

        if debug_mode:
            train_config['epochs'] = 3

        if os.path.exists(os.path.join(self.wd, '1_train_agat_model', 'agat_model')):
            rmtree(os.path.join(self.wd, '1_train_agat_model', 'agat_model'))
        # copytree(os.path.join(self.config['agat_model_dir'], 'agat_model_latest'),
        #          os.path.join(self.wd, '1_train_agat_model', 'agat_model'))
        os.chdir(os.path.join(self.wd, '1_train_agat_model'))
        f = Fit(**train_config)
        f.fit()
        os.rename('fit.log', f'fit_lr_{train_config["learning_rate"]}.log')

        train_config["learning_rate"] /= 3
        f = Fit(**train_config)
        f.fit()
        os.rename('fit.log', f'fit_lr_{train_config["learning_rate"]}.log')

        os.chdir(self.root_dir)
        if os.path.exists(os.path.join(self.config['agat_model_dir'], 'agat_model_latest')):
            rmtree(os.path.join(self.config['agat_model_dir'], 'agat_model_latest'))
        copytree(os.path.join(self.wd, '1_train_agat_model', 'agat_model'),
                 os.path.join(self.config['agat_model_dir'], 'agat_model_latest'))
        if os.path.exists(os.path.join(self.config['agat_model_dir'],
                                       f'agat_model_loop_{self.loop_num}')):
            rmtree(os.path.join(self.config['agat_model_dir'], f'agat_model_loop_{self.loop_num}'))
        copytree(os.path.join(self.wd, '1_train_agat_model', 'agat_model'),
                 os.path.join(self.config['agat_model_dir'],
                              f'agat_model_loop_{self.loop_num}'))
        print('Complete time for training AGAT model:', self.time, file=self.logIO)

    def predict_dft_results(self):
        print('Starting time for comparing to DFT results:',
              self.time, file=self.logIO)
        if not os.path.exists(os.path.join(self.wd, '2_compare_to_DFT')):
            os.mkdir(os.path.join(self.wd, '2_compare_to_DFT'))

        model_save_dir = os.path.join(self.config['agat_model_dir'],
                                      'agat_model_latest')
        graph_build_scheme_dir = self.config['binary_graphs_dir']
        paths_file = os.path.join(self.wd, '0_build_graphs', 'paths.log')
        log_file_name = os.path.join(self.wd, '2_compare_to_DFT',
                                     'compare_to_dft_results.csv')
        ctd = CompareToDFT(model_save_dir,
                           graph_build_scheme_dir,
                           paths_file=paths_file,
                           log_file_name = log_file_name,
                           device=self.device)
        # ctd.run()
        ctd.run_new()
        print('Complete time for comparing to DFT results:',
              self.time, file=self.logIO)

    def high_throughput_prediction(self, check_cgan=False, **kwargs):
        if check_cgan:
            cur_wd = '7_check_cgan_recommendation' # cur_wd: current working directory.
        else:
            cur_wd = '3_high_throughput_prediction'

        print(f'Starting time for high-throughput prediction ({cur_wd}):', self.time, file=self.logIO)
        if not os.path.exists(os.path.join(self.wd, cur_wd)):
            os.mkdir(os.path.join(self.wd, cur_wd))

        high_throughput_config = {
            'model_save_dir': os.path.join(self.config['agat_model_dir'],
                                           'agat_model_latest'),
            'graph_build_scheme_dir': os.path.join(self.config['binary_graphs_dir']),
            'device': self.device}
        high_throughput_config = {**default_high_throughput_config,
                                  **high_throughput_config,
                                  **kwargs}
        ase_calculator_config = high_throughput_config['opt_config']
        high_throughput_config['opt_config']['device'] = self.device

        formulas_all = np.loadtxt(os.path.join(self.root_dir,
                                               'high_throughput_formulas.txt'),
                                  dtype=str)
        formulas = formulas_all[-40:]
        previous_formulas = formulas_all[:-40] # we want the CGAN recommendation at last two loops.
        if len(previous_formulas) > 0:
            selected_formulas = np.random.choice(previous_formulas, size=40)
            formulas = np.hstack((formulas, selected_formulas))
            formulas = list(set(list(formulas)))

        if check_cgan:
            formulas = formulas_all[-20:]

        HA = HtAds(**high_throughput_config)

        if debug_mode:
            num_calc=2
            ase_calculator_config['fmax'] = 100
            high_throughput_config['sites' ] = 'ontop'
        else:
            num_calc = 30

        for j in range(0, num_calc):
            for f in formulas:
                path_tmp = os.path.join(self.wd, cur_wd, f)
                if not os.path.exists(path_tmp):
                    os.mkdir(path_tmp)
                os.chdir(path_tmp)
                HA.run(formula=f, calculation_index=str(j))
                os.chdir(self.root_dir)
                file_exit()

        # post processing
        mean_results = {k:[] for k in formulas}
        std_results = {k:[] for k in formulas}

        for f in formulas:
            surfs = [x for x in os.listdir(os.path.join(self.wd, cur_wd,
                                                        f)) if x.split('.')[-1] == 'txt']

            for s in surfs:
                data_tmp = np.loadtxt(os.path.join(self.wd, cur_wd,
                                                   f, s))
                # remove ill-converged results.
                accept_1 = data_tmp[:,2] == 1.0 # ase exit with poor force convergence.
                surf_ave = np.average(data_tmp[:,1])
                ads_hi = surf_ave + 50 # Generally, the total energy of adsorption structure is close to the clean surface, if you placed a small specie on the surface.
                accept_2 = data_tmp[:,0] < ads_hi
                ads_lo = surf_ave - 50
                accept_3 = data_tmp[:,0] > ads_lo
                accept = np.logical_and(accept_1, accept_2)
                accept = np.logical_and(accept, accept_3)
                reject = np.where(np.logical_not(accept))
                if len(reject[0]) > 0:
                    print(f'WARNING!!! Some results are rejected in {f}-{s}. {reject[0]} are discarded.')
                data_tmp = data_tmp[accept]

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
        df.to_csv(os.path.join(self.wd, cur_wd,
                               'average_delta_G.csv'))

        ## convergence
        mean_conv = pad_dict_list({k:increamental_mean(mean_results[k]) for k in formulas}, np.nan)
        std_conv = pad_dict_list({k:increamental_mean(std_results[k]) for k in formulas}, np.nan)
        pd.DataFrame(mean_conv).to_csv(os.path.join(self.wd, cur_wd, 'mean_conv.csv'))
        pd.DataFrame(std_conv).to_csv(os.path.join(self.wd, cur_wd, 'std_conv.csv'))
        print(f'Complete time for high-throughput prediction ({cur_wd}):',
              self.time, file=self.logIO)

    def generate_new_compositions(self):
        print('Starting time for training CGAN model:', self.time, file=self.logIO)
        if not os.path.exists(os.path.join(self.wd, '4_generate_new_compositions')):
            os.mkdir(os.path.join(self.wd, '4_generate_new_compositions'))

        opt_param = param_opt()
        with open(os.path.join(self.wd, '4_generate_new_compositions', 'opt_param.json'), 'w') as fp:
            json.dump(opt_param, fp, indent=4)

        active_results = np.loadtxt(os.path.join(self.wd,
                                                 '4_generate_new_compositions',
                                                 'active_results.txt'))

        active_results = active_results[active_results[:,1].argsort()]

        selected_dir = active_results[:5,0]
        mean_minus_final_mae = np.mean(active_results[:5,1])

        all_cons = []
        for d in selected_dir:
            d = str(int(d))
            data_tmp = np.loadtxt(os.path.join(
                self.wd, '4_generate_new_compositions', d,
                'generated_cons.txt'))
            all_cons.append(data_tmp)

        self.all_cons = np.vstack(all_cons)

        np.savetxt(os.path.join(self.wd,
                                '4_generate_new_compositions',
                                'best_5_cgan_model_predictions.txt'),
                   self.all_cons,
                   fmt='%.8f')

        # active_step_i = np.argmin(active_results[:,1])
        # self._active_step = int(active_results[active_step_i,0])
        # minus_final_mae = active_results[active_step_i,1]

        # plot

        # x_all_plot = [[x[0]+x[1]+x[2], x[3], x[4]] for x in self.all_cons]

        # fig,ax = plt.subplots()
        # scale = 1.0
        # fontsize = 12
        # offset = 0.14
        # figure, tax = ternary.figure(scale=scale,ax=ax)
        # tax.gridlines(multiple=5, color="blue")
        # figure.set_size_inches(6, 6)
        # tax.right_corner_label("NiCoFe", fontsize=fontsize)
        # tax.top_corner_label("Pd", fontsize=fontsize)
        # tax.left_corner_label("Pt", fontsize=fontsize)
        # tax.left_axis_label("Pt", fontsize=fontsize, offset=offset)
        # tax.right_axis_label("Pd", fontsize=fontsize, offset=offset)
        # tax.bottom_axis_label("NiCoFe", fontsize=fontsize, offset=offset)
        # tax.scatter(x_all_plot, marker='s', color='red', label="Initial pred")
        # # Remove default Matplotlib Axes
        # # tax.clear_matplotlib_ticks()
        # tax.get_axes().axis('off')
        # tax.boundary()
        # tax.set_title(f"MAE: {-mean_minus_final_mae}")
        # plt.savefig(os.path.join(self.wd, '4_generate_new_compositions',
        #                          'ternary_cons.png'))

        # === plot ===
        x1 = self.all_cons[:,0] + self.all_cons[:,1] + self.all_cons[:,2]
        x2, x3 = self.all_cons[:,3], self.all_cons[:,4]
        ax = plt.subplot(projection="ternary")
        ax.taxis.set_major_locator(MultipleLocator(0.20))
        ax.laxis.set_major_locator(MultipleLocator(0.20))
        ax.raxis.set_major_locator(MultipleLocator(0.20))
        ax.scatter(x1, x2, x3,
                   marker='o',
                   s=32.,
                   c=[np.array([36,140,175])/255],
                   alpha=0.3,
                   edgecolors=np.array([[45,45,45]])/255)
        ax.set_tlabel("NiCoFe", size=12)
        ax.set_llabel("Pd", size=12)
        ax.set_rlabel("Pt", size=12)


        ax.taxis.set_label_position("tick1")
        ax.laxis.set_label_position("tick1")
        ax.raxis.set_label_position("tick1")

        ax.grid(axis='t', linestyle='--')
        ax.grid(axis='l', linestyle='--')
        ax.grid(axis='r', linestyle='--')

        plt.title(f'Loop {self.loop_num}', loc='right')
        plt.savefig(os.path.join(self.wd, '4_generate_new_compositions',
                                  'ternary_cons.png'),
                                  dpi=600,
                                  bbox_inches='tight')
        print('Complete time for training CGAN model:', self.time, file=self.logIO)

    def knn_cluster(self):
        print('Starting time for KNN clustering:', self.time, file=self.logIO)
        if not os.path.exists(os.path.join(self.wd, '5_knn_cluster')):
            os.mkdir(os.path.join(self.wd, '5_knn_cluster'))

        # x_data = np.loadtxt(os.path.join(self.wd,
        #                                  '4_generate_new_compositions',
        #                                  str(self._active_step),
        #                                  'generated_cons.txt'))
        kmeans = KMeans(n_clusters=20, random_state=0)
        kmeans.fit(self.all_cons)
        centers = kmeans.cluster_centers_
        # score = kmeans.score(x_data)

        # ===find nearest structure of center===
        closest, _ = pairwise_distances_argmin_min(centers, self.all_cons)
        closest_samples = np.array([self.all_cons[x] for x in closest])

        # ===save results===
        np.savetxt(os.path.join(self.wd,
                                '5_knn_cluster',
                                'centers_20.txt'),
                   centers, fmt='%.8f')
        np.savetxt(os.path.join(self.wd,
                                '5_knn_cluster',
                                'closest_samples_20.txt'),
                   closest_samples, fmt='%.8f')

        # save formuals (to subdir and root_dir respectively.)
        high_throughput_formulas = concentration2formula(centers)
        np.savetxt(os.path.join(self.wd,
                                '5_knn_cluster',
                                'high_throughput_formulas.txt'),
                   high_throughput_formulas, fmt='%s')
        with open(os.path.join(self.root_dir, 'high_throughput_formulas.txt'), 'a+') as f:
            np.savetxt(f, high_throughput_formulas, fmt='%s')

        # save model
        joblib.dump(kmeans, os.path.join(self.wd, '5_knn_cluster', 'kmeans.model'))

        # === plot ===
        # x_all_plot = [[x[0]+x[1]+x[2], x[3], x[4]] for x in self.all_cons]
        # centers_plot = [[x[0]+x[1]+x[2], x[3], x[4]] for x in centers]

        # fig,ax = plt.subplots()
        # scale = 1.0
        # fontsize = 12
        # offset = 0.14
        # figure, tax = ternary.figure(scale=scale,ax=ax)
        # tax.gridlines(multiple=5, color="blue")
        # figure.set_size_inches(6, 6)
        # tax.right_corner_label("NiCoFe", fontsize=fontsize)
        # tax.top_corner_label("Pd", fontsize=fontsize)
        # tax.left_corner_label("Pt", fontsize=fontsize)
        # tax.left_axis_label("Pt", fontsize=fontsize, offset=offset)
        # tax.right_axis_label("Pd", fontsize=fontsize, offset=offset)
        # tax.bottom_axis_label("NiCoFe", fontsize=fontsize, offset=offset)

        # tax.scatter(x_all_plot, marker='s', color='red', label="Initial pred")
        # tax.scatter(centers_plot, marker='s', color='blue', label="Initial pred")
        # # Remove default Matplotlib Axes
        # # tax.clear_matplotlib_ticks()
        # tax.get_axes().axis('off')
        # tax.boundary()
        x1 = self.all_cons[:,0] + self.all_cons[:,1] + self.all_cons[:,2]
        x2, x3 = self.all_cons[:,3], self.all_cons[:,4]
        y1 = centers[:,0] + centers[:,1] + centers[:,2]
        y2, y3 = centers[:,3], centers[:,4]
        ax = plt.subplot(projection="ternary")
        ax.taxis.set_major_locator(MultipleLocator(0.20))
        ax.laxis.set_major_locator(MultipleLocator(0.20))
        ax.raxis.set_major_locator(MultipleLocator(0.20))
        ax.scatter(x1, x2, x3,
                   marker='o',
                   s=32.,
                   c=[np.array([36,140,175])/255],
                   alpha=0.3,
                   edgecolors=np.array([[45,45,45]])/255)
        ax.scatter(y1, y2, y3,
                   marker='h',
                   s=32.,
                   c=[np.array([219,59,51])/255],
                   alpha=1.0,
                   edgecolors='k')
        ax.set_tlabel("NiCoFe", size=12)
        ax.set_llabel("Pd", size=12)
        ax.set_rlabel("Pt", size=12)


        ax.taxis.set_label_position("tick1")
        ax.laxis.set_label_position("tick1")
        ax.raxis.set_label_position("tick1")

        ax.grid(axis='t', linestyle='--')
        ax.grid(axis='l', linestyle='--')
        ax.grid(axis='r', linestyle='--')

        plt.title(f'Loop {self.loop_num}', loc='right')
        plt.savefig(os.path.join(self.wd, '5_knn_cluster',
                                 'ternary_cons_after_KNN.png'),
                                 dpi=600,
                                 bbox_inches='tight')
        print('Complete time for KNN clustering:', self.time, file=self.logIO)

    def high_throughput_dft_calculation(self):
        print('Starting time for preparing high-throughput DFT calculations:',
              self.time, file=self.logIO)
        if not os.path.exists(os.path.join(self.wd, '6_high_throughput_dft_calculation')):
            os.mkdir(os.path.join(self.wd, '6_high_throughput_dft_calculation'))

        # HA = HtDftAds(calculation_index=0, sites='bridge', include_bulk_aimd=False,
        #       include_surface_aimd=False, include_adsorption_aimd=False,
        #       random_samples=1, vasp_bash_path='/home/jzhang988/scratch/adsorption-HEA/jobs/general_file_useful/vasp_run.sh')

        # cons = np.loadtxt(os.path.join(self.wd,
        #                                '4_generate_new_compositions',
        #                                str(self._active_step),
        #                                'generated_cons.txt'))
        # choices = np.random.choice(range(len(self.all_cons)), size=500)
        # cons = self.all_cons[choices]

        formulas = concentration2formula(self.all_cons)
        choices = np.random.choice(range(len(formulas)), size=200, replace=False)
        formulas = [formulas[x] for x in choices]

        for f in formulas:
            path_tmp = os.path.join(self.wd, '6_high_throughput_dft_calculation', f)
            if not os.path.exists(path_tmp):
                os.mkdir(path_tmp)
            copyfile(os.path.join(package_dir, 'bin', 'aug_data.py'),
                     os.path.join(path_tmp, 'aug_data.py'))
            copyfile(os.path.join(package_dir, 'bin', 'run.sh'),
                     os.path.join(path_tmp, 'run.sh'))
            # HA.run(f)
        print('Complete time for preparing high-throughput DFT calculations:',
              self.time, file=self.logIO)

    def run(self, **kwargs):
        self.config = {**self.config, **config_parser(kwargs)}
        self.build_graphs()
        self.train_agat()
        self.predict_dft_results()
        self.high_throughput_prediction()
        self.generate_new_compositions()
        self.knn_cluster()
        self.high_throughput_dft_calculation()
        self.high_throughput_prediction(check_cgan=True)

if __name__ == '__main__':
    sl = SingleLoop(loop_num=5,
                    raw_dataset_dir='raw_dataset_dir',
                    binary_graphs_dir='binary_graphs_dir',
                    device='cuda')
    sl.run()
    # self = sl
    # sl.build_graphs()
    # sl.train_agat()
    # sl.predict_dft_results()
    # sl.high_throughput_prediction()
    # sl.generate_new_compositions()
    # sl.knn_cluster()
    # sl.high_throughput_dft_calculation()
    # sl.high_throughput_prediction(check_cgan=True)
    # 
