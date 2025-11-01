# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:00:32 2023

@author: 18326
"""

import os
import ase
from ase.io import read
from ase import Atoms
import numpy as np
import pandas as pd

from ase.optimize import MDMin
from agat.app import AgatCalculator
from agat.lib.model_lib import config_parser

import torch
from acatal.lib import pad_dict_list
from acatal.default_parameters import default_ase_calculator_config

# os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

class CompareToDFT(object):
    def __init__(self, model_save_dir='agat_model',
                 graph_build_scheme_dir = 'agat_model',
                 log_file_name='compare_to_dft_results.csv',
                 paths_file='paths.log',
                 **kwargs):
        self.config = {**default_ase_calculator_config,
                       **config_parser(kwargs)}
        self.model_save_dir = model_save_dir
        self.graph_build_scheme_dir = graph_build_scheme_dir
        paths_file = paths_file
        self.paths_list = np.loadtxt(paths_file, dtype=str)
        self.log_file_name = log_file_name
        self.device = torch.device(self.config['device'])

    def parse_calculaition_type(self, ase_atoms):
        cell = ase_atoms.cell.array
        length_z = np.linalg.norm(cell, axis=1)[2]
        if length_z < 19.0:
            return 'bulk'
        else:
            symbols = ase_atoms.get_chemical_symbols()
            if 'H' in symbols:
                return 'adsorption'
            else:
                return 'surface'

    def parse_calculaition_type_via_path(self, path):
        base_path = os.path.basename(path)
        t = base_path.split('_')[0]
        if t.isdigit():
            return 'adsorption'
        else:
            return t

    def is_vasp_completed(self, fname='OUTCAR'):
        if not os.path.exists(fname):
            return False
        with open(fname, 'r') as f:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 13000, os.SEEK_SET)
            lines = f.readlines()
            return ' reached required accuracy - stopping structural energy minimisation\n' in lines

    def read_dft(self, fname='OUTCAR', file_IO=None):
        try:
            # fname = os.path.join('/home/jzhang/work/AGAT_HER/0_FeCoNiPdPt_HER_raw_data/5_high_throughput_dft_2/Ni24Co18Fe17Pd26Pt11/',
            #                      'surface_static', 'OUTCAR')
            atoms_list = read(fname, index=':')
            e_opt = atoms_list[-1].get_total_energy()
            poscar = atoms_list[0]
        except:
            poscar, e_opt = None, None
            print(f'User Warning: exception(s) captured in {fname}.')
        return poscar, e_opt

    def run(self):
        # flog = open(self.log_file_name, 'a+')

        results = {
            'bulk_dft': [],
            'bulk_agat': [],
            'surface_dft': [],
            'surface_agat': [],
            'adsorption_dft': [],
            'adsorption_agat': [],
            # 'path': []
            }

        calculator = AgatCalculator(self.model_save_dir,
                                    self.graph_build_scheme_dir,
                                    device=self.device)

        for path in self.paths_list:
            path = os.path.abspath(path)
            # results['path'].append(path)
            calc_type = self.parse_calculaition_type_via_path(path)
            atoms, e_dft = self.read_dft(fname=os.path.join(path, 'OUTCAR'))
            if bool(atoms):
                atoms = Atoms(atoms, calculator=calculator)
                dyn = MDMin(atoms)
                dyn.run(fmax=self.config['fmax'],
                        steps=self.config['steps'])
                e_agat = atoms.get_total_energy()
                # calc_type = self.parse_calculaition_type(atoms)
                results[f'{calc_type}_dft'].append(e_dft)
                results[f'{calc_type}_agat'].append(e_agat)
            else:
                results[f'{calc_type}_dft'].append(None)
                results[f'{calc_type}_agat'].append(None)

        df = pd.DataFrame(data=pad_dict_list(results, None))
        df['bulk_deviation'] = df['bulk_agat'] - df['bulk_dft']
        df['surface_deviation'] = df['surface_agat'] - df['surface_dft']
        df['adsorption_deviation'] = df['adsorption_agat'] - df['adsorption_dft']
        # df['delta_G_deviation'] = df['adsorption_deviation'] - df['surface_deviation']
        df.to_csv(self.log_file_name)

    def run_new(self):
        results = {}

        calculator = AgatCalculator(self.model_save_dir,
                                    self.graph_build_scheme_dir,
                                    device=self.device)

        for path in self.paths_list:
            path = os.path.normpath(path)
            if not self.is_vasp_completed:
                continue
            path_component = [x for x in path.split(os.sep) if x !='' and (not x.isdigit())]
            formula = None
            for pc in path_component:
                try:
                    ase.formula.Formula(pc)
                    formula = pc
                except ValueError:
                    pass
            if formula is None:
                print(f'Warning!!! Cannot detect chemical formula of this path string: {path}')
                continue

            if not results.__contains__(formula):
                results[formula] = {'bulk_dft': None, 'bulk_agat': None,
                                    'surface_dft': None, 'surface_agat': None,
                                    'adsorption_dft': None, 'adsorption_agat': None}
            # results['path'].append(path)
            atoms, e_dft = self.read_dft(fname=os.path.join(path, 'OUTCAR'))
            if bool(atoms):
                atoms = Atoms(atoms, calculator=calculator)
                dyn = MDMin(atoms)
                dyn.run(fmax=self.config['fmax'],
                        steps=self.config['steps'])
                e_agat = atoms.get_total_energy()
                calc_type = self.parse_calculaition_type(atoms)
                results[formula][f'{calc_type}_dft'] = e_dft
                results[formula][f'{calc_type}_agat'] = e_agat

        df = pd.DataFrame(data=results).T
        df['bulk_deviation'] = df['bulk_agat'] - df['bulk_dft']
        df['surface_deviation'] = df['surface_agat'] - df['surface_dft']
        df['adsorption_deviation'] = df['adsorption_agat'] - df['adsorption_dft']
        df['delta_G_deviation'] = df['adsorption_deviation'] - df['surface_deviation']
        df.to_csv(self.log_file_name)

if __name__ == '__main__':
    # model_save_dir = os.path.join('agat_model')
    # graph_build_scheme_dir = os.path.join('.')
    # paths_file = os.path.join('paths.log')
    # log_file_name = os.path.join('compare_to_dft_results.csv')
    # ctd = CompareToDFT(model_save_dir,
    #                    graph_build_scheme_dir,
    #                    paths_file=paths_file,
    #                    log_file_name = log_file_name,
    #                    device='cpu')
    # self = ctd
    # ctd.run()
    # ctd.run_new()

    # predict all DFT data with each model.
    loop_id = (1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    path_id = (1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    graph_build_scheme_dir = os.path.join('.')
    for loop in loop_id:
        for path in path_id:
            model_save_dir = os.path.join(f'loop_{loop}')
            paths_file = os.path.join(f'paths_loop_{path}.log')
            log_file_name = os.path.join('compare_to_all_dft_calcs_of_each_loop',
                                         f'loop_{loop}__path_{path}.csv')

            ctd = CompareToDFT(model_save_dir,
                               graph_build_scheme_dir,
                               paths_file=paths_file,
                               log_file_name = log_file_name,
                               device='cpu')
            ctd.run_new()

