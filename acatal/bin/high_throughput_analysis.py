# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:07:59 2024

@author: ZHANGJUN
"""

import os

import numpy as np
import pandas as pd
from ase.neighborlist import natural_cutoffs, NeighborList
import ase
from ase.io import read

from acatal.lib import get_d_G_H, increamental_mean, pad_dict_list, increamental_std
# from agat.lib.high_throughput_lib import get_concentration_from_ase_formula
from zjpf import get_concentration_from_ase_formula

'''
formulas <-- composition <-- surface <-- site
'''

class HtAdsAnalysis(object):
    def __init__(self, working_dir=os.path.abspath('.')):
        self.working_dir = working_dir
        self.coordination2site = {1: 'ontop',
                                  2: 'bridge',
                                  3: 'hollow'}
        self.get_delta_G = get_d_G_H
        self.elements = ['Ni', 'Co', 'Fe', 'Pd', 'Pt'] # element list of catalysts.

    def formula_parser(self, formulas):
        if isinstance(formulas, list):
            pass
        elif isinstance(formulas, str):
            assert os.path.exist(formulas), f'Path file not found: {formulas}.'
            formulas = np.loadtxt('formulas.txt', dtype=str)
        elif isinstance(formulas, type(None)):
            formulas = [x for x in os.listdir(self.working_dir) if os.path.isdir(x)]
        else:
            raise TypeError(f'Input type not supported: formulas {type(formulas)}')
        return formulas

    def is_formula(self, f_str):
        assert isinstance(f_str, str), 'The input should be a string.'
        try:
            ase.formula.Formula(f_str)
            if f_str == '' or f_str.isdigit():
                r = False
            else:
                r = True
        except ValueError:
            r = False
        return r

    def find_neighbors(self,
                       ase_atoms,
                       cutoff_mult=1.3,
                       tracing_atom=-1):
        tracing_atom = range(len(ase_atoms))[tracing_atom]
        ase_cutoffs = natural_cutoffs(ase_atoms, mult=cutoff_mult)
        i, j, d, D = ase.neighborlist.neighbor_list('ijdD', # i: sender; j: receiver; d: distance; D: direction
                                                    ase_atoms,
                                                    cutoff=ase_cutoffs,
                                                    self_interaction=False)
        mask = i==tracing_atom
        j = j[mask]
        return {'atom_id': list(j), 'atom_symbol': [ase_atoms.get_chemical_symbols()[x] for x in j]}

    def get_adsorption_site(self, neighbor_list: list):
        try:
            site_type = self.coordination2site[len(neighbor_list)]
        except KeyError:
            site_type = 'weird'
            print('Warning!!! Weird sites detected.')
        return site_type

    def dict2csv(self, data_dict: dict, out_name: str):
        data_dict = pad_dict_list(data_dict, np.nan)
        df = pd.DataFrame(data_dict)
        df.to_csv(out_name)

    def dict2nparray(self, data_dict: dict):
        return np.vstack([v for k,v in data_dict.items()])

    def get_surf_ads_data(self, surf_data_fname):
        # get adsorption data on one single surface. this function will lower the code efficiency.
        surf_data_dict = {}
        surf_data = np.loadtxt(surf_data_fname)
        # if np.shape(surf_data)[0] == 0: return {}
        site_id = np.reshape(range(len(surf_data)),(-1, 1))
        surf_data = np.hstack((surf_data, site_id))

        # remove ill-converged results.
        accept_1 = surf_data[:,2] == 1.0 # ase exit with poor force convergence.
        surf_ave = np.average(surf_data[:,1])
        ads_hi = surf_ave + 50 # Generally, the total energy of adsorption structure is close to the clean surface, if you placed a small specie on the surface.
        accept_2 = surf_data[:,0] < ads_hi
        ads_lo = surf_ave - 50
        accept_3 = surf_data[:,0] > ads_lo
        accept = np.logical_and(accept_1, accept_2)
        accept = np.logical_and(accept, accept_3)

        surf_data = surf_data[accept]

        reject = np.where(np.logical_not(accept))
        if len(reject[0]) > 0:
            print(f'WARNING!!! Some results are rejected in {surf_data_fname}. {reject[0]} are discarded.')
        for d in surf_data:
            site_id = int(d[3])
            surf_data_dict[f'site_{site_id}'] = d # [ads_energy, surf_energy, return_code, site_id]
        return surf_data_dict

    def get_surf_ads_site(self, surf_dir):
        # get adsorption sites on one signle surface.
        site_atoms, site_type = {}, {}
        # site_fnames = [x for x in os.listdir(surf_dir) if x.split('.')[-1] == 'gat' and 'ads' in x.split('_')]
        site_fnames = []
        for f in os.listdir(surf_dir):
            f_dot = f.split('.')
            f_und = f.split('_')
            if f_dot[-1] == 'gat' and 'ads' in f_und and 'CONTCAR' in f_und:
                site_fnames.append(f)

        for site in site_fnames:
            site_id = site.split('_')[4].split('.')[0]
            ads_atoms = read(os.path.join(surf_dir, site))
            neighbors = self.find_neighbors(ads_atoms)
            site_atoms[f'site_{site_id}'] = neighbors
            site_type[f'site_{site_id}'] = self.get_adsorption_site(
                neighbors['atom_id'])
        return {'site_atoms': site_atoms,
                'site_type': site_type}

    def get_comp_ads_data(self, comp_dir, read_site=False):
        # get adsorption data of one single composition (formula)
        surf_data_fnames = [x for x in os.listdir(
            comp_dir) if x.split('.')[-1] == 'txt']
        comp_data, comp_neighbor, comp_site = {}, {}, {}
        for surf in surf_data_fnames:
            surf_id = surf.split('_')[4].split('.')[0]
            comp_data[f'surface_{surf_id}'] = self.get_surf_ads_data(
                os.path.join(comp_dir, surf))
            if read_site:
                site_results = self.get_surf_ads_site(
                    os.path.join(comp_dir, f'{surf_id}_th_calculation'))
                neighbor_atoms, site_type = site_results['site_atoms'], site_results['site_type']
                comp_neighbor[f'surface_{surf_id}'] = neighbor_atoms
                comp_site[f'surface_{surf_id}'] = site_type
        return {'comp_data': comp_data,
                'comp_neighbor': comp_neighbor,
                'comp_site': comp_site}

    def get_all_ads_data(self):
        # get adsorption data of all compositions (formulas), no site analysis.
        comps = [x for x in os.listdir(self.working_dir) if os.path.isdir(
            os.path.join(self.working_dir, x)) and self.is_formula(x)]
        final_mean = {'formula': [],
                      'Ni': [],
                      'Co': [],
                      'Fe': [],
                      'NiCoFe': [],
                      'Pd': [],
                      'Pt': [],
                      'mean_d_G': [],
                      'std_d_G': []}

        mean_conv, std_conv = {}, {}

        for comp in comps:
            comp_ads_data = self.get_comp_ads_data(
                os.path.join(self.working_dir, comp), read_site=False)['comp_data']
            comp_ads_data = self.dict2nparray(
                {surf: self.dict2nparray(surf_data) for surf, surf_data in comp_ads_data.items()})

            final_mean['formula'].append(comp)
            cons = get_concentration_from_ase_formula(comp)
            cons = {k:(0.0 if k not in cons else cons[k]) for k in self.elements}
            final_mean['NiCoFe'].append(cons['Ni']+cons['Co']+cons['Fe'])
            final_mean['Ni'].append(cons['Ni'])
            final_mean['Co'].append(cons['Co'])
            final_mean['Fe'].append(cons['Fe'])
            final_mean['Pd'].append(cons['Pd'])
            final_mean['Pt'].append(cons['Pt'])
            delta_G = self.get_delta_G(comp_ads_data[:,0], comp_ads_data[:,1])
            final_mean['mean_d_G'].append(np.mean(delta_G))
            final_mean['std_d_G'].append(np.std(delta_G))
            mean_conv[comp] = increamental_mean(delta_G)
            std_conv[comp] = increamental_std(delta_G)

        self.dict2csv(final_mean, 'average_delta_G.csv')
        self.dict2csv(mean_conv, 'mean_conv.csv')
        self.dict2csv(std_conv, 'std_conv.csv')

    def get_all_ads_data_per_element(self, write_per_element_results=False):
        comps = [x for x in os.listdir(self.working_dir) if os.path.isdir(
            os.path.join(self.working_dir, x)) and self.is_formula(x)]

        final_mean = {k:[] for k in ['formula'] + self.elements +\
                                    [f'mean_d_G_{x}' for x in self.elements] +\
                                    [f'std_d_G_{x}' for x in self.elements]}
        mean_conv, std_conv = {}, {}

        for comp in comps:
            comp_ads_data = self.get_comp_ads_data(
                os.path.join(self.working_dir, comp), read_site=True)
            comp_data = comp_ads_data['comp_data']
            comp_neighbor = comp_ads_data['comp_neighbor']
            comp_site = comp_ads_data['comp_site']

            final_mean['formula'].append(comp)
            cons = get_concentration_from_ase_formula(comp)
            cons = {k:(0.0 if k not in cons else cons[k]) for k in self.elements}
            for e in self.elements:
                final_mean[e].append(cons[e])

            delta_G = {k:[] for k in self.elements}

            surfs = comp_neighbor.keys()
            for surf in surfs:
                sites = comp_data[surf].keys()
                for site in sites:
                    for atom_id, atom_symbol in zip(comp_neighbor[surf][site]['atom_id'],
                                                    comp_neighbor[surf][site]['atom_symbol']):
                        d = comp_data[surf][site]
                        delta_G[atom_symbol].append(self.get_delta_G(d[0], d[1]))
            for e in self.elements:
                final_mean[f'mean_d_G_{e}'].append(np.mean(delta_G[e]))
                final_mean[f'std_d_G_{e}'].append(np.std(delta_G[e]))
                mean_conv[f'{comp}_{e}'] = increamental_mean(delta_G[e])
                std_conv[f'{comp}_{e}'] = increamental_std(delta_G[e])

            if write_per_element_results:
                self.dict2csv(delta_G, f'ads_per_element_of_{comp}.csv')

        self.dict2csv(final_mean, 'average_delta_G_ele.csv')
        self.dict2csv(mean_conv, 'mean_conv_ele.csv')
        self.dict2csv(std_conv, 'std_conv_ele.csv')

    def get_all_ads_data_per_site(self):
        comps = [x for x in os.listdir(self.working_dir) if os.path.isdir(
            os.path.join(self.working_dir, x)) and self.is_formula(x)]

        for comp in comps:
            comp_ads_data = self.get_comp_ads_data(
                os.path.join(self.working_dir, comp), read_site=True)
            comp_data = comp_ads_data['comp_data']
            comp_neighbor = comp_ads_data['comp_neighbor']

            results = {'Neighbors': [], 'Number of neighbors': [], 'delta_G': []}
            surfs = comp_neighbor.keys()
            for surf in surfs:
                sites = comp_data[surf].keys()
                for site in sites:
                    d = comp_data[surf][site]
                    if d[2] == 1.0:
                        delta_G = self.get_delta_G(d[0], d[1])

                        results['delta_G'].append(delta_G)
                        results['Neighbors'].append('-'.join(comp_neighbor[surf][site]['atom_symbol']))
                        results['Number of neighbors'].append(len(comp_neighbor[surf][site]['atom_symbol']))
            self.dict2csv(results, f'ads_data_per_site_{comp}.csv')


    def plot(self):
        pass

if __name__ == '__main__':
    working_dir = r'C:\Users\junzh\Desktop\tmp files'
    working_dir = 'high_throughput_prediction'
    haa = HtAdsAnalysis(working_dir=working_dir)
    # haa.get_surf_ads_data(surf_data_fname=os.path.join(working_dir, 'NiCoFePdPt',
    #                                                    'ads_surf_energy_H_46.txt'))
    # haa.get_surf_ads_site(surf_dir=os.path.join(
    #     working_dir, 'NiCoFePdPt', '1_th_calculation', ))
    # comp_ads_data = haa.get_comp_ads_data(
    #     comp_dir=os.path.join(
    #         working_dir, 'NiCoFePdPt'), read_site=True)
    # all_ads_data = haa.get_all_ads_data(working_dir = working_dir)
    # haa.get_all_ads_data_per_element(write_per_element_results=True)
    haa.get_all_ads_data_per_site()
