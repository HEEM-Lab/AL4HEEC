# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:04:39 2024

@author: ZHANGJUN

This script is used to get ΔE(H*), which is calculated by: ΔE(H*) = E_surface+H - E_surface - 1/2E_H2
"""

import json
import os
import numpy as np
from ase.io import read
import pandas as pd
from zjpf import pad_dict_list

E_H2 = -6.766271 # eV
G_H2 = -6.810652 # eV
site_num = 48
element_list = ['Ru', 'Rh', 'Pd', 'Ir', 'Pt']

def get_delta_E(E_surf_H, E_surf):
    return E_surf_H - E_surf - 0.5*E_H2

def get_d_G_H(E_ads, E_surf):
    return E_ads - E_surf - -6.805418/2+0.141308467625899

def find_H_neighbors(atoms, H_index=[-1], cutoff=2.0):
    dists = atoms.get_distances(-1,
                                range(len(atoms)-1),
                                mic=True)
    neighbor = np.where(dists<cutoff)[0]
    if len(neighbor) > 1: print('Warning!!! More than one sites are found.')
    return neighbor

# load surface energies and d-band centers
with open('surface_total_energies.json', 'r') as f:
    surf_toten = json.load(f)
# with open('d-band_centers.json', 'r') as f:
#     d_band_centers = json.load(f)

df_all = pd.DataFrame()
# ele_results = {k:[] for k in element_list}
ele_results = {}
# delta_E_json = {f'structure_{k}':{f'surface_{s}':{} for s in list(range(6))} for k in list(range(1,11))}

for structure in range(4): # 10 independent random structures
    for surface in range(6): # we have 6 surfaces for each structure
        for site_i in range(site_num):
            print(structure, surface, site_i)
            # read
            try:
                fname = os.path.join(str(structure), str(surface), str(site_i), 'OUTCAR')
                atoms = read(fname)
            except: # FileNotFoundError:
                print('OUTCAR file not found: ', fname)
                continue
            neighbors = find_H_neighbors(atoms)

            # collect
            E_surf_H = atoms.get_potential_energy()
            E_surf = surf_toten[f'structure_{structure}'][f'surface_{surface}']
            # delta_E = get_delta_E(E_surf_H, E_surf)
            delta_G = get_d_G_H(E_surf_H, E_surf)
            # d_band_center_list = [d_band_centers[f'structure_{structure}'][f'surface_{surface}'][f'site_{n}'] for n in neighbors]

            # arrange
            eles = [atoms[n].symbol for n in neighbors]
            eles.sort()
            eles = '-'.join(eles)

            df = pd.DataFrame(data=[[structure, surface, neighbors,
                                     eles,
                                     delta_G]],
                              columns=['Structure', 'Surface', 'Site',
                                       'Element',
                                       'ΔE(H*)'])
            df_all = pd.concat([df_all, df])
            try:
                ele_results[eles].append(delta_G)
            except KeyError:
                ele_results[eles] = [delta_G]
            site = f'site_{"-".join([str(x) for x in neighbors])}'
            # delta_E_json[f'structure_{structure}'][f'surface_{surface}'][site] = delta_E

            # for ele,d,s in zip(eles, d_band_center_list, neighbors):
            #     df = pd.DataFrame(data=[[structure, surface, s,
            #                              ele, d,
            #                              delta_E]],
            #                       columns=['Structure', 'Surface', 'Site',
            #                                'Element', 'd-band_center',
            #                                'ΔE(H*)'])
            #     df_all = pd.concat([df_all, df])
            #     ele_results[ele].append(delta_E)
            #     delta_E_json[f'structure_{structure}'][f'surface_{surface}'][f'site_{s}'] = delta_E

# save to disk
df_all.to_csv('delta_G_H.csv')
pd.DataFrame.from_dict(pad_dict_list(ele_results)).to_csv('delta_E_H_per_element.csv')
# with open('delta_E_H.json', 'w') as f:
#     json.dump(delta_E_json, f, indent=2)

df = pd.read_csv('delta_E_H_per_element.csv')
print('Average')
print('==============================')
print(df.mean())
print('Std')
print('==============================')
print(df.std(ddof=0))
print('# DDOF: 0')
