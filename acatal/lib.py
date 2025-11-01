# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 21:07:40 2023

@author: ZHANG Jun
"""

import numpy as np

def get_d_G_H(E_ads, E_surf):
    return E_ads - E_surf - -6.805418/2+0.141308467625899

def increamental_mean(a_list: list):
    new_list = []
    for i,a in enumerate(a_list):
        new_list.append(np.mean(a_list[0:i+1]))
    return new_list

def increamental_std(a_list: list):
    new_list = []
    for i,a in enumerate(a_list):
        new_list.append(np.std(a_list[0:i+1]))
    return new_list

def pad_dict_list(dict_list, padel):
    ''' https://stackoverflow.com/questions/40442014/pandas-valueerror-arrays-must-be-all-same-length '''
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list

