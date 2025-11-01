# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:07:01 2023

@author: 18326
"""
from agat.lib.high_throughput_lib import get_concentration_from_ase_formula

def concentration2formula(cons, totoal_bulk_atoms=96):
    formulas = []
    for con in cons:
        Ni_num = round(con[0]*96)
        Co_num = round(con[1]*96)
        Fe_num = round(con[2]*96)
        Pd_num = round(con[3]*96)
        Pt_num = totoal_bulk_atoms - Ni_num - Co_num - Fe_num - Pd_num
        formula = f'Ni{Ni_num}Co{Co_num}Fe{Fe_num}Pd{Pd_num}Pt{Pt_num}'
        formulas.append(formula)
        # print(formula, end=' ')

    return list(set(formulas))

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

def con_dict2con_list(con_dict):
    keys = ['Ni', 'Co', 'Fe', 'Pd', 'Pt']
    con_list = []
    num_cons = len(con_dict['Ni'])
    for i in range(num_cons):
        con = []
        for k in keys:
            con.append(con_dict[k][i])
        con_list.append(con)
    return con_list
