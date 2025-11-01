# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:37:01 2024

@author: ZHANGJUN

https://www.lbma.org.uk/prices-and-data/precious-metal-prices#/
"""

import numpy as np
from ase.data import atomic_numbers, atomic_masses
from acatal.tool import formula2concentration

lbma_price = { # unit: US dollars per fine troy ounce
    'Pt': 980,
    'Pd': 1036
    }

def per_fine_troy_ounce_2_per_kg(ounce):
    return ounce / 0.0311035

def get_atomic_mass(ele: str):
    return atomic_masses[atomic_numbers[ele]]

def get_weight_concentration(molar_con: dict):
    weight_per_formula_unit = [c[0]*get_atomic_mass(k) for k,c in molar_con.items()]
    weight_sum = np.sum(weight_per_formula_unit)
    return {k: weight_per_formula_unit[i]/weight_sum for i,k in enumerate(molar_con)}

def get_precious_metal_price_per_kg_formula(*formulas):
    costs = []
    for f in formulas:
        cons = formula2concentration([f])
        weight_cons = get_weight_concentration(cons)
        cost = weight_cons['Pd'] * per_fine_troy_ounce_2_per_kg(lbma_price['Pd']) + \
               weight_cons['Pt'] * per_fine_troy_ounce_2_per_kg(lbma_price['Pt'])
        costs.append(cost)
    return costs

formulas = '''
Ni13Co17Fe15Pd37Pt14
Ni18Co10Fe23Pd16Pt29
Ni13Co9Fe13Pd52Pt9
Ni10Co14Fe12Pd32Pt28
Ni6Co8Fe11Pd7Pt64
Ni2Co8Fe4Pd79Pt3
Pt
NiCoFePdPt
Ni6Co7Fe7Pd40Pt40
Ni26Co27Fe27Pd10Pt10
'''.split()

formulas = '''
Ni8Co10Fe13Pd8Pt57
Ni8Co12Fe17Pd8Pt51
Ni15Co15Fe19Pd17Pt30
Ni6Co8Fe12Pd6Pt64
Ni12Co12Fe21Pd8Pt43
'''.split()

costs = get_precious_metal_price_per_kg_formula(*formulas)
for c in costs:
    print('{0:0.0f}'.format(c))

for f in formulas:
    wcs = get_weight_concentration(formula2concentration([f]))
    for wc in wcs:
        print('{0:0.2f}'.format(wcs[wc]), end='\t')
    print()
