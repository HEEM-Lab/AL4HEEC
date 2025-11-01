# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:32:43 2024

@author: ZHANGJUN
"""

import numpy as np
import pandas as pd
from acatal.tool import concentration2formula

fname = r'D:\OneDrive\Projects\ALAGAT_HER\data\20240924 data process\all_dft_concentrations\all_cons.csv'
df = pd.read_csv(fname)

nbins = 50+1
bin_size = 1/(nbins-1)
bins = np.linspace(0.0, 1.0, num=nbins)

count = []
for x1 in bins:
    x2_end = 1-x1
    for x2 in [xx for xx in bins if xx <= x2_end]:
        x3_end = 1-x1-x2
        for x3 in [xx for xx in bins if xx <= x3_end]:
            x4_end = 1-x1-x2-x3
            for x4 in [xx for xx in bins if xx <= x4_end]:
                x5 = 1.0 - x1 - x2 - x3 - x4
                count.append([x1, x2, x3, x4, x5, 0.0])
count = np.array(count)

num = len(df)
# count
for di, d in df.iterrows():
    print(di/num)
    for ci,c in enumerate(count):
        if d['Ni'] > c[0] - bin_size and d['Ni'] < c[0] + bin_size and\
           d['Co'] > c[1] - bin_size and d['Co'] < c[1] + bin_size and\
           d['Fe'] > c[2] - bin_size and d['Fe'] < c[2] + bin_size and\
           d['Pd'] > c[3] - bin_size and d['Pd'] < c[3] + bin_size and\
           d['Pt'] > c[4] - bin_size and d['Pt'] < c[4] + bin_size:
               count[ci][5] += 1.0
               continue

np.savetxt(r'D:\OneDrive\Projects\ALAGAT_HER\data\20240924 data process\all_dft_concentrations\con_count_21bins.csv',
           count, fmt='%.4f', delimiter=',')
###############
count = np.loadtxt(r'D:\OneDrive\Projects\ALAGAT_HER\data\20240924 data process\all_dft_concentrations\con_count_51bins.csv',
                   dtype=float, delimiter=',')
fre_id = np.where(count[:,5] > 6.9)
fre_count = count[fre_id]
formulas = concentration2formula(fre_count)

np.savetxt(r'D:\OneDrive\Projects\ALAGAT_HER\data\20240924 data process\all_dft_concentrations\frequent_cons_51bins.csv',
           fre_count, fmt='%.4f', delimiter=',')
np.savetxt(r'D:\OneDrive\Projects\ALAGAT_HER\data\20240924 data process\all_dft_concentrations\frequent_formulas_51bins.txt',
           formulas, fmt='%s')
