# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:43:32 2024

@author: ZHANGJUN
"""

import os
import pandas as pd

formulas_old = []

for l in [1,  1.5, 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]:
    fname = f'average_delta_G_loop_{l}.csv'
    data = pd.read_csv(os.path.join('average_delta_G_files', fname))
    formulas = data['formula']
    replica = [True if x in formulas_old else False for x in formulas]
    data['replica'] = replica
    data = data.sort_values('replica', axis=0)
    data.to_csv(f'average_delta_G_loop_{l}_with_replica.csv')

    formulas_old.extend(formulas)


