#!/bin/usr/env python

# This script is used to get the d-band center of each element under this directory.

import os
from ase.io import read
from zjpf import pad_dict_list
import pandas as pd

d_band_name = 'd_band_center.txt'

result = {'Ni': [],
        'Co': [],
        'Fe': [],
        'Pd': [],
        'Pt': []}

for d in range(1, 11):
    print(d)
    fname_pos = os.path.join(str(d), 'dos', 'POSCAR')
    fname_d = os.path.join(str(d), 'dos', d_band_name)
    atoms = read(fname_pos)
    symbols = atoms.get_chemical_symbols()
    f = open(fname_d, 'r')

    for s in symbols:
        l = f.readline()
        l = l.split()
        # print(l)
        e_d = float(l[5])
        result[s].append(e_d)


    f.close()

result = pad_dict_list(result)

df = pd.DataFrame.from_dict(result)

df.to_csv('d_band_center.csv')
print(df)
