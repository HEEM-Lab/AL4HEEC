# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:14:29 2024

@author: ZHANGJUN
"""

import os
from ase.io import read


fname = 'paths.log'

root_dir = os.getcwd()

with open(fname, 'r') as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]

total = 0

for l in lines:
    os.chdir(l)

    try:
        outcar = read('XDATCAR', index=':')
        num = len(outcar)
    except:
        num = 0
    total += num
    print(total)
    os.chdir(root_dir)




total = 0
from agat.data import LoadDataset

fnames = os.listdir('.')
fnames = [x for x in fnames if x.split('.')[1] == 'bin']

for f in fnames:
    dataset = LoadDataset(f)
    num = len(dataset)
    total += num
    print(total)
