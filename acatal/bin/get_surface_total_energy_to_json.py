# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:29:03 2024

@author: ZHANGJUN

This script is used to get the total energy of surface structures after optimization.
"""

import os
import json
from ase.io import read

surface_toten = {}
for structure in range(4):
    surface_toten[f'structure_{structure}'] = {}
    for surf in range(6):
        fname = os.path.join(str(structure), str(surf), 'OUTCAR')
        atoms = read(fname)
        energy = atoms.get_potential_energy()
        surface_toten[f'structure_{structure}'][f'surface_{surf}'] = energy

with open('surface_total_energies.json', 'w') as f:
    json.dump(surface_toten, f, indent=2)
