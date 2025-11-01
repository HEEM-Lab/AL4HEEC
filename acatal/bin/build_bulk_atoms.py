# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:51:48 2025

@author: ZHANGJUN
"""

import sys
from ase.build import sort
from agat.lib.high_throughput_lib import get_ase_atom_from_formula, get_v_per_atom

chemical_formula = sys.argv[1]
v_per_atom = get_v_per_atom(chemical_formula)
atoms = get_ase_atom_from_formula(chemical_formula, v_per_atom)
atoms = sort(atoms)
atoms.write('POSCAR')
