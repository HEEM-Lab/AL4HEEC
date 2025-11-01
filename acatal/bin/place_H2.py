# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:55:33 2024

@author: 18326
"""

from ase.io import read, write

fname = 'CONTCAR.txt'
dist = 2.4 # angstrom above the surface

atoms = read(fname)

positions = atoms.get_positions()

C1 = positions[-1]
C2 = positions[-2]
