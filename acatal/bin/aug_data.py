# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 20:44:46 2023

@author: ZHANG Jun
"""

import agat
import acatal
from agat.app.cata import HtDftAds
import os
import numpy as np

HA = HtDftAds(calculation_index=0, sites='bridge', include_bulk_aimd=False,
              include_surface_aimd=False, include_adsorption_aimd=False,
              random_samples=1, vasp_bash_path=os.path.join(os.path.dirname(acatal.__file__), 'bin', 'vasp_run.sh'))

formula = os.path.basename(os.path.abspath('.'))
HA.run(formula)

# formulas = np.loadtxt('formula.txt')
# curdir = os.getcwd()

# for f in formulas:
#     os.mkdir(f)
#     os.chdir(f)
#     HA.run(f)
#     os.chdir(curdir)
