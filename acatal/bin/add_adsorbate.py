# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:31:10 2024

@author: 18326
"""

import os
from agat.app.cata import  AddAtoms
from agat.lib.adsorbate_poscar import adsorbate_poscar

adder = AddAtoms(os.path.join('acatal', 'test', 'POSCAR_surf_CoFeNiPdPt'),
                     species='H-H_di',
                     sites='disigma',
                     dist_from_surf=1.6,
                     num_atomic_layer_along_Z=6)
num_sites = adder.write_file_with_adsorption_sites(adsorbate_poscar,
                                                   calculation_index=0)

adder = AddAtoms(os.path.join('acatal', 'test', 'POSCAR_surf_CoFeNiPdPt'),
                     species='H2_di',
                     sites='disigma',
                     dist_from_surf=2.44,
                     num_atomic_layer_along_Z=6)
num_sites = adder.write_file_with_adsorption_sites(adsorbate_poscar,
                                                   calculation_index=1)
