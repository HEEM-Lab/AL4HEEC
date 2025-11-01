# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:37:44 2023

@author: 18326
"""

import os
import torch.nn as nn

default_train_config = {
    'verbose': 1, # `0`: no train and validation output; `1`: Validation and test output; `2`: train, validation, and test output.
    # 'dataset_path': os.path.join(self.wd, '0_build_graphs', 'concated_graphs.bin'),
    # 'model_save_dir': os.path.join(self.wd, '1_train_agat_model', 'agat_model'),
    'epochs': 3000,
    # 'output_files': os.path.join(self.wd, '1_train_agat_model', 'agat_train_out_file'),
    'device': 'cuda:0',
    'validation_size': 0.15,
    'test_size': 0.15,
    'early_stop': True,
    'stop_patience': 300,
    'gat_node_dim_list': [6, 100, 100, 100],
    'head_list': ['mul', 'div', 'free'],
    'energy_readout_node_list': [300, 300, 100, 50, 30, 10, 3, 1],
    'force_readout_node_list': [300, 300, 100, 50, 30, 10, 3],
    'stress_readout_node_list': [300,300, 6],
    'bias': True,
    'negative_slope': 0.2,
    'criterion': nn.MSELoss(),
    'a': 1.0,
    'b': 1.0,
    'c': 0.0,
    'optimizer': 'adam', # Fix to adam.
    'learning_rate': 0.0001,
    'weight_decay': 0.0, # weight decay (L2 penalty)
    'batch_size': 64,
    'val_batch_size': 400,
    'mask_fixed': False,
    'tail_readout_no_act': [3,3,1],
    # 'adsorbate': False, #  or not when building graphs.
    'adsorbate_coeff': 20.0, # indentify and specify the importance of adsorbate atoms with respective to surface atoms. zero for equal importance.
    'transfer_learning': False
    }

default_ase_calculator_config = {'fmax'             : 0.1, # 0.05 can be better sometimes.
                                 'steps'            : 200,
                                 'maxstep'          : 0.05,
                                 'restart'          : None,
                                 'restart_steps'    : 0,
                                 'perturb_steps'    : 0,
                                 'perturb_amplitude': 0.05,
                                 'out'              : None,
                                 'device'           : 'cuda'}

default_high_throughput_config = {
        'formula_file': 'formulas.txt',
        'model_save_dir': 'agat_model',
        'graph_build_scheme_dir': '.',
        'opt_config': default_ase_calculator_config,
        'calculation_index'    : '0', # sys.argv[1],
        'fix_all_surface_atom' : False,
        'fix_H_x_y'            : False, # fix x and y of H atoms. If you want to use this tag with cuda, you need to modify ase/constraints.py#L889-895. See below:
'''
    def adjust_forces(self, atoms: Atoms, forces):
        # Forces are covariant to the coordinate transformation,
        # use the inverse transformations
        forces = forces.cpu().numpy()
        cell = atoms.cell
        scaled_forces = cell.cartesian_positions(forces[self.index])
        scaled_forces *= -(self.mask - 1)
        forces[self.index] = cell.scaled_positions(scaled_forces)
        forces = torch.tensor(forces, device='cuda')
'''

        'remove_bottom_atoms'  : False,
        'save_trajectory'      : False,
        'adsorbates'           : ['H'],
        'sites'                : ['bridge'],
        'dist_from_surf'       : 1.7,
        'using_template_bulk_structure': False,
        # 'graph_build_scheme_dir': os.path.join(self.config['binary_graphs_dir']),
        'device': 'cuda' # in our test results, the A6000 is about * times faster than EPYC 7763.
        }

default_active_learing_config = {
    'debug_mode': False,
    'loop_num': 0,
    'raw_dataset_dir': 'high_throughput_dft',
    'binary_graphs_dir': '0_binary_graphs',
    'agat_model_dir': '0_agat_model',
    }

default_graph_build_scheme_deprecated = {
    "species": [
        "H",
        "Ni",
        "Co",
        "Fe",
        "Pd",
        "Pt"
    ],
    "path_file": "paths.log",
    "build_properties": {
        "energy": True,
        "forces": True,
        "cell": True,
        "cart_coords": False,
        "frac_coords": True,
        "constraints": True,
        "stress": True,
        "distance": True,
        "direction": True,
        "path": False
    },
    "topology_only": False,
    "dataset_path": "dataset",
    "mode_of_NN": "ase_dist",
    "cutoff": 5.0,
    "load_from_binary": False,
    "num_of_cores": 32,
    "super_cell": False,
    "has_adsorbate": True,
    "keep_readable_structural_files": False,
    "mask_similar_frames": False,
    "mask_reversed_magnetic_moments": -0.5,
    "energy_stride": 0.05,
    "scale_prop": False
}
