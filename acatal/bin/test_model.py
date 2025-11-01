# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:01:36 2024

@author: ZHANGJUN
"""

import os
import json
import time

from agat.lib.model_lib import load_model
from agat.data.build_dataset import CrystalGraph
import torch

model_save_dir = os.path.join('agat_model')
graph_build_scheme_path = os.path.join('acatal', 'test', 'agat_model', 'graph_build_scheme.json')

with open(graph_build_scheme_path, 'r') as f:
    graph_build_scheme = json.load(f)

fname = os.path.join('acatal', 'test', 'POSCAR_NiCoFePdPt')

agat_model = load_model(model_save_dir, 'cpu')

graph_build_scheme['topology_only'] = True
cg = CrystalGraph(**graph_build_scheme)

bg, _ = cg.get_graph(fname)

with torch.no_grad():
    energy_pred, force_pred, stress_pred = agat_model.forward(bg)
    print(energy_pred)