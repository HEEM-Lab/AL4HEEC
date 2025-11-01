# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:56:28 2023

@author: ZHANG Jun
"""

from ase.calculators.calculator import Calculator
from agat.data.build_dataset import CrystalGraph
import json
from ase.calculators.calculator import Calculator
from agat.lib.model_lib import load_model
import torch
import os
from acatal.ase_torch.ase_torch_graph import AseGraph


class AgatCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters  = { }
    ignored_changes = set()
    def __init__(self, model_save_dir, graph_build_scheme_dir, device = 'cuda',
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        # self.atoms = None  # copy of atoms object from last calculation
        # self.results = {}  # calculated properties (energy, forces, ...)
        # self.parameters = None  # calculational parameters
        # self._directory = None  # Initialize

        self.model_save_dir = model_save_dir
        self.graph_build_scheme_dir = graph_build_scheme_dir
        self.device = device

        self.model = load_model(self.model_save_dir, self.device)
        self.graph_build_scheme = self.load_graph_build_scheme(self.graph_build_scheme_dir)

        build_properties = {'energy': False, 'forces': False, 'cell': False,
                            'cart_coords': False, 'frac_coords': False, 'path': False,
                            'stress': False} # We only need the topology connections.
        self.graph_build_scheme['build_properties'] = {**self.graph_build_scheme['build_properties'],
                                                       **build_properties}
        self.cg = CrystalGraph(**self.graph_build_scheme)

        # self.force_log = []
        # self.energy_log = []

    # def set(self):
    #     pass

    def load_graph_build_scheme(self, path):
        """ Load graph building scheme. This file is normally saved when you build your dataset.

        :param path: Directory for storing ``graph_build_scheme.json`` file.
        :type path: str
        :return: A dict denotes how to build the graph.
        :rtype: dict

        """
        json_file  = os.path.join(path, 'graph_build_scheme.json')
        assert os.path.exists(json_file), f"{json_file} file dose not exist."
        with open(json_file, 'r') as jsonf:
            graph_build_scheme = json.load(jsonf)
        return graph_build_scheme

    def calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc']):
        """

        :param atoms: ase.atoms object, defaults to None
        :type atoms: ase.atoms, optional
        :param properties: calculated properties, defaults to None
        :type properties: none, optional
        :param system_changes: DESCRIPTION, defaults to ['positions', 'numbers', 'cell', 'pbc']
        :type system_changes: TYPE, optional
        :return: calculated results
        :rtype: dict

        """

        # print('call calculate')
        if atoms is not None:
            self.atoms = atoms.copy()

        if properties is None:
            properties = self.implemented_properties

        # read graph
        graph, info = self.cg.get_graph(atoms)
        graph = graph.to(self.device)

        with torch.no_grad():
            energy_pred, force_pred, stress_pred = self.model.forward(graph)

        self.results = {'energy': energy_pred[0].item() * len(atoms),
                        'forces': force_pred.cpu().numpy(),
                        'stress': stress_pred[0].cpu().numpy()}

        # self.force_log.append(force_pred.cpu().numpy())
        # self.energy_log.append(energy_pred[0].item() * len(atoms))

class AgatCalculatorFastGraph(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters  = { }
    ignored_changes = set()
    def __init__(self, model_save_dir, graph_build_scheme_dir, device = 'cuda',
                 **kwargs):
        Calculator.__init__(self, **kwargs)

        self.model_save_dir = model_save_dir
        self.graph_build_scheme_dir = graph_build_scheme_dir
        self.device = torch.device(device)

        self.model = load_model(self.model_save_dir, self.device)
        self.graph_build_scheme = self.load_graph_build_scheme(self.graph_build_scheme_dir)

        build_properties = {'energy': False, 'forces': False, 'cell': False,
                            'cart_coords': False, 'frac_coords': False, 'path': False,
                            'stress': False} # We only need the topology connections.
        self.graph_build_scheme['build_properties'] = {**self.graph_build_scheme['build_properties'],
                                                       **build_properties}
        self.graph_build_scheme['device'] = self.device
        self.ag = AseGraph(**self.graph_build_scheme)

        # self.force_log = []
        # self.energy_log = []

    def load_graph_build_scheme(self, path):
        """ Load graph building scheme. This file is normally saved when you build your dataset.

        :param path: Directory for storing ``graph_build_scheme.json`` file.
        :type path: str
        :return: A dict denotes how to build the graph.
        :rtype: dict

        """
        json_file  = os.path.join(path, 'graph_build_scheme.json')
        assert os.path.exists(json_file), f"{json_file} file dose not exist."
        with open(json_file, 'r') as jsonf:
            graph_build_scheme = json.load(jsonf)
        return graph_build_scheme

    def calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc']):
        """

        :param atoms: ase.atoms object, defaults to None
        :type atoms: ase.atoms, optional
        :param properties: calculated properties, defaults to None
        :type properties: none, optional
        :param system_changes: DESCRIPTION, defaults to ['positions', 'numbers', 'cell', 'pbc']
        :type system_changes: TYPE, optional
        :return: calculated results
        :rtype: dict

        """

        # print('AGAT forward in the AgatCalculatorFastGraph.')

        # atoms_tmp = atoms.copy()
        if atoms is not None:
            self.atoms = atoms.copy()

        if properties is None:
            properties = self.implemented_properties

        # read graph
        graph = self.ag.get_graph(atoms)
        # graph = self.ag.build(atoms_tmp)
        # graph = graph.to(self.device)

        with torch.no_grad():
            energy_pred, force_pred, stress_pred = self.model.forward(graph)

        self.results = {'energy': energy_pred * len(atoms),
                        'forces': force_pred,
                        'stress': stress_pred[0]}

        # self.force_log.append(force_pred.cpu().numpy())
        # self.energy_log.append(energy_pred[0].item() * len(atoms_tmp))

class AgatCalculatorFastGraphNumpy(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters  = { }
    ignored_changes = set()
    def __init__(self, model_save_dir, graph_build_scheme_dir, device = 'cuda',
                 **kwargs):
        Calculator.__init__(self, **kwargs)

        self.model_save_dir = model_save_dir
        self.graph_build_scheme_dir = graph_build_scheme_dir
        self.device = torch.device(device)

        self.model = load_model(self.model_save_dir, self.device)
        self.graph_build_scheme = self.load_graph_build_scheme(self.graph_build_scheme_dir)

        build_properties = {'energy': False, 'forces': False, 'cell': False,
                            'cart_coords': False, 'frac_coords': False, 'path': False,
                            'stress': False} # We only need the topology connections.
        self.graph_build_scheme['build_properties'] = {**self.graph_build_scheme['build_properties'],
                                                       **build_properties}
        self.graph_build_scheme['device'] = self.device
        self.ag = AseGraph(**self.graph_build_scheme)

        # self.force_log = []
        # self.energy_log = []

    def load_graph_build_scheme(self, path):
        """ Load graph building scheme. This file is normally saved when you build your dataset.

        :param path: Directory for storing ``graph_build_scheme.json`` file.
        :type path: str
        :return: A dict denotes how to build the graph.
        :rtype: dict

        """
        json_file  = os.path.join(path, 'graph_build_scheme.json')
        assert os.path.exists(json_file), f"{json_file} file dose not exist."
        with open(json_file, 'r') as jsonf:
            graph_build_scheme = json.load(jsonf)
        return graph_build_scheme

    def calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc']):
        """

        :param atoms: ase.atoms object, defaults to None
        :type atoms: ase.atoms, optional
        :param properties: calculated properties, defaults to None
        :type properties: none, optional
        :param system_changes: DESCRIPTION, defaults to ['positions', 'numbers', 'cell', 'pbc']
        :type system_changes: TYPE, optional
        :return: calculated results
        :rtype: dict

        """

        # print('AGAT forward in the AgatCalculatorFastGraph.')

        if atoms is not None:
            self.atoms = atoms.copy()

        if properties is None:
            properties = self.implemented_properties

        # read graph
        graph = self.ag.get_graph(atoms)
        # graph = self.ag.build(atoms_tmp)
        # graph = graph.to(self.device)

        with torch.no_grad():
            energy_pred, force_pred, stress_pred = self.model.forward(graph)

        self.results = {'energy': energy_pred[0].item() * len(atoms),
                        'forces': force_pred.cpu().numpy(),
                        'stress': stress_pred[0].cpu().numpy()}

        # self.force_log.append(force_pred.cpu().numpy())
        # self.energy_log.append(energy_pred[0].item() * len(atoms_tmp))
