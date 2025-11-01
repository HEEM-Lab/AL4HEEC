# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:17:36 2023

@author: ZHANG Jun
"""

import platform
import numpy as np
import os
import stat
import shutil

from ase.formula import Formula
from ase.io import read
from ase.lattice.cubic import FaceCenteredCubic
from ase.data import covalent_radii, atomic_numbers

def perturb_positions(atoms, amplitude=0.1):
    calculator  = atoms.get_calculator()
    atoms       = atoms.copy()
    atoms.set_calculator(calculator)
    posistions  = atoms.arrays['positions']
    num_atoms   = len(atoms)
    increment   = np.clip(np.random.normal(0.0, amplitude / 3,  size=(num_atoms,3)), -amplitude, amplitude) # np.random.uniform(-amplitude, amplitude, (num_atoms,3))
    constraints = atoms.constraints
    if len(constraints) > 0:
        increment[constraints[0].index] = 0.0
    posistions += increment
    atoms.set_positions(posistions)
    return atoms

def scale_atoms(atoms, scale_factor=1.0):
    calculator = atoms.get_calculator()
    new_atoms  = atoms.copy()
    new_atoms.set_calculator(calculator)
    cell        = new_atoms.get_cell()
    frac_coords = new_atoms.get_scaled_positions()
    new_cell    = cell * scale_factor
    new_atoms.set_cell(new_cell)
    new_atoms.set_scaled_positions(frac_coords)
    return new_atoms

def get_concentration_from_ase_formula(formula):
    f_dict = Formula(formula).count()
    tot = np.sum(list(f_dict.values()))
    c_dict = {k: v/tot for k, v in f_dict.items()}
    c_dict = {k:c_dict[k] for k in c_dict if c_dict[k] > 0}
    return c_dict

def get_v_per_atom(chemical_formula):
    frac_dict = get_concentration_from_ase_formula(chemical_formula)
    frac_dict = {k:(0.0 if k not in frac_dict else frac_dict[k]) for k in ['Ni', 'Co', 'Fe', 'Pd', 'Pt']}
    return -282.7957531391954 * (frac_dict['Ni'] + frac_dict['Co'] + frac_dict['Fe'])\
        - 278.79605077419797 * frac_dict['Pd'] - 278.6228860885035 * frac_dict['Pt']\
            + 293.66128761358624

def get_ase_atom_from_formula(chemical_formula, v_per_atom=None):
    # interpret formula
    atomic_fracions = get_concentration_from_ase_formula(chemical_formula)
    elements = [x for x in atomic_fracions]
    element_number = [atomic_numbers[x] for x in elements]
    mean_radii       = np.sum([covalent_radii[n] * atomic_fracions[e] for n, e in zip(element_number, elements)]) #

    Pt_radii           = covalent_radii[78]
    latticeconstant    = mean_radii / Pt_radii * 3.92
    atoms              = FaceCenteredCubic('Pt', directions=[[1,-1,0], [1,1,-2], [1,1,1]], size=(4, 2, 2), latticeconstant=latticeconstant, pbc=True)
    total_atom         = len(atoms)
    num_atom_list      = np.array(list(atomic_fracions.values())) * total_atom
    num_atom_list      = np.around(num_atom_list, decimals=0)
    total_tmp          = np.sum(num_atom_list)
    deviation          = total_atom - total_tmp
    num_atom_list[np.random.randint(len(elements))] += deviation

    # shuffle atoms
    ase_number    = []
    for i_index, i in enumerate(num_atom_list):
        for j in range(int(i)):
            ase_number.append(element_number[i_index])
    np.random.shuffle(ase_number)
    atoms.set_atomic_numbers(ase_number)

    # scale atoms
    if isinstance(v_per_atom, (float, int)):
        volume = atoms.cell.volume
        volume_per_atom = volume / len(atoms)
        volume_ratio = v_per_atom / volume_per_atom
        scale_factor = pow(volume_ratio, 1/3)
        atoms = scale_atoms(atoms, scale_factor)
    return atoms

def get_ase_atom_from_formula_template(chemical_formula, v_per_atom=None,
                                       template_file='POSCAR_temp'):
    # interpret formula
    # the template file should be a bulk structure
    atomic_fracions    = get_concentration_from_ase_formula(chemical_formula)
    elements           = [x for x in atomic_fracions]
    element_number     = [atomic_numbers[x] for x in elements]
    atoms              = read(template_file)
    total_atom         = len(atoms)
    num_atom_list      = np.array(list(atomic_fracions.values())) * total_atom
    num_atom_list      = np.around(num_atom_list, decimals=0)
    total_tmp          = np.sum(num_atom_list)
    deviation          = total_atom - total_tmp
    num_atom_list[np.random.randint(len(elements))] += deviation

    # shuffle atoms
    ase_number    = []
    for i_index, i in enumerate(num_atom_list):
        for j in range(int(i)):
            ase_number.append(element_number[i_index])
    np.random.shuffle(ase_number)
    atoms.set_atomic_numbers(ase_number)

    # scale atoms
    if isinstance(v_per_atom, (float, int)):
        volume = atoms.cell.volume
        volume_per_atom = volume / len(atoms)
        volume_ratio = v_per_atom / volume_per_atom
        scale_factor = pow(volume_ratio, 1/3)
        atoms = scale_atoms(atoms, scale_factor)
    return atoms

def run_vasp(vasp_bash_path):
    """

    :raises ValueError: VASP can only run on a Linux platform


    .. warning:: Setup your own VAPS package and Intel libraries before using this function.

    """

    # os_type = platform.system()
    # if not os_type == 'Linux':
    #     raise ValueError(f'VASP can only be executed on Linux OS, instead of {os_type}.')
    # shell_script = '''#!/bin/bash
# . /home/jzhang/software/intel/oneapi/setvars.sh
# mpirun /home/jzhang/software/vasp/vasp_std
    # '''

    # with open('vasp_run.sh', 'w') as f:
    #     f.write(shell_script)
    #
    # os.chmod('vasp_run.sh', stat.S_IRWXU)

    shutil.copyfile(vasp_bash_path, os.path.join('./vasp_run.sh'))
    os.chmod('vasp_run.sh', stat.S_IRWXU)
    os.system('./vasp_run.sh')

import json

# from ase.optimize import BFGS
from ase.io import write
from ase.build import add_vacuum, sort
from ase.constraints import FixAtoms, FixScaled

# from agat.app import AgatCalculator
from agat.app.cata.generate_adsorption_sites import AddAtoms
from agat.lib.file_lib import file_exit
from agat.lib.model_lib import config_parser
from agat.default_parameters import default_high_throughput_config
from agat.lib.adsorbate_poscar import adsorbate_poscar

from acatal.ase_torch.ase_dyn import MDMinTorch
from acatal.ase_torch.calculators import AgatCalculatorFastGraph

class HtAds(object): # modified based on https://github.com/jzhang-github/AGAT/blob/main/agat/app/cata/generate_adsorption_sites.py
    def __init__(self, **hp_config):
        self.hp_config = {**default_high_throughput_config, **config_parser(hp_config)}
        self.device = self.hp_config['device']
        # model save path
        model_save_dir = self.hp_config['model_save_dir']

        # instantiate a calculator
        self.calculator=AgatCalculatorFastGraph(model_save_dir,
                                                self.hp_config['graph_build_scheme_dir'],
                                                device=self.device)

    def geo_opt(self, atoms_with_calculator, **kwargs):
        calculator = atoms_with_calculator.get_calculator()
        atoms  = atoms_with_calculator.copy()
        atoms.set_calculator(calculator)

        config = {**self.hp_config['opt_config'], **kwargs}

        if isinstance(config["out"], type(None)):
            logfile    = '-'
            trajectory = None

        else:
            logfile    = f'{config["out"]}.log'
            trajectory = f'{config["out"]}.traj'

        #######################################################################
        # Ignore a bug here. For now, the ase.io is incompatible with torch tensor.
        trajectory = None
        #######################################################################

        force_opt, energy_opt, atoms_list = [], [], []
        for i in range(config["perturb_steps"]+1):
            atoms.calc.ag.reset() # reset graph build parameters.
            dyn = MDMinTorch(atoms,
                       logfile=logfile,
                       trajectory=trajectory,
                       restart=config["restart"],
                       maxstep=config["maxstep"],
                       device=self.device)

            return_code  = dyn.run(fmax=config["fmax"], steps=config["steps"])
            restart_step = 1
            while not return_code and restart_step < config["restart_steps"]:
                atoms.calc.ag.reset() # reset graph build parameters.
                restart_step += 1
                maxstep_tmp = config["maxstep"]/2**restart_step
                dyn = MDMinTorch(atoms,
                           logfile=logfile,
                           trajectory=trajectory,
                           restart=config["restart"],
                           maxstep=maxstep_tmp,
                           device=self.device)
                return_code  = dyn.run(fmax=config["fmax"], steps=config["steps"])
            force_opt.append(atoms.get_forces().cpu().numpy())
            energy_opt.append(atoms.get_total_energy()[0].item()) # no difference between `atoms.get_total_energy` and `atoms.get_potential_energy`.
            atoms_list.append(atoms.copy())
            if config["perturb_steps"] > 0:
                atoms = self.perturb_positions(atoms, amplitude=config["perturb_amplitude"])
        argmin = np.argmin(energy_opt)
        energy, force, atoms = energy_opt[argmin], force_opt[argmin], atoms_list[argmin]
        force_max = np.linalg.norm(force, axis=1).max()
        return energy, force, atoms, force_max

    def ads_calc(self, formula, calculator, **kwargs):
        hp_config =  {**self.hp_config, **kwargs}

        if hp_config['save_trajectory']:
            out_dir = f'{hp_config["calculation_index"]}_th_calculation'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
        else:
            out_dir     = '.'
            outbasename = None

        # generate bulk structure
        chemical_formula = formula

        if self.hp_config['using_template_bulk_structure']:
            atoms = get_ase_atom_from_formula_template(chemical_formula,
                                                       get_v_per_atom(chemical_formula),
                                                       template_file='POSCAR_temp')
        else:
            atoms = get_ase_atom_from_formula(chemical_formula,
                                              v_per_atom=get_v_per_atom(chemical_formula))

        atoms.set_calculator(calculator)

        if hp_config['save_trajectory']:
            write(os.path.join(out_dir, 'POSCAR_bulk.gat'), atoms, format='vasp')
            outbasename = os.path.join(out_dir, 'bulk_opt')

        hp_config['opt_config']['out'] = outbasename

        energy_bulk, force_bulk, atoms_bulk, force_max_bulk = self.geo_opt(atoms,
                                                                           **hp_config['opt_config'])
        if hp_config['save_trajectory']:
            write(os.path.join(out_dir, 'CONTCAR_bulk.gat'), atoms_bulk)

        print('Bulk optimization done.')

        # add vacuum space and fix bottom atoms
        len_z = atoms_bulk.cell.array[2][2]
        atoms_bulk.positions += 1.3 # avoid PBC error
        atoms_bulk.wrap()
        c     = FixAtoms(indices=np.where(atoms_bulk.positions[:,2] < len_z / 2)[0])
        atoms_bulk.set_constraint(c)

        if hp_config['remove_bottom_atoms']:
            pop_list = np.where(atoms_bulk.positions[:,2] < 1.0)
            del atoms_bulk[pop_list]

        add_vacuum(atoms_bulk, 10.0)

        if hp_config['save_trajectory']:
            write(os.path.join(out_dir, 'POSCAR_surface.gat'), atoms_bulk)
            outbasename = os.path.join(out_dir, 'surface_opt')

        # surface optimization
        atoms_bulk.set_calculator(calculator)
        hp_config['opt_config']['out'] = outbasename

        energy_surf, force_surf, atoms_surf, force_max_surf = self.geo_opt(atoms_bulk,
                                                                           **hp_config['opt_config'])
        if hp_config['save_trajectory']:
            write(os.path.join(out_dir, 'CONTCAR_surface.gat'), atoms_surf)

        print('Surface optimization done.')

        if force_max_surf < hp_config['opt_config']['fmax']:
            if hp_config['fix_all_surface_atom']:
                c = FixAtoms(indices=[x for x in range(len(atoms_surf))])
                atoms_surf.set_constraint(c)
            write(f'POSCAR_surf_opt_{hp_config["calculation_index"]}.gat', sort(atoms_surf))

            # adsorbate_shift = {'bridge': 0.0, 'ontop': 0.35, 'hollow': -0.1}

            for ads in hp_config['adsorbates']:
                # generate adsorption configurations: OH adsorption
                adder     = AddAtoms(f'POSCAR_surf_opt_{hp_config["calculation_index"]}.gat',
                                     species=ads,
                                     sites=hp_config['sites'],
                                     dist_from_surf=hp_config['dist_from_surf'],
                                     num_atomic_layer_along_Z=6)
                all_sites = adder.write_file_with_adsorption_sites(adsorbate_poscar,
                                                                   calculation_index=hp_config['calculation_index'])

                # adsorption optimization
                ase_atoms = [read(f'POSCAR_{hp_config["calculation_index"]}_{x}') for x in range(all_sites)]
                [os.remove(f'POSCAR_{hp_config["calculation_index"]}_{x}') for x in range(all_sites)]
                # ase_atoms = ase_atoms[0:3] # !!!

                energy_ads_list, converge_stat = [], []
                for i, ads_atoms in enumerate(ase_atoms):
                    file_exit()

                    # fix x and y of H atoms if necessary
                    if hp_config['fix_H_x_y']:
                        c= FixScaled(a=[atom.index for atom in ads_atoms if atom.symbol == 'H'],
                                     mask=[True, True, False]) # (False: unfixed, True: fixed)
                        ads_atoms.constraints.append(c)

                    if hp_config['save_trajectory']:
                        write(os.path.join(out_dir, f'POSCAR_{ads}_ads_{hp_config["calculation_index"]}_{i}.gat'),
                              ads_atoms)
                        outbasename = os.path.join(out_dir, f'adsorption_{ads}_opt_{i}')

                    hp_config['opt_config']['out'] = outbasename
                    ads_atoms.set_calculator(calculator)
                    energy_ads, force_ads, atoms_ads, force_max_ads = self.geo_opt(ads_atoms, **hp_config['opt_config'])

                    if hp_config['save_trajectory']:
                        write(os.path.join(out_dir, f'CONTCAR_{ads}_ads_{hp_config["calculation_index"]}_{i}.gat'),
                              atoms_ads)

                    energy_ads_list.append(energy_ads)

                    if force_max_ads > hp_config['opt_config']['fmax']:
                        converge_stat.append(0.0)
                    else:
                        converge_stat.append(1.0)

                energy_surf_list = np.array([energy_surf] * len(energy_ads_list))
                energy_ads_list = np.array(energy_ads_list)

                out = np.vstack([energy_ads_list, energy_surf_list, converge_stat]).T
                if not np.shape(out)[0]:
                    np.savetxt(f'ads_surf_energy_{ads}_{hp_config["calculation_index"]}.txt',
                               out, fmt='%f')

    def run(self, formula, **kwargs):
        """

        :param formula: Input chemical formula
        :type formula: str
        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        with open('high_throughput_config.json', 'w') as f:
            json.dump(self.hp_config, f, indent=4)
        self.ads_calc(formula, self.calculator, **kwargs)
