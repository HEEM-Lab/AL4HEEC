# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:18:57 2023

@author: ZHANG Jun
"""

''' old school '''
# from agat.data import CrystalGraph
# from agat.default_parameters import default_data_config
# import time

# default_data_config['mode_of_NN'] = 'ase_dist'
# default_data_config['build_properties']['energy'] = False
# default_data_config['build_properties']['forces'] = False
# default_data_config['build_properties']['cell'] = False
# default_data_config['build_properties']['stress'] = False

# cg = CrystalGraph(**default_data_config)

# from ase.io import read
# ase_atoms = read('POSCAR')
# ase_atoms.write('POSCAR_50')
# ase_atoms.repeat((1,1,2)).write('POSCAR_100')
# ase_atoms.repeat((1,2,2)).write('POSCAR_200')
# ase_atoms.repeat((2,2,2)).write('POSCAR_400')
# ase_atoms.repeat((2,2,4)).write('POSCAR_800')
# ase_atoms.repeat((2,4,4)).write('POSCAR_1600')
# ase_atoms.repeat((4,4,4)).write('POSCAR_3200')

# if __name__ == '__main__':
#     for n in [50, 100, 200, 400, 800, 1600, 3200]:
#         start = time.time()
#         for i in range(30):
#             g = cg.get_graph(f'POSCAR_{n}')

#         print(time.time() - start)

''' New method '''
from agat.lib.model_lib import config_parser
from agat.data.atomic_feature import get_atomic_feature_onehot
from agat.default_parameters import default_data_config
import ase
from ase.io import read
import torch
import numpy as np
import dgl

class AseGraph(object):
    def __init__(self, **data_config):
        self.data_config = {**default_data_config,
                            **config_parser(data_config)}
        self.device = torch.device(self.data_config['device'])
        self.all_atom_feat = get_atomic_feature_onehot(
            self.data_config['species'])
        self.cutoff = torch.tensor(self.data_config['cutoff'],
                                   device = self.device)
        self.skin = torch.tensor(1.0, device = self.device)  # angstrom
        # self.adsorbate_skin = 2.0 # in many cases, the adsorbates experience larger geometry relaxation than surface atoms.

        self.step = 0
        self.new_graph_steps = 40  # buid new graph every * steps.
        self.inner_senders = None
        self.inner_receivers = None
        self.inner_receivers_image = None
        self.skin_senders = None
        self.skin_receivers = None
        self.skin_receivers_image = None
        self.graph = None

    def reset(self):
        # reset parameters for new optimization.
        # self.new_graph = True
        self.step = 0
        self.inner_senders = None
        self.inner_receivers = None
        self.inner_receivers_image = None
        self.skin_senders = None
        self.skin_receivers = None
        self.skin_receivers_image = None
        self.graph = None

    def get_ndata(self, ase_atoms):
        # print('get_ndata')
        ndata = []
        for i in ase_atoms.get_chemical_symbols():
            ndata.append(self.all_atom_feat[i])
        return torch.tensor(np.array(ndata), dtype=torch.float32,
                            device=self.device)

    def get_adsorbate_bool(self, element_list):
        # print('get_adsorbate_bool')
        """
       .. py:method:: get_adsorbate_bool(self)

          Identify adsorbates based on elements： H and O.

          :return: a list of bool values.
          :rtype: tf.constant

        """
        element_list = np.array(element_list)
        return torch.tensor(np.where((element_list == 'H') | (element_list == 'O'),
                                     1, 0), device=self.device)

    def get_scaled_positions_wrap(self, cell_I_tensor, positions):
        # print('get_scaled_positions')
        # cell_I: Returns the (multiplicative) inverse of invertible
        scaled_positions = torch.matmul(positions, cell_I_tensor)
        scaled_positions = torch.where(
            scaled_positions < 0.0, scaled_positions+1, scaled_positions)
        scaled_positions = torch.where(
            scaled_positions > 1.0, scaled_positions-1, scaled_positions)
        return scaled_positions

    def get_scaled_positions(self, cell_I_tensor, positions):
        # print('get_scaled_positions')
        # cell_I: Returns the (multiplicative) inverse of invertible
        return torch.matmul(positions, cell_I_tensor)

    def fractional2cartesian(self, cell_tensor, scaled_positions):
        # print('fractional2cartesian')
        positions = torch.matmul(scaled_positions, cell_tensor)
        return positions

    def safe_to_use(self, ase_atoms, critical=0.01):
        cell = ase_atoms.cell.array
        abs_cell = np.abs(cell)
        sum_cell = np.sum(abs_cell,axis=1)
        is_cubic = np.max((sum_cell - np.diagonal(abs_cell)) / sum_cell) < critical
        cutoff = self.cutoff + self.skin
        large_enough = cutoff.item() < np.linalg.norm(cell, axis=0)
        return is_cubic & large_enough.all()

    def get_pair_distances(self, a, b, ase_atoms):
        if not self.safe_to_use(ase_atoms):
            raise ValueError('Input structure is not a cubic system or not large enough, cannot use \
this function to calculate distances. Alternatively, you need to use \
``ase.Atoms.get_distances``: https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.get_distances')
        cell = ase_atoms.cell.array
        cell_tensor = torch.tensor(cell, dtype=torch.float32,
                                   device=self.device)
        cell_I_tensor = torch.tensor(
            np.array(np.mat(cell).I), dtype=torch.float32, device=self.device)
        positions = torch.tensor(ase_atoms.positions, dtype=torch.float32,
                                 device=self.device)
        scaled_positions = self.get_scaled_positions(cell_I_tensor, positions)
        a_positions = positions[a, :]
        a_scaled_positions = scaled_positions[a, :]
        b_scaled_positions = scaled_positions[b, :]
        diff_scaled_positions = b_scaled_positions - a_scaled_positions
        b_scaled_positions = torch.where(diff_scaled_positions > 0.5,
                                         b_scaled_positions-1.0,
                                         b_scaled_positions)
        b_scaled_positions = torch.where(diff_scaled_positions < -0.5,
                                         b_scaled_positions+1.0,
                                         b_scaled_positions)
        b_positions_new = self.fractional2cartesian(
            cell_tensor, b_scaled_positions)
        D = b_positions_new - a_positions
        d = torch.norm(D, dim=1)
        return d, D

    def update_pair_distances(self, a, b, b_image, ase_atoms):
#         if not self.safe_to_use(ase_atoms):
#             raise ValueError('Input structure is not a cubic system or not large enough, cannot use \
# this function to calculate distances. Alternatively, you need to use \
# ``ase.Atoms.get_distances``: https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.get_distances')
        cell = ase_atoms.cell.array
        cell_tensor = torch.tensor(cell, dtype=torch.float32,
                                    device=self.device)
        cell_I_tensor = torch.tensor(
            np.array(np.mat(cell).I), dtype=torch.float32, device=self.device)
        positions = torch.tensor(ase_atoms.positions, dtype=torch.float32,
                                  device=self.device)
        scaled_positions = self.get_scaled_positions(cell_I_tensor, positions)

        a_positions = positions[a, :]
        # a_scaled_positions = scaled_positions[a, :]
        b_scaled_positions = scaled_positions[b, :] + b_image
        b_positions_new = self.fractional2cartesian(
            cell_tensor, b_scaled_positions)
        D = b_positions_new - a_positions
        d = torch.norm(D, dim=1)
        return d, D

    def get_all_possible_distances(self, ase_atoms):
        # get senders and receivers, including inner and skin connections.
        # torch.from_numpy is memory effcient than torch.tensor, especially for large tensors.
        # No self loop and reverse direction.
        if not self.safe_to_use(ase_atoms):
            raise ValueError('Input structure is not a cubic system or not large enough, cannot use \
this function to calculate distances. Alternatively, you need to use \
``ase.Atoms.get_distances``: https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.get_distances')

        # prepare variables
        cell = ase_atoms.cell.array
        cell_tensor = torch.tensor(cell, dtype=torch.float32, device=self.device)
        cell_I_tensor = torch.tensor(
            np.array(np.mat(cell).I), dtype=torch.float32, device=self.device)
        positions = torch.tensor(ase_atoms.positions, dtype=torch.float32,
                                 device=self.device)
        scaled_positions = self.get_scaled_positions(cell_I_tensor, positions)

        a, b = np.triu_indices(len(ase_atoms), k=1)
        a, b = torch.tensor(a, device=self.device), torch.tensor(b, device=self.device)

        # ************************
        # a = torch.tensor([0,1,48,54,2,3])
        # b = torch.tensor([24,25,72,78,26,27])
        # dist_test_1 = ase_atoms.get_distance(0,24,mic=True)
        # dist_test_2 = ase_atoms.get_distance(0,24,mic=False)
        # ************************

        a_scaled_positions = scaled_positions[a, :]
        b_scaled_positions = scaled_positions[b, :]
        diff_scaled_positions = b_scaled_positions - a_scaled_positions
        abs_diff_scaled_positions = torch.absolute(diff_scaled_positions)
        cutoff = self.cutoff + self.skin
        scaled_cutoff = cutoff/torch.norm(cell_tensor, dim=1)

        # remove distant atomic pairs. Test results show that these lines have little effect on boosting efficiency.
        # if torch.all(scaled_cutoff < 0.5):
        #     # print('before:', len(a))
        #     mask1 = abs_diff_scaled_positions > scaled_cutoff
        #     mask2 = abs_diff_scaled_positions < 1-scaled_cutoff
        #     mask = mask1 & mask2
        #     mask = ~mask.all(dim=1)
        #     a = a[mask]
        #     b = b[mask]
        #     a_scaled_positions = scaled_positions[a, :]
        #     b_scaled_positions = scaled_positions[b, :]
        #     diff_scaled_positions = b_scaled_positions - a_scaled_positions
        #     abs_diff_scaled_positions = torch.absolute(diff_scaled_positions)
        #     # print('before:', len(a))

        # case one: connection at current image
        case_one_mask = torch.any(abs_diff_scaled_positions < scaled_cutoff, dim=1)
        case_one_i = torch.where(case_one_mask)[0]
        case_one_image = torch.zeros(len(case_one_i), 3,device=self.device)

        # case two: connection at other images
        case_two_mask = abs_diff_scaled_positions > 1 - scaled_cutoff
        x_mask = case_two_mask[:,0]
        y_mask = case_two_mask[:,1]
        z_mask = case_two_mask[:,2]
        xy_mask = x_mask & y_mask
        xz_mask = x_mask & z_mask
        yz_mask = y_mask & z_mask
        xyz_mask = x_mask & y_mask & z_mask

        xi = torch.where(x_mask)[0]
        yi = torch.where(y_mask)[0]
        zi = torch.where(z_mask)[0]
        xyi = torch.where(xy_mask)[0]
        xzi = torch.where(xz_mask)[0]
        yzi = torch.where(yz_mask)[0]
        xyzi = torch.where(xyz_mask)[0]

        image_x = torch.where(diff_scaled_positions[x_mask,0] > 0.0, -1.0, 1.0)
        image_y = torch.where(diff_scaled_positions[y_mask,1] > 0.0, -1.0, 1.0)
        image_z = torch.where(diff_scaled_positions[z_mask,2] > 0.0, -1.0, 1.0)
        image_xy = torch.where(diff_scaled_positions[xy_mask,0:2] > 0.0, -1.0, 1.0)
        image_xz = torch.where(diff_scaled_positions[xz_mask,:] > 0.0, -1.0, 1.0)
        image_xz_x = image_xz[:,0]
        image_xz_z = image_xz[:,2]
        image_yz = torch.where(diff_scaled_positions[yz_mask,1:3] > 0.0, -1.0, 1.0)
        image_xyz = torch.where(diff_scaled_positions[xyz_mask,:] > 0.0, -1.0, 1.0)

        image_x = torch.stack((image_x,
                               torch.zeros_like(image_x, device=self.device),
                               torch.zeros_like(image_x, device=self.device)))
        image_x = torch.transpose(image_x, 0, 1)
        image_y = torch.stack((torch.zeros_like(image_y, device=self.device),
                               image_y,
                               torch.zeros_like(image_y, device=self.device)))
        image_y = torch.transpose(image_y, 0, 1)
        image_z = torch.stack((torch.zeros_like(image_z, device=self.device),
                               torch.zeros_like(image_z, device=self.device),
                               image_z))
        image_z = torch.transpose(image_z, 0, 1)
        image_xy = torch.cat((image_xy,
                              torch.zeros(len(image_xy), 1,
                                          device=self.device)),
                             dim=1)
        image_xz = torch.stack((image_xz_x,
                                torch.zeros_like(image_xz_x, device=self.device),
                                image_xz_z))
        image_xz = torch.transpose(image_xz, 0, 1)
        image_yz = torch.cat((torch.zeros(len(image_yz), 1, device=self.device),
                              image_yz),
                             dim=1)
        image_xyz = image_xyz

        all_i = torch.cat((case_one_i,
                           xi, yi, zi,
                           xyi, xzi, yzi,
                           xyzi))

        all_image = torch.cat((case_one_image,
                               image_x, image_y, image_z,
                               image_xy, image_xz, image_yz,
                               image_xyz), dim=0)

        b_scaled_positions_all = b_scaled_positions[all_i] + all_image
        b_positions_all = self.fractional2cartesian(
            cell_tensor, b_scaled_positions_all)
        a_positions = positions[a, :]
        a_positions_all = a_positions[all_i]

        a_all, b_all = a[all_i], b[all_i]

        # calculate distance
        D = b_positions_all - a_positions_all
        d = torch.norm(D, dim=1)

        # position_0 = positions[0]
        # position_24 = positions[24]

        return a_all, b_all, d, D, all_image # what about return a dict? or image location

    def get_init_connections(self, ase_atoms):
        # No self loop and reverse direction.
        i, j, d, D, j_image = self.get_all_possible_distances(ase_atoms)
        inner_connections = torch.where(d < self.cutoff)
        skin_connections = torch.where((d > self.cutoff) & (d < self.cutoff+1))
        i_i, j_i, d_i, D_i, j_image_i = i[inner_connections], j[inner_connections], d[inner_connections], D[inner_connections], j_image[inner_connections]
        i_s, j_s, d_s, D_s, j_image_s = i[skin_connections], j[skin_connections], d[skin_connections], D[skin_connections], j_image[skin_connections]
        return i_i, j_i, d_i, D_i, j_image_i, i_s, j_s, d_s, D_s, j_image_s

    def update_connections(self, i_i, j_i, j_image_i, i_s, j_s, j_image_s, ase_atoms):
        # print('update instead of build')
        i, j, j_image = torch.cat((i_i, i_s)), torch.cat((j_i, j_s)), torch.cat((j_image_i, j_image_s))
        d, D = self.update_pair_distances(i, j, j_image, ase_atoms)
        inner_connections = torch.where(d < self.cutoff)
        skin_connections = torch.where(
            (d > self.cutoff) & (d < self.cutoff+self.skin))
        i_i, j_i, d_i, D_i = i[inner_connections], j[inner_connections], d[inner_connections], D[inner_connections]
        i_s, j_s, d_s, D_s = i[skin_connections], j[skin_connections], d[skin_connections], D[skin_connections]
        j_image_i = j_image[inner_connections]
        j_image_s = j_image[skin_connections]
        return i_i, j_i, d_i, D_i, j_image_i, i_s, j_s, d_s, D_s, j_image_s

    def build(self, ase_atoms):
        print('build new graph.')
        # build graph
        # Include all possible properties.

        ndata = self.get_ndata(ase_atoms)
        # No self loop and reverse direction.
        i_i, j_i, d_i, D_i, j_image_i, i_s, j_s, d_s, D_s, j_image_s = self.get_init_connections(
            ase_atoms)
        self.inner_senders = i_i
        self.inner_receivers = j_i
        self.inner_receivers_image = j_image_i
        self.skin_senders = i_s
        self.skin_receivers = j_s
        self.skin_receivers_image = j_image_s

        # add reverse direction and self loop connections.
        # Some properties will not change when update: h, constraints, adsorbate,
        bg = dgl.graph((torch.cat((self.inner_senders,
                                   self.inner_receivers,
                                   torch.arange(len(ase_atoms),
                                                device=self.device)),
                                  dim=0),
                       torch.cat((self.inner_receivers,
                                  self.inner_senders,
                                  torch.arange(len(ase_atoms),
                                               device=self.device)),
                                 dim=0)))
        bg.ndata['h'] = ndata
        bg.edata['dist'] = torch.cat((d_i, d_i, torch.zeros(len(ase_atoms),
                                                            device=self.device)),
                                     dim=0)
        bg.edata['direction'] = torch.cat(
            (D_i, -D_i, torch.zeros(len(ase_atoms), 3,
                                   device=self.device)), dim=0)
        constraints = [[1, 1, 1]] * len(ase_atoms)
        for c in ase_atoms.constraints:
            if isinstance(c, ase.constraints.FixScaled):
                for i in c.index:
                    constraints[i] = [int(c.mask[0]), int(c.mask[1]), int(c.mask[2])]
            elif isinstance(c, ase.constraints.FixAtoms):
                for i in c.index:
                    constraints[i] = [0, 0, 0]
            elif isinstance(c, ase.constraints.FixedLine):
                for i in c.index:
                    constraints[i] = c.dir.tolist()
            elif isinstance(c, ase.constraints.FixBondLengths):
                pass
            else:
                raise TypeError(
                    f'Wraning!!! Undefined constraint type: {type(c)}')
        bg.ndata['constraints'] = torch.tensor(
            constraints, dtype=torch.float32, device=self.device)
        element_list = ase_atoms.get_chemical_symbols()
        bg.ndata['adsorbate'] = self.get_adsorbate_bool(element_list)
        self.graph = bg
        return bg

    def update(self, ase_atoms):
        # print('update')
        # calculate d and D, reassign i and j
        i_i, j_i, d_i, D_i, j_image_i, i_s, j_s, d_s, D_s, j_image_s = self.update_connections(
            self.inner_senders, self.inner_receivers, self.inner_receivers_image,
            self.skin_senders, self.skin_receivers, self.skin_receivers_image,
            ase_atoms)
        self.inner_senders = i_i
        self.inner_receivers = j_i
        self.inner_receivers_image = j_image_i
        self.skin_senders = i_s
        self.skin_receivers = j_s
        self.skin_receivers_image = j_image_s
        bg = dgl.graph((torch.cat((self.inner_senders,
                                  self.inner_receivers,
                                  torch.arange(len(ase_atoms),
                                               device=self.device)),
                                  dim=0),
                       torch.cat((self.inner_receivers,
                                  self.inner_senders,
                                  torch.arange(len(ase_atoms),
                                               device=self.device)),
                                 dim=0)))
        bg.ndata['h'] = self.graph.ndata['h']
        bg.edata['dist'] = torch.cat((d_i, d_i, torch.zeros(len(ase_atoms),
                                                            device=self.device)),
                                     dim=0)
        bg.edata['direction'] = torch.cat((D_i, -D_i, torch.zeros(len(ase_atoms),
                                                                 3,
                                                                 device=self.device)),
                                          dim=0)
        bg.ndata['constraints'] = self.graph.ndata['constraints']
        bg.ndata['adsorbate'] = self.graph.ndata['adsorbate']
        self.graph = bg
        return bg

    def get_graph(self, ase_atoms):  # this is the high-level API.
        # if isinstance(ase_atoms, str):
        #     ase_atoms = read(ase_atoms)
        #     self.reset()
        # elif isinstance(ase_atoms, ase.atoms.Atoms):
        #     ase_atoms = ase_atoms
        # else:
        #     raise TypeError("Incorrect input structure type.")
        # ase_atoms_tmp = ase_atoms.copy()
        if self.step % self.new_graph_steps == 0:
            bg = self.build(ase_atoms)
        else:
            bg = self.update(ase_atoms)
        self.step += 1
        return bg

if __name__ == '__main__':
    # ase_atoms = read('XDATCAR')
    import os
    # os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
    import json
    graph_build_scheme_path = os.path.join('acatal', 'test', 'agat_model', 'graph_build_scheme.json')
    with open(graph_build_scheme_path, 'r') as f:
        graph_build_scheme = json.load(f)
    graph_build_scheme['device'] = 'cpu'
    graph_build_scheme['topology_only'] = True
    fname = os.path.join('acatal', 'test', 'CONTCAR_new_3')


    ###########################################################################
    # test
    from agat.lib.model_lib import load_model
    model_save_dir = os.path.join('acatal', 'test', 'agat_model')

    ase_atoms = read(fname)
    ag = AseGraph(**graph_build_scheme)

    g1 = ag.get_graph(ase_atoms)
    g2 = ag.get_graph(ase_atoms)
    agat_model = load_model(model_save_dir, 'cpu')
    with torch.no_grad():
        energy_pred1, force_pred1, stress_pred1 = agat_model.forward(g1)
        energy_pred2, force_pred2, stress_pred2 = agat_model.forward(g2)

    ###########################################################################
    # from agat.data import CrystalGraph
    # ase_atoms = read(fname)
    # cg = CrystalGraph(**graph_build_scheme)
    # g, _ = cg.get_graph(ase_atoms)
    # dist = g.edata['dist']
    # edges = g.edges()
    # len(dist)

    # ase_atoms = read(fname)
    # ag = AseGraph(**graph_build_scheme)
    # g_new = ag.get_graph(ase_atoms)
    # dist_new = g_new.edata['dist']
    # edges_new = g_new.edges()


    ###########################################################################
    # compare
    # num_edges = len(dist)
    # pair = []
    # for i in range(num_edges):
    #     pair.append((edges[0][i].item(),
    #                   edges[1][i].item()))

    # num_edges_new = len(dist_new)
    # pair_new = []
    # for i in range(num_edges_new):
    #     pair_new.append((edges_new[0][i].item(),
    #                       edges_new[1][i].item()))

    # for p in pair:
    #     repeat = pair.count(p)
    #     repeat_new = pair_new.count(p)
    #     if repeat > 1:
    #         print(p, repeat, repeat_new)


    # print('======================== PyTorch from scratch')
    # # self = ag
    # ase_atoms = read(fname)
    # ag = AseGraph(**graph_build_scheme)
    # import time
    # for n in [1,2,3,4]:
    #     start = time.time()
    #     ase_atoms_tmp = ase_atoms.copy().repeat(n)
    #     for i in range(30):
    #         bg = ag.build(ase_atoms_tmp)
    #     print(len(ase_atoms_tmp), time.time() - start)

    # print('======================== PyTorch with update')
    # ase_atoms = read(fname)
    # ag = AseGraph(**graph_build_scheme)
    # for n in [1,2,3,4]:
    #     start = time.time()
    #     ase_atoms_tmp = ase_atoms.copy().repeat(n)
    #     bg = ag.build(ase_atoms_tmp)
    #     for i in range(29):
    #         bg = ag.update(ase_atoms_tmp)
    #     print(len(ase_atoms_tmp), time.time() - start)

    # print('======================== ASE from scratch')
    # from agat.data import CrystalGraph
    # ase_atoms = read(fname)
    # cg = CrystalGraph(**graph_build_scheme)
    # for n in [1,2,3,4]:
    #     start = time.time()
    #     ase_atoms_tmp = ase_atoms.copy().repeat(n)
    #     for i in range(30):
    #         g = cg.get_graph(ase_atoms_tmp)
    #     print(len(ase_atoms_tmp), time.time() - start)

    # # start = time.time()
    # # for i in range(1000):
    # #     ase_atoms.get_positions()
    # # print(len(ase_atoms_tmp), time.time() - start)
    # # start = time.time()
    # # for i in range(1000):
    # #     ase_atoms.get_scaled_positions()
    # # print(len(ase_atoms_tmp), time.time() - start)
    # # for i in range(1000):
    # #     positions = torch.tensor(ase_atoms.get_positions(), dtype=torch.float32)
    # #     self.get_scaled_positions(cell_I_tensor, positions)
    # # print(len(ase_atoms_tmp), time.time() - start)
