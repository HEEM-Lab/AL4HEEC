# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:20:39 2023

@author: ZHANG Jun
"""

import json
import os

import numpy as np

import torch # initialize pytorch first.

from acatal.module.cgan import param_opt
from matplotlib import pyplot  as plt
import ternary

def generate_new_compositions():
    assert os.path.exists('average_delta_G.csv'), 'The `average_delta_G.csv` \
file is required in the current directory.'
    opt_param = param_opt()
    with open('opt_param.json', 'w') as fp:
        json.dump(opt_param, fp, indent=4)

    active_results = np.loadtxt('active_results.txt')
    active_step_i = np.argmin(active_results[:,1])
    _active_step = int(active_results[active_step_i,0])
    minus_final_mae = active_results[active_step_i,1]

    # plot
    x_data = np.loadtxt(os.path.join(str(_active_step), 'generated_cons.txt'))
    x_all_plot = [[x[0]+x[1]+x[2], x[3], x[4]] for x in x_data]

    fig,ax = plt.subplots()
    scale = 1.0
    fontsize = 12
    offset = 0.14
    figure, tax = ternary.figure(scale=scale,ax=ax)
    tax.gridlines(multiple=5, color="blue")
    figure.set_size_inches(6, 6)
    tax.right_corner_label("NiCoFe", fontsize=fontsize)
    tax.top_corner_label("Pd", fontsize=fontsize)
    tax.left_corner_label("Pt", fontsize=fontsize)
    tax.left_axis_label("Pt", fontsize=fontsize, offset=offset)
    tax.right_axis_label("Pd", fontsize=fontsize, offset=offset)
    tax.bottom_axis_label("NiCoFe", fontsize=fontsize, offset=offset)
    tax.scatter(x_all_plot, marker='s', c='red', label="Initial pred")
    # Remove default Matplotlib Axes
    # tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.boundary()
    tax.set_title(f"MAE: {-minus_final_mae}")
    plt.savefig(os.path.join(f'ternary_cons_from_{_active_step}.png'))

if __name__ == '__main__':
    generate_new_compositions()
