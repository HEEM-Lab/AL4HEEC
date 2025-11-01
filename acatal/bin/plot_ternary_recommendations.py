# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 18:45:25 2024

@author: ZHANGJUN
"""

import numpy as np
from matplotlib import pyplot  as plt

from matplotlib.ticker import MultipleLocator
import mpltern

for ii, i in enumerate((1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)):
    all_cons = np.loadtxt(f'generated_cons_loop_{i}.txt')
    centers = np.loadtxt(f'centers_20_loop_{i}.txt')
    x1 = all_cons[:,0] + all_cons[:,1] + all_cons[:,2]
    x2 = all_cons[:,3]
    x3 = all_cons[:,4]
    y1 = centers[:,0] + centers[:,1] + centers[:,2]
    y2 = centers[:,3]
    y3 = centers[:,4]

    ax = plt.subplot(projection="ternary")
    ax.taxis.set_major_locator(MultipleLocator(0.20))
    ax.laxis.set_major_locator(MultipleLocator(0.20))
    ax.raxis.set_major_locator(MultipleLocator(0.20))
    plt.title(f'Loop {ii+1}', loc='right')

    ax.scatter(x1, x2, x3,
               marker='o',
               s=32.,
               c=[np.array([36,140,175])/255],
               alpha=0.3,
               edgecolors=np.array([[45,45,45]])/255)
    ax.scatter(y1, y2, y3,
               marker='h',
               s=32.,
               c=[np.array([219,59,51])/255],
               alpha=1.0,
               edgecolors='k')
    ax.set_tlabel("NiCoFe", size=12)
    ax.set_llabel("Pd", size=12)
    ax.set_rlabel("Pt", size=12)
    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")


    # ax.laxis.set_minor_locator(MultipleLocator(0.1))
    # ax.raxis.set_minor_locator(AutoMinorLocator(5))

    ax.grid(axis='t', linestyle='--')
    ax.grid(axis='l', linestyle='--')
    ax.grid(axis='r', linestyle='--')

    plt.savefig(f'ternary_cons_after_KNN_loop_{i}_for_paper.png', bbox_inches='tight',
                dpi=600)
    plt.show()

