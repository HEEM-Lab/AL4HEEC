# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:50:40 2024

@author: ZHANGJUN
"""

import numpy as np
import pandas as pd
import mpltern
from matplotlib import pyplot  as plt
from matplotlib.ticker import MultipleLocator

fname = r'D:\OneDrive\Projects\ALAGAT_HER\data\20240924 data process\all_dft_concentrations\all_cons.csv'
t_label, l_label, r_label = 'NiCoFe', 'Pd', 'Pt'
save_files = False
df = pd.read_csv(fname)

nbins = 50+1
bin_size = 1/(nbins-1)
bins = np.linspace(0.0, 1.0, num=nbins)

df['NiCoFe'] = df['Ni'] + df['Co'] + df['Fe']
df['NiPdPt'] = df['Ni'] + df['Pd'] + df['Pt']
df['CoPdPt'] = df['Co'] + df['Pd'] + df['Pt']
count = []
for x in bins:
    for y in [xx for xx in bins if xx <= 1-x]:
        z = 1.0 - x - y
        count.append([x, y, z, 0.0])
count = np.array(count)

for di, d in df.iterrows():
    for ci,c in enumerate(count):
        if d[t_label] > c[0] - bin_size and d[t_label] < c[0] + bin_size and\
           d[l_label] > c[1] - bin_size and d[l_label] < c[1] + bin_size and\
           d[r_label] > c[2] - bin_size and d[r_label] < c[2] + bin_size:
               count[ci][3] += 1.0

if save_files:
    np.savetxt(f'count-{t_label}-{l_label}-{r_label}.csv', count, fmt='%.4f', delimiter=',')

##################################################################

t, l, r, v = count[:,0], count[:,1], count[:,2], count[:,3],
fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(projection="ternary")
cmap = "Blues"
shading = "gouraud"
cs = ax.tripcolor(t, l, r, v, cmap=cmap, shading=shading, rasterized=True)
ax.tricontour(t, l, r, v, colors="k", linewidths=0.5)

ax.set_tlabel(t_label, size=12)
ax.set_llabel(l_label, size=12)
ax.set_rlabel(r_label, size=12)
ax.taxis.set_major_locator(MultipleLocator(0.20))
ax.laxis.set_major_locator(MultipleLocator(0.20))
ax.raxis.set_major_locator(MultipleLocator(0.20))
ax.taxis.set_label_position("tick1")
ax.laxis.set_label_position("tick1")
ax.raxis.set_label_position("tick1")

ax.grid(axis='t', linestyle='--')
ax.grid(axis='l', linestyle='--')
ax.grid(axis='r', linestyle='--')

cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(cs, cax=cax)
colorbar.set_label("Count", rotation=270, va="baseline")


# plt.title(f'Loop {self.loop_num}', loc='right')
if save_files:
    plt.savefig(f'ternary_dft_density-{t_label}-{l_label}-{r_label}.png',
                              dpi=600,
                              bbox_inches='tight')
plt.show()
