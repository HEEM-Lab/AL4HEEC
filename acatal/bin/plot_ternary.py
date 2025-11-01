import numpy as np
from matplotlib import pyplot  as plt

from matplotlib.ticker import MultipleLocator
import mpltern
import pandas as pd

all_cons = pd.read_csv('average_delta_G.csv')
t = all_cons['Ni'] + all_cons['Co'] + all_cons['Fe']
l = all_cons['Pd']
r = all_cons['Pt']
v = all_cons['mean_d_G']
edgecolors=all_cons['mean_d_G'].apply(lambda x: 'red' if -0.2 <= x <= 0 else 'white')
linewidths=all_cons['mean_d_G'].apply(lambda x: 0.5 if -0.2 <= x <= 0 else 0.0)

fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(projection="ternary")
cs = ax.scatter(t, l, r,
           marker='h',
           s=40,
           c=v,
           alpha=1.0,
           edgecolors=edgecolors,
           linewidths=linewidths
           )

ax.set_tlabel('NiCoFe', size=12)
ax.set_llabel('Pd', size=12)
ax.set_rlabel('Pt', size=12)
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
colorbar.set_label("ΔG(H)", rotation=270, va="baseline")


# # plt.title(f'Loop {self.loop_num}', loc='right')
plt.savefig('average_delta_G_red.png',
                              dpi=600,
                              bbox_inches='tight')
plt.show()
