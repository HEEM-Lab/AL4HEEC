# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:42:45 2024

@author: ZHANGJUN
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

cons = np.loadtxt('best_cons_from_loop_final.txt')
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(cons)
centers = kmeans.cluster_centers_
# score = kmeans.score(x_data)

# ===find nearest structure of center===
closest, _ = pairwise_distances_argmin_min(centers, cons)
closest_samples = np.array([cons[x] for x in closest])

print(closest)
