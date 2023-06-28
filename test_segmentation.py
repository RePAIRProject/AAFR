import argparse
import importlib
import pdb
import open3d as o3d
import os 
import matplotlib.pyplot as plt
import numpy as np
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=5)


pcd = o3d.io.read_point_cloud("/home/seba/Projects/AAFR/3dvr_results/3dvr_30000/DrinkBottle_fractured_62/inner_obj2_challenge.ply")
labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=15))

max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label 
if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])


labels = clusterer.fit_predict(np.asarray(pcd.points))

max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label 
if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])


