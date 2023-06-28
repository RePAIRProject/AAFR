import argparse
import importlib
import pdb
import open3d as o3d
import os 
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
import test_modules.teaser as ts
import pandas as pd
from copy import deepcopy,copy
from utils.helpers import Rt2T
from evaluation_pairwise.utils import chamfer_distance

clusterer = hdbscan.HDBSCAN(min_cluster_size=5)

obj1 = o3d.io.read_point_cloud("/home/seba/Projects/AAFR/3dvr_results/3dvr_30000/DrinkBottle_fractured_62/pointclouds/obj_3dvr_30000_part1.ply")
obj2 = o3d.io.read_point_cloud("/home/seba/Projects/AAFR/3dvr_results/3dvr_30000/DrinkBottle_fractured_62/pointclouds/obj_3dvr_30000_part2.ply")

pcd1 = o3d.io.read_point_cloud("/home/seba/Projects/AAFR/3dvr_results/3dvr_30000/DrinkBottle_fractured_62/inner_obj1_challenge.ply")
pcd2 = o3d.io.read_point_cloud("/home/seba/Projects/AAFR/3dvr_results/3dvr_30000/DrinkBottle_fractured_62/inner_obj2_challenge.ply")

# labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=15))

# max_label = labels.max()
# colors = plt.get_cmap("tab20")(labels / (max_label 
# if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd])

seg1 = clusterer.fit_predict(np.asarray(pcd1.points))
seg2 = clusterer.fit_predict(np.asarray(pcd2.points))

# prepare the array containing statistics
candidates_registration = pd.DataFrame()
fitness = []
num_corrs_teaser = []
corr_set_size_icp = []
inlier_rmse = []
chamfer_distances = []
frag1_size = []
frag2_size = []
transf_teaser = []
transf_icp = []
rot_teaser = []
tra_teaser = []
o1s = []
o2s = []

MIN_PCD_SIZE=1000

for i in range(np.max(seg1)):
    idx1 = list(np.where((seg1 == (i)))[0])
    pcd_part1 = pcd1.select_by_index(idx1)
    for j in range(np.max(seg2)):
        idx2 = list(np.where((seg2 == (j)))[0])
        pcd_part2 = pcd2.select_by_index(idx2)

        cur_frag1_size = len(pcd_part1.points)
        cur_frag2_size = len(pcd_part2.points)
        frag1_size.append(cur_frag1_size)
        frag2_size.append(cur_frag2_size)
        print(f'Now registering part {i} of obj1 ({cur_frag1_size} points) with part {j} of obj2 ({cur_frag2_size} points)')
        o1s.append(i)
        o2s.append(j)
        
        if cur_frag1_size < MIN_PCD_SIZE or cur_frag2_size < MIN_PCD_SIZE:
            print('skip, too small')
            fitness.append(0)
            inlier_rmse.append(0)
            corr_set_size_icp.append(0)
            num_corrs_teaser.append(0)
            transf_teaser.append(np.eye(4))
            rot_teaser.append(np.eye(3))
            tra_teaser.append(np.zeros((3,1)))
            transf_icp.append(np.eye(4))
            chamfer_distances.append(MIN_PCD_SIZE)

        else:

            target = copy(pcd_part1)
            source = copy(pcd_part2)
            icp_sol, teaser_sol, num_corrs = ts.register_fragments(source, target)

            fitness.append(icp_sol.fitness)
            inlier_rmse.append(icp_sol.inlier_rmse)
            corr_set_size_icp.append(len(icp_sol.correspondence_set))
            num_corrs_teaser.append(num_corrs)
            transf_teaser.append(Rt2T(teaser_sol.rotation,teaser_sol.translation))
            rot_teaser.append(teaser_sol.rotation)
            tra_teaser.append(teaser_sol.translation)
            transf_icp.append(icp_sol.transformation)

            source.transform(icp_sol.transformation)
            cd = chamfer_distance(target.points, source.points)
            chamfer_distances.append(cd)

candidates_registration['o1s'] = o1s
candidates_registration['o2s'] = o2s
candidates_registration['fitness'] = fitness 
candidates_registration['inlier_rmse'] = inlier_rmse
candidates_registration['corr_set_size_icp'] = corr_set_size_icp
candidates_registration['num_corrs_teaser'] = num_corrs_teaser
candidates_registration['chamfer_distance'] = chamfer_distances
candidates_registration['frag1_size'] = frag1_size
candidates_registration['frag2_size'] = frag2_size
candidates_registration['transf_teaser'] = transf_teaser
candidates_registration['transf_icp'] = transf_icp
candidates_registration['rot_teaser'] = rot_teaser
candidates_registration['tra_teaser'] = tra_teaser

sorted_candidates_registration = candidates_registration.sort_values('chamfer_distance')
best_registration = sorted_candidates_registration.head(1)

folder_registration_results = os.path.join("./", "registration_seba")
os.makedirs(folder_registration_results, exist_ok=True)
sorted_candidates_registration.to_csv(os.path.join(folder_registration_results, 'sorted_candidates_registration.csv'))
best_registration.to_csv(os.path.join(folder_registration_results, 'best_registration.csv'))

obj1_to_draw = copy(obj1)
obj2_to_draw = copy(obj2)
obj1_to_draw.paint_uniform_color([1, 1, 0])
obj2_to_draw.paint_uniform_color([0, 0, 1])
obj2_to_draw = obj2_to_draw.transform(best_registration['transf_teaser'].item())
o3d.io.write_point_cloud(os.path.join(folder_registration_results, 'obj1.ply'), obj1_to_draw)
o3d.io.write_point_cloud(os.path.join(folder_registration_results, 'obj2.ply'), obj2_to_draw)


# max_label = labels.max()
# colors = plt.get_cmap("tab20")(labels / (max_label 
# if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd])

# """Register segmented regions"""
# print("_________________________Registration_________________________")
# self.candidates_registration = self.registration.run(self.obj1_seg_parts_array, self.obj2_seg_parts_array)
# self.sorted_candidates_registration = self.candidates_registration.sort_values('chamfer_distance')
# self.best_registration = self.sorted_candidates_registration.head(1)


