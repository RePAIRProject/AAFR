import os 
import numpy as np 
import matplotlib.pyplot as plt 
import open3d as o3d
from copy import copy
from utils.helpers import *
import pdb
import pandas as pd 
from evaluation_pairwise.utils import chamfer_distance

def register_fragments(pcd1, pcd2, voxel_size=2, resample_factor=5, verbose=False):


    dist1 = pcd1.compute_nearest_neighbor_distance()
    dist2 = pcd1.compute_nearest_neighbor_distance()
    #print(dist1, dist2)
    #pdb.set_trace()
    pcd1 = pcd1.voxel_down_sample(np.mean(dist1) * resample_factor)
    pcd2 = pcd2.voxel_down_sample(np.mean(dist2) * resample_factor)
    
    #o3d.visualization.draw_geometries([pcd1, pcd2])
    # should be taken from conf file
    #voxel_size=voxel_size 
    # the fpfh is very sensitive to this parameter! larger is better

    # Extract features
    A_xyz = pcd2xyz(pcd1) # np array of size 3 by N
    B_xyz = pcd2xyz(pcd2) # np array of size 3 by N

    # extract FPFH features
    A_feats = extract_fpfh(pcd1, voxel_size=voxel_size)
    B_feats = extract_fpfh(pcd2, voxel_size=voxel_size)

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(
        A_feats, B_feats, mutual_filter=True)
    A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
    B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs
    num_corrs = A_corr.shape[1]
    if verbose:
        print(f'matched {num_corrs} fpfh correspondences')

    # robust global registration using TEASER++
    NOISE_BOUND = voxel_size
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(A_corr,B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser,t_teaser)

    # Maybe it would be a good idea to include a flag in the run() 
    # to control whether to use ICP refinement after teaser or not
    # for now we include it anyway

    # if refinement == True:

    # local refinement using ICP
    icp_sol = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, NOISE_BOUND, T_teaser,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    refined_T = icp_sol.transformation

    if verbose:
        print("Teaser transformation:", T_teaser)
        print("Final solution after ICP:", refined_T)
    
    return icp_sol, solution, num_corrs, 

def main():

    MIN_PCD_SIZE = 200
    results_folder = '/home/lucap/code/AAFR/results/Experiments_March'
    exp_folders = os.listdir(results_folder)
    for exp_f in exp_folders: 
        print(exp_f)
    for exp_name in exp_folders:
        if 'bottles' in exp_name:
        # if exp_name == 'cookies_large_08_06_fractured_52_NO':
        #exp_name = 'bottles_small_08_06_fractured_3_UPSIDEDOWN'
            exp_folder = os.path.join(results_folder, exp_name)
            seg_parts_folder = os.path.join(exp_folder, 'segmented_parts')
            obj1_f = os.path.join(seg_parts_folder, 'obj1')
            obj1_parts = []
            for part in (os.listdir(obj1_f)):
                pcd = o3d.io.read_point_cloud(os.path.join(obj1_f, part))
                obj1_parts.append(pcd)
            obj2_f = os.path.join(seg_parts_folder, 'obj2')
            obj2_parts = []
            for part in (os.listdir(obj2_f)):
                pcd = o3d.io.read_point_cloud(os.path.join(obj2_f, part))
                obj2_parts.append(pcd)

            print(f'We have {len(obj1_parts)} parts of obj1 and {len(obj2_parts)} parts of obj2')
            
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
            for o1, pcd_part1 in enumerate(obj1_parts):
                for o2, pcd_part2 in enumerate(obj2_parts):
                    cur_frag1_size = len(pcd_part1.points)
                    cur_frag2_size = len(pcd_part2.points)
                    frag1_size.append(cur_frag1_size)
                    frag2_size.append(cur_frag2_size)
                    print(f'Now registerting part {o1} of obj1 ({cur_frag1_size} points) with part {o2} of obj2 ({cur_frag2_size} points)')
                    o1s.append(o1)
                    o2s.append(o2)
                    
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
                        icp_sol, teaser_sol, num_corrs = register_fragments(source, target)
                        #transformations.append((o1,o2,transf))
                        fitness.append(icp_sol.fitness)
                        inlier_rmse.append(icp_sol.inlier_rmse)
                        corr_set_size_icp.append(len(icp_sol.correspondence_set))
                        num_corrs_teaser.append(num_corrs)
                        transf_teaser.append(Rt2T(teaser_sol.rotation,teaser_sol.translation))
                        rot_teaser.append(teaser_sol.rotation)
                        tra_teaser.append(teaser_sol.translation)
                        transf_icp.append(icp_sol.transformation)
                        #target.paint_uniform_color([1, 0, 0])
                        #source.paint_uniform_color([0, 1, 0])
                        #pcd_part2.paint_uniform_color([0, 0, 1])
                        source.transform(icp_sol.transformation)
                        cd = chamfer_distance(target.points, source.points)
                        chamfer_distances.append(cd)
                    #o3d.visualization.draw_geometries([target, source, pcd_part2])

            #pdb.set_trace()
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

            candidates_registration.to_csv(os.path.join(exp_folder, 'candidates_registration.csv'))
            print(exp_folder)

            best = candidates_registration.sort_values('chamfer_distance').head(1)
            #print(best)
            #pdb.set_trace()
            full_frag1 = o3d.io.read_point_cloud(os.path.join(exp_folder, 'pointclouds', 'Obj1_before.ply'))
            full_frag2 = o3d.io.read_point_cloud(os.path.join(exp_folder, 'pointclouds', 'Obj2_before.ply'))
            full_frag2.transform(best['transf_icp'].item())
            full_frag1.paint_uniform_color([1, 1, 0])
            full_frag2.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([full_frag1, full_frag2])
            pdb.set_trace()

if __name__ == '__main__':
    main()