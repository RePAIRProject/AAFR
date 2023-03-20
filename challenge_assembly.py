import os, json
import numpy as np 
import matplotlib.pyplot as plt 
import open3d as o3d
from copy import copy
from utils.helpers import *
import pdb
import pandas as pd 
import time
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

    show_best_five = False
    save_image = False
    MIN_PCD_SIZE = 1000
    challenge_rot_angles = [30, 30, 30]
    challenge_trans_matrix = [3, 5, 2]
    results_folder = '/home/lucap/code/AAFR/segmentation_results_synth'
    exp_folders = os.listdir(results_folder)
    
    for exp_f in exp_folders: 
        print(exp_f)
    for exp_name in exp_folders:
        if True:
            #if 'OTHER_DATASET_03_13_presious_tombstone_pair1' in exp_name:
            
            exp_folder = os.path.join(results_folder, exp_name)
            json_files = [filename for filename in os.listdir(exp_folder) if filename.endswith('.json')]
            with open(os.path.join(exp_folder, json_files[0]), 'r') as jf:
                info_j = json.load(jf)

            pcl_out_folder = os.path.join(exp_folder, 'output_pcl')  

            # Read the full file 
            if info_j['Obj1_url'].endswith('.ply'):
                full_frag1 = o3d.io.read_point_cloud(info_j['Obj1_url'])
            else:
                target_mesh = o3d.io.read_triangle_mesh(info_j['Obj1_url'])
                full_frag1 = target_mesh.sample_points_uniformly(number_of_points=100000)

            if info_j['Obj2_url'].endswith('.ply'):
                full_frag2 = o3d.io.read_point_cloud(info_j['Obj2_url'])
            else:
                target_mesh = o3d.io.read_triangle_mesh(info_j['Obj2_url'])
                full_frag2 = target_mesh.sample_points_uniformly(number_of_points=100000)
                
            full_frag1.paint_uniform_color([1, 1, 0])
            full_frag2.paint_uniform_color([0, 0, 1])

            obj1_name = info_j['Obj1_url'].split('/')[-1][:-4]
            obj2_name = info_j['Obj2_url'].split('/')[-1][:-4]
            
            os.makedirs(pcl_out_folder, exist_ok=True)

            o3d.io.write_point_cloud(os.path.join(pcl_out_folder, f'{obj1_name}.ply'), full_frag1)
            o3d.io.write_point_cloud(os.path.join(pcl_out_folder, f'{obj2_name}.ply'), full_frag2)
            #pdb.set_trace()
            # TRANSFORMATION
            challenge_rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(challenge_rot_angles)
            challenge_transformation = Rt2T(challenge_rot_matrix, challenge_trans_matrix)
            full_frag2.transform(challenge_transformation)

            
            seg_parts_folder = os.path.join(exp_folder, 'segmented_parts')
            #objs_folders = os.listdir(seg_parts_folder)
            obj1_f = os.path.join(seg_parts_folder, obj1_name)
            obj1_parts = []
            for part in (os.listdir(obj1_f)):
                pcd = o3d.io.read_point_cloud(os.path.join(obj1_f, part))
                obj1_parts.append(pcd)
            obj2_f = os.path.join(seg_parts_folder, obj2_name)
            obj2_parts = []
            for part in (os.listdir(obj2_f)):
                pcd = o3d.io.read_point_cloud(os.path.join(obj2_f, part))
                # TRANSFORMATION
                pcd.transform(challenge_transformation)
                obj2_parts.append(pcd)

            #pdb.set_trace()
            print(f'We have {len(obj1_parts)} parts of obj1 and {len(obj2_parts)} parts of obj2')

            # o3d.visualization.draw_geometries([obj1_parts[9], obj1_parts[10]])
            # T_obj2_2_origin = - np.mean(np.asarray(full_frag2.points), axis=0)
            # #pcl2 = pcl2.translate(T_obj2_2_origin)
            # R_obj2_challenge = o3d.geometry.get_rotation_matrix_from_axis_angle([45, 45, 45])
            # #pcl2 = pcl2.rotate(R_obj2_challenge)
            # TM = Rt2T(R_obj2_challenge, T_obj2_2_origin)
            # GT_inv_M = np.eye(4)
            # GT_inv_M[:3, :3] = np.linalg.inv(R_obj2_challenge)
            # GT_inv_M[:3, 3] = - T_obj2_2_origin
            # o3d.visualization.draw_geometries([obj1_parts[9], obj1_parts[10]])
            # pdb.set_trace()

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
                    print(f'Now registering part {o1} of obj1 ({cur_frag1_size} points) with part {o2} of obj2 ({cur_frag2_size} points)')
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
            sorted_cands = candidates_registration.sort_values('chamfer_distance')
            sorted_cands.to_csv(os.path.join(exp_folder, 'candidates_registration_sorted.csv'))
            print(exp_folder)


            if show_best_five and len(candidates_registration) > 5:
                best_five = candidates_registration.sort_values('chamfer_distance').head(5)
                #pdb.set_trace()
                for index, top_cand in best_five.iterrows():
                    f2_copy = copy(full_frag2)
                    f2_copy.transform(top_cand['transf_icp'])
                    #print(top_cand)
                    i1 = top_cand['o1s']
                    i2 = top_cand['o2s']
                    part1 = obj1_parts[i1]
                    part2 = obj2_parts[i2]
                    part1.paint_uniform_color([1, 1, 0])
                    part2.paint_uniform_color([0, 0, 1])
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name=f'Segmented Parts ({i1}, {i2})', width=960, height=540, left=0, top=0, visible=True)
                    vis.add_geometry(part1)
                    vis.add_geometry(part2)
                    

                    vis2 = o3d.visualization.Visualizer()
                    vis2.create_window(window_name=f"Assembly (CD={top_cand['chamfer_distance']:.03f})", width=960, height=540, left=960, top=0)
                    vis2.add_geometry(full_frag1)
                    vis2.add_geometry(f2_copy)

                    while True:
                        #vis.update_geometry()
                        if not vis.poll_events():
                            break
                        vis.update_renderer()
                        #vis.capture_screen_image('test_update.png')
                        

                        #vis2.update_geometry()
                        if not vis2.poll_events():
                            break
                        vis2.update_renderer()

                    
                    #vis.capture_screen_image('test_after.png')
                    vis.destroy_window()
                    vis2.destroy_window()
                    #o3d.visualization.draw_geometries([full_frag1, full_frag2], f"{i1}, {i2}")
                    pdb.set_trace()
            elif save_image:
                best_five = candidates_registration.sort_values('chamfer_distance').head(5)
                #pdb.set_trace()
                segmented_parts_imgs = []
                full_fragments_imgs = []
                plt.figure(figsize=(32,12))
                #pdb.set_trace()
                counter = 1
                for index, top_cand in best_five.iterrows():
                    candidate_transformation = top_cand['transf_icp']
                    f2_copy = copy(full_frag2)
                    f2_copy.transform(candidate_transformation)
                    #print(top_cand)
                    i1 = top_cand['o1s']
                    i2 = top_cand['o2s']
                    part1 = obj1_parts[i1]
                    part2 = obj2_parts[i2]
                    part2.transform(candidate_transformation)
                    part1.paint_uniform_color([1, 1, 0])
                    part2.paint_uniform_color([0, 0, 1])
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name=f'Segmented Parts ({i1}, {i2})', width=960, height=540, left=0, top=0, visible=False)
                    vis.add_geometry(part1)
                    vis.add_geometry(part2)
                    plt.subplot(2, 5, counter)
                    plt.title(f'Segmented Parts ({i1}, {i2})')
                    sp_img = vis.capture_screen_float_buffer(do_render=True)
                    segmented_parts_imgs.append(sp_img)
                    plt.imshow(sp_img)
                    plt.axis('off')

                    vis2 = o3d.visualization.Visualizer()
                    vis2.create_window(window_name=f"Assembly (CD={top_cand['chamfer_distance']:.03f})", width=960, height=540, left=960, top=0, visible=False)
                    vis2.add_geometry(full_frag1)
                    vis2.add_geometry(f2_copy)
                    plt.subplot(2, 5, 5+counter)
                    plt.title(f"Assembly (CD={top_cand['chamfer_distance']:.03f})")
                    ff_img = vis2.capture_screen_float_buffer(do_render=True)
                    full_fragments_imgs.append(ff_img)
                    plt.imshow(ff_img)
                    plt.axis('off')

                    vis.clear_geometries()
                    vis2.clear_geometries()
                    counter += 1
                
                    vis.destroy_window()
                    vis2.destroy_window()
                    time.sleep(1)
                plt.tight_layout()
                plt.savefig(os.path.join(exp_folder, 'top_five_registration.jpg'))
                
            best = candidates_registration.sort_values('chamfer_distance').head(1)       
            
            o3d.io.write_point_cloud(os.path.join(pcl_out_folder, f'{obj2_name}_challenge.ply'), full_frag2)
            full_frag2.transform(best['transf_icp'].item())
            # full_frag1.paint_uniform_color([1, 1, 0])
            # full_frag2.paint_uniform_color([0, 0, 1])
            o3d.io.write_point_cloud(os.path.join(pcl_out_folder, f'{obj2_name}_predicted.ply'), full_frag2)
            #pdb.set_trace()
            part1 = obj1_parts[best['o1s'].item()]
            o3d.io.write_point_cloud(os.path.join(pcl_out_folder, f'{obj1_name}_part_{best["o1s"].item()}.ply'), part1)
            part2 = obj2_parts[best['o2s'].item()]
            part2.transform(best['transf_icp'].item())
            o3d.io.write_point_cloud(os.path.join(pcl_out_folder, f'{obj2_name}_part_{best["o2s"].item()}_predicted.ply'), part2)

            #o3d.visualization.draw_geometries([full_frag1, full_frag2])
            #pdb.set_trace()
            print(f'\nDone with {exp_name}\n')

if __name__ == '__main__':
    main()