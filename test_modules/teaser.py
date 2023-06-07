import open3d as o3d
import numpy as np
from utils.helpers import *

def run(obj1_seg_parts_array, obj2_seg_parts_array):

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
    
    for o1, pcd_part1 in enumerate(obj1_seg_parts_array):
        for o2, pcd_part2 in enumerate(obj2_seg_parts_array):
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

    return candidates_registration
