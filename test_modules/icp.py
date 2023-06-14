"""
UNDER CONSTRUCTION 

Check the teaser.py file, it should have a run function with the same parameters!
Copy the registration part from the ICP_test

WARNING: 
This below is unfinished code!
"""
import open3d as o3d 
import numpy as np 
import pandas as pd

def run(obj1_seg_parts_array, obj2_seg_parts_array, MIN_PCD_SIZE=1000):

    # create dataframe
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

    # loop over parts
    for o1, seg_part1 in enumerate(obj1_seg_parts_array):
        for o2, seg_part2 in enumerate(obj2_seg_parts_array):
            pcd_part1 = seg_part1.pcd
            pcd_part2 = seg_part2.pcd
            cur_frag1_size = len(pcd_part1.points)
            cur_frag2_size = len(pcd_part2.points)
            frag1_size.append(cur_frag1_size)
            frag2_size.append(cur_frag2_size)
            print(f'Now registering part {o1} of obj1 ({cur_frag1_size} points) with part {o2} of obj2 ({cur_frag2_size} points)')
            o1s.append(o1)
            o2s.append(o2)


            # compute transformation with ICP 
            target = copy(pcd_part1)
            source = copy(pcd_part2)
            init_trans = np.eye(4)
            result_icp = o3d.pipelines.registration.registration_icp(
                    source,target,5000,init_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling =False),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
            transformations.append((o1,o2,result_icp.transformation))


    # example from teaser.py file 
    # need to fill the panda dataframe 
    # candidates_registration['o1s'] = o1s
    # candidates_registration['o2s'] = o2s
    # candidates_registration['fitness'] = fitness 
    # candidates_registration['inlier_rmse'] = inlier_rmse
    # candidates_registration['corr_set_size_icp'] = corr_set_size_icp
    # candidates_registration['num_corrs_teaser'] = num_corrs_teaser
    # candidates_registration['chamfer_distance'] = chamfer_distances
    # candidates_registration['frag1_size'] = frag1_size
    # candidates_registration['frag2_size'] = frag2_size
    # candidates_registration['transf_teaser'] = transf_teaser
    # candidates_registration['transf_icp'] = transf_icp
    # candidates_registration['rot_teaser'] = rot_teaser
    # candidates_registration['tra_teaser'] = tra_teaser

    

    return candidates_registration # a pandas dataframe with the keys commented above

    
    # return candidates_registration



