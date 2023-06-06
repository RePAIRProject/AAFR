import open3d as o3d
import numpy as np
from copy import copy
from utils.helpers import *
import pdb
def estimate_transform(pcd1, pcd2, voxel_size=2, verbose=False):

    # should be taken from conf file
    voxel_size=voxel_size 
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
    
    return refined_T

def run(Obj1, Obj2, _):

    transformations = []
    #pdb.set_trace()
    for o1,obj1 in enumerate(Obj1):
        for o2,obj2 in enumerate(Obj2):
            pdb.set_trace()
            target = copy(obj1.pcd)
            source = copy(obj2.pcd)
            transf = estimate_transform(source, target)
            transformations.append((o1,o2,transf))

    return transformations
