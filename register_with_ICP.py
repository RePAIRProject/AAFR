import open3d as o3d 
import numpy as np 

source = o3d.io.read_point_cloud('/home/lucap/code/AAFR/results/Experiments_March/cookies_large_08_06_fractured_73/pointclouds/Obj2_before.ply')
target = o3d.io.read_point_cloud('/home/lucap/code/AAFR/results/Experiments_March/cookies_large_08_06_fractured_73/pointclouds/Obj1_before.ply')

# tf_param, _, _ = cpd.registration_cpd(copy(source), copy(target))

init_trans = np.eye(4)
# init_trans[:3, :3] = tf_param.rot

result_icp = o3d.pipelines.registration.registration_icp(
        source,target,5000,init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling =False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))

source.transform(result_icp.transformation)
o3d.io.write_point_cloud('/media/lucap/big_data/datasets/pairwise/ali/EXPERIMENTS/COMPARISON/Mug_fractured_73/ICP_pred_piece_0.ply', target)
o3d.io.write_point_cloud('/media/lucap/big_data/datasets/pairwise/ali/EXPERIMENTS/COMPARISON/Mug_fractured_73/ICP_pred_piece_1.ply', source)