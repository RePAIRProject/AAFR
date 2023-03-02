""" 
WORKFLOW:
1. load a pair of fragments (obj1, obj2)
2. place them at the origin (obj1 at the origin, obj2 assembled to it):
    - place obj1 in (0, 0, 0) with a translation T_obj1_2_origin. 
    - apply T_obj1_2_origin to the obj2 (now they are assembled again)
3. get the ground truth transformation 
    - move obj2 to the origin (translation) and apply rotation (range?)
    - save the inverse transformation as "solution"
4. feed the pipeline with the two objects placed at the origin
    - get S segmented faces
    - try SxS (or KxK, with K=min(S, 5) to save time) registrations
    - evaluate the best one (how? RMSE? CD?)
    - return best one as result
5. save results
    - pcls assembled with obj1 in (0,0)
    - pcls both in (0,0)
    - pcls after transformation from the pipeline
    - calculate RMSE (R), RMSE (T), CD 
""" 
import open3d as o3d 

save_everything = True
output_dir = os.path.join(os.getcwd(), 'output_evaluation_pairwise')
os.makedirs(output_dir, exist_ok=True)
axis_angles = [45, 45, 45] # config

# 1. load a pair of fragments (obj1, obj2)
pair_folder = '/media/lucap/big_data/datasets/pairwise/repair_group15_pair1'
Obj1_url = os.path.join(pair_folder, 'RPf_00105_gt.ply')
Obj2_url = os.path.join(pair_folder, 'RPf_00108_gt.ply')
pcl1 = o3d.io.read_point_cloud(Obj1_url)
pcl2 = o3d.io.read_point_cloud(Obj2_url)

# 2. place them at the origin 
# (obj1 at the origin, obj2 assembled to it):
T_obj1_2_origin = - np.mean(np.asarray(pcl1.points), axis=1)
pcl1 = pcl1.translate(T_obj1_2_origin)
pcl2 = pcl2.translate(T_obj1_2_origin)
if save_everything:
    o3d.io.write_point_cloud(os.path.join(output_dir, 'pcl1_gt.ply'), pcl1)
    o3d.io.write_point_cloud(os.path.join(output_dir, 'pcl2_gt.ply'), pcl2)

# 3. get the ground truth transformation 
T_obj2_2_origin = - np.mean(np.asarray(pcl2.points), axis=1)
R_obj2_challenge = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
M_obj2_gt_2_challenge = np.eye(4)
M_obj2_gt_2_challenge[:3, :3] = R_obj2_challenge
M_obj2_gt_2_challenge[:3, 3] = T_obj2_2_origin
pcl2 = pcl2.transform(M_obj2_gt_2_challenge)
GT_inv_M = np.eye(4)
GT_inv_M[:3, :3] = np.transpose(R_obj2_challenge) # or np.linalg.inv ?
GT_inv_M[:3, 3] = - T_obj2_2_origin
if save_everything:
    o3d.io.write_point_cloud(os.path.join(output_dir, 'pcl2_origin.ply'), pcl2)
    np.savetxt(os.path.join(output_dir, 'gt_inverse_transformation_obj2.json'), GT_inv_M)

# 4. feed the pipeline with the two objects placed at the origin
