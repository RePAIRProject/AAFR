import scipy
from copy import copy, deepcopy
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import open3d as o3d
import pandas as pd 
import pdb 

def sort_results(reassembly):

    i = 0
    winner_index = -1
    winner_value = 10000000
    arr = []
    #pdb.set_trace()
    for o1, o2, R_T in reassembly.result_transformation_arr:
        obj1 = reassembly.Obj1_array[o1]
        obj2 = reassembly.Obj2_array[o2]
        obj2_tmp = copy(obj2)
        obj2_tmp.pcd = copy(obj2.pcd)
        obj2_tmp.pcd.transform(R_T)
        R_error = reassembly.results[i]['R_error']
        T_error = reassembly.results[i]['T_error']
        chamfer_value = chamfer_distance(list(obj2_tmp.pcd.points),list(obj1.pcd.points))
        arr.append({"index":i, \
                    "o1":o1, \
                    "o2":o2, \
                    "R":R_error, \
                    "T":T_error, \
                    "CD":chamfer_value})
        if winner_value>chamfer_value:
            winner_index = i
            winner_value = chamfer_value
        i+=1
    winner = {
        "index":winner_index, \
        "o1":reassembly.results[winner_index]['o1'], \
        "o2":reassembly.results[winner_index]['o2'], \
        "R":reassembly.results[winner_index]['R_error'], \
        "T":reassembly.results[winner_index]['T_error'], \
        "CD":chamfer_value
    }
    results_df = pd.DataFrame(arr)
    return winner, results_df.sort_values('CD')
    

def get_winner_pair(test):

    i = 0
    winner_index = -1
    winner_value = 10000000
    arr = []
    #pdb.set_trace()
    for o1,o2,R_T in test.result_transformation_arr:
    #     print(o1,o2)
        obj1 = test.Obj1_array[o1]
        obj2 = test.Obj2_array[o2]
        obj2_tmp = copy(obj2)
        obj2_tmp.pcd = copy(obj2.pcd)
        obj2_tmp.pcd.transform(R_T)
#     print(scipy.spatial.distance.directed_hausdorff(list(obj2_tmp.pcd.points), list(obj1.pcd.points), seed=0))
#     print(scipy.spatial.distance.directed_hausdorff(list(obj1.pcd.points),list(obj2_tmp.pcd.points) , seed=0))
        chamfer_value = chamfer_distance(list(obj2_tmp.pcd.points),list(obj1.pcd.points))
        arr.append({"index":i,"o1":o1,"o2":o2,"chamfer":chamfer_value})
        if winner_value > chamfer_value:
            winner_index = i
            winner_value = chamfer_value
        i+=1
    chamfer_df = pd.DataFrame(arr)
    return chamfer_df.sort_values('chamfer')

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist

def save(output_dir, test, winner_index):

    # pcl folder
    path_pcls = os.path.join(output_dir, "pointclouds")
    os.makedirs(path_pcls, exist_ok=True)

    pcd1 = deepcopy(test.Obj1.pcd)
    pcd2 = deepcopy(test.Obj2.pcd)
    pcd1.colors = o3d.utility.Vector3dVector(np.asarray([(0,1,0) for _ in pcd1.points]).astype("float"))
    pcd2.colors = o3d.utility.Vector3dVector(np.asarray([(0,0,1) for _ in pcd2.points]).astype("float"))
    pcd2.transform(np.eye(4))
    o3d.io.write_point_cloud(os.path.join(path_pcls, "Obj1_before.ply"), pcd1, compressed=True)
    o3d.io.write_point_cloud(os.path.join(path_pcls, "Obj2_before.ply"), pcd2, compressed=True)

    pcd2.transform(test.result_transformation_arr[winner_index][2])
    o3d.io.write_point_cloud(os.path.join(path_pcls, "Obj1_after.ply"), pcd1, compressed=True)
    o3d.io.write_point_cloud(os.path.join(path_pcls, "Obj2_after.ply"), pcd2, compressed=True)

def save_parts(parts, output_dir, prefix_name='obj'):

    os.makedirs(output_dir, exist_ok=True)

    for j, part in enumerate(parts):
        o3d.io.write_point_cloud(os.path.join(output_dir, f"{prefix_name}_part_{j}.ply"), part.pcd, compressed=True)


# def save_arr(name,test):
#     import os
#     from copy import deepcopy
#     import open3d as o3d
#     # Directory
#     directory = name

#     # Parent Directory path
#     parent_dir = "results/"

#     # Path
#     path = os.path.join(parent_dir, directory)

#     os.makedirs(path, exist_ok=True)
#     for i,(obj1,obj2) in enumerate(zip(test.Obj1_array,test.Obj2_array)):
#         pcd1 = deepcopy(obj1.pcd)
#         pcd2 = deepcopy(obj2.pcd)
#         pcd1.colors = o3d.utility.Vector3dVector(np.asarray([(0,1,0) for _ in pcd1.points]).astype("float"))
#         pcd2.colors = o3d.utility.Vector3dVector(np.asarray([(0,0,1) for _ in pcd2.points]).astype("float"))
#         pcd2.transform(np.eye(4))
#         o3d.io.write_point_cloud(os.path.join(path, str(i)+"_Obj1.ply"), pcd1, compressed=True)
#         o3d.io.write_point_cloud(os.path.join(path, str(i)+"Obj2.ply"), pcd2, compressed=True)
#     print("saved")