import scipy
from copy import copy
import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_winner_pair(test):

    i = 0
    winner_index = -1
    winner_value = 10000000
    arr = []
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
        if winner_value>chamfer_value:
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
