import helper
import numpy as np
from Fragment import FeatureLines
from tqdm import tqdm
import open3d as o3d
import numpy as np
def pipline_1(Obj_url,N,shortest_cycle_length,smallest_isolated_island_length,shortest_allowed_branch_length ):
    Obj = FeatureLines(Obj_url,voxel_size=0.9)
    # print("Size :",len(Obj.pcd.points))
    Obj.init(N)
    valid = []
    for idx,val in enumerate(Obj.w_co):
        if val<0.97:
            valid.append(idx)

    # print("Size valid :",len(valid))


    shortest_cycle_length = np.sqrt(len(Obj.pcd.points))//shortest_cycle_length
    smallest_isolated_island_length = np.sqrt(len(Obj.pcd.points))//smallest_isolated_island_length
    shortest_allowed_branch_length = np.sqrt(len(Obj.pcd.points))//shortest_allowed_branch_length




    Obj.pcd.points = o3d.utility.Vector3dVector(np.asarray(Obj.pcd.points)[valid])
    isolated_islands_pruned_graph, F_lines, isolated_islands = helper.create_graph(Obj,\
    2,shortest_cycle_length, smallest_isolated_island_length)
    # print("After graph",len([point for branch in F_lines for point in branch]))
    pruned_graph, removed_nodes, valid_nodes = helper.prune_branches(Obj,F_lines,isolated_islands_pruned_graph,\
    shortest_allowed_branch_length)
    # print("After Pruning",len([node for branch in valid_nodes for node in branch]))

    mask = np.isin(np.arange(0, len(Obj.pcd.points), 1).tolist(),[node for branch in valid_nodes for node in branch])
    Obj.pcd.points = o3d.utility.Vector3dVector(np.asarray(Obj.pcd.points)[mask])

    return Obj
