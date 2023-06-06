import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from copy import copy
import helper
import numpy as np
from Fragment import FeatureLines
from tqdm import tqdm
import open3d as o3d
import numpy as np

import random
import random
def get_sides(Graph, w_co, thre):
    faces = []
    all_visited = set()
    nodes = list(Graph.nodes)
    while len(nodes):
        random_point = nodes.pop(0)
        while random_point in all_visited:
            if not len(nodes):
                sorted_faces = sorted(faces,key = lambda key:key[0], reverse = True)
                return sorted_faces[:10]
            random_point = nodes.pop(0)

        queue = [random_point]
        visited = set()
        while len(queue):
            point = queue.pop(0)
            if point not in visited:
                visited.add(point)
                all_visited.add(point)
                neighbors = Graph.neighbors(point)
                for neighbor in neighbors:
                    all_visited.add(neighbor)
                    if w_co[neighbor]>thre:
                        queue.append(neighbor)
        faces.append((len(visited),visited))
    sorted_faces = sorted(faces,key = lambda key:key[0], reverse = True)
    return sorted_faces[:20]

def run(Obj_url, pipline_variables):
    (N, shortest_cycle_length, smallest_isolated_island_length, shortest_allowed_branch_length, thre) = pipline_variables

    print("start")
    Obj = FeatureLines(Obj_url,voxel_size=15000)
    # print("Size :",len(Obj.pcd.points))
    print("starting init")
    Obj.init(int(N))

    # print("Size valid :",len(valid))

    shortest_cycle_length = np.sqrt(len(Obj.pcd.points))//shortest_cycle_length
    smallest_isolated_island_length = np.sqrt(len(Obj.pcd.points))//smallest_isolated_island_length
    shortest_allowed_branch_length = np.sqrt(len(Obj.pcd.points))//shortest_allowed_branch_length


    print("create graph")
    isolated_islands_pruned_graph, F_lines, isolated_islands = helper.create_graph(Obj,\
    2,shortest_cycle_length, smallest_isolated_island_length)
    print("After graph",len([point for branch in F_lines for point in branch]))
    pruned_graph, removed_nodes, valid_nodes = helper.prune_branches(F_lines,isolated_islands_pruned_graph,\
    shortest_allowed_branch_length)
    Obj.pruned_graph = pruned_graph
    print("After Pruning",len([node for branch in valid_nodes for node in branch]))

    faces = get_sides(Obj.pruned_graph, Obj.w_co, thre)

    Obj_arr = []
    for size, face in faces:
        mask = np.isin(np.arange(0, len(Obj.pcd.points), 1).tolist(),[point for point in face])
        Obj_copy = copy(Obj)
        Obj_copy.pcd= copy(Obj.pcd)
        Obj_copy.pcd.points = o3d.utility.Vector3dVector(np.asarray(Obj.pcd.points)[mask])
        print(len(Obj_copy.pcd.points))
        Obj_arr.append(Obj_copy)
    return Obj,Obj_arr
