from copy import deepcopy,copy
import random

import open3d as o3d
import numpy as np
from copy import copy
import helper
import numpy as np
from Fragment import FeatureLines
from tqdm import tqdm
import open3d as o3d
import numpy as np

import random
import random
def dilate_border(my_obj,border,size):
    tmp_Obj = copy(my_obj)
    tmp_Obj.pcd = copy(my_obj.pcd)
    pcd_tree = o3d.geometry.KDTreeFlann(tmp_Obj.pcd)
    w_co = copy(tmp_Obj.w_co)
    new_borders = []
    for idx in border:
        point = tmp_Obj.pcd.points[idx]
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point,size)
        new_borders.extend(idx)
    return new_borders


def find_dilattion_size(my_obj,border):
    tmp_Obj = copy(my_obj)
    tmp_Obj.pcd = copy(my_obj.pcd)
    pcd_tree = o3d.geometry.KDTreeFlann(tmp_Obj.pcd)
    w_co = copy(tmp_Obj.w_co)
    size = 0
    upper_size = 2
    lower_size = 0
    avg_number_ngh = 0
    while lower_size < upper_size:
        size = (lower_size+upper_size)/2.0
        ngh = 0
        for i,idx in enumerate(border):
            point = tmp_Obj.pcd.points[idx]
            [k, idx, _] = pcd_tree.search_radius_vector_3d(point,size)
            ngh+=len(idx)
        if abs(ngh-20*len(border))<5:
                return size
        elif ngh > 20*len(border):
                upper_size = size-0.00000001
        else:
                lower_size = size
    return size

def get_sides(Graph, borders):
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
                    if neighbor not in borders:
                        queue.append(neighbor)
        faces.append((len(visited),visited))
    sorted_faces = sorted(faces,key = lambda key:key[0], reverse = True)
    return sorted_faces[:20]

def knn_expand(Obj,my_borders,node_face,face_nodes,size):

    points_u = []
    tmp_tree = o3d.geometry.KDTreeFlann(Obj.pcd)
    for i in range(len(Obj.pcd.points)):
            point = Obj.pcd.points[i]
            [k, idx, _] = tmp_tree.search_knn_vector_3d(point, 100)
            q_points = np.asarray(Obj.pcd.points).take(idx,axis=0)
            points_u.append(np.mean(np.abs(q_points[1:] - point)))
    radius = np.mean(points_u)
    print("KNN Radius : ",radius)
    borders = deepcopy(my_borders)
    face_nodes_new = deepcopy(face_nodes)
    tree = o3d.geometry.KDTreeFlann(Obj.pcd)
    no_change = 0
    while borders:
        len_before = len(borders)

        idx = borders.pop(0)
        point = Obj.pcd.points[idx]
        [k, q_idxs, _] = tree.search_knn_vector_3d(point, size)
        #vote
        my_faces = {}
        for q_idx in q_idxs:
            if q_idx not in node_face:
                continue

            if node_face[q_idx] not in my_faces:
                my_faces[node_face[q_idx]] = 1
            else:
                my_faces[node_face[q_idx]] += 1

        if len(my_faces)==0:
            borders.append(idx)
        else:
            winner = max(my_faces, key=my_faces.get)
            face_nodes_new[winner].add(idx)
            node_face[idx] = winner

        if len(borders) == len_before:
            no_change += 1
        else:
            no_change = 0

        if no_change >= len_before+10000000:
            break
    print(len(borders))
    return face_nodes_new

def run(Obj_url,pipline_variables):

    (N, shortest_cycle_length, smallest_isolated_island_length, shortest_allowed_branch_length, thre) = pipline_variables

    print("start")
    Obj = FeatureLines(Obj_url,voxel_size=30000)
    # print("Size :",len(Obj.pcd.points))
    print("starting init")
    Obj.init(int(N))

    # print("Size valid :",len(valid))

    shortest_cycle_length = np.sqrt(len(Obj.pcd.points))//shortest_cycle_length
    smallest_isolated_island_length = np.sqrt(len(Obj.pcd.points))//smallest_isolated_island_length
    shortest_allowed_branch_length = np.sqrt(len(Obj.pcd.points))//shortest_allowed_branch_length

    isolated_islands_pruned_graph, F_lines, isolated_islands = helper.create_graph(Obj, \
    shortest_cycle_length, smallest_isolated_island_length)
    print("After graph",len([point for branch in F_lines for point in branch]))
    pruned_graph_original, removed_nodes, valid_nodes = helper.prune_branches(F_lines,isolated_islands_pruned_graph,\
    shortest_allowed_branch_length)
    print("After Pruning",len([node for branch in valid_nodes for node in branch]))

    print("constructing borders")

    tmp_Obj = copy(Obj)
    tmp_Obj.pcd = copy(Obj.pcd)

    N = 15
    t1 = 0.1
    t2 = 1
    t3 = 0.1
    pipline_variables = (N, t1, t2, t3, thre)
    (N, shortest_cycle_length, smallest_isolated_island_length, shortest_allowed_branch_length, thre) = pipline_variables
    # print("Size valid :",len(valid))
    valid = []
    for idx,val in enumerate(tmp_Obj.w_co):
        if val<0.96:
            valid.append(idx)
    shortest_cycle_length = np.sqrt(len(valid))//shortest_cycle_length
    smallest_isolated_island_length = np.sqrt(len(valid))//smallest_isolated_island_length
    shortest_allowed_branch_length = np.sqrt(len(valid))//shortest_allowed_branch_length
    print("shortest_allowed_branch_length",shortest_allowed_branch_length)
    print("smallest_isolated_island_length",smallest_isolated_island_length)
    print("shortest_cycle_length",shortest_cycle_length)
    isolated_islands_pruned_graph, F_lines, isolated_islands = helper.create_graph(tmp_Obj, \
    shortest_cycle_length, smallest_isolated_island_length,mask=valid,radius=None)
    print("After graph",len([point for branch in F_lines for point in branch]))
    pruned_graph, removed_nodes, valid_nodes = helper.prune_branches(F_lines,isolated_islands_pruned_graph,\
    shortest_allowed_branch_length)
    print("After Pruning",len([node for branch in valid_nodes for node in branch]))

    print("dilation and segmentation")

    border_nodes = [node for branch in valid_nodes for node in branch]

    size = find_dilattion_size(Obj,border_nodes)
    print(f"my size : {size}")
    dilated_border = dilate_border(Obj,border_nodes,0.009)

    print("borders dilated")

    dilated_faces = get_sides(pruned_graph_original, dilated_border)

    print("got faces")

    node_face = {}
    face_nodes = []
    for idx,(_,face) in enumerate(dilated_faces):
        for node in face:
            node_face[node] = idx
        face_nodes.append(face)
    # for key,value in node_face.items():
    #     if value>=10:
    #         print(value)

    all_dilated = [node for  _,face in dilated_faces for node in face]
    all_dilated.extend([node for node in dilated_border])


    left_overs = list(set(range(len(Obj.pcd.points)))-set(all_dilated))
    border_left_overs = left_overs+dilated_border


    expanded_faces = knn_expand(Obj,border_left_overs,node_face,face_nodes,size=5)

    print("expanded")


    Objects = []
    for expanded_face in expanded_faces:
        mask = np.isin(np.arange(0, len(Obj.pcd.points), 1).tolist(),[node for node in expanded_face])
        Obj_tmp = copy(Obj)
        Obj_tmp.pcd = copy(Obj.pcd)
        Obj_tmp.pcd.points = o3d.utility.Vector3dVector(np.asarray(Obj.pcd.points)[mask])
        Objects.append(Obj_tmp)

    print("coloring")
    colors = [(0,255,0) for _ in Obj.pcd.points]
    for face in expanded_faces:
        color = tuple(random.choices(range(256), k=3))
        for point in face:
            colors[point] = color
    Obj.pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors).astype("float") / 255.0)
    # o3d.visualization.draw_geometries([Obj.pcd])

    return Obj,Objects
