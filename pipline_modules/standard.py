from copy import deepcopy,copy
import random
import os 
import open3d as o3d
import numpy as np
import helper
from Fragment import FeatureLines
from tqdm import tqdm
import pdb 
import random

colors = [
    [0, 204, 0],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
    [76, 153, 0],
    [153, 0, 76],
    [76, 0, 153],
    [153, 76, 0],
    [76, 0, 153],
    [153, 0, 76],
    [204, 51, 127],
    [204, 51, 127],
    [51, 204, 127],
    [51, 127, 204],
    [127, 51, 204],
    [127, 204, 51],
    [76, 76, 178],
    [76, 178, 76],
    [178, 76, 76],
    [0, 204, 0],
    [204, 0, 0],
]

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


def get_sides(Graph, borders):
    faces = []
    all_visited = set()
    nodes = list(Graph.nodes)
    shortest_cycle_length = np.sqrt(len(borders))//5
    with tqdm(total=len(nodes)) as pbar:
        while len(nodes):
            random_point = nodes.pop(0)
            while random_point in all_visited or random_point in borders:
                if not len(nodes):
                    sorted_faces = sorted(faces,key = lambda key:key[0], reverse = True)
                    my_faces = []
                    for size,face in sorted_faces:
                        if size>shortest_cycle_length:
                            my_faces.append((size,face))
                    return my_faces
                random_point = nodes.pop(0)

            queue = [random_point]
            visited = set()
            while len(queue):
                point = queue.pop(0)
                if point not in visited and point not in borders:
                    visited.add(point)
                    all_visited.add(point)
                    neighbors = Graph.neighbors(point)
                    pbar.update(len(list(copy(neighbors))))
                    for neighbor in neighbors:
                        all_visited.add(neighbor)
                        if neighbor not in borders:
                            queue.append(neighbor)
            #print(len(visited))
            faces.append((len(visited),visited))
        sorted_faces = sorted(faces,key = lambda key:key[0], reverse = True)
        my_faces = []
        for size,face  in sorted_faces:
            if size>shortest_cycle_length:
                my_faces.append((size,face) )
        return my_faces

def knn_expand(Obj,my_borders,node_face,face_nodes,size):

    points_u = []
    tmp_tree = o3d.geometry.KDTreeFlann(Obj.pcd)
    for i in range(len(Obj.pcd.points)):
            point = Obj.pcd.points[i]
            [k, idx, _] = tmp_tree.search_knn_vector_3d(point, 100)
            q_points = np.asarray(Obj.pcd.points).take(idx,axis=0)
            points_u.append(np.mean(np.abs(q_points[1:] - point)))
    radius = np.mean(points_u)
    #print("KNN Radius : ",radius)
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
    #print(len(borders))
    return face_nodes_new

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
            point = Obj.pcd.points[idx]
            [k, idx, _] = pcd_tree.search_radius_vector_3d(point,size)
            ngh+=len(idx)
        if abs(ngh-20*len(border))<5:
                return size
        elif ngh > 20*len(border):
                upper_size = size-0.00000001
        else:
                lower_size = size

def run(Obj_url, pipline_variables, folder_path=''):

    (small, large, N, to1, to2, to3, tb1, tb2, tb3, dilation_size, thre) = pipline_variables

    variables = {}
    print("start")
    print(Obj_url.split("_")[-1])
    if Obj_url.split("_")[-1] == "0.obj":
        print("big object")
        if Obj_url.endswith('.obj'):
            Obj = FeatureLines(Obj_url,"mesh", voxel_size=large)
        else:
            Obj = FeatureLines(Obj_url,voxel_size=large)
    else:
        print("small object")
        if Obj_url.endswith('.obj'):
            Obj = FeatureLines(Obj_url,"mesh", voxel_size=small)
        else:
            Obj = FeatureLines(Obj_url, voxel_size=small)
    # print("Size :",len(Obj.pcd.points))
    print("starting init")
    Obj.init(int(N))

    # print("Size valid :",len(valid))

    shortest_cycle_length = np.sqrt(len(Obj.pcd.points))//to1
    variables['shortest_cycle_length'] = shortest_cycle_length
    smallest_isolated_island_length = np.sqrt(len(Obj.pcd.points))//to2
    variables['smallest_isolated_island_length'] = smallest_isolated_island_length
    shortest_allowed_branch_length = np.sqrt(len(Obj.pcd.points))//to3
    variables['shortest_allowed_branch_length'] = shortest_allowed_branch_length

    isolated_islands_pruned_graph, F_lines, isolated_islands = helper.create_graph(Obj, \
    shortest_cycle_length, smallest_isolated_island_length)
    print("After graph",len([point for branch in F_lines for point in branch]))
    print("constructing borders")

    tmp_Obj = copy(Obj)
    tmp_Obj.pcd = copy(Obj.pcd)


    t1 = tb1
    t2 = tb2
    t3 = tb3
    pipline_variables = (N, t1, t2, t3, thre)
    (N, shortest_cycle_length, smallest_isolated_island_length, shortest_allowed_branch_length, thre) = pipline_variables
    # print("Size valid :",len(valid))
    valid = []
    for idx,val in enumerate(tmp_Obj.w_co):
        if val < thre:
            valid.append(idx)
    print(len(valid))
    shortest_cycle_length = np.sqrt(len(valid))/shortest_cycle_length
    smallest_isolated_island_length = np.sqrt(len(valid))/smallest_isolated_island_length
    shortest_allowed_branch_length = np.sqrt(len(valid))/shortest_allowed_branch_length
    print("shortest_allowed_branch_length",shortest_allowed_branch_length)
    print("smallest_isolated_island_length",smallest_isolated_island_length)
    print("shortest_cycle_length",shortest_cycle_length)
    isolated_islands_pruned_graph_border, F_lines, isolated_islands = helper.create_graph(tmp_Obj, \
    shortest_cycle_length, smallest_isolated_island_length,mask=valid,radius=None)
    print("After graph",len([point for branch in F_lines for point in branch]))
    #pdb.set_trace()
    pruned_graph, removed_nodes, valid_nodes = helper.prune_branches(F_lines,isolated_islands_pruned_graph_border,\
    shortest_allowed_branch_length)
    print("After Pruning",len([node for branch in valid_nodes for node in branch if len(branch)>smallest_isolated_island_length]))
    print("After Pruning",len([node for branch in valid_nodes for node in branch]))

    #pdb.set_trace()
    print("dilation and segmentation")

    border_nodes = [node for branch in valid_nodes for node in branch]

    dilated_border = dilate_border(Obj,border_nodes,dilation_size)
    dilated_faces = get_sides(isolated_islands_pruned_graph, dilated_border)

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

def load_obj(Obj_url, small, large, N):
    if Obj_url.split("_")[-1] == "0.obj":
        #print("big object")
        if Obj_url.endswith('.obj'):
            Obj = FeatureLines(Obj_url,"mesh", voxel_size=large)
        else:
            Obj = FeatureLines(Obj_url,voxel_size=large)
    else:
        if Obj_url.endswith('.obj'):
            Obj = FeatureLines(Obj_url,"mesh", voxel_size=small)
        else:
            Obj = FeatureLines(Obj_url, voxel_size=small)
    # print("Size :",len(Obj.pcd.points))
    Obj.init(int(N))
    return Obj


def detect_breaking_curves(obj, pipeline_variables):
    (small, large, N, to1, to2, to3, tb1, tb2, tb3, dilation_size, thre) = pipeline_variables

    variables = {}
    var_general = {}
    var_object = {}
    var_borders = {}    
    var_general['N'] = N
    shortest_cycle_length = np.sqrt(len(obj.pcd.points))//to1
    var_object['shortest_cycle_length'] = shortest_cycle_length
    smallest_isolated_island_length = np.sqrt(len(obj.pcd.points))//to2
    var_object['smallest_isolated_island_length'] = smallest_isolated_island_length
    shortest_allowed_branch_length = np.sqrt(len(obj.pcd.points))//to3
    var_object['shortest_allowed_branch_length'] = shortest_allowed_branch_length
    print('Creating point cloud graph..')
    isolated_islands_pruned_graph, F_lines, isolated_islands = helper.create_graph(obj, \
        shortest_cycle_length, smallest_isolated_island_length)
    #print("After graph",len([point for branch in F_lines for point in branch]))
    print("Constructing borders graph..")
    tmp_Obj = copy(obj)
    tmp_Obj.pcd = copy(obj.pcd)
    t1 = tb1
    t2 = tb2
    t3 = tb3
    pipline_variables = (N, t1, t2, t3, thre)
    (N, shortest_cycle_length, smallest_isolated_island_length, shortest_allowed_branch_length, thre) = pipline_variables
    valid = []
    for idx,val in enumerate(tmp_Obj.w_co):
        if val < thre:
            valid.append(idx)
    shortest_cycle_length = np.sqrt(len(valid))/shortest_cycle_length
    smallest_isolated_island_length = np.sqrt(len(valid))/smallest_isolated_island_length
    shortest_allowed_branch_length = np.sqrt(len(valid))/shortest_allowed_branch_length
    isolated_islands_pruned_graph_border, F_lines, isolated_islands = helper.create_graph(tmp_Obj, \
    shortest_cycle_length, smallest_isolated_island_length,mask=valid,radius=None)

    print("Pruning..")
    pruned_graph, removed_nodes, valid_nodes = helper.prune_branches(F_lines,isolated_islands_pruned_graph_border,\
    shortest_allowed_branch_length)

    print("Dilating..")
    border_nodes = [node for branch in valid_nodes for node in branch]
    dilated_border = dilate_border(obj,border_nodes,dilation_size)

    return dilated_border, isolated_islands_pruned_graph

def write_breaking_curves(obj, borders_indices, output_dir, obj_name):
    border_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(obj.pcd.points)[borders_indices]))
    ppd = copy(obj.pcd)
    p_colors = np.zeros((len(ppd.points), 3))
    p_colors[:, 1] = 1
    p_colors[borders_indices] = [1, 0, 0]
    ppd.colors = o3d.utility.Vector3dVector(p_colors)
    o3d.io.write_point_cloud(os.path.join(output_dir, f'col_borders_{obj_name}.ply'), ppd)
    o3d.io.write_point_cloud(os.path.join(output_dir, f'borders_{obj_name}.ply'), border_pcd)

def segment_regions(obj, borders_indices, isolated_islands_pruned_graph):   
    seg_regions_indices = get_sides(isolated_islands_pruned_graph, borders_indices)
    node_face = {}
    face_nodes = []
    for idx,(_,face) in enumerate(seg_regions_indices):
        for node in face:
            node_face[node] = idx
        face_nodes.append(face)
    all_dilated = [node for  _,face in seg_regions_indices for node in face]
    all_dilated.extend([node for node in borders_indices])
    left_overs = list(set(range(len(obj.pcd.points)))-set(all_dilated))
    border_left_overs = left_overs+borders_indices
    print('Assigning border nodes..')
    expanded_faces = knn_expand(obj,border_left_overs,node_face,face_nodes,size=5)
    #print("expanded")
    seg_parts_array = []
    for f, expanded_face in enumerate(expanded_faces):
        mask = np.isin(np.arange(0, len(obj.pcd.points), 1).tolist(),[node for node in expanded_face])
        Obj_tmp = copy(obj)
        Obj_tmp.pcd = copy(obj.pcd)
        Obj_tmp.pcd.points = o3d.utility.Vector3dVector(np.asarray(obj.pcd.points)[mask])
        Obj_tmp.pcd.paint_uniform_color(np.asarray(colors[f % len(colors)]).astype("float") / 255.0)
        seg_parts_array.append(Obj_tmp)

    print("Creating colored version..")
    colored_regions = copy(obj.pcd)
    regions_col = np.zeros((len(obj.pcd.points), 3))
    for k, face in enumerate(expanded_faces):
        for point in face:
            regions_col[point] = colors[k % len(colors)]
    colored_regions.colors = o3d.utility.Vector3dVector(np.asarray(regions_col).astype("float") / 255.0)
    #o3d.visualization.draw_geometries([colored_regions])
    #pdb.set_trace()
    return seg_parts_array, seg_regions_indices, colored_regions

def write_segmented_regions(seg_parts_array, colored_regions, output_dir, obj_name):
    o3d.io.write_point_cloud(os.path.join(output_dir, f'col_regions_{obj_name}.ply'), colored_regions)
    saving_folder = os.path.join(output_dir, 'segmented_parts', obj_name)
    os.makedirs(saving_folder, exist_ok=True)
    for j, region in enumerate(seg_parts_array):
        o3d.io.write_point_cloud(os.path.join(saving_folder, f'{obj_name}_part_{j}_.ply'), region.pcd)