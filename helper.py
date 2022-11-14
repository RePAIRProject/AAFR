import open3d as o3d
import numpy as np
import scipy.spatial
from tqdm import tqdm
import math
import networkx as nx
import heapq
from disjoint import DisjointSetExtra
import random
import matplotlib
from collections import Counter
from itertools import count


def down_sample_to(obj,num):
    if len(obj.points) < num:
        raise Exception("Sorry, num should be less than number of voxels in the objects")
    diff = 0
    answer = None
    g_counter = 0
    while not answer and g_counter<5:
        g_counter+=1
        diff  += 0.01
        right = 1
        left = 0
        curr = 0.5
        curr_num = len(obj.points)
        counter = 0

        while right>left and counter<50:
            counter+=1
            curr = float(right+left)/2
            downpcd = obj.voxel_down_sample(voxel_size=curr)
            if abs(len(downpcd.points)-num)/num < diff:
                answer = curr
                break
            elif len(downpcd.points) < num:
                right = curr - 0.000001
            else:
                left = curr
        answer = curr
    return answer

def load_cloud(url,voxel_size=30000):
      pcd = o3d.io.read_point_cloud(url)
      voxel_percentage = down_sample_to(pcd,voxel_size)
      downpcd = pcd.voxel_down_sample(voxel_size=voxel_percentage)
      pcd_tree = o3d.geometry.KDTreeFlann(downpcd)
      return downpcd,pcd_tree

def remove_point(G,p,visited):
    counter = 0
    short_branches = []
    stack = [[p]]
    while stack:
        counter+=1
        curr_branch = stack.pop(0)
        curr_p  = curr_branch[-1]
        neighbors = set(G.neighbors(curr_p))-visited
        neighbors = list(neighbors)
        if len(neighbors) == 0:
            short_branches.append(curr_branch)
            continue
        for n in neighbors:
            visited.add(n)
            stack.append(curr_branch+[n])
    return short_branches
def short_branch(G,p,visited,threshold):
    counter = 0
    short_branches = []
    stack = [[p]]
    while stack:
        counter+=1
        if counter >= threshold:
            return False
        curr_branch = stack.pop(0)
        curr_p  = curr_branch[-1]
        neighbors = set(G.neighbors(curr_p)) - visited
        neighbors = list(neighbors)
        for n in neighbors:
            if n not in visited:
                visited.add(n)
                stack.append(curr_branch+[n])
    return True
def path_length(G,p,q):
    counter = 0
    short_branches = []
    stack = [[p]]
    visited = set()
    while stack:
        counter+=1
        # if counter >= threshold:
        #     break
        curr_branch = stack.pop(0)
        curr_p  = curr_branch[-1]
        neighbors = list(G.neighbors(curr_p))
#         print(len(neighbors),"->",neighbors)
        if len(neighbors) == 1 and neighbors[0] in visited:
            short_branches.append(curr_branch[2:])
            continue
        for n in neighbors:
            if n == q:
                return counter+1
            if n not in visited:
                visited.add(n)
                stack.append(curr_branch+[n])

    return counter+1
def prune_branches(F_lines,Graph,shortest_allowed_branch_length):
    nodes = {}
    for branch in F_lines:
        for point in branch:
            neighbors = set(Graph.neighbors(point))
            if len(list(neighbors)) >= 2:
                nodes[point]=neighbors

    T = shortest_allowed_branch_length
    all_nodes_rem = []
    visited = set()
    for point,my_nodes in nodes.items():
        score = 0
        point_rem_nodes = []
        for node in my_nodes:
            if not short_branch(Graph,node,{point},T):
                score+=1
            else:
                point_rem_nodes.append(node)

        if score >= 2:
            for node in point_rem_nodes:
                tmp_rem_nodes = remove_point(Graph,node,{point})
                all_nodes_rem.extend(tmp_rem_nodes)
    removed_nodes = {node for branch in all_nodes_rem for node in branch}
    pruned_graph = Graph
    pruned_graph.remove_nodes_from(list(removed_nodes))
    valid_nodes = [[node for node in group-removed_nodes] for group in F_lines]
    return pruned_graph, removed_nodes, valid_nodes
#Create the Graph
def create_graph(Obj, radius, shortest_cycle_length, smallest_isolated_island_length):
    ds = DisjointSetExtra()
    Graph = nx.Graph()
    tree = o3d.geometry.KDTreeFlann(Obj.pcd)
    radius = np.mean(self.points_u)
    print("my radius is : ",radius)
    for idx,p in enumerate(tqdm(Obj.pcd.points)):
        [k, points_q, _] = tree.search_radius_vector_3d(Obj.pcd.points[idx],radius)
        distance = np.abs(np.linalg.norm(p-np.asarray(Obj.pcd.points).take(points_q,axis=0),axis=1))
        arr1inds = distance.argsort()
        points_q_sorted = np.asarray(points_q)[arr1inds]
        for q in points_q_sorted:
            if ds.exists(q) and ds.exists(int(idx)):
                if not ds.connected(q,int(idx)):
                    ds.connect(q,int(idx))
                    Graph.add_edge(q,int(idx))
                else:
                    if path_length(Graph,int(idx),q)>shortest_cycle_length:
                        ds.add(q,int(idx))
                        Graph.add_edge(q,int(idx))
            else:
                ds.add(q,int(idx))
                Graph.add_edge(q,int(idx))

    F_lines = []
    isolated_islands = []
    for group in list(ds.ds.itersets()):
        if len(group) < smallest_isolated_island_length:
            isolated_islands.extend(group)
        else:
            F_lines.append(group)

    Graph.remove_nodes_from(isolated_islands)

    return Graph, F_lines, isolated_islands
