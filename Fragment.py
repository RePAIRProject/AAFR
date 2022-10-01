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
import helper
# a global
tiebreaker = count()

class FeatureLines(object):
    """docstring for ."""

    def __init__(self, url, voxel_size=0.1):
        self.pcd,self.pcd_tree = helper.load_cloud(url,voxel_size)

    def init(self,num_points):

        self.num_points = num_points
        self.points_q_idxs, self.points_q_points, self.points_u, self.points_c, \
        self.points_eig_vecs, self.points_eig_vals, self.k_points,  = self.cal_all_points_main_atts(self.pcd,self.pcd_tree,num_points = self.num_points)


        self.w_cr_v = self.cal_crease_penalty_vector(self.points_eig_vals,self.points_eig_vecs)
        self.w_co = self.cal_corner_penalty(self.points_eig_vals,self.points_eig_vecs)
        self.e_vectors_mag, self.e_vectors_dir = self.cal_p_q_vectors(self.pcd.points,self.points_q_points)

        #problem
        self.w_k = self.cal_curvature_estimate(self.pcd.points,self.points_eig_vals,self.points_eig_vecs,self.points_c,self.points_u)

        #works
        self.w_b2 = self.cal_max_angle()
        self.w_b1 = self.cal_border_penalty_vector(self.points_eig_vals,self.points_eig_vecs)

    def NormalizeData(self,data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def get_dist(self,a,b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def corner_weight(self,eig_vals):
        w_corner = (eig_vals[2]-eig_vals[0])/eig_vals[2]
        return w_corner


    def cal_all_points_main_atts(self,pcd, pcd_tree,num_points):
        def theta(v, w): return np.arccos(v.dot(w)/(np.linalg.norm(v)*np.linalg.norm(w)))
        points_q_points = []
        points_q_idxs = []
        points_u = []
        points_c = []
        points_eig_vecs = []
        points_eig_vals = []
        points_k = []
        centroid = self.pcd.get_center()
        for i in range(len(pcd.points)):
            point = pcd.points[i]
            [k, idx, _] = pcd_tree.search_knn_vector_3d(point, num_points)

            points_q_idxs.append(idx)
            q_points = np.asarray(pcd.points).take(idx,axis=0)
            points_q_points.append(q_points)

            points_u.append(np.mean(np.abs(q_points[1:] - point)))
            ui = points_u[i]
            points_c.append(np.mean(q_points[1:], axis=0))
            ci = points_c[i]

            term1 = ci - q_points[1:]
            res = np.zeros((3,3))
            for qt in term1:
                res += np.outer(qt,qt.T)
            res /= q_points.shape[0]
            CI = res
            # CI = np.corrcoef(q_points[1:],rowvar=False)
            eig_vals, eig_vecs = np.linalg.eig(CI)

            arr1inds = eig_vals.argsort()
            eig_vals = eig_vals[arr1inds]
            eig_vecs = eig_vecs[arr1inds]
            points_eig_vecs.append(eig_vecs)
            points_eig_vals.append(eig_vals)
            e0 = eig_vecs[0]
            p = point
            d1 = np.abs(np.linalg.norm(e0.T.dot(p-ci)))
            e01 = e0*-1
            d2 = np.abs(np.linalg.norm(e01.T.dot(p-ci)))
            # if d1 != 0:
            #     print(e0,p-ci,e0.T.dot(p-ci))
            points_k.append(d1)
        return np.asarray(points_q_idxs), np.asarray(points_q_points), np.asarray(points_u), np.asarray(points_c), np.asarray(points_eig_vecs), np.asarray(points_eig_vals), np.asarray(points_k)

    def cal_max_angle(self):
        all_angles = []
        for i in range(len(self.points_q_points)):
            p,q_points = self.pcd.points[i],self.points_q_points[i]
            pq_vecs = q_points
            P_xy = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
            projected_vecs = []
            for vec in pq_vecs:
                projected_vecs.append(np.asarray((P_xy*np.matrix(vec).T)).flatten())
            # first vector is the P vector
            projected_vecs = projected_vecs-projected_vecs[0]
            angles = [np.arctan2(vec[0],vec[1]) for vec in projected_vecs]
            sorted_angles = sorted(angles)
            abs_angle = float("-inf")
            true_angle = float("-inf")
            for angle in range(len(sorted_angles)+4):
                angle_1 = sorted_angles[0]
                sorted_angles.append(sorted_angles.pop(0))
                angle_2 = sorted_angles[0]
                sorted_angles.append(sorted_angles.pop(0))
                if abs_angle<abs(angle_2-angle_1):
                    abs_angle = abs(angle_2-angle_1)
                    true_angle = angle_2-angle_1
            all_angles.append(true_angle)
        beta = 1-(np.abs(np.asarray(all_angles))/(2*np.pi))
        return beta

    # W_k
    def cal_curvature_estimate(self,points,points_eig_vals,points_eig_vecs,points_c,points_u):
        d = np.abs(((np.asarray(points)-np.asarray(points_c)) * np.asarray(points_eig_vecs)[:,0]).sum(axis=1))
        w_k = np.abs((2*d)/(np.asarray(points_u)**2))
        w_k_max = np.amax(w_k)
        w_k = 1-w_k/w_k_max
        w_k = self.NormalizeData(w_k)
        return w_k

    #  W_cr
    def cal_crease_penalty_vector(self,points_eig_vals,points_eig_vecs):
        #primary ellipsoid direction penalty function
        term1 = points_eig_vals[:,2]-points_eig_vals[:,1]
        term2 = points_eig_vals[:,2]-np.abs(points_eig_vals[:,1]+points_eig_vals[:,0])
        max_t1_t2 = np.maximum(term1,term2)/points_eig_vals[:,2]
        w_points_crease_vector = np.repeat(max_t1_t2.reshape(max_t1_t2.shape[0],1),3,axis=1)*points_eig_vecs[:,2]
        return w_points_crease_vector

    # W_b1
    def cal_border_penalty_vector(self,points_eig_vals,points_eig_vecs):
        term1 =(points_eig_vals[:,2]-2*points_eig_vals[:,1])/points_eig_vals[:,2]
        w_b1 = term1.reshape(term1.shape[0],1).repeat(3,axis=1) * points_eig_vecs[:,2]
        return w_b1

    # W_co
    def cal_corner_penalty(self,points_eig_vals,points_eig_vecs):
        points_eig_vals = np.asarray(points_eig_vals)
        w_points_corner = (points_eig_vals[:,2]-points_eig_vals[:,0])/points_eig_vals[:,2]
        w_points_corner = self.NormalizeData(w_points_corner)
        return w_points_corner

    def cal_p_q_vectors(self,points,points_q_points):
        points = np.asarray(points).reshape(np.asarray(points).shape[0],1,np.asarray(points).shape[1])
        e_vectors = points_q_points - points
        e_vectors_mag = np.sqrt((e_vectors * e_vectors).sum(axis=2))
        e_vectors_dir = e_vectors/np.repeat(e_vectors_mag.reshape(e_vectors_mag.shape[0],e_vectors_mag.shape[1],1),3,axis=2)
        return e_vectors_mag,e_vectors_dir

    def weight_crease_penalty(self,alpha=0.2):
        term1 = alpha * (self.w_k.take(self.points_q_idxs) + self.w_k.reshape(self.w_k.shape[0],1).repeat(self.num_points,axis=1))
        w_cr_v_reshaped = self.w_cr_v.reshape(self.w_cr_v.shape[0],1,self.w_cr_v.shape[1]).repeat(self.num_points,axis=1)
        w_co_reshaped = self.w_co.reshape(self.w_co.shape[0],1).repeat(self.num_points,axis=1)
        p_term = np.minimum(np.abs((w_cr_v_reshaped * self.e_vectors_dir).sum(axis=2)),w_co_reshaped)

        w_cr_v_reshaped =self. w_cr_v.take(self.points_q_idxs,axis=0)
        w_co_reshaped = w_co_reshaped.take(self.points_q_idxs)
        q_term = np.minimum(np.abs((w_cr_v_reshaped * self.e_vectors_dir).sum(axis=2)),w_co_reshaped)
        term2 = (1-alpha) * (p_term+q_term)
        points_u_p = self.points_u.reshape(self.points_u.shape[0],1).repeat(self.num_points,axis=1)
        points_u_q = self.points_u.take(self.points_q_idxs)
        term3 = 2*np.abs(self.e_vectors_mag)/((np.abs(points_u_p))+np.abs((points_u_q)))
        w_c = term1+term2+term3
        w_c_vertex_penalty = term1+term2
        return  np.nan_to_num(w_c),np.nan_to_num(w_c_vertex_penalty)

    def weight_border_penalty(self,gamma=0.5):
        term1 = gamma*(self.w_b2.reshape(self.w_b2.shape[0],1).repeat(self.num_points,axis=1)+self.w_b2.take(self.points_q_idxs))

        w_b1_reshaped_p = self.w_b1.reshape(self.w_b1.shape[0],1,self.w_b1.shape[1]).repeat(self.num_points,axis=1)

        w_b1_reshaped_q = self.w_b1.take(self.points_q_idxs,axis=0)
        term2 =(1-gamma)* (np.abs(np.sum(np.nan_to_num(w_b1_reshaped_p*self.e_vectors_dir),axis=2)) + np.abs(np.sum(np.nan_to_num(w_b1_reshaped_q*self.e_vectors_dir),axis=2)))

        points_u_p = self.points_u.reshape(self.points_u.shape[0],1).repeat(self.num_points,axis=1)
        points_u_q = self.points_u.take(self.points_q_idxs)
        term3 = 2*np.abs(self.e_vectors_mag)/(points_u_p+points_u_q)

        w_b = term1+term2+term3
        w_b_vertex_penalty = term1+term2
        return np.nan_to_num(w_b), np.nan_to_num(w_b_vertex_penalty)

    def array_to_graph(self):
        Graph = nx.Graph()
        edges = []
        for p in range(len(self.pcd.points)):
            for q_ind,q in enumerate(self.points_q_idxs[p][1:]):
                Graph.add_edge(int(p),int(q))
                Graph[int(p)][int(q)]["crease_penalty"] = self.w_c[int(p)][int(q_ind)]
                Graph[int(p)][int(q)]["border_penalty"] = self.w_b[int(p)][int(q_ind)]
                Graph[int(p)][int(q)]["border_vertex_penalty"] = self.w_b_vertex_penalty[int(p)][int(q_ind)]
                Graph[int(p)][int(q)]["crease_vertex_penalty"] = self.w_c_vertex_penalty[int(p)][int(q_ind)]
            Graph.nodes[int(p)]["w_cr_v"] = (1-self.NormalizeData(np.sqrt(np.sum(self.w_cr_v * self.w_cr_v ,axis=1))))[p]
            Graph.nodes[int(p)]["w_co"] = self.w_co[p]
            Graph.nodes[int(p)]["w_b1"] = (1-self.NormalizeData(np.sqrt(np.sum(self.w_b1 * self.w_b1 ,axis=1))))[p]
            Graph.nodes[int(p)]["w_b2"] = self.w_b2[p]
        return Graph

    def path_length(self,G,p,q,threshold):
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

    def save(self,path=""):
        nx.write_gpickle(self.my_graph,path+"my_network.gpickle")

    def load(self,path=""):
        self.my_graph = nx.read_gpickle(path+"my_network.gpickle")

    # create the graph with params (inner functions used by crease_prune and border_prune)
    def init_graph(self,alpha=0.2,gamma=0.5):
        self.w_c, self.w_c_vertex_penalty = self.weight_crease_penalty(alpha=alpha)
        self.w_b, self.w_b_vertex_penalty = self.weight_border_penalty(gamma=gamma)
        self.my_graph = self.array_to_graph()
    def __create_graph(self,pattern="crease",T=0.9,pattern_length_T=2):
            pattern_length_T = np.sqrt(len(self.pcd.points))//pattern_length_T
            ds_queue = DisjointSetExtra()
            nodes = set()
            edges = []
            other_edges = []
            for edge in np.asarray(self.my_graph.edges):
                if self.my_graph[edge[0]][edge[1]][pattern+"_vertex_penalty"] < T:
                    heapq.heappush(edges,(self.my_graph[edge[0]][edge[1]][pattern+"_penalty"],next(tiebreaker),(edge)))
                    ds_queue.add(edge[0],edge[1])
            tmp_Graph = nx.Graph()
            ds = DisjointSetExtra()
            pbar = tqdm(total=len(edges), unit = "B", unit_scale=True, position=0, leave=True)
            while len(edges)>0:
                w,_,edge = heapq.heappop(edges)

                # a cycle can occure only if the two nodes actually exists in the graph
                # so if either of the two nodes doesnot exists then we can add the edges
                if ds.exists(edge[0]) and ds.exists(edge[1]):
                    if not ds.connected(edge[0],edge[1]):
                        ds.connect(edge[0],edge[1])
                        tmp_Graph.add_edge(edge[0],edge[1])
                    elif helper.path_length(tmp_Graph,edge[0],edge[1],pattern_length_T)>pattern_length_T//2:
                        ds.add(edge[0],edge[1])
                        tmp_Graph.add_edge(edge[0],edge[1])
                else:
                    ds.add(edge[0],edge[1])
                    tmp_Graph.add_edge(edge[0],edge[1])

            F_lines = []
            for group in list(ds.ds.itersets()):
                if len(group) < pattern_length_T//2:
                    continue
                F_lines.append(group)
            return F_lines,tmp_Graph
    def __create_crease_graph(self, T=0.9,pattern_length_T=2):
        return self.__create_graph("crease",T,pattern_length_T)
    def __create_border_graph(self, T=0.9,pattern_length_T=2):
        return self.__create_graph("border",T,pattern_length_T)

    #prunning the graph
    def create_crease(self,T=0.9,pattern_length_T=2):
            self.crease_pattern,self.tmp_graph = self.__create_crease_graph(T,pattern_length_T)
            nodes = {}
            for branch in self.crease_pattern:
                for point in branch:
                    neighbors = set(self.tmp_graph.neighbors(point))
                    if len(list(neighbors)) >= 2:
                        nodes[point]=neighbors

            T = np.sqrt(np.asarray(self.pcd.points).shape[0])//2
            all_nodes_rem = []
            visited = set()
            for point,my_nodes in nodes.items():
                score = 0
                tmp_rem_nodes = []
                for node in my_nodes:
                    if not helper.short_branch(self.tmp_graph,node,{point},T):
                        score+=1
                    else:
                        tmp_rem_nodes.append(node)

                if score >= 2:
                    for node in tmp_rem_nodes:
                        tmp_rem_nodes = helper.remove_point(self.tmp_graph,node,{point})
                        all_nodes_rem.extend(tmp_rem_nodes)
            self.crease_pruned_points = {node for branch in all_nodes_rem for node in branch}
            self.crease_pattern_pruned = [[node for node in group-self.crease_pruned_points] for group in self.crease_pattern]
            # print(len(self.crease_pruned_points)/len(list(self.pcd.points)))
    def create_border(self,T=0.9,pattern_length_T=2):
        self.border_pattern,self.tmp_graph = self.__create_border_graph(T,pattern_length_T)
        nodes = {}
        for branch in self.border_pattern:
            for point in branch:
                neighbors = set(self.tmp_graph.neighbors(point))
                if len(list(neighbors)) >= 2:
                    nodes[point]=neighbors

        T = np.sqrt(np.asarray(self.pcd.points).shape[0])//2
        all_nodes_rem = []
        visited = set()
        for point,my_nodes in nodes.items():
            score = 0
            tmp_rem_nodes = []
            for node in my_nodes:
                if not helper.short_branch(self.tmp_graph,node,{point},T):
                    score+=1
                else:
                    tmp_rem_nodes.append(node)

            if score >= 2:
                for node in tmp_rem_nodes:
                    tmp_rem_nodes = helper.remove_point(self.tmp_graph,node,{point})
                    all_nodes_rem.extend(tmp_rem_nodes)

        self.border_pruned_points = {node for branch in all_nodes_rem for node in branch}
        self.border_pattern_pruned = [[node for node in group-self.border_pruned_points] for group in self.border_pattern]
        # print(len(self.border_pruned_points)/len(list(self.pcd.points)))
    def show_heat(self,weights,thre = 0):
        if len(weights.shape)>weights.shape[-1] > 1:
            weights = 1-self.NormalizeData(np.sqrt(np.sum(weights * weights,axis=1)))
        weights = weights.copy()
        # weights[weights>=thre] = 1
        cmap = matplotlib.cm.get_cmap('viridis')
        rgba = cmap(weights)
        rgb = rgba[:,:3]
        self.pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb).astype(np.float))
        o3d.visualization.draw_geometries([self.pcd])
    def show_pattern(self,pattern):
            groups = pattern
            colors = [(0,255,0) for _ in self.pcd.points]
            for group in groups:
                color = (random.randrange(0, 255),0,random.randrange(0, 255))
                for nodes in group:
                    colors[nodes] = color
            self.pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors).astype(np.float) / 255.0)
            o3d.visualization.draw_geometries([self.pcd])
