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

class FeatureLines(object):
    """docstring for ."""

    def __init__(self, url, voxel_size=0.1):
        self.pcd,self.pcd_tree = self.load_cloud(url,voxel_size)

    def set_params(self,alpha=0.2,gamma=0.5,crease_threshold=0.9,border_threshold=0.9,pattern_length_T=-1):

            if pattern_length_T == -1:
                pattern_length_T = np.sqrt(len(self.pcd.points))//2
            self.w_c = self.weight_crease_penalty(alpha=alpha)

            self.w_b = self.weight_border_penalty(gamma=gamma)

            self.crease_pattern = self.get_pattern("crease",crease_threshold,pattern_length_T)
            self.border_pattern = self.get_pattern("border",border_threshold,pattern_length_T)


    def init(self,num_points):

        self.num_points = num_points
        self.points_q_idxs, self.points_q_points, self.points_u, self.points_c, \
        self.points_eig_vecs, self.points_eig_vals, self.k_points,  = self.cal_all_points_main_atts(self.pcd,self.pcd_tree,num_points = self.num_points)


        self.w_cr_v = self.cal_crease_penalty_vector(self.points_eig_vals,self.points_eig_vecs)
        self.w_co = self.cal_corner_penalty(self.points_eig_vals,self.points_eig_vecs)
        self.e_vectors_mag, self.e_vectors_dir = self.cal_p_q_vectors(self.pcd.points,self.points_q_points)
        self.w_k = self.cal_curvature_estimate(self.pcd.points,self.points_eig_vals,self.points_eig_vecs,self.points_c,self.points_u)
        self.w_b2 = self.cal_max_angle(self.pcd.points,self.points_q_points)
        self.w_b1 = self.cal_border_penalty_vector(self.points_eig_vals,self.points_eig_vecs)


    def NormalizeData(self,data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def get_dist(self,a,b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def corner_weight(self,eig_vals):
        w_corner = (eig_vals[2]-eig_vals[0])/eig_vals[2]
        return w_corner

    def load_cloud(self,url,voxel_size=0.1):
      pcd = o3d.io.read_point_cloud(url)
      downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
      pcd_tree = o3d.geometry.KDTreeFlann(downpcd)
      return downpcd,pcd_tree

    def cal_all_points_main_atts(self,pcd, pcd_tree,num_points):
        points_q_points = []
        points_q_idxs = []
        points_u = []
        points_c = []
        points_eig_vecs = []
        points_eig_vals = []
        points_k = []
        for i in tqdm(range(len(pcd.points))):
            point = pcd.points[i]
            [k, idx, _] = pcd_tree.search_knn_vector_3d(point, num_points)

            points_q_idxs.append(idx)
            q_points = np.asarray(pcd.points).take(idx,axis=0)
            points_q_points.append(q_points)

            points_u.append(np.mean(np.abs(q_points[1:] - point)))
            ui = points_u[i]
            points_c.append(np.mean(q_points[1:], axis=0))
            ci = points_c[i]
            CI = np.corrcoef(q_points.T)
            eig_vals, eig_vecs = np.linalg.eig(np.nan_to_num(CI))

            arr1inds = eig_vals.argsort()
            eig_vals = eig_vals[arr1inds]
            eig_vecs = eig_vecs[arr1inds]
            points_eig_vecs.append(eig_vecs)
            points_eig_vals.append(eig_vals)
            e0 = eig_vecs[0]
            p = point
            d = abs(np.dot(e0,(p-ci)))
            points_k.append((2*d)/(ui**2))
        return np.asarray(points_q_idxs), np.asarray(points_q_points), np.asarray(points_u), np.asarray(points_c), np.asarray(points_eig_vecs), np.asarray(points_eig_vals), np.asarray(points_k)

    def cal_max_angle(self,points_p,points_q_points):
        max_angle = []
        for i in tqdm(range(len(points_q_points))):
            p,q_points = points_p[i],points_q_points[i]
            n = self.points_eig_vecs[i][0]
            vec = np.repeat(np.asarray([(np.dot(q_points,n))/(np.sqrt(n.dot(n)))**2]),3,axis=0)
            projection = np.multiply(vec.T, n)
            q_points_projected = np.asarray(q_points)-np.asarray(projection)
            q_points_projected_vectors = list(p-q_points_projected)
            angle = float("-inf")
            for j in range(len(q_points_projected_vectors)+2):
                vector_1 = q_points_projected_vectors[0]
                q_points_projected_vectors.append(q_points_projected_vectors.pop(0))
                vector_2 = q_points_projected_vectors[0]
                q_points_projected_vectors.append(q_points_projected_vectors.pop(0))
                unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                dot_product = np.dot(unit_vector_1.T, unit_vector_2)
                angle = max(angle,abs(math.degrees(np.arccos(dot_product))))
            max_angle.append(angle)
        beta = self.NormalizeData(1-(np.asarray(max_angle)/360))
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
        return  np.nan_to_num(w_c)

    def weight_border_penalty(self,gamma=0.5):

        term1 = gamma*(self.w_b2.reshape(self.w_b2.shape[0],1).repeat(self.num_points,axis=1)+self.w_b2.take(self.points_q_idxs))

        w_b1_reshaped_p = self.w_b1.reshape(self.w_b1.shape[0],1,self.w_b1.shape[1]).repeat(self.num_points,axis=1)

        w_b1_reshaped_q = self.w_b1.take(self.points_q_idxs,axis=0)
        term2 =(1-gamma)* (np.abs(np.sum(np.nan_to_num(w_b1_reshaped_p*self.e_vectors_dir),axis=2)) + np.abs(np.sum(np.nan_to_num(w_b1_reshaped_q*self.e_vectors_dir),axis=2)))

        points_u_p = self.points_u.reshape(self.points_u.shape[0],1).repeat(self.num_points,axis=1)
        points_u_q = self.points_u.take(self.points_q_idxs)
        term3 = 2*np.abs(self.e_vectors_mag)/(points_u_p+points_u_q)

        w_b = term1+term2+term3
        return np.nan_to_num(w_b)

    def array_to_graph(self,weights):
        G = nx.Graph()
        edges = []
        for p in tqdm(range(len(weights))):
            p_edges = [(int(p),int(q),float(w)) for q,w in zip(self.points_q_idxs[p][1:],weights[p][1:])]
            G.add_weighted_edges_from(p_edges)
        return G


    def get_pattern(self,pattern="crease",T=0.9,pattern_length_T=-1):

        ds_queue = DisjointSetExtra()
        if pattern == "crease":
            my_graph = self.array_to_graph(self.w_c)
        elif pattern == "border":
            my_graph = self.array_to_graph(self.w_b)
        else:
            raise Exception("invalid pattern type")
        nodes = set()
        other_nodes = set()
        my_edge = set()
        edges = []
        other_edges = []
        for edge in np.asarray(my_graph.edges):
            if my_graph[edge[0]][edge[1]]["weight"] < T:
                heapq.heappush(edges,(my_graph[edge[0]][edge[1]]["weight"],(edge)))
                ds_queue.add(edge[0],edge[1])

        ds = DisjointSetExtra()
        pbar = tqdm(total=len(edges), unit = "B", unit_scale=True, position=0, leave=True)
        while len(edges)>0:
            w,edge = heapq.heappop(edges)
            if ds.exists(edge[0]) and ds.exists(edge[1]):
                if not ds.connected(edge[0],edge[1]):
                    if ds.count(edge[0]) + ds.count(edge[1]) >= pattern_length_T//2:
                        ds.connect(edge[0],edge[1])
                    # ds_queue.connect(edge[0],edge[1])
            else:
                ds.add(edge[0],edge[1])
            pbar.update(1)

        F_lines = []
        for group in list(ds.ds.itersets()):
            if len(group) < pattern_length_T//4:
                continue
            F_lines.append(group)
        return F_lines

    def show_heat(self,weights):
        if weights.shape[-1] == 3:
            weights = self.NormalizeData(np.sqrt(np.sum(weights * weights,axis=1)))
        cmap = matplotlib.cm.get_cmap('viridis')
        rgba = cmap(weights)
        rgb = rgba[:,:3]
        self.pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb).astype(np.float))
        o3d.visualization.draw_geometries([self.pcd])

    def show(self,pattern="common_crease_border"):
        if pattern == "crease":
            groups = self.crease_pattern
            colors = [(0,255,0) for _ in self.pcd.points]
            for group in groups:
                color = (random.randrange(0, 255),0,random.randrange(0, 255))
                for nodes in group:
                    colors[nodes] = color
        elif    pattern == "border":
            groups = self.border_pattern
            colors = [(0,255,0) for _ in self.pcd.points]
            for group in groups:
                color = (random.randrange(0, 255),0,random.randrange(0, 255))
                for nodes in group:
                    colors[nodes] = color
        elif pattern == "common_crease_border":
            crease = {node for group in self.crease_pattern for node in group}
            border =  {node for group in self.border_pattern for node in group}
            common = list(set(crease).intersection(border))
            only_crease  = list(set(crease) - set(common))
            only_border  = list(set(border) - set(common))
            colors = [(0,255,0) for _ in self.pcd.points]
            for node in common:
                colors[node] = (0,0,0)
            for node in only_crease:
                colors[node] = (255,0,0)
            for node in only_border:
                colors[node] = (0,0,255)
        elif pattern == "heat_border":
            cmap = matplotlib.cm.get_cmap('viridis')
            rgba = cmap(np.sqrt(np.sum(self.w_b1 * self.w_b1,axis=1)))
            rgb = rgba[:,:3]
            colors = rgb*255.0

        elif pattern == "heat_crease":
            cmap = matplotlib.cm.get_cmap('viridis')
            rgba = cmap(np.sqrt(np.sum(self.w_cr_v * self.w_cr_v,axis=1)))
            rgb = rgba[:,:3]
            colors = rgb*255.0

        self.pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors).astype(np.float) / 255.0)
        o3d.visualization.draw_geometries([self.pcd])
