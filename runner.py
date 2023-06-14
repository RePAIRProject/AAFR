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
from copy import copy
import feature_piplines
import importlib
from p_tqdm import p_umap,p_uimap
import os
import os.path
import pandas as pd
import itertools
import re
import yaml
from yaml.loader import SafeLoader
import datetime
from copy import copy,deepcopy
import pickle
np.random.seed(seed=0)
import json 
import pdb 

class experiment(object):
    """docstring for ."""

    def __init__(self, conf_file_name):


        t = datetime.datetime.now()
        time = t.strftime("[%d.%m.%y] Time - %H_%M_%S")
        self.logs_file = open("logs/"+ time + ".log", "a+")
        self.experiment_dir_name = None

        self.conf_file_name = conf_file_name
        self.log("conf file : "+self.conf_file_name)
        conf = self.read_conf(conf_file_name)
        self.log("Configurations : "+str(conf))
        test_combinations = self.combination_creation(conf)
        self.log("number of tests : "+str(len(test_combinations[0])))

        self.save_results = conf["save_results"]
        if self.save_results:
            date = datetime.datetime.now()
            timestampStr = date.strftime("%d-%b-%Y__%H_%M_%S_%f")
            self.experiment_dir_name = "experiment_"+timestampStr
            path = os.path.join("results",self.experiment_dir_name)
            os.mkdir(path)


        self.tests_objects = []
        for test_atts in test_combinations :
            for num,(((data,rot_tran),pipline),test_name) in enumerate(test_atts):
                Obj1_url,Obj2_url = data
                pipline_name,pipline_variables = pipline[0],pipline[1:]

                path = self.experiment_dir_name+"/test_"+test_name+str(num)
                self.tests_objects.append(test(Obj1_url, Obj2_url,rot_tran, pipline_name,pipline_variables, test_name, conf["evaluation"], show_results=False, save_results = self.save_results, dir_name = path ))

    def run_test(self,test_obj):
        result = {"obj1":test_obj.Obj1_url, "obj2":test_obj.Obj2_url,\
                "pipeline":test_obj.pipline_name,"params":test_obj.pipline_variables,\
                 "R & T":test_obj.init_R_T, \
                 "test":test_obj.test_name }

        try:
            evals = test_obj.run()
            for key,val in evals.items():
                result[key] = val

            result["transformation"] = test_obj.result_transformation
            if test_obj.save_results:
                result["dir"] = test_obj.results_path.split("/")[-1]
            result["status"] = "success"

        except Exception as e:
            result["status"] = "Failed"
            result["error"] = str(e)
        self.log(result)
        return result

    def log(self,msg):
        t = datetime.datetime.now()
        time = t.strftime("\n [%d.%m.%y] Time - %H_%M_%S")
        log_msg = str(msg)
        self.logs_file.write(time+" : "+ log_msg)

    def run(self):
        results_tmp = p_umap(self.run_test, self.tests_objects)
        self.results = pd.DataFrame(results_tmp)
        if self.save_results:
            try:
                path = os.path.join("results",self.experiment_dir_name,'results.csv')
                self.results.to_csv(path)
            except:
                print("ERROR in saving results")

    def get_number(self,NumberString):
        if NumberString.isdigit():
            Number = int(NumberString)
        else:
            Number = float(NumberString)
            if Number.is_integer():
                Number = int(Number)

        return Number

    def compile_string(self, string):
        string = str(string)
        fixed_number = re.compile("\d+(?:\.\d+)?")
        range_pattern = re.compile("range\(\d+(?:\.\d+)?,\d+(?:\.\d+)?,\d+(?:\.\d+)?\)")
        random_pattern = re.compile("random\(\d+(?:\.\d+)?,\d+(?:\.\d+)?,\d+(?:\.\d+)?\)")
        array_pattern = re.compile("array\(\[\d+(?:\.\d+)?,\d+(?:\.\d+)?,\d+(?:\.\d+)?\]\)")
        my_values = []
        if range_pattern.match(string):
            start,end,step = re.findall(r"\d+(?:\.\d+)?", string, re.I)
            my_values = np.arange(float(start), float(end), float(step))
        elif fixed_number.match(string):
            my_values = [self.get_number(string)]
        elif random_pattern.match(string):
            start,end,num = re.findall(r"\d+(?:\.\d+)?", string, re.I)
            start = self.get_number(start)
            end = self.get_number(end)
            my_values = np.random.uniform(low=start, high=end, size=(int(num),))
        elif array_pattern.match(string):
            arr = re.findall(r"\d+(?:\.\d+)?", string, re.I)
            my_values = [self.get_number(num) for num in arr]
        else:
            print("ERROR")
        return my_values

    def combination_creation(self, conf):
        final_lists = []
        data = conf["data"]
        rotation = [conf["rotation"]["x"],conf["rotation"]["y"],conf["rotation"]["z"]]
        translation = [conf["translation"]["x"],conf["translation"]["y"],conf["translation"]["z"]]
        rot_list = itertools.product(*rotation)

        tran_list= itertools.product(*translation)

        rot_tran = itertools.product(*[rot_list,tran_list])
        rot_tran_data = itertools.product(*[data,rot_tran])

        pipline_lists = []
        for pipline in conf["piplines"]:
            pipline_list = itertools.product(*[val for key,val in pipline.items()])
            rot_tran_data_pipline = itertools.product(*[rot_tran_data,pipline_list])
            pipline_lists.append(list(rot_tran_data_pipline))

        for test in conf["tests"]:
            for lst in pipline_lists:
                final_lists.append(list(itertools.product(*[lst,test])))

        return list(final_lists)

    def read_conf(self, config_file):
        my_conf = {}
        with open('expirments_conf/'+config_file) as f:
            data = yaml.load(f, Loader=SafeLoader)

            for key,val in data.items():
                if key == "data":
                    all_data = []
                    data = []
                    for dirpath, dirnames, filenames in os.walk(val):
                        for filename in [f for f in filenames if f.endswith(".ply")]:
                            data.append((dirpath, os.path.join(dirpath, filename)))
                    key_func = lambda x: x[0]
                    for key, group in itertools.groupby(data, key_func):
                        all_data.append([filename for dirname, filename in list(group)])
                    my_conf["data"] = all_data

                if key == "rotation":
                    my_conf["rotation"] = {}
                    for axis_name, axis_val in val.items():
                        my_conf["rotation"][axis_name] = None
                        my_conf["rotation"][axis_name] = self.compile_string(axis_val)

                if key == "translation":
                    my_conf["translation"] = {}
                    for axis_name, axis_val in val.items():
                        my_conf["translation"][axis_name] = None
                        my_conf["translation"][axis_name] = self.compile_string(axis_val)

                if key == "piplines":
                    my_conf["piplines"] = []
                    for pipline in val:
                        my_pipline = {"name":[pipline["name"]]}
                        del pipline["name"]
                        for axis_name, axis_val in pipline.items():
                            my_pipline[axis_name] = self.compile_string(axis_val)
                        my_conf["piplines"].append(my_pipline)

                if key == "tests":
                    tests = []
                    for test in val:
                        tests.append([test["name"]])
                    my_conf["tests"] = tests

                if key == "evaluation":
                    my_conf[key] = val

                if key == "save_results":
                    my_conf[key] = val
            print(my_conf)
            return my_conf

class test(object):

    def __init__(self, Obj1_url, Obj2_url, init_R_T, pipline, pipline_variables, my_test, evaluation_list, show_results=False, save_results=False, dir_name=None):
        if not os.path.exists("pipline_modules/"+pipline+".py") :
            print("Error !  cannot find "+pipline+" module in pipline_modules")

        if save_results:
            if not os.path.exists("results"):
                os.mkdir("results")

            # date = datetime.datetime.now()
            # timestampStr = date.strftime("%d-%b-%Y__%H_%M_%S_%f")
            # if not dir_name:
            #     dir = "test_"+timestampStr
            #     path = os.path.join("results", dir)
            # else:
            #     path = os.path.join("results",dir_name)
            self.results_path = "results" #path
            # if not os.path.exists(path):
            #     os.mkdir(path)
        self.Obj1_url = Obj1_url
        self.Obj2_url = Obj2_url
        self.pipline_variables = pipline_variables
        self.init_R_T = init_R_T
        self.pipline_name = pipline
        self.test_name = my_test
        self.evaluation_list_names = evaluation_list
        self.my_pipline = importlib.import_module(str("pipline_modules."+pipline))

        self.my_test = importlib.import_module(str("test_modules."+my_test))
        self.evaluation_list = [importlib.import_module(str("evaluation_modules."+my_eval)) for my_eval in evaluation_list]
        self.show_results = show_results
        self.save_results = save_results

    

    def run(self, folder_path=''):
        # try:
        print("_________________________First Object_________________________")

        self.Obj1,self.Obj1_array = self.my_pipline.run(self.Obj1_url,self.pipline_variables, folder_path)
        
        print("_________________________Second Object_________________________")

        self.Obj2,self.Obj2_array = self.my_pipline.run(self.Obj2_url,self.pipline_variables, folder_path)


        R,T = self.init_R_T
        RM_trial = np.eye(4)
        RM = self.Obj2.pcd.get_rotation_matrix_from_xyz(R)
        RM_trial[:3, :3] = RM
        RM_trial[0, 3] = T[0]
        RM_trial[1, 3] = T[1]
        RM_trial[2, 3] = T[2]

        RM_ground = np.linalg.inv(RM_trial)
        self.RM_ground = RM_ground

        print("_________________________Registeration_________________________")
        self.Obj2 = self.change_rotation_translation(self.Obj2,self.init_R_T)
        self.Obj2_array = self.change_rotation_translation(self.Obj2_array,self.init_R_T)
        self.result_transformation_arr = self.my_test.run(self.Obj1_array,self.Obj2_array,RM_trial)

        # print(np.matmul(np.transpose(RM_trial),self.result_transformation_1))
        # if self.show_results:
        #     self.show_after()

        print("_________________________Evaluation_________________________")
        self.results = [{**{"o1":o1, "o2":o2},**self.evalute(RM_ground,result_transformation)} for o1, o2, result_transformation in self.result_transformation_arr]

        return self.results
        # except:
        #     return -1,-1

    def evalute(self,RotationMatrix_and_Translation,result_transformation):
        results = dict()
        for eval in self.evaluation_list:
            tmp_res = eval.run(RotationMatrix_and_Translation,result_transformation)
            for key,val in tmp_res.items():
                results[key] = val
        return  results

    def change_rotation_translation(self,Obj_arr,init_R_T):
        R,T = init_R_T
        if isinstance(Obj_arr, list):
            for i in range(len(Obj_arr)):
                RM = Obj_arr[i].pcd.get_rotation_matrix_from_xyz((R[0], R[1], R[2]))
                Obj_arr[i].pcd.rotate(RM, center=(0, 0, 0))
                Obj_arr[i].pcd.translate((T[0], T[1], T[2]))
            return Obj_arr
        else:
            RM = Obj_arr.pcd.get_rotation_matrix_from_xyz((R[0], R[1], R[2]))
            Obj_arr.pcd.rotate(RM, center=(0, 0, 0))
            Obj_arr.pcd.translate((T[0], T[1], T[2]))
            return Obj_arr
    def draw_registration_result_original_color(self, obj1, obj2, transformation):
        pcd1 = deepcopy(obj1.pcd)
        pcd2 = deepcopy(obj2.pcd)
        # pcd1.colors = o3d.utility.Vector3dVector(np.asarray([(0,1,0) for _ in pcd1.points]).astype(np.float))
        # pcd2.colors = o3d.utility.Vector3dVector(np.asarray([(0,0,1) for _ in pcd2.points]).astype(np.float))
        pcd2.transform(transformation)
        o3d.visualization.draw_geometries([pcd1, pcd2])

    def save_objects_with_registration(self, obj1, obj2, transformation):
        # obj1.pcd.colors = o3d.utility.Vector3dVector(np.asarray([(0,1,0) for _ in obj1.pcd.points]).astype(np.float))
        # obj2.pcd.colors = o3d.utility.Vector3dVector(np.asarray([(0,0,1) for _ in obj2.pcd.points]).astype(np.float))
        obj2.pcd.transform(transformation)
        o3d.io.write_point_cloud(self.results_path+"/Obj1.ply", obj1.pcd, compressed=True)
        o3d.io.write_point_cloud(self.results_path+"/Obj2.ply", obj2.pcd, compressed=True)

    def show_before(self):
        self.draw_registration_result_original_color(copy(self.Obj1), copy(self.Obj2), np.identity(4))
    def show_after(self):
        self.draw_registration_result_original_color(copy(self.Obj1),  copy(self.Obj2), self.result_transformation)


    def save_test(self):
        test_vars = [self.Obj1_url, self.Obj2_url, self.pipline_variables, self.init_R_T,\
         self.pipline_name, self.test_name, self.my_test, self.evaluation_list_names, self.show_results, self.save_results,\
         self.results_path, self.result_transformation_arr, self.results]

        pipline_module = open(str("pipline_modules/"+self.pipline_name+".py"), "r")
        test_vars.append(pipline_module.read())

        test_module = open(str("test_modules/"+self.test_name+".py"), "r")
        test_vars.append(test_module.read())


        test_txt = open('my_ob', 'wb')
        file = open('my_ob', 'wb')
        pickle.dump(test_vars, file)
        file.close()

    def load_test(self,url):
        import sys,imp
        self.Obj1_url, self.Obj2_url, self.pipline_variables, self.init_R_T,\
        self.pipline_name, self.test_name, self.my_test, self.evaluation_list_names, self.show_results, self.save_results,\
        self.results_path, self.result_transformation_arr, self.results, pipline_module_code, test_module_code = pickle.load(url)

        self.pipline_module = imp.new_module(pipline_module_code)
        exec(pipline_module_code, self.pipline_module.__dict__)

        self.test_module = imp.new_module(test_module_code)
        exec(test_module_code, self.test_module.__dict__)


class fragment_reassembler(object):
    """
    An extension of the test class with function for each step of the pipeline 
    to debug and understand the codebase better and for further extension/modification
    of the pipeline components 
    """
    def __init__(self, broken_objects, variables_as_list, parameters, name='broken_objects', show_results=False, save_results=False, dir_name=None):
        
        if not os.path.exists("pipline_modules/"+parameters['processing_module']+".py") :
            print("Error !  cannot find "+parameters['processing_module']+" module in pipline_modules")

        self.pair_name = parameters
        self.obj1_url = broken_objects['path_obj1']
        self.obj2_url = broken_objects['path_obj2']
        self.pipeline_variables = variables_as_list
        self.pipeline_name = parameters['processing_module']
        self.registration_name = parameters['registration_module']
        self.evaluation_list_names = parameters['evaluation_metrics']
        self.processing_pipeline = importlib.import_module(str("pipline_modules."+self.pipeline_name))
        self.registration = importlib.import_module(str("test_modules."+self.registration_name))
        self.evaluation_list = [importlib.import_module(str("evaluation_modules."+my_eval)) for my_eval in self.evaluation_list_names]
        self.show_results = show_results
        self.save_results = save_results
        self.obj1_name = self.obj1_url.split('/')[-1][:-4]
        self.obj2_name = self.obj2_url.split('/')[-1][:-4]

    def load_objects(self):
        small = self.pipeline_variables[0]
        large = self.pipeline_variables[1]
        N = self.pipeline_variables[2]

        print('Loading object 1:', self.obj1_url)
        self.obj1 = self.processing_pipeline.load_obj(self.obj1_url, small, large, N)
        print('Loading object 2:', self.obj2_url)
        self.obj2 = self.processing_pipeline.load_obj(self.obj2_url, small, large, N)
        print('done') 

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        
    def detect_breaking_curves(self):
        if not self.obj1:
            self.load_objects()
        print('Detecting breaking curves for object 1..')
        self.obj1_borders_indices, self.obj1_isolated_islands_pruned_graph = self.processing_pipeline.detect_breaking_curves(self.obj1, self.pipeline_variables)
        print('Detecting breaking curves for object 2..')
        self.obj2_borders_indices, self.obj2_isolated_islands_pruned_graph = self.processing_pipeline.detect_breaking_curves(self.obj2, self.pipeline_variables)
        print('done')

    def save_breaking_curves(self):
        print('Saving breaking curves for object 1..')
        self.processing_pipeline.write_breaking_curves(self.obj1, self.obj1_borders_indices, self.output_dir, self.obj1_name)
        print('Saving breaking curves for object 2..')
        self.processing_pipeline.write_breaking_curves(self.obj2, self.obj2_borders_indices, self.output_dir, self.obj2_name)
        print('done')

    def segment_regions(self):
        print("The segmentation process is very slow at the moment: it visits all nodes in a queue to assign them to a region")
        print("The code is in the processing module in the get_sides method: feel free to help improving it (:")
        print('Segmenting object 1.. ')
        self.obj1_seg_parts_array, seg_regions_indices, self.obj1_colored_regions = self.processing_pipeline.segment_regions(self.obj1, self.obj1_borders_indices, self.obj1_isolated_islands_pruned_graph)
        print('Segmenting object 2.. ')
        self.obj2_seg_parts_array, seg_regions_indices, self.obj2_colored_regions = self.processing_pipeline.segment_regions(self.obj2, self.obj2_borders_indices, self.obj2_isolated_islands_pruned_graph)
        print('done')

    def save_segmented_regions(self):
        print('Saving segmented parts for object 1..')
        self.processing_pipeline.write_segmented_regions(self.obj1_seg_parts_array, self.obj1_colored_regions, self.output_dir, self.obj1_name)
        print('Saving segmented parts for object 2..')
        self.processing_pipeline.write_segmented_regions(self.obj2_seg_parts_array, self.obj2_colored_regions, self.output_dir, self.obj2_name)
        print('done')

    def save_fragments(self, pcl_name=''):
        """Save the fragments to ply files"""
        os.makedirs(os.path.join(self.output_dir, 'pointclouds'), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(self.output_dir, 'pointclouds', f'obj_{pcl_name}_part1.ply'), self.obj1.pcd)
        o3d.io.write_point_cloud(os.path.join(self.output_dir, 'pointclouds', f'obj_{pcl_name}_part2.ply'), self.obj2.pcd)

    def save_info(self):
        info = {
            'name_o1': self.obj1_name,
            'name_o2': self.obj2_name
        }
        with open(os.path.join(self.output_dir, 'info.json'), 'w') as ij:
            json.dump(info, ij, indent=2) 

    def set_gt(self, gt):
        """Manually set the ground truth matrices"""
        self.gt = gt
        # self.gt_r = gt[:3, :3] 
        # self.gt_t = gt[:3, 3]

    def register_segmented_regions(self):
        """Register segmented regions"""
        print("_________________________Registration_________________________")
        self.candidates_registration = self.registration.run(self.obj1_seg_parts_array, self.obj2_seg_parts_array)
        self.sorted_candidates_registration = self.candidates_registration.sort_values('chamfer_distance')
        self.best_registration = self.sorted_candidates_registration.head(1)

    def save_registration_results(self):
        self.folder_registration_results = os.path.join(self.output_dir, "registration")
        os.makedirs(self.folder_registration_results, exist_ok=True)
        self.sorted_candidates_registration.to_csv(os.path.join(self.folder_registration_results, 'sorted_candidates_registration.csv'))
        self.best_registration.to_csv(os.path.join(self.folder_registration_results, 'best_registration.csv'))
        
    def save_registered_pcls(self):
        obj1_to_draw = copy(self.obj1.pcd)
        obj2_to_draw = copy(self.obj2.pcd)
        obj1_to_draw.paint_uniform_color([1, 1, 0])
        obj2_to_draw.paint_uniform_color([0, 0, 1])
        obj2_to_draw = obj2_to_draw.transform(self.best_registration['transf_teaser'].item())
        o3d.io.write_point_cloud(os.path.join(self.folder_registration_results, 'obj1.ply'), obj1_to_draw)
        o3d.io.write_point_cloud(os.path.join(self.folder_registration_results, 'obj2.ply'), obj2_to_draw)

    def evaluate_error(self):
        """Evaluate against gt and estimate rmse(R) and rmse(T)"""
        return 1

    def evaluate_against_gt(self):
        if not self.gt:
            print("Set the ground truth first!")
            self.error = -1
        else:
            self.error = self.evaluate_error()

    def save_evaluation_results(self):
        folder_eval_res = os.path.join(self.output_dir, f"evaluation_{self.pair_name}")
        os.makedirs(folder_results, exist_ok=True)
        with open(os.path.join(folder_eval_res, 'evaluation.json'), 'w') as ejf:
            json.dump(self.error, ejf, indent=3)


    ### NOT SURE IF NEEDED
    ### parked here for now

    def set_fragments_in_place(self, T_1, T_2):
        """Move both objects"""
        self.obj1 = self.apply_transformation(self.obj1, T_1)
        if self.obj1_seg_parts_array:
            self.obj1_seg_parts_array = self.apply_transformation(self.obj2_seg_parts_array, T_1)
        self.obj2 = self.apply_transformation(self.obj2, T_2)
        if self.obj2_seg_parts_array:
            self.obj2_seg_parts_array = self.apply_transformation(self.obj2_seg_parts_array, T_2)

    def apply_transformation(self, objects, T):
        if isinstance(objects, list):
            for i in range(len(objects)):
                objects = objects[i].pcd.transform(T)
        else:
            objects = objects.pcd.transform(T)
        return objects

    def show_fragments(self):
        """Just visualize the fragments"""
        o3d.visualization.draw_geometries([self.obj1, self.obj2], 'Fragments')

    def evaluate_results(self):
        print("_________________________Evaluation_________________________")
        self.results = [{**{"o1":o1, "o2":o2},**self.evaluate(self.RM_ground, result_transformation)} for o1, o2, result_transformation in self.result_transformation_arr]

    def evaluate(self, gt_transf, estimated_transformation):
        results = dict()
        for eval_module in self.evaluation_list:
            tmp_res = eval_module.run(gt_transf, estimated_transformation)
            for key,val in tmp_res.items():
                results[key] = val
        return results

    def get_winner(self, topk=5):
        self.winner, self.sorted_results = sort_results(self)
        print(f"Results (top {topk}):\n")
        print(self.sorted_results.head(topk))
        print("Winner:")
        for k in self.winner.keys():
            print(f"{k}: {self.winner[k]}")
        
 