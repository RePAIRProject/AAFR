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

    

    def run(self):
        # try:
        print("_________________________First Object_________________________")

        self.Obj1,self.Obj1_array = self.my_pipline.run(self.Obj1_url,self.pipline_variables)
        print("_________________________Second Object_________________________")

        self.Obj2,self.Obj2_array = self.my_pipline.run(self.Obj2_url,self.pipline_variables)


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
    def __init__(self, obj1_url, obj2_url, init_R_T, processing_pipeline, pipeline_variables, \
                 registration_module, evaluation_list, show_results=False, save_results=False, dir_name=None):
        if not os.path.exists("pipline_modules/"+pipeline+".py") :
            print("Error !  cannot find "+pipline+" module in pipline_modules")

        if save_results:
            if not os.path.exists("results"):
                os.mkdir("results")
            # date = datetime.datetime.now()
            # timestamp_str = date.strftime("%d-%b-%Y__%H_%M_%S_%f")
            # if not dir_name:
            #     dir_timestamp = "test_" + timestamp_str
            #     path = os.path.join("results", dir_timestamp)
            # else:
            #     path = os.path.join("results",dir_name)
            self.results_path = "results"
            # if not os.path.exists(path):
            #     os.mkdir(path)
        self.pair_name = obj1_url.split('/')[-2]
        self.obj1_url = obj1_url
        self.obj2_url = obj2_url
        self.pipeline_variables = pipeline_variables
        self.init_R_T = init_R_T
        self.pipeline_name = pipeline
        self.registration_name = registration_module
        self.evaluation_list_names = evaluation_list
        self.processing_pipeline = importlib.import_module(str("pipline_modules."+processing_pipeline))
        self.registration = importlib.import_module(str("test_modules."+registration_module))
        self.evaluation_list = [importlib.import_module(str("evaluation_modules."+my_eval)) for my_eval in evaluation_list]
        self.show_results = show_results
        self.save_results = save_results

    def process_fragments(self):
        """Read and process fragments (breaking curve segmentation)"""
        print("_________________________First Object_________________________")
        self.Obj1, self.Obj1_array = self.processing_pipeline.run(self.Obj1_url,self.pipeline_variables)
        print("_________________________Second Object_________________________")
        self.Obj2, self.Obj2_array = self.processing_pipeline.run(self.Obj2_url,self.pipeline_variables)

    def set_fragments_in_place(self, T_1, T_2):
        """Move both objects"""
        self.Obj1 = self.change_rotation_translation(self.Obj2,self.T_1)
        self.Obj1_array = self.change_rotation_translation(self.Obj2_array,self.T_1)
        self.Obj2 = self.change_rotation_translation(self.Obj2,self.T_2)
        self.Obj2_array = self.change_rotation_translation(self.Obj2_array,self.T_2)

    def apply_transformation(self, objects, T):
        if isinstance(Obj_arr, list):
            for i in range(len(Obj_arr)):
                objects = objects[i].pcd.transform(T)
            return objects
        else:
            objects = objects.pcd.transform(T)
            return objects

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

    def show_fragments(self):
        """Just visualize the fragments"""
        o3d.visualization.draw_geometries([self.Obj1, self.Obj2], 'Fragments')

    def save_fragments(self, output_dir, pcl_name=''):
        """Save the fragments to ply files"""
        o3d.io.write_point_cloud(os.path.join(output_dir, f'fragment1_{pcl_name}.ply'), self.Obj1)
        o3d.io.write_point_cloud(os.path.join(output_dir, f'fragment2_{pcl_name}.ply'), self.Obj1)

    def set_gt_M(self, GT):
        """Manually set the ground truth matrix"""
        self.RM_ground = GT

    def set_gt_R_T(self, R, T):
        """Manually set the ground truth matrix (rotation and translation separate)"""
        RM_ground = np.eye(4)
        RM_ground[:3, :3] = R
        RM_ground[:3, 3] = T
        self.RM_ground = RM_ground

    def register_fragments(self, init_T=np.eye(4)):
        """Register fragments (apply T before registration, if given)"""
        print("_________________________Registration_________________________")
        self.result_transformation_arr = self.registration.run(self.Obj1_array, self.Obj2_array, init_T)

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
        
    def save_results(self, save_all=True, save_parameters=True):
        if not self.sorted_results:
            self.get_winner()
        path_error = os.path.join(self.results_path, f"error_{self.pair_name}.csv")
        df_test = pd.DataFrame(self.sorted_results)
        df_test["transformations"] = [el[2] for el in self.result_transformation_arr]
        df_test.to_csv(path_error, encoding='utf-8')


