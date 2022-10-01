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
from p_tqdm import p_umap
import os
import os.path
import pandas as pd
import itertools
import re
import yaml
from yaml.loader import SafeLoader

class experiment(object):
    """docstring for ."""

    def __init__(self, conf_file_name):
        self.conf_file_name = conf_file_name
        conf = self.read_conf(conf_file_name)
        test_combinations = self.combination_creation(conf)
        self.tests_objects = []
        for test_atts in test_combinations:
            for ((data,rot_tran),pipline),test_name in test_atts:
                Obj1_url,Obj2_url = data
                pipline_name,pipline_variables = pipline[0],pipline[1:]
                self.tests_objects.append(test(Obj1_url, Obj2_url,rot_tran, pipline_name,pipline_variables, test_name, conf["evaluation"] ))

    def run_test(self,test_obj):
        evals = test_obj.run()
        result = {"obj1":test_obj.Obj1_url, "obj2":test_obj.Obj2_url,\
                "pipeline":test_obj.pipline_name,"params":test_obj.pipline_variables,\
                 "R & T":test_obj.init_R_T,  "transformation": test_obj.result_transformation,\
                 "test":test_obj.test_name }
        for key,val in evals.items():
            result[key] = val
        return result


    def run(self):
        results_tmp = p_umap(self.run_test, self.tests_objects)
        self.results = pd.DataFrame(results_tmp)

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
            my_values = [float(string)]
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
            print(my_conf)
            return my_conf

class test(object):

    def __init__(self, Obj1_url, Obj2_url, init_R_T, pipline, pipline_variables, my_test, evaluation_list):
        self.Obj1_url = Obj1_url
        self.Obj2_url = Obj2_url
        self.pipline_variables = pipline_variables
        self.init_R_T = init_R_T
        self.pipline_name = pipline
        self.test_name = my_test
        self.my_pipline = importlib.import_module(str("pipline_modules."+pipline))
        self.my_test = importlib.import_module(str("test_modules."+my_test))
        self.evaluation_list = [importlib.import_module(str("evaluation_modules."+my_eval)) for my_eval in evaluation_list]

    def run(self):
        # try:
        Obj1 = self.my_pipline.run(self.Obj1_url,self.pipline_variables)
        Obj2 = self.my_pipline.run(self.Obj2_url,self.pipline_variables)
        Obj2 = self.change_rotation_translation(Obj2,self.init_R_T)
        self.result_transformation = self.my_test.run(Obj1,Obj2)
        return self.evalute(self.init_R_T,self.result_transformation)
        # except:
        #     return -1,-1

    def evalute(self,init_R_T,result_transformation):
        results = dict()
        for eval in self.evaluation_list:
            tmp_res = eval.run(self.init_R_T,self.result_transformation)
            for key,val in tmp_res.items():
                results[key] = val
        return  results


    def change_rotation_translation(self,Obj,init_R_T):
        R,T = init_R_T
        RM = Obj.pcd.get_rotation_matrix_from_xyz((R[0], R[1], R[2]))
        Obj.pcd.rotate(RM, center=(0, 0, 0))
        Obj.pcd.translate((T[0], T[1], T[2]))
        return Obj
