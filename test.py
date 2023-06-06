import os
os.environ['OMP_NUM_THREADS'] = '1'
import ast

from runner import test
import numpy as np

number = 192

#objects
Obj1_url = "data_test/breaking_bad/teacup/fractured_39/piece_0.obj"
Obj2_url = "data_test/breaking_bad/teacup/fractured_39/piece_1.obj"

#Before Evaluate do a Rotation and translation of
init_R_T = ((0.2, 0.2, 0.1), (-0.5, -0.5, -0.5))


#pipline name and pipline parameters
pipline_name = "pipline_breaking_teacup"


N = 15
t1 = 100
t2 = 100
t3 = 100


#threshold for corner
thre = 0.93

pipline_variables = (N, t1, t2, t3, thre)

#test
test_name = "ICP_test"

# eval

eval_list = ["rms"]

test_breaking_thin = test(Obj2_url, Obj1_url, init_R_T, pipline_name, pipline_variables, test_name, eval_list, show_results = True, save_results=False)

my_results = test_breaking_thin.run()
import pandas as pd
df_breaking_thin = pd.DataFrame(test_breaking_thin.results)
df_breaking_thin["total"] = df_breaking_thin["R_error"]+df_breaking_thin["T_error"]
df_breaking_thin.sort_values('T_error')
