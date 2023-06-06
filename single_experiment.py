#from evaluation_pairwise.objects import objects
from evaluation_pairwise.utils import get_winner_pair, chamfer_distance, save, save_arr, sort_results
import pandas as pd
import ast
from runner import test, fragment_reassembler
import numpy as np
import pdb, json, os

exp_params = {
    "obj1_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad_statue_pair1/breaking_bad_cat_statue_piece1.ply",
    "obj2_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad_statue_pair1/breaking_bad_cat_statue_piece2.ply",
    "small_object":1500,"large_object":2000,
    "to1":100 ,"to2":100, "to3":100,
    "tb1":0.1 ,"tb2":0.1, "tb3":0.1,
    "dilation_size":0.05,"thre":0.90,
    "N":15, "pipeline_name":"general_pipeline"
}

pair_name = exp_params['obj1_url'].split('/')[-2]
print('Working on', pair_name)

f_name = f'exp_{pair_name}'
output_dir = os.path.join('results', f_name)
os.makedirs(output_dir, exist_ok=True)

#Before Evaluate do a Rotation and translation of
init_R_T = ((0.2, 0.2, 0.1), (-0.5, -0.5, -0.5))
pipeline_variables = (exp_params['small_object'], exp_params['large_object'], exp_params['N'], \
                      exp_params['to1'], exp_params['to2'], exp_params['to3'], 
                      exp_params['tb1'], exp_params['tb2'], exp_params['tb3'], 
                      exp_params['dilation_size'], exp_params['thre'])
#test
test_name = "ICP_test"
# eval
eval_list = ["rms"]

frag_ass = test(exp_params['obj1_url'], exp_params['obj2_url'], \
                init_R_T, exp_params['pipeline_name'], pipeline_variables, \
                test_name, eval_list, show_results=True, save_results=True, \
                dir_name=pair_name)

# run the pipeline
results = frag_ass.run()

# get results
winner, sorted_results = sort_results(frag_ass)
sorted_results.to_csv(f'{output_dir}/results_{pair_name}_sorted.csv')
with open(f'{output_dir}/winner_{pair_name}.json', 'w') as jp:
    json.dump(winner, jp, indent=2)

save(f_name, frag_ass, winner['index'])

frag_ass.draw_registration_result_original_color(frag_ass.Obj1, frag_ass.Obj2, \
    frag_ass.result_transformation_arr[winner['index']][2])

