from objects.statue import objects
from evaluation_pairwise.utils import get_winner_pair, chamfer_distance, \
    save, sort_results, save_parts
import pandas as pd
import ast
from runner import test
import numpy as np
import pdb, json, os

prefix_run = 'STATUE'
for object_number in range(len(objects)):

    #objects
    Obj1_url = objects[object_number]["Obj1_url"]
    Obj2_url = objects[object_number]["Obj2_url"]
    print(Obj1_url,Obj2_url)
    pair_name = Obj1_url.split('/')[-2]
    f_name = f'{prefix_run}_{pair_name}'
    output_dir = os.path.join('results', f_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/parameters_{pair_name}.json', 'w') as jp:
        json.dump(objects[object_number], jp, indent=2)
    #Before Evaluate do a Rotation and translation of
    init_R_T = ((0.2, 0.2, 0.1), (-0.5, -0.5, -0.5))

    #pipline name and pipline parameters
    pipline_name = "general_pipeline"

    #sampling print(f'saved in {output_dir}')
    large_object = objects[object_number]["large_object"]
    small_object = objects[object_number]["small_object"]

    N = 15
    Object_t1 = objects[object_number]["to1"]
    Object_t2 = objects[object_number]["to2"]
    Object_t3 = objects[object_number]["to3"]

    border_t1 = objects[object_number]["tb1"]
    border_t2 = objects[object_number]["tb1"]
    border_t3 = objects[object_number]["tb1"]

    #threshold for corner
    thre = objects[object_number]["thre"]

    dilation_size = objects[object_number]["dilation_size"]

    pipline_variables = (small_object, large_object, N, Object_t1, Object_t2,
                        Object_t3, border_t1, border_t2, border_t3, dilation_size, thre)

    #test
    test_name = "ICP_test"

    # eval
    eval_list = ["rms"]

    test_bbad = test(Obj2_url, Obj1_url, init_R_T, pipline_name, pipline_variables, test_name, eval_list, show_results = True, save_results=True)

    my_results = test_bbad.run()

    winner, sorted_results = sort_results(test_bbad)
    sorted_results.to_csv(f'{output_dir}/results_{pair_name}_sorted.csv')
    with open(f'{output_dir}/winner_{pair_name}.json', 'w') as jp:
        json.dump(winner, jp, indent=2)

    save(output_dir, test_bbad, winner['index'])

    root_path = os.path.join(output_dir, "segmented_parts")
    parts_obj1_path = os.path.join(root_path, "obj1")
    parts_obj2_path = os.path.join(root_path, "obj2")
    save_parts(test_bbad.Obj1_array, parts_obj1_path, 'obj1')
    save_parts(test_bbad.Obj2_array, parts_obj2_path, 'obj2')
    

    print(f"Results (top {5}):\n")
    print(sorted_results.head(5))
    print("Winner:")
    for k in winner.keys():
        print(f"{k}: {winner[k]}")

    print(f'\nsaved in {output_dir}\n')
    pdb.set_trace()
    # test_breaking_thin.draw_registration_result_original_color(test_breaking_thin.Obj1,test_breaking_thin.Obj2,test_breaking_thin.result_transformation_arr[5][2])
