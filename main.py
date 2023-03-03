from evaluation_pairwise.objects import objects
from evaluation_pairwise.utils import get_winner_pair, chamfer_distance, save, save_arr
import pandas as pd
import ast
from runner import test
import numpy as np
import pdb, json 

for object_number in range(len(objects)):

    with open(f'results/json/parameters_{object_number}.json', 'w') as jp:
        json.dump(objects[object_number], jp, indent=2)

    #objects
    Obj1_url = objects[object_number]["Obj1_url"]
    Obj2_url = objects[object_number]["Obj2_url"]
    print(Obj1_url,Obj2_url)

    #Before Evaluate do a Rotation and translation of
    init_R_T = ((0.2, 0.2, 0.1), (-0.5, -0.5, -0.5))

    #pipline name and pipline parameters
    pipline_name = "general_pipeline"

    #sampling
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

    test_breaking_thin = test(Obj2_url, Obj1_url, init_R_T, pipline_name, pipline_variables, test_name, eval_list, show_results = True, save_results=True)

    my_results = test_breaking_thin.run()
    sorted_results = get_winner_pair(test_breaking_thin)
    #print(sorted_results[0])
    df_breaking_3 = pd.DataFrame(test_breaking_thin.results)
    df_breaking_3["total"] = df_breaking_3["R_error"]+df_breaking_3["T_error"]
    df_breaking_3.sort_values('T_error')
    df_breaking_3.to_csv(f'results/csv/results_{object_number}.csv')

    save_arr(f'results_array_{object_number}', test_breaking_thin)
    save(f'results_single_{object_number}', test_breaking_thin, object_number)

    #test_breaking_thin.draw_registration_result_original_color(test_breaking_thin.Obj1,test_breaking_thin.Obj2,test_breaking_thin.result_transformation_arr[5][2])
