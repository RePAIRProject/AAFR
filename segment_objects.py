#from objects. import objects
from evaluation_pairwise.utils import get_winner_pair, chamfer_distance, \
    save, sort_results, save_parts
import pandas as pd
import ast
from runner import fragment_reassembler
import numpy as np
import pdb, json, os

# from objects.bottles import objects
# prefix_run = 'BOTTLES_03_14'

# from objects.other_datasets import objects
# prefix_run = 'ODS_v2_03_15'

# from objects.real_objects import objects
# prefix_run = 'scanned_03_16'

from objects.repair_objects import objects
prefix_run = 'repair_2-100-100_03_16'

# from objects.synth_objects import objects
# prefix_run = 'synth_10k_03_16'

# from objects.objects_quick import objects
# prefix_run = 'QUICK_TEST'

for object_number in range(len(objects)):

    #objects
    Obj1_url = objects[object_number]["Obj1_url"]
    Obj2_url = objects[object_number]["Obj2_url"]
    print(Obj1_url,Obj2_url)
    pair_name = Obj1_url.split('/')[-2]
    f_name = f'{prefix_run}_{pair_name}'
    output_dir = os.path.join('segmentation_results_synth', f_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/parameters_{pair_name}.json', 'w') as jp:
        json.dump(objects[object_number], jp, indent=2)
    #Before Evaluate do a Rotation and translation of
    init_R_T = ((0.2, 0.2, 0.1), (-0.5, -0.5, -0.5))

    #pipline name and pipline parameters
    pipeline_name = "visualization_pipeline"

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
    #pdb.set_trace()
    pipeline_variables = (small_object, large_object, N, Object_t1, Object_t2,
                        Object_t3, border_t1, border_t2, border_t3, dilation_size, thre)

    #test
    test_name = "ICP_test"

    # eval
    eval_list = ["rms"]

    fr_bbad = fragment_reassembler(Obj2_url, Obj1_url, init_R_T, pipeline_name, pipeline_variables, test_name, eval_list, show_results = True, save_results=True)
    print_line_length = 65
    print('-' * print_line_length)
    fr_bbad.load_objects()
    print('-' * print_line_length)
    fr_bbad.detect_breaking_curves()
    fr_bbad.save_breaking_curves(output_dir)
    print('-' * print_line_length)
    fr_bbad.segment_regions()
    fr_bbad.save_segmented_regions(output_dir)
    print('-' * print_line_length)

    print(f'\nsaved in {output_dir}\n')
    # test_breaking_thin.draw_registration_result_original_color(test_breaking_thin.Obj1,test_breaking_thin.Obj2,test_breaking_thin.result_transformation_arr[5][2])
