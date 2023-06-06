# parameters for the pipeline
pipeline_parameters = {
    'init_R_T' : ((0.2, 0.2, 0.1), (-0.5, -0.5, -0.5)), # it could be whatever
    'processing_module' : "standard",
    'registration_module' : "teaser",
    'evaluation_metrics' : ["rms"]
}

# data_list
num_of_points = 15000 # resampling to n points
# see paper for details (graph creation)
to = 100
tb = 0.1
dil = 0.01
thre = 0.93
N = 15

import os
name = f'drinkbottle_{num_of_points}'
output_dir = os.path.join('3dvr_results', name)
os.makedirs(output_dir, exist_ok=True)

# list of broken objects (for now pairs)
data_list = [
    {
        "path_obj1":"/media/lucap/big_data/datasets/pairwise/ali/DrinkBottle/fractured_62/piece_0.obj",
        "path_obj2":"/media/lucap/big_data/datasets/pairwise/ali/DrinkBottle/fractured_62/piece_1.obj",
        "small_object":num_of_points,"large_object":num_of_points,
        "to1":to ,"to2":to, "to3":to,
        "tb1":tb ,"tb2":tb, "tb3":tb,
        "dilation_size":dil,"thre":thre,
        "N":N,
        "variables_as_list": [num_of_points, num_of_points, N, to, to, to, tb, tb, tb, dil, thre]
    },
    {
        "path_obj1":"/media/lucap/big_data/datasets/pairwise/ali/DrinkBottle/fractured_70/piece_0.obj",
        "path_obj2":"/media/lucap/big_data/datasets/pairwise/ali/DrinkBottle/fractured_70/piece_1.obj",
        "small_object":num_of_points,"large_object":num_of_points,
        "to1": to ,"to2": to, "to3": to,
        "tb1":tb ,"tb2":tb, "tb3":tb,
        "dilation_size":dil,"thre":thre,
        "N":N,
        "variables_as_list": [num_of_points, num_of_points, N, to, to, to, tb, tb, tb, dil, thre]
    }
]