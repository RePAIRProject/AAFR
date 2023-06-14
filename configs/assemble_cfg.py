# parameters for the pipeline
pipeline_parameters = {
    'processing_module' : "standard",
    'registration_module' : "teaser",
    'evaluation_metrics' : ["rms"]
}

# data_list
num_of_points = 20000 # resampling to n points
# see paper for details (graph creation)
to = 100
tb = 0.1
dil = 0.01
thre = 0.93
N = 15
variables_as_list =  [num_of_points, num_of_points, N, to, to, to, tb, tb, tb, dil, thre]

import os, pdb, json
name = f'3dvr_{num_of_points}'
output_dir = os.path.join('3dvr_results', name)
os.makedirs(output_dir, exist_ok=True)

# list of broken objects (for now pairs)
data_folder = 'data'
data_list = []

for category_folder in os.listdir(data_folder):
    cat_ff = os.path.join(data_folder, category_folder)
    for fracture_folder in os.listdir(cat_ff):
        objects_folder = os.path.join(cat_ff, fracture_folder, 'objects')
        p_o1 = os.path.join(objects_folder, 'obj1_challenge.ply')
        p_o2 = os.path.join(objects_folder, 'obj2_challenge.ply')
        solution_folder = os.path.join(cat_ff, fracture_folder, 'solution')
        broken_obj_dict = {
            "path_obj1": p_o1,
            "path_obj2": p_o2,
            "category": category_folder,
            "fracture": fracture_folder
        }
        if os.path.exists(solution_folder):
            with open(os.path.join(solution_folder, 'solution.json'), 'r') as sj:
                solution = json.load(sj)
            broken_obj_dict['solution'] = solution

        data_list.append(broken_obj_dict)
