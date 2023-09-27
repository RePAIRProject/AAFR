import numpy as np 
from helper import down_sample_to
import open3d as o3d 
import json 
import argparse 
import yaml 
import os 

def read_obj(path, voxel_size):

    if path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(path)
        voxel_percentage = down_sample_to(pcd,voxel_size)
        #print("my number is -> ",voxel_percentage)
        downpcd1 = pcd.voxel_down_sample(voxel_size=voxel_percentage)
    elif path.endswith('.obj'):
        mesh = o3d.io.read_triangle_mesh(path)
        downpcd1 = mesh.sample_points_uniformly(number_of_points=voxel_size)
    else:
        print("we only support .ply or .obj at the moment!\nFound", cfg['path_obj1'])
        return None, -1

    return downpcd1, 0

def main(args):

    print('Config file:', args.cfg)
    with open(args.cfg, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    for obj_pair in cfg['data_list']:
        
        print('Preparing:')
        path_obj1 = obj_pair[0]
        path_obj2 = obj_pair[1]
        print(path_obj1)
        print(path_obj2)
        
        pcd1, ok1 = read_obj(path_obj1, voxel_size = cfg['voxel_size'])
        pcd2, ok2 = read_obj(path_obj2, voxel_size = cfg['voxel_size'])

        if ok1 == 0 and ok2 == 0:

            # moving to origin if we need it (see config file, default False)
            if cfg['move_to_origin']:
                c1 = np.mean(np.asarray(pcd1.points), axis=1)
                pcd1.translate(-c1)
                c2 = np.mean(np.asarray(pcd2.points), axis=1)
                pcd1.translate(-c2)

            r = np.asarray(cfg['r'])
            t = np.asarray(cfg['t'])
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(r)
            trsf_mat = np.eye(4)
            trsf_mat[:3, :3] = rot_mat
            trsf_mat[:3, 3] = t
            r_inv = np.linalg.inv(rot_mat)
            t_inv = -t
            gt = np.eye(4)
            gt[:3, :3] = r_inv
            gt[:3, 3] = t_inv

            pcd2.transform(trsf_mat)

            category = path_obj1.split("/")[-3]
            fracture = path_obj1.split("/")[-2]
            output_dir = os.path.join(cfg['output_folder'], category, fracture)

            output_pcds = os.path.join(output_dir, 'objects')
            output_sol = os.path.join(output_dir, 'solution')
            os.makedirs(output_pcds, exist_ok=True)
            os.makedirs(output_sol, exist_ok=True)
            solution = {
                'T' : trsf_mat.tolist(),
                'r' : rot_mat.tolist(),
                't' : t.tolist(),
                'GT': gt.tolist(),
                'gt_r': r_inv.tolist(),
                'gt_t': t_inv.tolist()
            }
            with open(os.path.join(output_sol, 'solution.json'), 'w') as osj:
                json.dump(solution, osj, indent=3)
            o3d.io.write_point_cloud(os.path.join(output_pcds, 'obj1_challenge.ply'), pcd1)
            o3d.io.write_point_cloud(os.path.join(output_pcds, 'obj2_challenge.ply'), pcd2)
            print('Saved in', output_dir)
        else:
            print('something went wrong! please check config file and input paths.')
    print('FINISHED')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preparing challenge (moving objects)')
    parser.add_argument('--cfg', type=str, default='base', help='config file (.yaml)')
    args = parser.parse_args()
    main(args)