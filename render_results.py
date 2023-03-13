import open3d as o3d 
import os
import numpy as np 

root_folder= '/media/lucap/big_data/datasets/pairwise/ali/EXPERIMENTS/COMPARISON/Mug_fractured_73'
method = 'OURS'
ext = 'ply'
if ext == 'ply':
    source = o3d.io.read_point_cloud(os.path.join(root_folder, f"{method}_pred_piece_0.ply"))
    target = o3d.io.read_point_cloud(os.path.join(root_folder, f"{method}_pred_piece_1.ply"))
    target.paint_uniform_color([0, 0, 1])
    source.paint_uniform_color([0.9, 0.9, 0])
    o3d.visualization.draw_geometries([source, target], zoom=0.7,
                                  front=[0, -1, 0],
                                  lookat=[0, 0, 0],
                                  up=[0, 0, 1])
else:
    source_mesh = o3d.io.read_triangle_mesh(os.path.join(root_folder, f"{method}_piece_0.obj"))
    source = source_mesh.sample_points_uniformly(number_of_points=300000)
    target_mesh = o3d.io.read_triangle_mesh(os.path.join(root_folder, f"{method}_piece_1.obj"))
    target = target_mesh.sample_points_uniformly(number_of_points=100000)
    source_mesh.paint_uniform_color([0, 0, 1])
    target_mesh.paint_uniform_color([0.9, 0.9, 0])

    o3d.visualization.draw_geometries([source_mesh, target_mesh], zoom=0.7,
                                  front=[0, -1, 0],
                                  lookat=[0, 0, 0],
                                  up=[0, 0, 1])
#o3d.visualization.capture_screen_image('test.png', do_render=True)
                                    # front=[0, -1, 0],
                                    # up=[0, 0, 1],
                                    # zoom=[0.7])

# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 0.50000602006912231, 0.33127701282501221, 0.4359230101108551 ],
# 			"boundingbox_min" : [ -0.49998098611831665, -0.33128198981285095, -0.4359000027179718 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.068040778766282159, -0.99708720040010768, 0.034461097242449001 ],
# 			"lookat" : 
# 			[
# 				1.2516975402832031e-05,
# 				-2.4884939193725586e-06,
# 				1.1503696441650391e-05
# 			],
# 			"up" : [ -0.056734363721074599, 0.038352159032831588, 0.9976524063373402 ],
# 			"zoom" : 0.69999999999999996
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }