import open3d as o3d 
import os, pdb
import numpy as np 

#results_folder= '/home/lucap/code/AAFR/segmentation_results'
orig_folder = '/media/lucap/big_data/datasets/pairwise/ali/EXPERIMENTS/SEGMENTATION/cookie_part'
#exp_name = 'OTHER_DS_03_14_tuwien_brick_pair1'
#pcl_name = 'brick_part02.ply'
target_mesh = o3d.io.read_triangle_mesh(os.path.join(orig_folder, 'orig.obj'))
orig_pcl = target_mesh.sample_points_uniformly(number_of_points=50000)
orig_pcl.paint_uniform_color([0.5, 0.5, 0.5])
#orig_pcl = o3d.io.read_point_cloud(os.path.join(orig_folder, 'orig.ply'))
borders_pcl = o3d.io.read_point_cloud(os.path.join(orig_folder, "regions.ply"))
regions_pcl = o3d.io.read_point_cloud(os.path.join(orig_folder, "borders.ply"))

# o3d.visualization.draw_geometries([borders_pcl])
# pdb.set_trace()
# cookie 
front = [ 0.52289497871127666, 0.67088661505843528, 0.5258250573850286 ]
lookat = [ 0.13017669366140427, -0.37205301116060352, -0.021907060254475568 ]
up = [ -0.47064324742040053, -0.28708955129906194, 0.8343108073089327 ]
zoom = 1.26
# brick
# front = [ -0.96665089432787166, -0.21217119300606224, 0.1434204774553148 ]
# lookat = [ -2.5125162142857143, 0.45615583333333332, 0.52928675000000003 ]
# up = [ 0.25580869403044471, -0.82654404926244773, 0.50138492865988193 ]
# zoom = 0.86
# head
# front = [ -0.28896506992845827, 0.91486341750323474, 0.28200020509840662 ]
# lookat = [ 0.047834496945142746, 0.066043850034475327, 0.33967249840497971 ]
# up = [ 0.36858180851579014, -0.16554237199020558, 0.91473666894198891 ]
# zoom = 2
# manual 
o3d.visualization.draw_geometries([orig_pcl], zoom=zoom,
                                  front=front, lookat=lookat, up=up)
pdb.set_trace()

o3d.visualization.draw_geometries([borders_pcl], zoom=zoom,
                                  front=front, lookat=lookat, up=up)

o3d.visualization.draw_geometries([regions_pcl], zoom=zoom,
                                  front=front, lookat=lookat, up=up)

pdb.set_trace()

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Original', width=960, height=540, left=200, top=200)
vis.add_geometry(orig_pcl)
#vis.load_view_point("single_fragment_vp.json")

vis2 = o3d.visualization.Visualizer()
vis2.create_window(window_name='Breaking Curves', width=960, height=540, left=200+960, top=200)
vis2.add_geometry(borders_pcl)

vis3 = o3d.visualization.Visualizer()
vis3.create_window(window_name='Segmented Parts', width=960, height=540, left=0, top=0)
vis3.add_geometry(regions_pcl)
#vis3.capture_screen_image('test.png')

while True:
    #vis.update_geometry()
    if not vis.poll_events():
        break
    vis.update_renderer()

    #vis2.update_geometry()
    if not vis2.poll_events():
        break
    vis2.update_renderer()

    if not vis3.poll_events():
        break
    vis3.update_renderer()

vis.destroy_window()
vis2.destroy_window()
vis3.destroy_window()

#     source = o3d.io.read_point_cloud(os.path.join(root_folder, f"{method}_pred_piece_0.ply"))
#     target = o3d.io.read_point_cloud(os.path.join(root_folder, f"{method}_pred_piece_1.ply"))
#     target.paint_uniform_color([0, 0, 1])
#     source.paint_uniform_color([0.9, 0.9, 0])
#     o3d.visualization.draw_geometries([source, target], zoom=0.7,
#                                   front=[0, -1, 0],
#                                   lookat=[0, 0, 0],
#                                   up=[0, 0, 1])
# else:
#     source_mesh = o3d.io.read_triangle_mesh(os.path.join(root_folder, f"{method}_piece_0.obj"))
#     source = source_mesh.sample_points_uniformly(number_of_points=300000)
#     target_mesh = o3d.io.read_triangle_mesh(os.path.join(root_folder, f"{method}_piece_1.obj"))
#     target = target_mesh.sample_points_uniformly(number_of_points=100000)
#     source_mesh.paint_uniform_color([0, 0, 1])
#     target_mesh.paint_uniform_color([0.9, 0.9, 0])

#     o3d.visualization.draw_geometries([source_mesh, target_mesh], zoom=0.7,
#                                   front=[0, -1, 0],
#                                   lookat=[0, 0, 0],
#                                   up=[0, 0, 1])
# #o3d.visualization.capture_screen_image('test.png', do_render=True)
#                                     # front=[0, -1, 0],
#                                     # up=[0, 0, 1],
#                                     # zoom=[0.7])

# # {
# # 	"class_name" : "ViewTrajectory",
# # 	"interval" : 29,
# # 	"is_loop" : false,
# # 	"trajectory" : 
# # 	[
# # 		{
# # 			"boundingbox_max" : [ 0.50000602006912231, 0.33127701282501221, 0.4359230101108551 ],
# # 			"boundingbox_min" : [ -0.49998098611831665, -0.33128198981285095, -0.4359000027179718 ],
# # 			"field_of_view" : 60.0,
# # 			"front" : [ -0.068040778766282159, -0.99708720040010768, 0.034461097242449001 ],
# # 			"lookat" : 
# # 			[
# # 				1.2516975402832031e-05,
# # 				-2.4884939193725586e-06,
# # 				1.1503696441650391e-05
# # 			],
# # 			"up" : [ -0.056734363721074599, 0.038352159032831588, 0.9976524063373402 ],
# # 			"zoom" : 0.69999999999999996
# # 		}
# # 	],
# # 	"version_major" : 1,
# # 	"version_minor" : 0
# # }