import open3d as o3d
import numpy as np
from copy import copy
np.random.seed(0)
def run(Obj1,Obj2,RT_Matrix):
    transformations = []
    for o1,obj1 in enumerate(Obj1):
        print(o1,len(obj1.pcd.points))
    for o2,obj2 in enumerate(Obj2):
        print(o2,len(obj2.pcd.points))
    for o1,obj1 in enumerate(Obj1):
        for o2,obj2 in enumerate(Obj2):

            target = copy(obj1.pcd)
            source = copy(obj2.pcd)

            # tf_param, _, _ = cpd.registration_cpd(copy(source), copy(target))

            init_trans = np.eye(4)
            # init_trans[:3, :3] = tf_param.rot

            result_icp_2 = o3d.pipelines.registration.registration_icp(
                    source,target,5000,init_trans,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling =False),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))

            transformations.append((o1,o2,result_icp_2.transformation))
            # print("_________________________Registeration test_________________________")
            # print(o1,o2)
            # print(result_icp_2)
    return transformations
