from probreg import cpd
import numpy as np
from copy import copy
import open3d as o3d
import transforms3d
def run(Obj1, Obj2, RM):

    source = copy(Obj2[0].pcd)
    target = copy(Obj1[0].pcd)

    # tf_param, _, _ = cpd.registration_cpd(copy(source), copy(target))

    init_trans = np.eye(4)
    # init_trans[:3, :3] = tf_param.rot

    result_icp_2 = o3d.pipelines.registration.registration_icp(
            source,target,40, init_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling =False),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))


    print(result_icp_2.transformation)


    T_ICP, R_ICP, _, _ = transforms3d.affines.decompose44(result_icp_2.transformation)
    tf_param, _, _ = cpd.registration_cpd(copy(source), copy(target))
    rot = tf_param.rot
    T = tf_param.t
    RM_trial = np.eye(4)
    RM_trial[:3, :3] = rot
    RM_trial[0, 3] = T[0]
    RM_trial[1, 3] = T[1]
    RM_trial[2, 3] = T[2]
    print(RM_trial)
    return result_icp_2.transformation,result_icp_2.transformation
