import open3d as o3d
import numpy as np

def run(Obj1,Obj2,RT_Matrix):
    source = Obj2.pcd
    source.colors = o3d.utility.Vector3dVector(np.asarray([(0,1,0) for _ in source.points]).astype(np.float))

    target = Obj1.pcd
    target.colors = o3d.utility.Vector3dVector(np.asarray([(0,0,1) for _ in target.points]).astype(np.float))

    current_transformation = RT_Matrix

    # result_icp_1 = o3d.pipelines.registration.registration_icp(
    #     source,target,10000000, np.eye(4),
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling =False),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50000000))

    
    result_icp_2 = o3d.pipelines.registration.registration_icp(
            source,target,21, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling =False),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))

    return result_icp_2.transformation, result_icp_2.transformation
