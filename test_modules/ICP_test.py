import open3d as o3d
import numpy as np

def run(Obj1,Obj2):
    source = Obj1.pcd
    source.colors = o3d.utility.Vector3dVector(np.asarray([(0,1,0) for _ in source.points]).astype(np.float))

    target = Obj2.pcd
    target.colors = o3d.utility.Vector3dVector(np.asarray([(0,0,1) for _ in target.points]).astype(np.float))

    current_transformation = np.identity(4)

    result_icp = o3d.pipelines.registration.registration_icp(
        source,target,0.000000000000000000001, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling =False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))

    return result_icp.transformation
