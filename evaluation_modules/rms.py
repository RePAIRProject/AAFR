import numpy as np
import transforms3d
def run(RM_ground,result_transformation):

    
    T_ground, R_ground, _, _ = transforms3d.affines.decompose44(RM_ground)
    T_result, R_result, _, _ = transforms3d.affines.decompose44(result_transformation)
    inv = np.linalg.inv(result_transformation)

    T_result_test, R_result_test, _, _ = transforms3d.affines.decompose44(inv)
    print(inv)
    R_error = (np.sqrt(np.sum(np.square(R_ground - R_result_test))))
    # T_Matrix = inv[:3,3]
    T_error = (np.sqrt(np.sum(np.square(T_ground - T_result_test))))

    print({"R_error":R_error,"T_error":T_error})

    R_error = (np.sqrt(np.sum(np.square(R_ground - R_result))))
    # T_Matrix = inv[:3,3]
    T_error = (np.sqrt(np.sum(np.square(T_ground - T_result))))
    # print("Rotation",R,inv[0:3,0:3])
    # print("Transalation",T,T_Matrix)
    return {"R_error":R_error,"T_error":T_error}
