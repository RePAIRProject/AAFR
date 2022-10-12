import numpy as np
def run(R_T,result_transformation):
    R,T = R_T
    inv = np.linalg.inv(result_transformation)
    R_error = (np.sum(np.sqrt(np.square(R - inv[0:3,0:3]))))
    T_Matrix = inv[:3,3]
    T_error = (np.sum(np.sqrt(np.square((T[0], T[1], T[2]) - T_Matrix))))
    # print("Rotation",R,inv[0:3,0:3])
    # print("Transalation",T,T_Matrix)
    return {"R_error":R_error,"T_error":T_error}
