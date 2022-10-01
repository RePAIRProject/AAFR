import numpy as np
def run(init_R_T,result_transformation):
    R,T = init_R_T
    inv = np.linalg.inv(result_transformation)
    R_error = (np.sum(np.sqrt(np.square(R - inv[0:3,0:3]))))
    T_Matrix = inv[:3,3]
    T_error = (np.sum(np.sqrt(np.square((T[0], T[1], T[2]) - T_Matrix))))
    return {"R_error":R_error,"T_error":T_error}
