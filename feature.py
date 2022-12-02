import open3d as o3d
import matplotlib
import numpy as np
class feature():
    def __init__(self,val):
        self.val = val

    def show(self):
        weights = self.val
        if weights.shape[-1] == 3:
            weights = 1-self.NormalizeData(np.sqrt(np.sum(weights * weights,axis=1)))
        weights = weights.copy()
        # weights[weights>=thre] = 1
        cmap = matplotlib.cm.get_cmap('viridis')
        rgba = cmap(weights)
        rgb = rgba[:,:3]
        self.pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb).astype(np.float))
        o3d.visualization.draw_geometries([self.pcd])
