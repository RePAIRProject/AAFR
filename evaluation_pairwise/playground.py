from helper import FeatureLines
import pdb

datapath = '/home/palma/Unive/RePAIR/Datasets/RePAIR_dataset/group_19/processed/RPf_00152.ply'
#Obj1 = FeatureLines("data/group19/RPf_00152.ply",voxel_size=0.7)
Obj1 = FeatureLines(datapath,voxel_size=0.7)

# calculates all penalty functions after creating the graph where N = 16 (nearest neighbor)
Obj1.init(16)

# set the parameters as specified in the paper
# alpha, gamma,crease_threshold, and border_threshold and pattern_length_threshold
# pattern_length_T is set by default to half sqrt of the number of nodes
Obj1.set_params(alpha=0.2,gamma=0.5,crease_threshold=1.7,border_threshold=2)

Obj1.show("crease")
pdb.set_trace()

data = Obj1.w_k
Obj1.show_heat(data)
pdb.set_trace()
