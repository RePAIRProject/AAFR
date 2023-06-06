objects = [
    {
        "Obj1_url":"/media/lucap/big_data/datasets/pairwise/ali/breaking_bad_statue_pair1/breaking_bad_cat_statue_piece1.ply",
        "Obj2_url":"/media/lucap/big_data/datasets/pairwise/ali/breaking_bad_statue_pair1/breaking_bad_cat_statue_piece2.ply",
        "small_object":30000,"large_object":30000,
        #to1,to2,to3 -> always fixed for the object creation (i fixed it for almost of them)
        "to1":100 ,"to2":100, "to3":100,
        #tb1,tb2,tb3 -> border paramters depend on the nature of the object
        "tb1":0.1 ,"tb2":1, "tb3":0.1,
        #dilation_size , thre -> size of the dilation and threshold applied depend on each object
        "dilation_size":0.02,"thre":0.96},
    {
        "Obj1_url":"/media/lucap/big_data/datasets/pairwise/ali/breaking_bad_cat_artifact_pair2/breaking_bad_cat_artifact_piece2.ply",
        "Obj2_url":"/media/lucap/big_data/datasets/pairwise/ali/breaking_bad_cat_artifact_pair2/breaking_bad_cat_artifact_piece3.ply",
        "small_object":30000,"large_object":30000,
        "to1":100 ,"to2":100, "to3":100,
        "tb1":0.1 ,"tb2":1, "tb3":0.1,
        "dilation_size":0.02,"thre":0.96},
    {
        "Obj1_url":"/media/lucap/big_data/datasets/pairwise/ali/breaking_bad_cat_artifact_pair1/breaking_bad_cat_artifact_piece1.ply",
        "Obj2_url":"/media/lucap/big_data/datasets/pairwise/ali/breaking_bad_cat_artifact_pair1/breaking_bad_cat_artifact_piece3.ply",
        "small_object":30000,"large_object":30000,
        "to1":100 ,"to2":100, "to3":100,
        "tb1":0.1 ,"tb2":1, "tb3":0.1,
        "dilation_size":0.02,"thre":0.96}
]
