objects = [
    {
        "Obj1_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad_statue_pair1/breaking_bad_cat_statue_piece1.ply",
        "Obj2_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad_statue_pair1/breaking_bad_cat_statue_piece2.ply",
        "small_object":30000,"large_object":30000,
        #to1,to2,to3 -> always fixed for the object creation (i fixed it for almost of them)
        "to1":100 ,"to2":100, "to3":100,
        #tb1,tb2,tb3 -> border paramters depend on the nature of the object
        "tb1":0.1 ,"tb2":1, "tb3":0.1,
        #dilation_size , thre -> size of the dilation and threshold applied depend on each object
        "dilation_size":0.02,"thre":0.96
    },
    # # Run 06 march
    # {
    #     "Obj1_url":"/media/lucap/big_data/datasets/pairwise/Mirror/mode_1/piece_0.obj",
    #     "Obj2_url":"/media/lucap/big_data/datasets/pairwise/Mirror/mode_1/piece_1.obj",
    #     "small_object":300000,"large_object":300000,
    #     "to1":100 ,"to2":100, "to3":100,
    #     "tb1":0.1 ,"tb2":0.1, "tb3":0.1,
    #     "dilation_size":0.004,"thre":0.93
    # },
    # {
    #     "Obj1_url":"/media/lucap/big_data/datasets/pairwise/Mug/fractured_73/piece_0.obj",
    #     "Obj2_url":"/media/lucap/big_data/datasets/pairwise/Mug/fractured_73/piece_1.obj",
    #     "small_object":300000,"large_object":300000,
    #     "to1":100 ,"to2":100, "to3":100,
    #     "tb1":0.1 ,"tb2":0.1, "tb3":0.1,
    #     "dilation_size":0.01,"thre":0.93
    # },
    # {
    #     "Obj1_url":"/media/lucap/big_data/datasets/pairwise/Cookie/fractured_46/piece_0.obj",
    #     "Obj2_url":"/media/lucap/big_data/datasets/pairwise/Cookie/fractured_46/piece_1.obj",
    #     "small_object":40000,"large_object":80000,
    #     "to1":100 ,"to2":100, "to3":100,
    #     "tb1":100 ,"tb2":100, "tb3":100,
    #     "dilation_size":0.004,"thre":0.93
    # },
    # {
    #     "Obj1_url":"/media/lucap/big_data/datasets/pairwise/Cookie/fractured_52/piece_0.obj",
    #     "Obj2_url":"/media/lucap/big_data/datasets/pairwise/Cookie/fractured_52/piece_1.obj",
    #     "small_object":40000,"large_object":80000,
    #     "to1":100 ,"to2":100, "to3":100,
    #     "tb1":100 ,"tb2":100, "tb3":100,
    #     "dilation_size":0.004,"thre":0.93
    # },
    # {
    #     "Obj1_url":"/media/lucap/big_data/datasets/pairwise/Cookie/fractured_70/piece_0.obj",
    #     "Obj2_url":"/media/lucap/big_data/datasets/pairwise/Cookie/fractured_70/piece_1.obj",
    #     "small_object":40000,"large_object":80000,
    #     "to1":100 ,"to2":100, "to3":100,
    #     "tb1":100 ,"tb2":100, "tb3":100,
    #     "dilation_size":0.004,"thre":0.93
    # },
    {
        "Obj1_url":"/media/lucap/big_data/datasets/pairwise/DrinkBottle/fractured_62/piece_0.obj",
        "Obj2_url":"/media/lucap/big_data/datasets/pairwise/DrinkBottle/fractured_62/piece_1.obj",
        "small_object":100000,"large_object":200000,
        "to1":100 ,"to2":100, "to3":100,
        "tb1":0.1 ,"tb2":0.1, "tb3":0.1,
        "dilation_size":0.008,"thre":0.93
    },
    # initial ones
    {
        "Obj1_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad_cat_artifact_pair2/breaking_bad_cat_artifact_piece2.ply",
        "Obj2_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad_cat_artifact_pair2/breaking_bad_cat_artifact_piece3.ply",
        "small_object":30000,"large_object":30000,
        "to1":100 ,"to2":100, "to3":100,
        "tb1":0.1 ,"tb2":1, "tb3":0.1,
        "dilation_size":0.02,"thre":0.96
    },
    {
        "Obj1_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad_cat_artifact_pair1/breaking_bad_cat_artifact_piece1.ply",
        "Obj2_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad_cat_artifact_pair1/breaking_bad_cat_artifact_piece3.ply",
        "small_object":30000,"large_object":30000,
        "to1":100 ,"to2":100, "to3":100,
        "tb1":0.1 ,"tb2":1, "tb3":0.1,
        "dilation_size":0.02,"thre":0.96
    },
    # waiting
    #  {
    # "Obj1_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad/WineBottle/fractured_14/piece_0.obj",
    # "Obj2_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad/WineBottle/fractured_14/piece_1.obj",
    # "small_object":40000,"large_object":100000,
    # "to1":100 ,"to2":100, "to3":100,
    # "tb1":0.1 ,"tb2":0.1, "tb3":0.1,
    # "dilation_size":0.006,"thre":0.97
    # },
    #  {
    # "Obj1_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad/WineBottle/fractured_3/piece_0.obj",
    # "Obj2_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad/WineBottle/fractured_3/piece_1.obj",
    # "small_object":30000,"large_object":80000,
    # "to1":100 ,"to2":100, "to3":100,
    # "tb1":0.1 ,"tb2":0.1, "tb3":0.1,
    # "dilation_size":0.01,"thre":0.97
    # },
    #  {
    # "Obj1_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad/PillBottle/fractured_40/piece_0.obj",
    # "Obj2_url":"/media/lucap/big_data/datasets/pairwise/breaking_bad/PillBottle/fractured_40/piece_1.obj",
    # "small_object":100000,"large_object":200000,
    # "to1":100 ,"to2":100, "to3":100,
    # "tb1":0.1 ,"tb2":0.1, "tb3":0.1,
    # "dilation_size":0.01,"thre":0.95
    # },

]
