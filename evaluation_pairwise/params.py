# config
axis_angles = [45, 45, 45]
from objects import objects
object_number = 0
#sampling
large_object = objects[object_number]["large_object"]
small_object = objects[object_number]["small_object"]
N = 15
Object_t1 = objects[object_number]["to1"]
Object_t2 = objects[object_number]["to2"]
Object_t3 = objects[object_number]["to3"]
border_t1 = objects[object_number]["tb1"]
border_t2 = objects[object_number]["tb1"]
border_t3 = objects[object_number]["tb1"]
#threshold for corner
thre = objects[object_number]["thre"]
dilation_size = objects[object_number]["dilation_size"]
pipeline_variables = (small_object, large_object, N, Object_t1, Object_t2,
                     Object_t3, border_t1, border_t2, border_t3, dilation_size, thre)
pipeline_name = "general_pipeline"
#test
test_name = "ICP_test"
# eval
eval_list = ["rms"]
