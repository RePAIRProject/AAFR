# Configuration 

In our, we use several parameters. 

In our `assemble_cfg.py`, we have:
```python
num_of_points = 30000 # resampling to n points
# see paper for details (graph creation)
to = 100
tb = 0.1
dil = 0.01
thre = 0.93
N = 15
variables_as_list =  [num_of_points, num_of_points, N, to, to, to, tb, tb, tb, dil, thre]
```

Here below we provide a more detailed explanation:

#### `num_of_points`
We down sample each point cloud to (approximately) `num_of_points`. If the point cloud has less points, nothing is done.
The related function in the code is [down_sample_to()](https://github.com/RePAIRProject/AAFR/blob/master/helper.py#L17).

#### `to` and `tb`
These parameters are used in the creation and pruning of the graph (created from the points). In the paper, we mention "_The final version is obtained after applying a refinement step similar to the morphological operation of opening. A pruning step is followed by a dilation to remove small isolated branches and promote the creation of closed breaking curves_".
They are linked to (and explained in greater details) in Section 3.2 of the [paper _Feature extraction from point clouds_, Gumhold et al., 2001](https://graphics.stanford.edu/courses/cs164-10-spring/Handouts/papers_gumhold.pdf), and they actually used to set the `shortest_cycle_length`, `smallest_isolated_island_length` and `shortest_allowed_branch_length` in the graph.
You can see the code [here](https://github.com/RePAIRProject/AAFR/blob/master/pipline_modules/standard.py#L187).
We use `to1 = to2 = to3` and `tb1 = tb2 = tb3`. 
They can be changed if we want to allow different sizes of island, cycles or branch in the created graph.

#### `dil`
The `dil` parameter is the dilation size when enlarging the breaking points detected to _fill_ a curve. It is used to make sure that there are no holes between points of the breaking curve. As before, in the paper we mention "_a dilation to remove small isolated branches and promote the creation of closed breaking curves_" and the dilation is conditioned on this parameter.
The related function in the code is [dilate_border()](https://github.com/RePAIRProject/AAFR/blob/master/pipline_modules/standard.py#L36).

#### `thre`
This parameter `thre` (threshold) is linked to the nodes we select as _breaking_ (part of the breaking curves). Given our corner penalty $w_co$, we accept the one with $w_co < \tau$, where $\tau$ is exactly this `thre` parameter. 
In the paper, we mention "_We select all nodes whose corner penalty is less than a threshold to obtain a noisy initial version of the breaking curves._"
In the code it is used [here](https://github.com/RePAIRProject/AAFR/blob/master/pipline_modules/standard.py#L211)

#### `N`
It should be the number of objects, but it is not used anymore, it can be ignored.