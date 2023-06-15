# Demo run for 3DVR 2023

We prepared a couple of examples for reassembly.

The data is taken from the [breaking bad dataset](https://breaking-bad-dataset.github.io/) and is preprocessed in `data/DrinkBottle` and we have `fractured_62` and `fractured_70`.

If you run (after having installed everything) the `assemble_fragments` script:

```bash
python assemble_fragments.py --cfg assemble_cfg
```

You should get an output similar to:

```bash
> python assemble_fragments.py --cfg assemble_cfg

Will try to assemble:
0) DrinkBottle_fractured_62

#################################################################
Current broken object: fractured_62 (DrinkBottle)
-----------------------------------------------------------------
Loading object 1: data/DrinkBottle/fractured_62/objects/obj1_challenge.ply
100%|███████████████████████████████████████████████████████████████████████████| 29883/29883 [00:05<00:00, 5920.63it/s]
Loading object 2: data/DrinkBottle/fractured_62/objects/obj2_challenge.ply
100%|███████████████████████████████████████████████████████████████████████████| 29767/29767 [00:04<00:00, 5999.79it/s]
done
-----------------------------------------------------------------
Detecting breaking curves for object 1..
Creating point cloud graph..

```

And it will continue (after breaking curves there is segmentation and registration..).
It may take a while (depending on computer). 
The segmentation step is the slowest at the moment, and it may take some minutes. 
The runtime changed a lot on different computers, from some minutes to almost half an hour.

This is done for one object (two parts) but everything is configured to accept lists of data, so that it can be run on a batch of data

This will create an output folder, whose name is decided in `configs/assemble_cfg.py`, as:
```python
name = f'3dvr_{num_of_points}'
output_dir = os.path.join('3dvr_results', name)
os.makedirs(output_dir, exist_ok=True)
```

Feel free to change the name, of course.

In the output folder you will see:
- `pointclouds` folder (with pointcloud in initial position)
- `segmented_parts` folder (with segmented parts divided)
- `registration` folder (with pointcloud in final position plus the transformation in a `.csv` file)
- `borders_objN_challenge.ply`, the pointcloud containing only the breaking curves
- `col_borders_objN_challenge.ply`, a visualization with the full pointcloud and the breaking curves colored
- `col_regions_objN_challenge.ply`, a visualization with the full pointcloud and the segmented regions colored
- `info.json` a file with information (nothing interesting at the moment)

If you visualize (here screenshots from Meshlab) the results you should see something like this:

| Initial position (in `pointclouds`) | After assembly (in `registration`) |
|:---:|:----:|
| ![demo position](static/images/demo_position.png) | ![demo position](static/images/demo_solved.png) |

A demo result is saved in `3dvr_results/demo_results`, but you can do it on your pc to test it!


