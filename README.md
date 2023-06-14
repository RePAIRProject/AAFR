# AAFR (Ali Alagrami Fresco Reconstruction)

The AAFR is a framework for virtually reassembling broken objects.
[Click here to check out the project page for more details](https://repairproject.github.io/AAFR/)

It was developed within the [RePAIR european project](https://github.com/RePAIRProject) but the reconstruction works also for other broken objects.
We tested it on [the breaking bad dataset](https://breaking-bad-dataset.github.io/) and on [the brick from the TU-Wien dataset](https://www.geometrie.tuwien.ac.at/geom/nbkdir/hofer/3dpuzzles/brick/brick.html).

# 1) Description

Receiving as input the point cloud of each broken part of the object (initially we start with two, but it will be extended to multiple, the work is already in progress)).

The pipeline consists of several steps:
- it creates two graphs from each pointcloud (one for the whole point cloud, one for the borders), then it computes corner penalties to detect points belonging to *breaking curves* (edges in 3D data).
- following these breaking curves, the pointcloud is segmented into regions.
- each possible pair of regions (excluding super small regions) is registered and the registration is evaluated. The best registration is selected and applied to the pointcloud.

# 2) Installation

We do not have yet a clean environment, but we have a `requirements.txt` file which can be installed using `pip`:
```bash
pip install -r requirements.txt
```
The code does not rely on GPU and uses widely used libraries plus the code in this folder (so make sure the root path is included).
It was tested using `python 3.9.0` on `Ubuntu 20.04`..

# 3) Usage

The code does many computations, so we created a couple of scripts to be able to run it quickly without complications.

These are based on the breaking bad dataset because this is available and easier to use.
The data collection process from the RePAIR project is yet to be finished and the data will be released when ready, so at the moment it is not possible to make experiments with these objects.
However, the code can be easily modified to work with different kind of data (it uses pointcloud, but resampling a mesh in a pointcloud is easy and you find an example in the `prepare_challenge.py` script if needed).
If there are issues or you need help, please do open an issue and ask.

## Data preparation
If you already have the pointclouds of the broken objects in some random position, you can skip this part.

If you want to test with two pointclouds which are initially aligned, you can run the preparation script (please change the path )
We prepared an example based on the breaking bad dataset, please change the `/path_to_breaking_bad_dataset/` to the real path on your local computer.
Then run:

```bash
prepare_challenge.py --cfg configs/prepare_cfg.yaml
```
##### WARNING:
This is prepared for the breaking bad dataset, so it assumes the data has subfolder (category, fracture_id, objects_files) and re-creates this structures in the output folder. If you have different structure, change the prepare challenge code (for example lines 65 and 66 of `prepare_challenge.py` extract category and fracture_id from the path)

The script reads the configuration from `configs/prepare_cfg.yaml` (check the file for more info), it reads the data (either `.obj` mesh or `.ply` pointcloud) and resample it.
It contains a list of the broken objects (this works for a list of data)
It has `r` and `t` vectors which are the angles and units for rotation and translation.

Everything will be prepared and saved in the output folder (which is by default `data` in the root folder where you run the code).
You can check there that everything worked (it will create subfolders and write there the resampled pointclouds with some rotation and translation as a *challenge*).

## Assembling fragments

Assuming the data is ready, we can run the script to assemble the data. It works on a loop, so it will reconstruct all the folders.
We created a script for the assembly, which can be run as:
```bash
assemble_fragments.py --cfg assemble_cfg
```
This will assemble all the broken objects and create a lot (maybe too much? The call to the saving functions (detached from the computational one for ease of use) are easy to disable, check the code in `assemble_fragments.py` and look for `fr_ass.save_`)

The results will be in the output folder (check the name, line 19 of `configs/assembly_cfg.py` file) and will contain segmented data (colored) registered data and a copy of the pointcloud (and also colored borders) so you should be able to easily visualize all intermediate steps (if you have no idea, use Meshlab to visualize `.ply` files)

##### Question: why before a `.yaml` file and now a `.py` file as config (with different syntax)?
*Answer: If you raised it, you have a point. They were created at different times, and the `.py` file is very useful to create nested folders. They are relatively easy to use so it should be possible to understand both, sorry for the change`.*

# 4) Known Issues

The repo contains *a lot* of experiments, which will be organized with time. Use the scripts mentioned above for a guaranteed execution and use the rest of the code at your own risk.

# 5) Relevant publications

The code is related to the [paper accepted in the 2023 3DVR CVPR Workshop](https://arxiv.org/abs/2306.02782).

To cite the paper:
```
@misc{alagrami2023reassembling,
      title={Reassembling Broken Objects using Breaking Curves},
      author={Ali Alagrami and Luca Palmieri and Sinem Aslan and Marcello Pelillo and Sebastiano Vascon},
      year={2023},
      eprint={2306.02782},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 6) License

The code is released under the Creative Commons Zero v1.0 Universal License.
For more details, refer to [the license itself](https://github.com/RePAIRProject/AAFR/blob/master/LICENSE).

## 7) Project Page 

The project page is available [here](https://repairproject.github.io/AAFR/)

*Reassembling Broken Objects using Breaking Curves*

Ali Alagrami*, Luca Palmieri*, Sinem Aslan*, Marcello Pelillo, Sebastiano Vascon

Ca' Foscari University of Venice, Department of Computer Science (DAIS)

CVPR 2023 Workshop on 3D Vision and Robotics (3DVR)

This page was built using the <a href="https://github.com/eliahuhorwitz/Academic-project-page-template" target="_blank">Academic Project Page Template</a>.
You are free to borrow the of this website, we just ask that you link back to this page in the footer. <br> This website is licensed under a <a rel="license"  href="http://creativecommons.org/licenses/by-sa/4.0/" target="_blank">Creative
Commons Attribution-ShareAlike 4.0 International License</a>.

Shootout to <a href="https://github.com/eliahuhorwitz" target="_blank">Eliahu Horwitz</a> for the template. Thanks.
