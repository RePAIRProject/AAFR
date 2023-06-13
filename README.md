# AAFR (Ali Alagrami Fresco Reconstruction)

The AAFR is a framework for virtually reassembling broken objects. 
[Click here to check out the project page for more details](https://repairproject.github.io/AAFR_web/)

It was developed within the [RePAIR european project](https://github.com/RePAIRProject) but the reconstruction works also for other broken objects. 
We tested it on [the breaking bad dataset](https://breaking-bad-dataset.github.io/) and on [the brick from the TU-Wien dataset](https://www.geometrie.tuwien.ac.at/geom/nbkdir/hofer/3dpuzzles/brick/brick.html).

# 1) Description

Receiving as input the point cloud of each broken part of the object (initially we start with two, but it will be extended to multiple, the work is already in progress)).

The pipeline consists of several steps:
- it creates two graphs from each pointcloud (one for the whole point cloud, one for the borders), then it computes corner penalties to detect points belonging to *breaking curves* (edges in 3D data). 
- following these breaking curves, the pointcloud is segmented into regions. 
- each possible pair of regions (excluding super small regions) is registered and the registration is evaluated. The best registration is selected and applied to the pointcloud.

# 2) Installation

Build/Installation instructions (including requirements and dataset).

# 3) Usage

We created a sample version in `assemble_fragments.py`.

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