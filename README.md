# AAFR (Ali Alagrami Fresco Reconstruction)

The AAFR is a framework for virtually reassembling broken objects. 
[Click here to check out the project page for more details](https://repairproject.github.io/AAFR_web/)

It was developed within the [RePAIR european project](https://github.com/RePAIRProject) but the reconstruction works also for other broken objects. 
We tested it on [the breaking bad dataset](https://breaking-bad-dataset.github.io/) and on [the brick from the TU-Wien dataset](https://www.geometrie.tuwien.ac.at/geom/nbkdir/hofer/3dpuzzles/brick/brick.html).

# 1) Description

It consists of several steps:
- it creates two graphs from the pointcloud (one for the whole point cloud, one for the borders), then it computes corner penalties to detect points belonging to *breaking curves* (edges in 3D data). 
- following these breaking curves,


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