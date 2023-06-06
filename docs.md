# Overview of the scripts

The code assembles two (matching) broken parts of an objects.

The workflow can be simplified in 3 steps, as shown in the image below:

![workflow](workflow.jpg)

There are several steps from the input to the output, so there are some useful scripts:

### Segmentation (`segment_objects.py`)

It extracts breaking curves and segment the regions, saving them.

### Assembly (`challenge_assembly.py`)

It tries to assemble the broken parts. This assumes that `segment_objects` was launched and uses the output from that script. 

### Objects Folder (`objects/some_datasets.py`)

They are separated files that contains url and parameters for objects, separated sometimes for different datasets, useful to test (they are imported from segmentation/assembly scripts)

### Partial scripts 

The other scripts may be helpful for debugging:
- `register_seg_parts` is actually a subset of `challenge_assembly` without the transformation (so it register them in place)
- `register_with_icp` uses ICP without any pre-processing, useful for baseline
- `show_something` (where `something` can be `pipeline`, `segmentation`, `assembly`) are some scripts for visualizing results (to produce images for the paper)
