# GazeSearch: Radiology Findings Search Benchmark (accepted to WACV 2025)
![alt text](imgs/qualitative-results-github-repo.png)

## Overview

GazeSearch is a curated visual search dataset designed for evaluating search algorithms in radiology findings. The dataset leverages medical eye-tracking data ([REFLACX](https://physionet.org/content/reflacx-xray-localization/1.0.0/0) and [EGD](https://physionet.org/content/egd-cxr/1.0.0/)) to understand how radiologists visually interpret medical images, with the goal of improving both the accuracy and interpretability of deep learning models for X-ray analysis.

<!-- ### Key Features
- Purposefully aligned eye-tracking data focused on target-present visual search.
- Comprehensive benchmark for visual search in medical imaging between state-of-the-art visual search models.
- Includes ChestSearch: a baseline scan path prediction model -->


## Data Access
We provide the processed scanpath at data/finding_visual_search_coco_format_train_test_filtered_max_6_split_train_valid_test_2024-07-22.json 
This data has max length of 6 fixations. To get the length of 7 as in the paper, please add the center point at the beginning of the sequence.

To gain access to the full GazeSearch dataset (including the images from MIMIC-CXR), please contact tp030@uark.edu. You can expect a response within 24-48 hours with further instructions on how to obtain the data.

## ChestSearch Baseline
The dataset includes source code for ChestSearch, a scan path prediction baseline model specifically designed for GazeSearch. See `src/` for more details.



## Citation

If you use the GazeSearch dataset in your research, please cite the following paper:

```
@article{GazeSearch2023,
    title={GazeSearch: Radiology Findings Search Benchmark},
    author={Trong Thang Pham, Tien-Phat Nguyen, Yuki Ikebe, Akash Awasthi, Zhigang Deng, Carol C. Wu, Hien Nguyen and Ngan Le},
    journal={IEEE Winter Conference on Applications of Computer Vision (WACV)},
    year={2025}
}
```
## License
All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.
