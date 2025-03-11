# GazeSearch: Radiology Findings Search Benchmark (accepted to WACV 2025)
![alt text](imgs/qualitative-results-github-repo.png)

## Overview

GazeSearch is a curated visual search dataset designed for evaluating search algorithms in radiology findings. The dataset leverages medical eye-tracking data ([REFLACX](https://physionet.org/content/reflacx-xray-localization/1.0.0/0) and [EGD](https://physionet.org/content/egd-cxr/1.0.0/)) to understand how radiologists visually interpret medical images, with the goal of improving both the accuracy and interpretability of deep learning models for X-ray analysis.


## Data Access
We provide the processed scanpath at data/finding_visual_search_coco_format_train_test_filtered_max_6_split_train_valid_test_2024-07-22.json 
This data has max length of 6 fixations. To get the length of 7 as in the paper, please add the center point at the beginning of the sequence.

To gain access to the full GazeSearch dataset (including the images from MIMIC-CXR), please contact tp030@uark.edu. You can expect a response within 24-48 hours with further instructions on how to obtain the data.

## ChestSearch Baseline
The dataset includes source code for ChestSearch, a scan path prediction baseline model specifically designed for GazeSearch. See `src/` for more details.

## Demo 
I provide an example image in the `example/` folder. You can run the demo by using the `src/demo_medical.ipynb` notebook. 
The checkpoints are in [here](https://uark-my.sharepoint.com/:u:/g/personal/tp030_uark_edu/EYST3kkJNJpAuadtgt5UILcBaZ8_UFAF0o95adk2p15FvQ?e=lK5Wdm).


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
MIT License

Copyright (c) 2024 AICV@University of Arkansas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
