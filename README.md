# GazeSearch
GazeSearch: Radiology Findings Search Benchmark (accepted to WACV 2025 Oral Presentation)

## Overview

GazeSearch is a curated visual search dataset designed for evaluating search algorithms in radiology findings. The dataset leverages medical eye-tracking data to understand how radiologists visually interpret medical images, with the goal of improving both the accuracy and interpretability of deep learning models for X-ray analysis.

### Key Features
- Refined eye-tracking data focused on target-present visual search
- Purposefully aligned fixation sequences for specific findings
- Comprehensive benchmark for visual search in medical imaging
- Includes ChestSearch: a baseline scan path prediction model

### Methodology
Our dataset is created through a refinement method inspired by the target-present visual search challenge, where:
- Each fixation sequence has a specific radiological finding target
- Fixations are guided to locate the target
- Eye-tracking data is processed and standardized for clarity

## Data Access
We provide the processed scanpath at data/finding_visual_search_coco_format_train_test_filtered_max_6_split_train_valid_test_2024-07-22.json 
This data has max length of 6 fixations. To get the length of 7 as in the paper, please add the center point at the beginning of the sequence.

To gain access to the full GazeSearch dataset (including the images from MIMIC-CXR), please contact tp030@uark.edu. You can expect a response within 24-48 hours with further instructions on how to obtain the data.

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

## Models

### ChestSearch Baseline
The dataset includes ChestSearch, a scan path prediction baseline model specifically designed for GazeSearch. This provides a starting point for comparing new approaches.



## License

The GazeSearch dataset is licensed under the Creative Commons Attribution 4.0 International License. You are free to use, share, and adapt the dataset, provided you give appropriate credit to the original authors.

## Contact

For any questions or issues regarding the dataset, please contact tp030@uark.edu.