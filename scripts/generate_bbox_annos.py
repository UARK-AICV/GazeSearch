"""Generate `bbox_annos_filled_dummy.npy`.

Rebuilds the bounding-box annotation dict expected by
`src/chestsearch/builder.py` and `src/chestsearch/inference_builder.py`
directly from the released scanpath annotations, which carry a per-trial
`bbox` field ([x, y, w, h]; dummy-filled zeros in the public release).

Usage:
    python scripts/generate_bbox_annos.py \
        --dataset_root data \
        --data_path finding_visual_search_coco_format_train_test_filtered_max_6_split_train_valid_test_2024-07-22.json
"""

import argparse
import json
from os.path import join

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="data")
    parser.add_argument(
        "--data_path",
        default=(
            "finding_visual_search_coco_format_train_test_filtered_max_6"
            "_split_train_valid_test_2024-07-22.json"
        ),
    )
    parser.add_argument("--out", default="bbox_annos_filled_dummy.npy")
    args = parser.parse_args()

    with open(join(args.dataset_root, args.data_path), "r") as f:
        scanpaths = json.load(f)

    # Keys follow the convention used across the codebase:
    # utils.compute_search_cdf / FFN_IRL use `task + "_" + name`.
    bbox_annos = {}
    for traj in scanpaths:
        key = traj["task"] + "_" + traj["name"]
        if key in bbox_annos and bbox_annos[key] != traj["bbox"]:
            raise ValueError(f"inconsistent bbox for {key}")
        bbox_annos[key] = traj["bbox"]

    out_path = join(args.dataset_root, args.out)
    np.save(out_path, bbox_annos)
    print(f"wrote {len(bbox_annos)} entries to {out_path}")


if __name__ == "__main__":
    main()
