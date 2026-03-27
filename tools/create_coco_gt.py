import argparse
import json
import os

import numpy as np
from tqdm import tqdm

thing_classes = list(range(0, 30))
id_cnt = 0
inst_id_cnt = 0
ann_json = dict(images=[], annotations=[], categories=[])
ann_json["categories"] = [
    {"id": 0, "name": "single door", "supercategory": "cadspotting"},
    {"id": 1, "name": "double door", "supercategory": "cadspotting"},
    {"id": 2, "name": "sliding door", "supercategory": "cadspotting"},
    {"id": 3, "name": "folding door", "supercategory": "cadspotting"},
    {"id": 4, "name": "revolving door", "supercategory": "cadspotting"},
    {"id": 5, "name": "rolling door", "supercategory": "cadspotting"},
    {"id": 6, "name": "window", "supercategory": "cadspotting"},
    {"id": 7, "name": "bay window", "supercategory": "cadspotting"},
    {"id": 8, "name": "blind window", "supercategory": "cadspotting"},
    {"id": 9, "name": "opening symbol", "supercategory": "cadspotting"},
    {"id": 10, "name": "sofa", "supercategory": "cadspotting"},
    {"id": 11, "name": "bed", "supercategory": "cadspotting"},
    {"id": 12, "name": "chair", "supercategory": "cadspotting"},
    {"id": 13, "name": "table", "supercategory": "cadspotting"},
    {"id": 14, "name": "TV cabinet", "supercategory": "cadspotting"},
    {"id": 15, "name": "Wardrobe", "supercategory": "cadspotting"},
    {"id": 16, "name": "cabinet", "supercategory": "cadspotting"},
    {"id": 17, "name": "gas stove", "supercategory": "cadspotting"},
    {"id": 18, "name": "sink", "supercategory": "cadspotting"},
    {"id": 19, "name": "refrigerator", "supercategory": "cadspotting"},
    {"id": 20, "name": "airconditioner", "supercategory": "cadspotting"},
    {"id": 21, "name": "bath", "supercategory": "cadspotting"},
    {"id": 22, "name": "bath tub", "supercategory": "cadspotting"},
    {"id": 23, "name": "washing machine", "supercategory": "cadspotting"},
    {"id": 24, "name": "squat toilet", "supercategory": "cadspotting"},
    {"id": 25, "name": "urinal", "supercategory": "cadspotting"},
    {"id": 26, "name": "toilet", "supercategory": "cadspotting"},
    {"id": 27, "name": "stairs", "supercategory": "cadspotting"},
    {"id": 28, "name": "elevator", "supercategory": "cadspotting"},
    {"id": 29, "name": "escalator", "supercategory": "cadspotting"},
    {"id": 30, "name": "row chairs", "supercategory": "cadspotting"},
    {"id": 31, "name": "parking spot", "supercategory": "cadspotting"},
    {"id": 32, "name": "wall", "supercategory": "cadspotting"},
    {"id": 33, "name": "curtain wall", "supercategory": "cadspotting"},
    {"id": 34, "name": "railing", "supercategory": "cadspotting"},
    {"id": 35, "name": "bg", "supercategory": "cadspotting"},
]


def process_file(file_path):
    global id_cnt, inst_id_cnt

    ann_json["images"].append({"id": id_cnt, "file_name": os.path.basename(file_path), "width": 140, "height": 140})
    data = np.load(file_path)
    temp_inst_ids = np.unique(data["instanceIds"])

    for inst_id in temp_inst_ids:
        if inst_id == -1:
            continue
        category_id = np.unique(data["semanticIds"][data["instanceIds"] == inst_id])[0]
        if category_id not in thing_classes:
            continue

        temp_ids = np.unique(data["elementIds"][data["instanceIds"] == inst_id])
        bboxes = data["bboxes"][temp_ids]
        x_min, y_min = bboxes[:, 0].min(), bboxes[:, 1].min()
        x_max, y_max = bboxes[:, 2].max(), bboxes[:, 3].max()
        w, h = x_max - x_min, y_max - y_min
        ann_json["annotations"].append(
            {
                "id": inst_id_cnt,
                "image_id": id_cnt,
                "category_id": int(category_id),
                "segmentation": [],
                "area": w * h,
                "bbox": [x_min, y_min, w, h],
                "iscrowd": 0,
            }
        )
        inst_id_cnt += 1

    id_cnt += 1


def main(root_dir, output_json):
    npz_files = []
    for subdir, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith(".npz"):
                npz_files.append(os.path.join(subdir, file_name))

    for file_path in tqdm(sorted(npz_files), desc="Processing files"):
        process_file(file_path)

    with open(output_json, "w") as file:
        json.dump(ann_json, file, indent=4)
    print(f"Annotations saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create COCO-style GT annotations from npz files.")
    parser.add_argument("--root-dir", default="data/floorplan_color/test_color/npz_gt", help="Directory containing npz GT files.")
    parser.add_argument("--output-json", default="aaai_test/annotations_test.json", help="Output annotation json path.")
    args = parser.parse_args()
    main(args.root_dir, args.output_json)
