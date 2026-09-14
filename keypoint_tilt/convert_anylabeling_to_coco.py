"""
Convert AnyLabeling / Labelme JSON annotations to COCO Keypoints JSON format (4-point scheme).
Based on Paper Fig. 3: P1(left_horn), P2(left_support), P3(right_support), P4(right_horn).
"""

import os
import glob
import json
import argparse
import numpy as np
from PIL import Image

KEYPOINT_NAMES = [
    "p1_left_horn",
    "p2_left_support",
    "p3_right_support",
    "p4_right_horn"
]

ALIAS_MAP = {
    # P1: Left horn
    "p1": 0, "p1_left_horn": 0, "left_horn": 0, "p1_left_horn_vertex": 0, "1": 0,
    # P2: Left support
    "p2": 1, "p2_left_support": 1, "left_support": 1, "p2_left_support_joint": 1, "2": 1,
    # P3: Right support
    "p3": 2, "p3_right_support": 2, "right_support": 2, "p3_right_support_joint": 2, "3": 2,
    # P4: Right horn
    "p4": 3, "p4_right_horn": 3, "right_horn": 3, "p4_right_horn_vertex": 3, "4": 3,
}


def convert_dataset(input_dir, output_json):
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    images = []
    annotations = []

    anno_id = 1
    image_id = 1

    for jf in sorted(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_rel_path = data.get("imagePath", "")
        img_name = os.path.basename(img_rel_path)
        img_full_path = os.path.join(input_dir, img_name)

        if not os.path.exists(img_full_path):
            base_no_ext = os.path.splitext(jf)[0]
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
                if os.path.exists(base_no_ext + ext):
                    img_full_path = base_no_ext + ext
                    img_name = os.path.basename(img_full_path)
                    break

        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if not width or not height:
            if os.path.exists(img_full_path):
                with Image.open(img_full_path) as im:
                    width, height = im.size
            else:
                width, height = 2048, 1536

        images.append({
            "id": image_id,
            "file_name": img_name,
            "width": width,
            "height": height,
        })

        # 4 keypoints: [x, y, v] * 4
        kpts = [[0.0, 0.0, 0] for _ in range(4)]

        for s in data.get("shapes", []):
            label = s.get("label", "").strip()
            pts = s.get("points", [])
            if not pts:
                continue

            idx = None
            if label in ALIAS_MAP:
                idx = ALIAS_MAP[label]
            else:
                prefix = label.split("_")[0].lower()
                if prefix in ALIAS_MAP:
                    idx = ALIAS_MAP[prefix]

            if idx is not None and 0 <= idx < 4:
                pt = pts[0]
                vis = 2
                flags = s.get("flags", {})
                if flags.get("occluded", False) or "occluded" in label:
                    vis = 1
                kpts[idx] = [float(pt[0]), float(pt[1]), vis]

        valid_pts = [p[:2] for p in kpts if p[2] > 0]
        if valid_pts:
            xs = [p[0] for p in valid_pts]
            ys = [p[1] for p in valid_pts]
            xmin, xmax = max(0, min(xs) - 20), min(width, max(xs) + 20)
            ymin, ymax = max(0, min(ys) - 20), min(height, max(ys) + 20)
            bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
            area = bbox[2] * bbox[3]
        else:
            bbox = [0.0, 0.0, float(width), float(height)]
            area = float(width * height)

        flat_kpts = [coord for pt in kpts for coord in pt]
        num_kpts = sum(1 for p in kpts if p[2] > 0)

        annotations.append({
            "id": anno_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": bbox,
            "area": area,
            "keypoints": flat_kpts,
            "num_keypoints": num_kpts,
            "iscrowd": 0,
        })

        anno_id += 1
        image_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "pantograph",
            "supercategory": "pantograph",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": [
                [0, 1], [1, 2], [2, 3], [0, 3]
            ]
        }]
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2, ensure_ascii=False)

    print(f"[Success] Converted {len(annotations)} AnyLabeling files to 4-keypoint COCO format: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images and .json files")
    parser.add_argument("--output_json", type=str, required=True, help="Output COCO format json path")
    args = parser.parse_args()
    convert_dataset(args.input_dir, args.output_json)
