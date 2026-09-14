"""
Dataset Preparation & Stratified 8:2 Split for Pantograph Keypoints.
Filters out unannotated/blurry images and splits into train/val maintaining video domain balance.
"""

import os
import glob
import json
import random
from collections import defaultdict
from PIL import Image

KEYPOINT_NAMES = [
    "p1_left_horn",
    "p2_left_support",
    "p3_right_support",
    "p4_right_horn"
]

ALIAS_MAP = {
    "p1": 0, "p1_left_horn": 0, "left_horn": 0,
    "p2": 1, "p2_left_support": 1, "left_support": 1,
    "p3": 2, "p3_right_support": 2, "right_support": 2,
    "p4": 3, "p4_right_horn": 3, "right_horn": 3,
}


def build_coco_split(input_dir, output_dir="data", val_ratio=0.2, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    jpg_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    by_source = defaultdict(list)
    skipped_blurry = []

    for jpg in jpg_files:
        base = os.path.splitext(jpg)[0]
        jf = base + ".json"
        if not os.path.exists(jf):
            skipped_blurry.append(os.path.basename(jpg))
            continue

        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            skipped_blurry.append(os.path.basename(jpg))
            continue

        shapes = data.get("shapes", [])
        if len(shapes) < 4:
            skipped_blurry.append(os.path.basename(jpg))
            continue

        # Extract video domain e.g. spark_2, tilt_1
        fname = os.path.basename(jpg)
        parts = fname.split("_")
        src_tag = f"{parts[2]}_{parts[3]}" if len(parts) >= 4 else "other"

        by_source[src_tag].append((jpg, jf, data))

    train_samples = []
    val_samples = []

    print(f"=== Dataset Stratified 8:2 Split ===")
    for src_tag, items in sorted(by_source.items()):
        random.shuffle(items)
        num_val = int(len(items) * val_ratio)
        v_set = items[:num_val]
        t_set = items[num_val:]
        train_samples.extend(t_set)
        val_samples.extend(v_set)
        print(f"  [{src_tag}]: Total={len(items)}, Train={len(t_set)}, Val={len(v_set)}")

    def export_coco(sample_list, out_json_path):
        images = []
        annotations = []
        anno_id = 1
        img_id = 1

        for jpg_path, jf_path, data in sample_list:
            img_name = os.path.basename(jpg_path)
            width = data.get("imageWidth") or 2048
            height = data.get("imageHeight") or 1536

            images.append({
                "id": img_id,
                "file_name": img_name,
                "width": width,
                "height": height,
            })

            kpts = [[0.0, 0.0, 0] for _ in range(4)]
            for s in data.get("shapes", []):
                lbl = s.get("label", "").strip()
                pts = s.get("points", [])
                if not pts:
                    continue
                idx = ALIAS_MAP.get(lbl) or ALIAS_MAP.get(lbl.split("_")[0].lower())
                if idx is not None and 0 <= idx < 4:
                    vis = 1 if s.get("flags", {}).get("occluded", False) else 2
                    kpts[idx] = [float(pts[0][0]), float(pts[0][1]), vis]

            xs = [p[0] for p in kpts if p[2] > 0]
            ys = [p[1] for p in kpts if p[2] > 0]
            xmin, xmax = max(0, min(xs) - 20), min(width, max(xs) + 20)
            ymin, ymax = max(0, min(ys) - 20), min(height, max(ys) + 20)
            bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]

            annotations.append({
                "id": anno_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": float(bbox[2] * bbox[3]),
                "keypoints": [c for p in kpts for c in p],
                "num_keypoints": sum(1 for p in kpts if p[2] > 0),
                "iscrowd": 0,
            })
            anno_id += 1
            img_id += 1

        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": [{
                "id": 1,
                "name": "pantograph",
                "supercategory": "pantograph",
                "keypoints": KEYPOINT_NAMES,
                "skeleton": [[0, 1], [1, 2], [2, 3], [0, 3]],
            }],
        }
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out_json_path} (Images: {len(images)})")

    train_json = os.path.join(output_dir, "pantograph_train.json")
    val_json = os.path.join(output_dir, "pantograph_val.json")
    export_coco(train_samples, train_json)
    export_coco(val_samples, val_json)

    print(f"\nTotal Valid Annotated Images : {len(train_samples) + len(val_samples)}")
    print(f"  Train Set                  : {len(train_samples)} (80%)")
    print(f"  Val Set                    : {len(val_samples)} (20%)")
    print(f"  Excluded Blurry/No-label   : {len(skipped_blurry)} images")
    return train_json, val_json


if __name__ == "__main__":
    train_dir = "/home/ibd/jyb/gongwang_hrnet_tilt/train"
    build_coco_split(train_dir, output_dir="/mnt/T4/jyb/gongwang_hrnet_tilt/data")
