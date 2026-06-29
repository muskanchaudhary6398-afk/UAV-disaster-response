"""
data/augmentation.py

Offline data-augmentation pipeline used to expand the disaster-simulation
training set before YOLOv5 fine-tuning.

Why offline (in addition to YOLOv5's built-in on-the-fly augmentation in
hyp_disaster.yaml)? Disaster footage has failure modes that generic
hue/scale/mosaic jitter doesn't reproduce well: smoke haze, motion blur from
UAV vibration, and partial occlusion of humans under debris. We bake a fixed
number of these "hard" variants into the dataset so they reliably show up in
every training epoch, rather than relying on probability alone.

Usage:
    python data/augmentation.py \
        --images dataset/images/train \
        --labels dataset/labels/train \
        --out dataset_augmented \
        --multiplier 3
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError as e:
    raise SystemExit(
        "albumentations is required: pip install albumentations --break-system-packages"
    ) from e


def build_transform(img_size: int = 640) -> A.Compose:
    """Augmentation pipeline tuned for disaster-zone aerial footage.

    Bounding boxes are carried through in YOLO format (normalized
    [class, x_center, y_center, w, h]) via Albumentations' bbox_params,
    so labels stay correct after geometric transforms.
    """
    return A.Compose(
        [
            # --- Simulated environmental degradation ---
            A.OneOf(
                [
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.35, alpha_coef=0.08, p=1.0),
                    A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1,
                                   num_shadows_upper=3, p=1.0),
                    A.RandomToneCurve(scale=0.2, p=1.0),
                ],
                p=0.45,
            ),
            # --- UAV motion / vibration blur ---
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 9), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.3,
            ),
            # --- Sensor noise (low-light gimbal camera) ---
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
            A.ISONoise(p=0.15),
            # --- Lighting variance: dawn/dusk SAR ops, fire glare ---
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            # --- Partial occlusion (debris covering victims) ---
            A.CoarseDropout(max_holes=4, max_height=0.12, max_width=0.12,
                             min_holes=1, fill_value=0, p=0.25),
            # --- Geometric: altitude / gimbal angle variance ---
            A.Affine(scale=(0.7, 1.3), rotate=(-15, 15), shear=(-5, 5), p=0.4),
            A.Resize(img_size, img_size),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2),
    )


def read_yolo_labels(label_path: Path):
    boxes, classes = [], []
    if not label_path.exists():
        return boxes, classes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, w, h = line.split()
        boxes.append([float(xc), float(yc), float(w), float(h)])
        classes.append(int(cls))
    return boxes, classes


def write_yolo_labels(label_path: Path, boxes, classes):
    lines = []
    for cls, (xc, yc, w, h) in zip(classes, boxes):
        # clip to valid range to avoid degenerate boxes after augmentation
        xc, yc, w, h = (max(0.0, min(1.0, v)) for v in (xc, yc, w, h))
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines))


def augment_dataset(images_dir: str, labels_dir: str, out_dir: str,
                     multiplier: int = 3, img_size: int = 640, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    images_dir, labels_dir, out_dir = Path(images_dir), Path(labels_dir), Path(out_dir)
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    transform = build_transform(img_size)
    image_paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in
                           {".jpg", ".jpeg", ".png"}])

    if not image_paths:
        print(f"No images found in {images_dir}. Nothing to augment.")
        return

    total_written = 0
    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [skip] could not read {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, classes = read_yolo_labels(label_path)

        # Always copy the original (unaugmented) pair forward
        cv2.imwrite(str(out_images / img_path.name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if label_path.exists():
            write_yolo_labels(out_labels / label_path.name, boxes, classes)
        total_written += 1

        for i in range(multiplier):
            try:
                if boxes:
                    result = transform(image=image, bboxes=boxes, class_labels=classes)
                else:
                    result = transform(image=image, bboxes=[], class_labels=[])
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"  [warn] augmentation failed on {img_path.name} (variant {i}): {exc}")
                continue

            aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            cv2.imwrite(str(out_images / aug_name), aug_img)
            write_yolo_labels(out_labels / f"{img_path.stem}_aug{i}.txt",
                               result["bboxes"], result["class_labels"])
            total_written += 1

    print(f"Done. Wrote {total_written} image/label pairs to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Disaster-zone augmentation pipeline")
    parser.add_argument("--images", required=True, help="path to source images dir")
    parser.add_argument("--labels", required=True, help="path to source YOLO labels dir")
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument("--multiplier", type=int, default=3,
                         help="number of augmented variants generated per source image")
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    augment_dataset(args.images, args.labels, args.out, args.multiplier, args.img_size)


if __name__ == "__main__":
    main()
