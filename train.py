"""
scripts/train.py

Fine-tunes a pre-trained YOLOv5 model (transfer learning) on the disaster-
response dataset (humans / debris / vehicles).

This script is a thin, reproducible wrapper around Ultralytics' YOLOv5
`train.py`. We don't reimplement YOLOv5's training loop -- that would be
pointless and error-prone -- we drive it with the project's own configs
(configs/data.yaml, configs/hyp_disaster.yaml) and a sensible transfer-
learning recipe:

  1. Start from the COCO-pretrained checkpoint (yolov5s.pt by default).
  2. Freeze the backbone for the first few epochs (`--freeze`) so the head
     adapts to the new 3-class problem before the backbone features drift.
  3. Unfreeze and fine-tune end-to-end for the remaining epochs.

Prerequisite (one-time setup):
    git clone https://github.com/ultralytics/yolov5.git third_party/yolov5
    pip install -r third_party/yolov5/requirements.txt

Usage:
    python scripts/train.py \
        --data configs/data.yaml \
        --hyp configs/hyp_disaster.yaml \
        --weights yolov5s.pt \
        --epochs 100 \
        --freeze-epochs 15 \
        --img-size 640 \
        --batch-size 16 \
        --name disaster_uav_v1
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
YOLOV5_DIR = REPO_ROOT / "third_party" / "yolov5"


def ensure_yolov5_present():
    if not (YOLOV5_DIR / "train.py").exists():
        raise SystemExit(
            f"YOLOv5 repo not found at {YOLOV5_DIR}.\n"
            "Run:\n"
            f"  git clone https://github.com/ultralytics/yolov5.git {YOLOV5_DIR}\n"
            f"  pip install -r {YOLOV5_DIR}/requirements.txt --break-system-packages"
        )


def run_phase(args, *, weights: str, epochs: int, freeze: int, resume: bool, phase_name: str):
    """Invoke YOLOv5's own train.py as a subprocess for one training phase."""
    cmd = [
        sys.executable, str(YOLOV5_DIR / "train.py"),
        "--data", str(Path(args.data).resolve()),
        "--hyp", str(Path(args.hyp).resolve()),
        "--weights", weights,
        "--epochs", str(epochs),
        "--imgsz", str(args.img_size),
        "--batch-size", str(args.batch_size),
        "--name", f"{args.name}_{phase_name}",
        "--project", str(Path(args.project).resolve()),
        "--workers", str(args.workers),
    ]
    if freeze > 0:
        # YOLOv5 layer indices: 0-9 backbone, 10+ head, for yolov5s/m/l/x
        cmd += ["--freeze", str(freeze)]
    if resume:
        cmd += ["--resume"]
    if args.device:
        cmd += ["--device", args.device]

    print(f"\n[phase: {phase_name}] running:\n  " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True, cwd=str(YOLOV5_DIR))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv5 for disaster-response UAV perception")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--hyp", default="configs/hyp_disaster.yaml")
    parser.add_argument("--weights", default="yolov5s.pt",
                         help="pretrained checkpoint to start transfer learning from")
    parser.add_argument("--epochs", type=int, default=100, help="total fine-tuning epochs")
    parser.add_argument("--freeze-epochs", type=int, default=15,
                         help="epochs to train with the backbone frozen before unfreezing")
    parser.add_argument("--freeze-layers", type=int, default=10,
                         help="number of leading layers to freeze (10 = full backbone on yolov5s)")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="", help="cuda device, e.g. '0' or 'cpu'")
    parser.add_argument("--name", default="disaster_uav")
    parser.add_argument("--project", default="runs/train")
    args = parser.parse_args()

    ensure_yolov5_present()

    remaining_epochs = max(args.epochs - args.freeze_epochs, 0)

    # Phase 1: head-only adaptation with frozen backbone
    if args.freeze_epochs > 0:
        run_phase(args, weights=args.weights, epochs=args.freeze_epochs,
                  freeze=args.freeze_layers, resume=False, phase_name="frozen")
        phase1_best = (Path(args.project) / f"{args.name}_frozen" / "weights" / "best.pt")
        next_weights = str(phase1_best) if phase1_best.exists() else args.weights
    else:
        next_weights = args.weights

    # Phase 2: full fine-tuning, backbone unfrozen
    if remaining_epochs > 0:
        run_phase(args, weights=next_weights, epochs=remaining_epochs,
                  freeze=0, resume=False, phase_name="full")

    print("\nTraining complete. Best weights are under "
          f"{args.project}/{args.name}_full/weights/best.pt "
          "(or _frozen/ if freeze_epochs == total epochs).")


if __name__ == "__main__":
    main()
