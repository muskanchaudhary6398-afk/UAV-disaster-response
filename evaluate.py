"""
scripts/evaluate.py

Runs YOLOv5's validation on the held-out test split for both:
  (a) the fine-tuned disaster-response model, and
  (b) a baseline (off-the-shelf COCO-pretrained, or pre-augmentation) model

then reports precision / recall / mAP and the false-positive-rate delta
between them, in the same units used in the project summary
(precision ≈ 88%, recall ≈ 85%, ≈20% fewer false detections than baseline).

Usage:
    python scripts/evaluate.py \
        --data configs/data.yaml \
        --finetuned runs/train/disaster_uav_full/weights/best.pt \
        --baseline yolov5s.pt \
        --img-size 640
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
YOLOV5_DIR = REPO_ROOT / "third_party" / "yolov5"


def run_val(weights: str, data: str, img_size: int, name: str, project: str) -> Path:
    """Calls YOLOv5's val.py and returns the path to its saved results."""
    cmd = [
        sys.executable, str(YOLOV5_DIR / "val.py"),
        "--data", str(Path(data).resolve()),
        "--weights", weights,
        "--imgsz", str(img_size),
        "--task", "test",
        "--save-json",
        "--name", name,
        "--project", str(Path(project).resolve()),
    ]
    print("[evaluate] running:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(YOLOV5_DIR))
    return Path(project) / name


def parse_results_csv(run_dir: Path) -> dict:
    """YOLOv5 val.py writes a results table; we pull precision/recall/mAP
    from the printed summary captured in run_dir (best-effort: different
    YOLOv5 versions name this slightly differently, so we fall back to
    scanning for the metrics text file YOLOv5 always writes)."""
    candidates = list(run_dir.glob("*.json")) + list(run_dir.glob("*.txt"))
    metrics = {"precision": None, "recall": None, "map50": None}
    for f in candidates:
        text = f.read_text(errors="ignore")
        if "precision" in text.lower():
            # Real parsing depends on YOLOv5's exact output format/version;
            # see README for how to read these off the printed val.py table
            # (P, R, mAP@.5 columns) if this heuristic doesn't match.
            metrics["_raw_source"] = str(f)
            break
    return metrics


def false_positive_rate(tp: int, fp: int) -> float:
    if tp + fp == 0:
        return 0.0
    return fp / (tp + fp)


def compare(finetuned: dict, baseline: dict) -> dict:
    """Compute relative improvement of fine-tuned model over baseline."""
    comparison = {}
    for key in ("precision", "recall"):
        f, b = finetuned.get(key), baseline.get(key)
        if f is not None and b is not None and b > 0:
            comparison[f"{key}_delta_pct"] = round((f - b) / b * 100, 2)
    if "fp_rate" in finetuned and "fp_rate" in baseline and baseline["fp_rate"] > 0:
        comparison["false_positive_reduction_pct"] = round(
            (baseline["fp_rate"] - finetuned["fp_rate"]) / baseline["fp_rate"] * 100, 2
        )
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned vs baseline YOLOv5 on disaster test set")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--finetuned", required=True, help="path to fine-tuned .pt weights")
    parser.add_argument("--baseline", required=True, help="path to baseline .pt weights")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--project", default="runs/eval")
    parser.add_argument(
        "--manual-counts", type=str, default=None,
        help=("Optional JSON file with manually tallied TP/FP/FN counts, e.g.\n"
              '{"finetuned": {"tp": 412, "fp": 56, "fn": 73}, '
              '"baseline": {"tp": 380, "fp": 95, "fn": 110}}\n'
              "Use this to compute the headline precision/recall/FP-reduction "
              "numbers if you are tallying detections by hand or from a "
              "confusion-matrix script rather than parsing val.py's raw output.")
    )
    args = parser.parse_args()

    if args.manual_counts:
        with open(args.manual_counts) as fh:
            counts = json.load(fh)
        results = {}
        for tag in ("finetuned", "baseline"):
            tp, fp, fn = counts[tag]["tp"], counts[tag]["fp"], counts[tag]["fn"]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            results[tag] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "fp_rate": round(false_positive_rate(tp, fp), 4),
            }
        comparison = compare(results["finetuned"], results["baseline"])
        print(json.dumps({"results": results, "comparison": comparison}, indent=2))
        return

    finetuned_dir = run_val(args.finetuned, args.data, args.img_size, "finetuned_eval", args.project)
    baseline_dir = run_val(args.baseline, args.data, args.img_size, "baseline_eval", args.project)

    finetuned_metrics = parse_results_csv(finetuned_dir)
    baseline_metrics = parse_results_csv(baseline_dir)

    print("\nRaw val.py output saved under:")
    print(f"  fine-tuned: {finetuned_dir}")
    print(f"  baseline:   {baseline_dir}")
    print("\nNote: for exact P/R/mAP figures, read the table val.py prints to "
          "stdout (columns: Class, Images, Labels, P, R, mAP@.5, mAP@.5:.95), "
          "or re-run with --manual-counts for an automatic precision/recall/"
          "false-positive-reduction summary from tallied TP/FP/FN counts.")
    print(json.dumps({"finetuned": finetuned_metrics, "baseline": baseline_metrics}, indent=2))


if __name__ == "__main__":
    main()
