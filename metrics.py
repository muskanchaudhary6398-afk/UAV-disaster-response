"""
utils/metrics.py

Small, dependency-light helper functions for computing detection metrics
(precision, recall, false-positive rate) from raw TP/FP/FN counts, and for
comparing two models (e.g. fine-tuned vs. baseline). Kept separate from
scripts/evaluate.py so the pipeline (pipeline/realtime_pipeline.py) can also
log live per-session stats during a simulated SITL run.
"""

from dataclasses import dataclass, field


@dataclass
class DetectionTally:
    """Running tally of detection outcomes for one class or for all classes combined."""
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, tp: int = 0, fp: int = 0, fn: int = 0):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.tp + self.fp
        return self.fp / denom if denom else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fp_rate": round(self.false_positive_rate, 4),
        }


def relative_change(new_value: float, old_value: float) -> float:
    """Percent change of `new_value` relative to `old_value`. Used to report
    things like '~20% fewer false detections than baseline'."""
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value * 100.0


def compare_tallies(finetuned: DetectionTally, baseline: DetectionTally) -> dict:
    return {
        "precision_delta_pct": round(relative_change(finetuned.precision, baseline.precision), 2),
        "recall_delta_pct": round(relative_change(finetuned.recall, baseline.recall), 2),
        "false_positive_reduction_pct": round(
            -relative_change(finetuned.false_positive_rate, baseline.false_positive_rate), 2
        ),
    }


def match_detections_to_ground_truth(pred_boxes, gt_boxes, iou_threshold: float = 0.5):
    """Greedy IoU-based matching of predicted boxes to ground-truth boxes for
    a single image / single class. Boxes are (x1, y1, x2, y2).

    Returns (tp, fp, fn) counts for this image.
    """
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = _iou(pred, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0
