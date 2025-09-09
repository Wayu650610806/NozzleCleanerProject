# detector.py
"""
NozzleDetector (4×4 Grid Detection)
-----------------------------------
Detects 16 nozzle boxes from a single BGR image using YOLO with 4×4 grid enforcement.

Usage:
    det = NozzleDetector("weights/best.pt", imgsz=1280, conf=0.25, iou=0.5)
    boxes = det.detect_16(image_bgr)
    # -> list[NozzleBox] of length 16 with (x1, y1, x2, y2, conf, cx, cy, R, grid_index)

Workflow:
    1) YOLO detect → drop extreme-size outliers (relative-to-median filter)
    2) If <12 boxes → retry once with slightly lower conf
    3) If still <12 → raise InvalidInputImageError
    4) Enforce a robust 4×4 lattice (snap + synthesize to reach 16)
    5) Sort row-major → assign grid_index (0..15) and (cx, cy, R)

Raises:
    InvalidInputImageError – when too few valid nozzles are detected to form a 4×4 grid
"""

from __future__ import annotations

from typing import List

import numpy as np
import cv2
from ultralytics import YOLO

from nozzle_types import NozzleBox


class InvalidInputImageError(Exception):
    """Raised when the input image is not a valid 4×4 nozzle grid."""
    pass


class NozzleDetector:
    def __init__(self, weights_path: str, imgsz: int = 1280, conf: float = 0.25, iou: float = 0.5):
        self.model = YOLO(weights_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    # ======================
    # === PUBLIC METHOD  ===
    # ======================
    def detect_16(self, image_bgr: np.ndarray) -> List[NozzleBox]:
        """
        Run detection and return exactly 16 row-major boxes with geometry populated.
        """
        # 1) First pass with configured confidence
        boxes = self._adaptive_detect_once(image_bgr, self.conf)
        boxes = self._filter_by_relative_area(boxes, min_ratio=0.30, max_ratio=3.0)

        # 1.1) Retry once with a slightly lower confidence if fewer than 12 remain
        if len(boxes) < 12:
            soften_conf = max(0.16, self.conf - 0.05)
            if soften_conf < self.conf:
                boxes2 = self._adaptive_detect_once(image_bgr, soften_conf)
                boxes2 = self._filter_by_relative_area(boxes2, min_ratio=0.30, max_ratio=3.0)  # fixed
                if len(boxes2) > len(boxes):
                    boxes = boxes2

        # 2) Guard rail
        if len(boxes) < 12:
            raise InvalidInputImageError(
                f"Detected only {len(boxes)} nozzles. "
                "Expected a clear 4×4 nozzle grid. Please retake the photo."
            )

        # 3) Enforce 16 outputs via grid snapping/filling
        boxes = self._enforce_16(image_bgr, boxes)

        # 4) Row-major sort + grid_index (0..15)
        boxes = self._sort_row_major(boxes)
        for idx, b in enumerate(boxes):
            b.grid_index = idx
            # 5) Populate circle geometry from bbox
            w = b.x2 - b.x1
            h = b.y2 - b.y1
            b.cx = b.x1 + w // 2
            b.cy = b.y1 + h // 2
            b.R = max(8, int(min(w, h) * 0.4))

        return boxes

    # ======================
    # === HELPER METHODS ===
    # ======================
    def _adaptive_detect_once(self, image_bgr: np.ndarray, conf: float) -> List[NozzleBox]:
        r = self.model.predict(
            image_bgr, imgsz=self.imgsz, conf=conf, iou=self.iou, verbose=False
        )[0]
        out: List[NozzleBox] = []
        if r.boxes is not None:
            xs = r.boxes.xyxy.cpu().numpy()
            cs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cf in zip(xs, cs):
                out.append(NozzleBox(int(x1), int(y1), int(x2), int(y2), float(cf)))
        return out

    def _filter_by_relative_area(
        self,
        boxes: List[NozzleBox],
        min_ratio: float = 0.30,
        max_ratio: float = 3.0,
        min_boxes_for_filter: int = 6
    ) -> List[NozzleBox]:
        """
        Keep boxes whose area is within [min_ratio, max_ratio] × median area.
        Falls back to original list if filtering would remove too many.
        """
        if len(boxes) < min_boxes_for_filter:
            return boxes

        areas = np.array([(b.x2 - b.x1) * (b.y2 - b.y1) for b in boxes], dtype=np.float64)
        med = float(np.median(areas))
        if med <= 0:
            return boxes

        lo = med * min_ratio
        hi = med * max_ratio if max_ratio is not None else float("inf")

        kept = [b for b in boxes if lo <= (b.x2 - b.x1) * (b.y2 - b.y1) <= hi]
        # Safety: avoid over-pruning on odd images
        return kept if len(kept) >= max(4, int(0.5 * len(boxes))) else boxes

    def _sort_row_major(self, boxes: List[NozzleBox]) -> List[NozzleBox]:
        """
        Buckets detections into 4 horizontal rows via Y-quantiles, then sorts each row by X.
        Returns detections in row-major order (top→bottom, left→right).
        """
        if not boxes:
            return boxes

        centers = np.array([((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2) for b in boxes], dtype=np.float32)
        ys = centers[:, 1]
        q25, q50, q75 = np.quantile(ys, [0.25, 0.5, 0.75])

        rows = [[] for _ in range(4)]
        for i, (xc, yc) in enumerate(centers):
            r = 0 if yc <= q25 else 1 if yc <= q50 else 2 if yc <= q75 else 3
            rows[r].append((float(xc), i))

        ordered: List[NozzleBox] = []
        for r in range(4):
            rows[r].sort(key=lambda t: t[0])
            ordered.extend([boxes[i] for _, i in rows[r]])

        return ordered

    def _enforce_16(self, img: np.ndarray, boxes: List[NozzleBox]) -> List[NozzleBox]:
        """
        Global grid snapping to produce exactly 16 boxes:

        1) Estimate overall bounds & build a robust 4×4 grid using percentiles (resistant to outliers)
        2) Greedy one-to-one assignment from detections to grid targets by shortest distance
        3) Unassigned grid targets → synthesize median-sized boxes at those targets
        4) Return 16 boxes in row-major (grid) order
        """
        if not boxes:
            return []

        centers = np.array([((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0) for b in boxes], dtype=np.float32)
        ws = np.array([b.x2 - b.x1 for b in boxes], dtype=np.float32)
        hs = np.array([b.y2 - b.y1 for b in boxes], dtype=np.float32)
        w_med = int(max(8, np.median(ws))) if len(ws) else 50
        h_med = int(max(8, np.median(hs))) if len(hs) else 50

        # Robust bounds (5th–95th percentiles)
        x_lo = float(np.percentile(centers[:, 0], 5)) if len(centers) else w_med
        x_hi = float(np.percentile(centers[:, 0], 95)) if len(centers) else w_med * 4
        y_lo = float(np.percentile(centers[:, 1], 5)) if len(centers) else h_med
        y_hi = float(np.percentile(centers[:, 1], 95)) if len(centers) else h_med * 4

        # Fallback when percentile collapses
        if x_hi <= x_lo:
            x_lo, x_hi = float(np.min(centers[:, 0])), float(np.max(centers[:, 0]))
        if y_hi <= y_lo:
            y_lo, y_hi = float(np.min(centers[:, 1])), float(np.max(centers[:, 1]))

        xs = np.linspace(x_lo, x_hi, 4)
        ys = np.linspace(y_lo, y_hi, 4)
        grid = np.array([(xs[c], ys[r]) for r in range(4) for c in range(4)], dtype=np.float32)  # row-major targets

        # Distance cap ≈ 0.8 × cell diagonal
        dx = (x_hi - x_lo) / max(3.0, 1.0)
        dy = (y_hi - y_lo) / max(3.0, 1.0)
        max_d = float(np.hypot(0.8 * dx, 0.8 * dy))

        # Greedy assignment
        n_det, n_tar = len(centers), 16
        pairs = []
        for i in range(n_det):
            for j in range(n_tar):
                d = float(np.hypot(centers[i, 0] - grid[j, 0], centers[i, 1] - grid[j, 1]))
                pairs.append((d, i, j))
        pairs.sort(key=lambda t: t[0])

        used_det, used_tar = set(), set()
        assign = [None] * n_tar  # detection index for target j

        for d, i, j in pairs:
            if i in used_det or j in used_tar:
                continue
            # If too far and we already matched plenty, skip far outliers
            if d > max_d and len(used_det) >= 8:
                continue
            assign[j] = i
            used_det.add(i)
            used_tar.add(j)
            if len(used_tar) == 16:
                break

        def synth_at(xc: float, yc: float) -> NozzleBox:
            """Synthesize a median-sized box centered at (xc, yc)."""
            x1 = int(xc - w_med / 2)
            x2 = x1 + w_med
            y1 = int(yc - h_med / 2)
            y2 = y1 + h_med
            return NozzleBox(x1, y1, x2, y2, conf=0.0)

        out: List[NozzleBox] = []
        for j in range(n_tar):
            det_idx = assign[j]
            out.append(boxes[det_idx] if det_idx is not None else synth_at(grid[j, 0], grid[j, 1]))

        return out
