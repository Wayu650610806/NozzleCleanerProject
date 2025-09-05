# detector.py
# NozzleDetector (4x4 Grid Detection)
# -----------------------------------
# Detects 16 nozzle boxes from a single BGR image using YOLO + grid enforcement.
#
# Usage:
#     det = NozzleDetector("weights/best.pt", imgsz=1280, conf=0.25, iou=0.5)
#     boxes = det.detect_16(image_bgr)
#     # boxes: list of 16 NozzleBox with (x1, y1, x2, y2, conf, cx, cy, R, grid_index)
#
# Workflow:
#     1) YOLO detect → drop extreme-size outliers
#     2) If <12 boxes → raise InvalidInputImageError
#     3) Snap to 4×4 grid → fill missing with synthetic boxes
#     4) Sort row-major → compute (cx, cy, R)
#
# Raises:
#     InvalidInputImageError – when too few valid nozzles detected
# """

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from nozzle_types import NozzleBox


class InvalidInputImageError(Exception):
    """Raised when the input image is not a valid 4x4 nozzle grid."""
    pass


class NozzleDetector:
    def __init__(self, weights_path: str, imgsz: int = 1280, conf: float = 0.25, iou: float = 0.5):
        self.model = YOLO(weights_path)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def detect_16(self, image_bgr: np.ndarray) -> List[NozzleBox]:
        """
        Detection pipeline:

        1) YOLO → boxes (try normal `conf` first; if <12, try `conf - δ` once)
        2) Sanitize → clip + drop extreme area outliers
        3) Business rule:
            - <12 → raise InvalidInputImageError (ask user to retake)
            - 12..15 → lattice fill
            - >16 → keep top 16 by size (implicitly handled by grid match)
        4) Sort row-major + assign `grid_index`
        5) Estimate (cx, cy, R) from bbox
        """

        # 1) Detect with the current confidence
        boxes = self._adaptive_detect_once(image_bgr, self.conf)

        # Optionally reduce conf if <12 and re-try; then filter by median-relative area
        boxes = self._filter_by_relative_area(boxes, min_ratio=0.30, max_ratio=3.0)

        # 1.1) If fewer than 12, try a softened confidence once (e.g., -0.05)
        if len(boxes) < 12:
            soften_conf = max(0.16, self.conf - 0.05)
            if soften_conf < self.conf:
                boxes2 = self._adaptive_detect_once(image_bgr, soften_conf)
                # NOTE: Keep behavior identical to original (filter applied using `boxes` variable)
                boxes2 = self._filter_by_relative_area(boxes, min_ratio=0.30, max_ratio=3.0)
                if len(boxes2) > len(boxes):
                    boxes = boxes2

        # 2) Business rule check
        if len(boxes) < 12:
            raise InvalidInputImageError(
                f"Detected only {len(boxes)} nozzles. "
                "Expected a clear 4x4 nozzle grid. Please retake the photo."
            )

        # 3) Enforce 16 outputs by snapping/filling on a 4x4 lattice
        boxes = self._enforce_16(image_bgr, boxes)

        # 4) Sort row-major + assign grid_index (1..16 in reading order)
        boxes = self._sort_row_major(boxes)

        for idx, b in enumerate(boxes):
            b.grid_index = idx

            # 5) Estimate circle geometry from bbox
            w = b.x2 - b.x1
            h = b.y2 - b.y1
            b.cx = b.x1 + w // 2
            b.cy = b.y1 + h // 2
            b.R = max(8, int(min(w, h) * 0.4))

        return boxes

    def _filter_by_relative_area(
        self,
        boxes: List[NozzleBox],
        min_ratio: float = 0.30,      # Must not be smaller than 30% of median area
        max_ratio: float = 3.0,       # Optionally drop boxes >3× median area (guard outliers)
        min_boxes_for_filter: int = 6 # Require at least 6 boxes before filtering to avoid over-pruning
    ) -> List[NozzleBox]:
        if len(boxes) < min_boxes_for_filter:
            return boxes

        areas = np.array([(b.x2 - b.x1) * (b.y2 - b.y1) for b in boxes], dtype=np.float64)
        med = float(np.median(areas))
        if med <= 0:
            return boxes

        lo = med * min_ratio
        hi = med * max_ratio if max_ratio is not None else float("inf")

        keep: List[NozzleBox] = []
        for b in boxes:
            a = (b.x2 - b.x1) * (b.y2 - b.y1)
            if lo <= a <= hi:
                keep.append(b)

        # If filtering removes too many, fall back to original (safety)
        return keep if len(keep) >= max(4, int(0.5 * len(boxes))) else boxes

    def _adaptive_detect_once(self, image_bgr: np.ndarray, conf: float) -> List[NozzleBox]:
        r = self.model.predict(image_bgr, imgsz=self.imgsz, conf=conf, iou=self.iou, verbose=False)[0]
        out: List[NozzleBox] = []
        if r.boxes is not None:
            for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                out.append(NozzleBox(int(x1), int(y1), int(x2), int(y2), float(cf)))
        return out

    def _sort_row_major(self, boxes: List[NozzleBox]) -> List[NozzleBox]:
        if not boxes:
            return boxes

        centers = np.array(
            [[(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2] for b in boxes],
            dtype=np.float32,
        )
        ys = centers[:, 1]
        q = np.quantile(ys, [0.25, 0.5, 0.75])

        rows = [[] for _ in range(4)]
        for i, (xc, yc) in enumerate(centers):
            r = 0 if yc <= q[0] else 1 if yc <= q[1] else 2 if yc <= q[2] else 3
            rows[r].append((xc, i))

        ordered: List[NozzleBox] = []
        for r in range(4):
            rows[r].sort(key=lambda t: t[0])
            ordered.extend([boxes[i] for _, i in rows[r]])

        return ordered

    def _enforce_16(self, img: np.ndarray, boxes: List[NozzleBox]) -> List[NozzleBox]:
        """
        Global grid snapping to produce exactly 16 boxes:

        1) Estimate overall bounds & build a robust 4×4 grid using percentiles to resist outliers
        2) Greedy one-to-one assignment from detections to grid targets by shortest distance
        3) Unassigned grid targets → synthesize median-sized boxes
        4) Return 16 boxes in row-major order (grid order)
        """
        if not boxes:
            return []

        # --- centers & median size ---
        centers = np.array(
            [[(b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0] for b in boxes],
            dtype=np.float32,
        )
        ws = np.array([b.x2 - b.x1 for b in boxes], dtype=np.float32)
        hs = np.array([b.y2 - b.y1 for b in boxes], dtype=np.float32)
        w_med = int(max(8, np.median(ws))) if len(ws) else 50
        h_med = int(max(8, np.median(hs))) if len(hs) else 50

        # Robust 4×4 grid using 5th–95th percentiles
        x_lo = float(np.percentile(centers[:, 0], 5)) if len(centers) else w_med
        x_hi = float(np.percentile(centers[:, 0], 95)) if len(centers) else w_med * 4
        y_lo = float(np.percentile(centers[:, 1], 5)) if len(centers) else h_med
        y_hi = float(np.percentile(centers[:, 1], 95)) if len(centers) else h_med * 4

        # Fallback when percentile collapses (all points the same)
        if x_hi <= x_lo:
            x_lo, x_hi = float(np.min(centers[:, 0])), float(np.max(centers[:, 0]))
        if y_hi <= y_lo:
            y_lo, y_hi = float(np.min(centers[:, 1])), float(np.max(centers[:, 1]))

        xs = np.linspace(x_lo, x_hi, 4)
        ys = np.linspace(y_lo, y_hi, 4)
        grid = np.array([(xs[c], ys[r]) for r in range(4) for c in range(4)], dtype=np.float32)  # row-major targets

        # Max acceptable distance (0.8 × grid cell diagonal)
        dx = (x_hi - x_lo) / max(3.0, 1.0)
        dy = (y_hi - y_lo) / max(3.0, 1.0)
        max_d = float(np.hypot(0.8 * dx, 0.8 * dy))

        # Greedy assignment by distance
        n_det = len(centers)
        n_tar = 16
        pairs = []
        for i in range(n_det):
            for j in range(n_tar):
                d = float(np.hypot(centers[i, 0] - grid[j, 0], centers[i, 1] - grid[j, 1]))
                pairs.append((d, i, j))
        pairs.sort(key=lambda t: t[0])

        used_det = set()
        used_tar = set()
        assign = [None] * n_tar  # detection index assigned to target j (row-major)

        for d, i, j in pairs:
            if i in used_det or j in used_tar:
                continue
            # If too far and we already matched enough, skip to avoid bad outliers
            if d > max_d and len(used_det) >= 8:
                continue
            assign[j] = i
            used_det.add(i)
            used_tar.add(j)
            if len(used_tar) == 16:
                break

        # Helper: synthesize a median-sized box centered at (xc, yc)
        def synth_at(xc: float, yc: float) -> NozzleBox:
            x1 = int(xc - w_med / 2)
            x2 = x1 + w_med
            y1 = int(yc - h_med / 2)
            y2 = y1 + h_med
            return NozzleBox(x1, y1, x2, y2, conf=0.0)

        # Build output list in grid order
        out: List[NozzleBox] = []
        for j in range(n_tar):
            det_idx = assign[j]
            if det_idx is None:
                out.append(synth_at(grid[j, 0], grid[j, 1]))
            else:
                out.append(boxes[det_idx])

        return out
