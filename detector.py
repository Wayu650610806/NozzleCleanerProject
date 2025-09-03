# """
# NozzleDetector (4x4 Grid Detection)
# -----------------------------------
# Detects 16 nozzle boxes from a single BGR image using YOLO + grid enforcement.

# Usage:
#     det = NozzleDetector("weights/best.pt", imgsz=1280, conf=0.25, iou=0.5)
#     boxes = det.detect_16(image_bgr)
#     # boxes: list of 16 NozzleBox with (x1,y1,x2,y2,conf,cx,cy,R,grid_index)

# Workflow:
#     1) YOLO detect → drop extreme-size outliers
#     2) If <12 boxes → raise InvalidInputImageError
#     3) Snap to 4×4 grid → fill missing with synthetic boxes
#     4) Sort row-major → compute (cx, cy, R)

# Raises:
#     InvalidInputImageError – when too few valid nozzles detected
# """

from __future__ import annotations
from typing import List
import numpy as np
import cv2
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
        1) YOLO → boxes (ลอง conf ปกติก่อน ถ้าต่ำกว่า 12 ลอง conf-δ อีกครั้ง)
        2) sanitize → clip + ตัดเล็ก
        3) กฎ: <12 → ยก InvalidInputImageError (ให้ user ถ่ายใหม่)
        12..15 → lattice fill
        >16 → คัด 16 ใหญ่สุด
        4) sort row-major + ใส่ grid_index
        5) ประมาณ (cx,cy,R) จาก bbox
        """

        # 1) detect (conf ปกติ)
        boxes = self._adaptive_detect_once(image_bgr, self.conf)
        # (ออปชัน) ลอง conf ลดลงถ้าน้อยกว่า 12 … แล้วรวมค่าที่ดีกว่า
        # จากนั้นกรองพื้นที่ตาม median:
        boxes = self._filter_by_relative_area(boxes, min_ratio=0.30, max_ratio=3.0)

        # 1.1) ถ้าได้น้อยกว่า 12 ลองลด conf หนึ่งสเต็ป (เช่น -0.05)
        if len(boxes) < 12:
            soften_conf = max(0.16, self.conf - 0.05)
            if soften_conf < self.conf:
                boxes2 = self._adaptive_detect_once(image_bgr, soften_conf)
                boxes2 = self._filter_by_relative_area(boxes, min_ratio=0.30, max_ratio=3.0)
                if len(boxes2) > len(boxes):
                    boxes = boxes2

        # 2) ตรวจเกณฑ์ธุรกิจ
        if len(boxes) < 12:
            raise InvalidInputImageError(
                f"Detected only {len(boxes)} nozzles. "
                "Expected a clear 4x4 nozzle grid. Please retake the photo."
            )

        # 3) บังคับ 16
        boxes = self._enforce_16(image_bgr, boxes)

        # 4) sort + grid_index (order it to 1 2 3 4 / 5 6 7 8 9 10....)
        boxes = self._sort_row_major(boxes)

        for idx, b in enumerate(boxes):
            b.grid_index = idx
            # 5) ประมาณ circle จาก bbox
            w = b.x2 - b.x1; h = b.y2 - b.y1
            b.cx = b.x1 + w // 2
            b.cy = b.y1 + h // 2
            b.R  = max(8, int(min(w, h) * 0.4))
        return boxes
    
    def _filter_by_relative_area(
        self,
        boxes: List[NozzleBox],
        min_ratio: float = 0.30,   # กล่องต้องไม่เล็กกว่า 30% ของ median
        max_ratio: float = 3.0,    # (ออปชัน) ตัดกล่องที่ใหญ่เกิน 3× median (กัน outlier)
        min_boxes_for_filter: int = 6  # มีอย่างน้อย 6 กล่องก่อนค่อยกรอง เพื่อกัน over-prune
    ) -> List[NozzleBox]:
        if len(boxes) < min_boxes_for_filter:
            return boxes
        areas = np.array([(b.x2 - b.x1) * (b.y2 - b.y1) for b in boxes], dtype=np.float64)
        med = float(np.median(areas))
        if med <= 0:
            return boxes
        lo = med * min_ratio
        hi = med * max_ratio if max_ratio is not None else float("inf")
        keep = []
        for b in boxes:
            a = (b.x2 - b.x1) * (b.y2 - b.y1)
            if lo <= a <= hi:
                keep.append(b)
        # ถ้าเผลอตัดจนเหลือน้อยมาก ให้ถอยกลับไปไม่กรอง
        return keep if len(keep) >= max(4, int(0.5 * len(boxes))) else boxes

    def _adaptive_detect_once(self, image_bgr: np.ndarray, conf: float) -> List[NozzleBox]:
        r = self.model.predict(image_bgr, imgsz=self.imgsz, conf=conf, iou=self.iou, verbose=False)[0]
        out: List[NozzleBox] = []
        if r.boxes is not None:
            for (x1, y1, x2, y2), cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                out.append(NozzleBox(int(x1), int(y1), int(x2), int(y2), float(cf)))
        return out

    def _sort_row_major(self, boxes: List[NozzleBox]) -> List[NozzleBox]:
        if not boxes: return boxes
        centers = np.array([[(b.x1+b.x2)/2, (b.y1+b.y2)/2] for b in boxes], dtype=np.float32)
        ys = centers[:,1]
        q = np.quantile(ys, [0.25, 0.5, 0.75])
        rows = [[] for _ in range(4)]
        for i,(xc,yc) in enumerate(centers):
            r = 0 if yc<=q[0] else 1 if yc<=q[1] else 2 if yc<=q[2] else 3
            rows[r].append((xc,i))
        ordered=[]
        for r in range(4):
            rows[r].sort(key=lambda t:t[0])
            ordered.extend([boxes[i] for _,i in rows[r]])
        return ordered

    def _enforce_16(self, img: np.ndarray, boxes: List[NozzleBox]) -> List[NozzleBox]:
        """
        Global Grid Snap:
        1) ประมาณกรอบรวม & สร้างกริด 4x4 ด้วย linspace (ใช้เปอร์เซ็นไทล์กัน outlier)
        2) จับคู่กล่องจริงกับจุดกริดแบบ one-to-one โดยใช้ระยะทางน้อยสุด (greedy)
        3) จุดกริดที่ว่าง → เติมกล่องสังเคราะห์ขนาด median
        4) คืนลิสต์ 16 กล่องเรียง row-major (ตำแหน่งกริด)
        """
        if not boxes:
            return []

        # --- centers & median size ---
        centers = np.array([[(b.x1 + b.x2)/2.0, (b.y1 + b.y2)/2.0] for b in boxes], dtype=np.float32)
        ws = np.array([b.x2 - b.x1 for b in boxes], dtype=np.float32)
        hs = np.array([b.y2 - b.y1 for b in boxes], dtype=np.float32)
        w_med = int(max(8, np.median(ws))) if len(ws) else 50
        h_med = int(max(8, np.median(hs))) if len(hs) else 50

        # --- สร้างกริด 4x4 แบบ robust ต่อ outlier (ใช้ 5th-95th percentile) ---
        x_lo = float(np.percentile(centers[:, 0], 5)) if len(centers) else w_med
        x_hi = float(np.percentile(centers[:, 0], 95)) if len(centers) else w_med * 4
        y_lo = float(np.percentile(centers[:, 1], 5)) if len(centers) else h_med
        y_hi = float(np.percentile(centers[:, 1], 95)) if len(centers) else h_med * 4

        # กันกรณี percentile ล้มเหลว (ทุกจุดเท่ากัน)
        if x_hi <= x_lo:
            x_lo, x_hi = float(np.min(centers[:,0])), float(np.max(centers[:,0]))
        if y_hi <= y_lo:
            y_lo, y_hi = float(np.min(centers[:,1])), float(np.max(centers[:,1]))

        xs = np.linspace(x_lo, x_hi, 4)
        ys = np.linspace(y_lo, y_hi, 4)
        grid = np.array([(xs[c], ys[r]) for r in range(4) for c in range(4)], dtype=np.float32)  # row-major targets

        # เกณฑ์ระยะสูงสุดที่ยอมรับ (0.8 ของระยะช่องกริด)
        dx = (x_hi - x_lo) / max(3.0, 1.0)
        dy = (y_hi - y_lo) / max(3.0, 1.0)
        max_d = np.hypot(0.8 * dx, 0.8 * dy)

        # --- จับคู่แบบ greedy: sort pair by distance แล้วจับคู่ไม่ซ้ำ ---
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
        assign = [None] * n_tar  # index detection → ต่อ target j (row-major)

        for d, i, j in pairs:
            if i in used_det or j in used_tar:
                continue
            # ถ้าหลุดกรอบมากเกินไป ให้เว้นไว้ให้เติมทีหลัง (กันกล่องผิดที่)
            if d > max_d and len(used_det) >= 8:
                # เมื่อจับคู่ได้พอสมควรแล้ว อย่าให้ outlier มาแทรก
                continue
            assign[j] = i
            used_det.add(i)
            used_tar.add(j)
            if len(used_tar) == 16:
                break

        # helper: สร้างกล่องสังเคราะห์ที่ศูนย์กลาง (xc,yc)
        def synth_at(xc, yc):
            x1 = int(xc - w_med / 2); x2 = x1 + w_med
            y1 = int(yc - h_med / 2); y2 = y1 + h_med
            return NozzleBox(x1, y1, x2, y2, conf=0.0)

        # --- สร้างลิสต์ผลลัพธ์ตามลำดับกริด ---
        out: List[NozzleBox] = []
        for j in range(n_tar):
            det_idx = assign[j]
            if det_idx is None:
                # เติมตามตำแหน่งกริด
                out.append(synth_at(grid[j, 0], grid[j, 1]))
            else:
                b = boxes[det_idx]
                out.append(b)

        return out
