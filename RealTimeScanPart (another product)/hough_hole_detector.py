# hough_hole_detector.py
# ----------------------
# ใช้ HoughCircle หา "รูเล็ก" ใน ROI ของ nozzle ที่ detect ได้

import cv2
import numpy as np
from typing import List, Tuple, Optional

from nozzle_specs_mm import NozzleSpec, NOZZLE_4, NOZZLE_2
from scale_model import InverseHeightScale

def detect_holes_hough_in_nozzle(
    frame_bgr: np.ndarray,
    nozzle_bbox_xyxy: Tuple[int, int, int, int],
    nozzle_spec: NozzleSpec,
    height_mm: float,
    scale_model: InverseHeightScale,
    *,
    dp: float = 1.2,
    param1: int = 100,
    param2: int = 12,
    tol_ratio: Tuple[float, float] = (0.7, 1.3),
    mask_to_circle: bool = True,
) -> List[Tuple[int, int, int]]:
    """
    คืนลิสต์วงกลมของ "รูเล็ก" ในพิกัดภาพใหญ่: [(cx, cy, r_px), ...]
    - nozzle_bbox_xyxy: กล่อง nozzle จาก YOLO (x1,y1,x2,y2) ในพิกัดภาพ
    - nozzle_spec: ใช้รู้ 'hole_diameter_mm'
    - height_mm: ความสูงกล้อง ณ ตอนนี้
    - scale_model: ไว้คำนวณ px/mm จากความสูง
    - tol_ratio: ค่าความยืดหยุ่นของรัศมี Hough (เช่น 0.7~1.3 เท่าของรัศมีคาดการณ์)
    """
    x1, y1, x2, y2 = nozzle_bbox_xyxy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_bgr.shape[1]-1, x2), min(frame_bgr.shape[0]-1, y2)

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    # คำนวณช่วงรัศมี (px) สำหรับ Hough จาก "ขนาดรูจริง (mm) × สเกล(px/mm)"
    rmin, rmax = scale_model.radius_px_range(
        diameter_mm=nozzle_spec.hole_diameter_mm,
        height_mm=height_mm,
        tol_ratio=tol_ratio
    )

    # เตรียมภาพสำหรับ Hough
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # (ออปชัน) mask ให้เป็นวงกลมโดยประมาณเพื่อตัดขอบกล่อง
    if mask_to_circle:
        h, w = gray.shape
        mask = np.zeros_like(gray)
        cxr, cyr = w//2, h//2
        radius = int(min(w, h) * 0.48)  # วงกลมใหญ่ประมาณขอบจาน
        cv2.circle(mask, (cxr, cyr), radius, 255, -1)
        gray = cv2.bitwise_and(gray, mask)

    # HoughCircles หา "รูเล็ก"
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=max(2, int(0.5 * (rmin + rmax))),  # ระยะแยกระหว่างวง—พอประมาณ (ปรับได้)
        param1=param1,
        param2=param2,   # ยิ่งต่ำยิ่งหาเยอะ/เสียงเยอะ
        minRadius=rmin,
        maxRadius=rmax
    )

    out: List[Tuple[int, int, int]] = []
    if circles is not None and len(circles) > 0:
        circles = np.round(circles[0]).astype(int)
        for (cx, cy, r) in circles:
            # map พิกัดกลับภาพใหญ่
            out.append((cx + x1, cy + y1, r))

    return out
