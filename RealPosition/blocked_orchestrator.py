# blocked_orchestrator.py
"""
Blocked Nozzle Orchestrator — circle-ROI quadrant check
------------------------------------------------------
Purpose (for users):
    Read an image → detect nozzles → build per-nozzle *circular* ROI →
    split into 4 quadrants (TL/TR/BL/BR) → call `isBlockedHole(roi)` →
    return centers of blocked quadrants as points.

Quick Use:
    from blocked_orchestrator import detect_blocked_nozzles, developerTest

    points = detect_blocked_nozzles(
        image_path,
        imgsz=1280, conf=0.25, iou=0.5, pad_ratio=0.05,
        save_rois_dir=r"C:\out_rois"  # saves the exact patches passed into isBlockedHole
    )

    dev_points = developerTest(
        image_path, weights_path=None,  # None = auto-find 'best.pt'
        imgsz=1280, conf=0.25, iou=0.5, pad_ratio=0.05,
        show=True, wait_ms=1, save_path=None,
        save_rois_dir=r"C:\out_rois"
    )

Inputs:
    image_path: str
    imgsz: int (YOLO inference size), conf: float, iou: float
    pad_ratio: float (padding added to the circle radius before splitting)
    save_rois_dir: str | None (directory to dump the *raw* quadrant ROIs)

Outputs:
    detect_blocked_nozzles(...) -> List[Tuple[int, int]]   # (x, y) pixel coords
    developerTest(...)          -> List[Tuple[int, int]]   # same as user API

Raises:
    FileNotFoundError, InvalidInputImageError

Notes for Developers:
    - `quad_statuses`: List[[TL, TR, BL, BR] of bool] aligned with detected boxes.
    - Saved ROIs are the *exact* masked inputs to isBlockedHole (no overlays/watermarks).
    - Circle parameters are consolidated in `_circle_from_box()` for reuse and consistency.
    - Code is divided into sections: CONSTANTS/HELPERS, CORE (internal), MAIN (public API),
      DEVELOPER utilities, and a CLI example.
"""

from __future__ import annotations

from typing import List, Tuple
import os
import glob

import cv2
import numpy as np

from detector import NozzleDetector, InvalidInputImageError
from holecheckAI import isBlockedHole
from nozzle_types import NozzleBox


# =========================
# === CONSTANTS/HELPERS ===
# =========================

# Quadrant name order for labeling results
_QUAD_NAMES = ("TopLeft", "TopRight", "BottomLeft", "BottomRight")


def _ensure_dir(p: str) -> None:
    """Create directory if not exists; ignore empty/None."""
    if p:
        os.makedirs(p, exist_ok=True)


def _auto_find_weights(image_path: str, filename: str = "bestNozzleV8.pt") -> str:
    """
    Find 'best.pt' automatically in priority order:
        1) Folder of this Python module
        2) Folder of the given image_path
        3) Current working directory
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.dirname(os.path.abspath(image_path))
    cwd = os.getcwd()

    candidates: List[str] = []
    seen = set()
    for c in (
        os.path.join(module_dir, filename),
        os.path.join(image_dir, filename),
        os.path.join(cwd, filename),
    ):
        if c not in seen:
            candidates.append(c)
            seen.add(c)

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Could not find '{filename}'. Looked in:\n- " + "\n- ".join(candidates)
    )


def _save_rois(quads: List[np.ndarray], out_dir: str, stem: str, nozzle_num: int) -> None:
    """
    Save 4 raw quadrant ROIs (TL/TR/BL/BR) using consistent filenames.
    These are the exact masked inputs passed into isBlockedHole.
    """
    if not out_dir:
        return
    _ensure_dir(out_dir)
    for q_idx, q in enumerate(quads):
        fname = f"{stem}_nozzle{nozzle_num}_{_QUAD_NAMES[q_idx]}.png"
        cv2.imwrite(os.path.join(out_dir, fname), q)


def _circle_from_box(box: NozzleBox) -> Tuple[int, int, int]:
    """
    Extract the circle parameters (cx, cy, R) from a YOLO box.
    - If cx, cy, R exist in the box, use them.
    - Otherwise, estimate R = 0.4 * min(w, h) with a minimum of 8 px.
    """
    x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
    w, h = (x2 - x1), (y2 - y1)
    cx = int((x1 + x2) / 2) if box.cx is None else int(box.cx)
    cy = int((y1 + y2) / 2) if box.cy is None else int(box.cy)
    R = int(0.4 * min(w, h)) if (box.R is None or box.R <= 0) else int(box.R)
    return cx, cy, max(8, R)

def _quadrant_centers(
    box: NozzleBox,
    pad_ratio: float,
    mm_per_px: Optional[float] = None,
    angle_deg: float = 0.0
) -> List[Tuple[int, int]]:
    cx, cy, R = _circle_from_box(box)
    r = float(R) * (1.0 + float(pad_ratio))
    d = max(1.0, r * 0.5)

    th = math.radians(float(angle_deg))
    ux, uy = math.cos(th), math.sin(th)          # แกน X ของกริด
    vx, vy = -math.sin(th), math.cos(th)         # แกน Y ของกริด

    # TL = (-ux - vx), TR = (+ux - vx), BL = (-ux + vx), BR = (+ux + vx)
    return [
        (int(round(cx - d*ux - d*vx)), int(round(cy - d*uy - d*vy))),  # TL
        (int(round(cx + d*ux - d*vx)), int(round(cy + d*uy - d*vy))),  # TR
        (int(round(cx - d*ux + d*vx)), int(round(cy - d*uy + d*vy))),  # BL
        (int(round(cx + d*ux + d*vx)), int(round(cy + d*uy + d*vy))),  # BR
    ]


# ===== scale utils =====
import math
import numpy as np
from typing import Dict, Optional, List, Tuple

def estimate_scale_from_boxes(
    boxes: List[NozzleBox],
    nominal_spacing_mm: float = 173.0,
    angle_tol_deg: float = 20.0
) -> Dict[str, Optional[float]]:
    """
    ประมาณสเกลจากระยะห่าง 'เพื่อนบ้านที่ใกล้สุด' ของศูนย์กลาง nozzle
    คืน dict มี mm_per_px, px_per_mm, ค่าแกน x/y และ anisotropy_pct
    """
    if len(boxes) < 2:
        return {"mm_per_px": None, "px_per_mm": None,
                "mm_per_px_x": None, "mm_per_px_y": None,
                "n_pairs_x": 0, "n_pairs_y": 0,
                "anisotropy_pct": None}

    pts = np.array([(float(b.cx), float(b.cy)) for b in boxes], dtype=np.float64)
    nearest_all, nearest_h, nearest_v = [], [], []

    ang_tol = float(angle_tol_deg)
    n = len(pts)
    for i in range(n):
        best_d = float("inf"); best_dx = best_dy = 0.0
        for j in range(n):
            if i == j: continue
            dx = pts[j, 0] - pts[i, 0]
            dy = pts[j, 1] - pts[i, 1]
            d  = math.hypot(dx, dy)
            if 1e-6 < d < best_d:
                best_d, best_dx, best_dy = d, dx, dy
        if not math.isfinite(best_d): continue
        nearest_all.append(best_d)
        a = abs(math.degrees(math.atan2(best_dy, best_dx)))  # 0°=แนวนอน, 90°=แนวตั้ง
        if a <= ang_tol: nearest_h.append(best_d)
        elif abs(90.0 - a) <= ang_tol: nearest_v.append(best_d)

    med_h = float(np.median(nearest_h)) if nearest_h else None
    med_v = float(np.median(nearest_v)) if nearest_v else None
    med_all = float(np.median(nearest_all)) if nearest_all else None

    mm_per_px_x = (nominal_spacing_mm / med_h) if med_h else None
    mm_per_px_y = (nominal_spacing_mm / med_v) if med_v else None

    if mm_per_px_x and mm_per_px_y:
        mm_per_px = 0.5 * (mm_per_px_x + mm_per_px_y)
        anisotropy_pct = 100.0 * abs(mm_per_px_x - mm_per_px) / mm_per_px
    elif mm_per_px_x:
        mm_per_px, anisotropy_pct = mm_per_px_x, None
    elif mm_per_px_y:
        mm_per_px, anisotropy_pct = mm_per_px_y, None
    else:
        mm_per_px = (nominal_spacing_mm / med_all) if med_all else None
        anisotropy_pct = None

    px_per_mm = (1.0 / mm_per_px) if mm_per_px else None
    # --- ใน estimate_scale_from_boxes(...)

    # หลังคำนวณ nearest_h / nearest_v เสร็จ เราจะคำนวณมุมเด่นของแกน X
    angles_h_raw = []
    for i in range(n):
        # หาเพื่อนบ้านที่ใกล้สุดอีกที เพื่อดึงมุมจริงของเส้นที่จัดเป็นแนวนอน
        best_d = float("inf"); best_dx = best_dy = 0.0; jj = -1
        for j in range(n):
            if i == j: continue
            dx = pts[j, 0] - pts[i, 0]
            dy = pts[j, 1] - pts[i, 1]
            d  = math.hypot(dx, dy)
            if 1e-6 < d < best_d:
                best_d, best_dx, best_dy, jj = d, dx, dy, j
        if not math.isfinite(best_d): 
            continue
        a = math.degrees(math.atan2(best_dy, best_dx))  # (-180, 180]
        aa = abs(a)
        if aa <= ang_tol:            # มองเป็นเพื่อนบ้านแนวนอน
            # นอร์มัลไลซ์ให้มาอยู่ในช่วง [-90, +90] เพื่อค่ากลางนิ่ง
            if a > 90:   a -= 180
            if a < -90:  a += 180
            angles_h_raw.append(a)


    angle_deg = float(np.median(angles_h_raw)) if angles_h_raw else 0.0
    angle_std_deg = float(np.std(angles_h_raw)) if angles_h_raw else None

    # unit vectors ของแกนกริด (แกน X คือแนวนอนของกริด)
    th = math.radians(angle_deg)
    ux, uy = math.cos(th), math.sin(th)          # แกน X (แนวนอนของกริด)
    vx, vy = -math.sin(th), math.cos(th)         # แกน Y (แนวตั้งของกริด)

    return {
        "mm_per_px": mm_per_px,
        "px_per_mm": px_per_mm,
        "mm_per_px_x": mm_per_px_x,
        "mm_per_px_y": mm_per_px_y,
        "n_pairs_x": len(nearest_h),
        "n_pairs_y": len(nearest_v),
        "anisotropy_pct": anisotropy_pct,
        # ใหม่เพิ่ม:
        "angle_deg": angle_deg,
        "angle_std_deg": angle_std_deg,
        "ux": ux, "uy": uy, "vx": vx, "vy": vy,
    }




# ===========================
# === CORE (INTERNAL API) ===
# ===========================

def _crop_circle_quadrants(
    img: np.ndarray,
    box: NozzleBox,
    pad_ratio: float = 0.05,
    mm_per_px: Optional[float] = None,
    angle_deg: float = 0.0,
) -> Tuple[List[np.ndarray], Tuple[int, int, int, int]]:
    H, W = img.shape[:2]
    cx, cy, R = _circle_from_box(box)

    pr    = int(R * pad_ratio)
    r_pad = R + pr
    # ขยายเพิ่มเผื่อการหมุน (diagonal)
    r_rot = int(math.ceil(r_pad * math.sqrt(2))) + 2

    sx, ex = max(0, cx - r_rot), min(W, cx + r_rot)
    sy, ey = max(0, cy - r_rot), min(H, cy + r_rot)
    if ex <= sx or ey <= sy:
        x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
        sx, sy, ex, ey = x1, y1, x2, y2

    patch = img[sy:ey, sx:ex].copy()
    if patch.size == 0:
        return [np.zeros((1, 1, 3), dtype=np.uint8)] * 4, (sx, sy, ex, ey)

    # center ในพาッチ
    pcx, pcy = (cx - sx), (cy - sy)

    # หมุนพาッチให้แกนกริดตรงกับแกนภาพ
    M = cv2.getRotationMatrix2D((float(pcx), float(pcy)), float(-angle_deg), 1.0)
    patch_rot = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_LINEAR)

    # แบ่ง 2×2 ที่ขนาด r_pad (ไม่ใช่ r_rot) เพื่อให้ ROI กระชับ
    # สร้าง “กรอบสี่เหลี่ยมจัตุรัส” แนบศูนย์กลางรัศมี r_pad หลังหมุน
    sx2, ex2 = max(0, int(pcx - r_pad)), min(patch_rot.shape[1], int(pcx + r_pad))
    sy2, ey2 = max(0, int(pcy - r_pad)), min(patch_rot.shape[0], int(pcy + r_pad))
    if ex2 <= sx2 or ey2 <= sy2:
        # fallback: ใช้พาッチหมุนทั้งภาพ
        sx2, sy2, ex2, ey2 = 0, 0, patch_rot.shape[1], patch_rot.shape[0]

    sub = patch_rot[sy2:ey2, sx2:ex2].copy()
    if sub.size == 0:
        return [np.zeros((1, 1, 3), dtype=np.uint8)] * 4, (sx, sy, ex, ey)

    H2, W2 = sub.shape[:2]
    mx, my = W2 // 2, H2 // 2

    rects = [
        (0, 0, mx, my),        # TL
        (mx, 0, W2, my),       # TR
        (0, my, mx, H2),       # BL
        (mx, my, W2, H2),      # BR
    ]

    # center ในพิกัด sub (หลังหมุน + ตัดซ้อนชั้น)
    scx, scy = (pcx - sx2), (pcy - sy2)

    out_quads: List[np.ndarray] = []
    for (x1, y1, x2, y2) in rects:
        quad = sub[y1:y2, x1:x2].copy()
        qH, qW = quad.shape[:2]
        if qH <= 0 or qW <= 0:
            out_quads.append(np.zeros((1,1,3), dtype=np.uint8))
            continue

        # mask วงกลมรัศมี R โดยศูนย์กลางอยู่ที่ (scx,scy)
        mask = np.zeros((qH, qW), dtype=np.uint8)
        lc_x = int(round(scx - x1))
        lc_y = int(round(scy - y1))
        cv2.circle(mask, (lc_x, lc_y), int(R), 255, thickness=-1)

        quad_masked = cv2.bitwise_and(quad, quad, mask=mask)
        out_quads.append(quad_masked)

    # หมายเหตุ: (sx, sy, ex, ey) ที่คืนยังเป็นกรอบดั้งเดิมก่อนหมุน (พิกัดภาพต้นฉบับ)
    return out_quads, (sx, sy, ex, ey)



from typing import Dict

def _analyze_nozzles(
    image_path: str,
    weights_path: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    pad_ratio: float,
    save_rois_dir: str | None = None,
    modelClassifyname : str = "bestBlockV8.pt"
) -> Tuple[np.ndarray, List[NozzleBox], List[List[bool]], List[List[Tuple[int,int]]], Dict[str, Optional[float]]]:

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    det = NozzleDetector(weights_path, imgsz=imgsz, conf=conf, iou=iou)
    boxes = det.detect(img)

    # === คำนวณสเกลจากกล่อง ===
    scale = estimate_scale_from_boxes(boxes, nominal_spacing_mm=173.0)
    mm_per_px = scale.get("mm_per_px", None)
    angle_deg = float(scale.get("angle_deg", 0.0))

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(save_rois_dir, stem) if save_rois_dir else None
    if out_dir: _ensure_dir(out_dir)

    quad_statuses: List[List[bool]] = []
    quad_centers:  List[List[Tuple[int,int]]] = []

    for idx, box in enumerate(boxes, start=1):
        quads, _ = _crop_circle_quadrants(
            img, box, pad_ratio=pad_ratio, mm_per_px=mm_per_px, angle_deg=angle_deg
        )
        if out_dir: _save_rois(quads, out_dir, stem, idx)

        centers = _quadrant_centers(
            box, pad_ratio, mm_per_px=mm_per_px, angle_deg=angle_deg
        )
        quad_centers.append(centers)

        statuses: List[bool] = []
        for roi in quads:
            try:
                statuses.append(bool(isBlockedHole(roi, modelName=modelClassifyname)))
            except Exception:
                statuses.append(False)
        quad_statuses.append(statuses)

    # **ส่งออก scale ออกไปด้วย**
    return img, boxes, quad_statuses, quad_centers, scale




# ===========================
# === MAIN (PUBLIC  API)  ===
# ===========================

def detect_blocked_nozzles(
    image_path: str,
    *,
    imgsz: int = 1280,
    conf: float = 0.9,
    iou: float = 0.5,
    pad_ratio: float = 0.05,
    save_rois_dir: str | None = None,
    detectionModel : str = "bestNozzleV8.pt",
    classifyModel  : str = "bestBlockV8.pt"
) -> List[Tuple[int, int]]:
    """
    Return list of (x, y) centers in original image coordinates
    for quadrants judged as BLOCKED.
    """
    weights_path = _auto_find_weights(image_path=image_path, filename=detectionModel)

    # ถ้าแก้ _analyze_nozzles ให้คืน centers มาแล้ว:
    _, boxes, quad_statuses, quad_centers , scale = _analyze_nozzles(
        image_path, weights_path,
        imgsz=imgsz, conf=conf, iou=iou, pad_ratio=pad_ratio,
        save_rois_dir=save_rois_dir,
        modelClassifyname=classifyModel
    )

    blocked_points: List[Tuple[int, int]] = []
    for statuses, centers in zip(quad_statuses, quad_centers):
        for is_blocked, (x, y) in zip(statuses, centers):
            if bool(is_blocked):
                blocked_points.append((int(x), int(y)))
    return blocked_points


# # ================================
# # === DEVELOPER / VISUAL UTILS ===
# # ================================
def _get_box_confidence(box, default: float | None = None) -> float | None:
    """
    Try to extract confidence score from a NozzleBox using common field names.
    Returns a float in [0,1] or None if not available.
    """
    for attr in ("conf", "score", "confidence", "prob"):
        v = getattr(box, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default

def _put_label_with_bg(img: np.ndarray, text: str, org: Tuple[int, int],
                       font_scale: float = 0.5, thickness: int = 1,
                       text_color: Tuple[int, int, int] = (255, 255, 255),
                       bg_color: Tuple[int, int, int] = (0, 0, 0),
                       pad: int = 3) -> None:
    """
    Draw text with a filled background rectangle for readability.
    org is the bottom-left corner of the text (same as cv2.putText).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    # background rect (top-left to bottom-right)
    cv2.rectangle(img, (x - pad, y - th - baseline - pad),
                       (x + tw + pad, y + baseline + pad),
                       bg_color, -1)
    cv2.putText(img, text, org, font, font_scale, text_color, thickness, cv2.LINE_AA)

def _resize_for_display(img: np.ndarray, max_w: int = 1600, max_h: int = 900) -> np.ndarray:
    """Downscale image for display while keeping aspect ratio."""
    H, W = img.shape[:2]
    scale = min(max_w / W, max_h / H, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    return img


def _blend_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: Tuple[int, int, int], alpha: float = 0.28) -> None:
    """Draw a translucent filled rectangle on `img` in-place."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def developerTest(
    image_path: str,
    weights_path: str | None = None,   # None = auto-find
    *,
    imgsz: int = 1280,
    conf: float = 0.9,
    iou: float = 0.5,
    pad_ratio: float = 0.05,
    show: bool = True,
    wait_ms: int = 0,
    save_path: str | None = None,
    save_rois_dir: str | None = None,
) -> List[Tuple[int, int]]:
    """
    Developer utility (mirror user API):
        - Runs the same analysis as `detect_blocked_nozzles`
        - Overlays: YOLO boxes, circle, *rotated* grid axes (per angle_deg)
        - Marks the centers of BLOCKED quadrants with labels
        - Returns the list of blocked points [(x, y), ...]
    """
    if weights_path is None:
        weights_path = _auto_find_weights(image_path, filename="bestNozzleV8.pt")

    # รัน logic จริงให้ได้ผลเดียวกับ user API
    img, boxes, quad_statuses, quad_centers, scale = _analyze_nozzles(
        image_path, weights_path,
        imgsz=imgsz, conf=conf, iou=iou, pad_ratio=pad_ratio,
        save_rois_dir=save_rois_dir,
        modelClassifyname="bestBlockV8.pt"
    )

    vis = img.copy()
    green = (60, 200, 60)
    red   = (0, 0, 255)
    cyan  = (255, 200, 0)
    yellow= (0, 220, 255)
    white = (255, 255, 255)

    # =========================
    # อ่าน scale & angle
    # =========================
    mm_per_px      = scale.get("mm_per_px")
    angle_deg      = float(scale.get("angle_deg", 0.0))
    angle_std_deg  = scale.get("angle_std_deg", None)

    # แสดงค่าด้านบนซ้าย
    base_y = 24
    if mm_per_px:
        _put_label_with_bg(
            vis,
            f"{mm_per_px:.4f} mm/px  ({1.0/mm_per_px:.2f} px/mm)",
            (10, base_y), font_scale=0.6, thickness=2
        )
        base_y += 22
    _put_label_with_bg(
        vis,
        f"angle: {angle_deg:.2f} deg" + (f" (+-{angle_std_deg:.2f})" if angle_std_deg is not None else ""),
        (10, base_y), font_scale=0.6, thickness=2
    )

    blocked_points: List[Tuple[int, int]] = []

    # พรีคำนวณ unit vectors ของแกนกริดที่เอียง
    th = math.radians(angle_deg)
    ux, uy = math.cos(th), math.sin(th)          # แกน X ของกริด
    vx, vy = -math.sin(th), math.cos(th)         # แกน Y ของกริด

    for n, (box, statuses, centers) in enumerate(zip(boxes, quad_statuses, quad_centers), start=1):
        x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), cyan, 2)

        conf_val = _get_box_confidence(box)
        label_text = f"#{n}  {conf_val*100:.1f}%" if conf_val is not None else f"#{n}"
        _put_label_with_bg(
            vis, label_text, (x1, max(0, y1 - 6)),
            font_scale=0.6, thickness=2, text_color=cyan, bg_color=(0,0,0)
        )

        # วาดวงกลม nozzle
        cx, cy, R = _circle_from_box(box)
        cv2.circle(vis, (cx, cy), int(R), yellow, 2)

        # วาดเส้นแกน "ที่เอียง" ตรงกับกริดจริง (ยาว ~ r_pad)
        pr    = int(R * pad_ratio)
        r_pad = int(R + pr)
        xA = (int(round(cx - r_pad*ux)), int(round(cy - r_pad*uy)))
        xB = (int(round(cx + r_pad*ux)), int(round(cy + r_pad*uy)))
        yA = (int(round(cx - r_pad*vx)), int(round(cy - r_pad*vy)))
        yB = (int(round(cx + r_pad*vx)), int(round(cy + r_pad*vy)))
        cv2.line(vis, xA, xB, white, 1)
        cv2.line(vis, yA, yB, white, 1)

            # ... ด้านบนเหมือนเดิมจนได้ boxes, quad_statuses, quad_centers, scale, angle_deg ...

    mm_per_px = scale.get("mm_per_px")
    # helper ทำข้อความพิกัด
    def _coord_label(px, py):
        # if mm_per_px:
        #     x_mm = px * mm_per_px
        #     y_mm = py * mm_per_px
        #     return f"{int(px)} {int(py)}  |  {x_mm:.1f}mm {y_mm:.1f}mm"
        # else:
        #     return f"{int(px)} {int(py)}"
        
        return f"{int(px)} {int(py)}"

    for n, (box, statuses, centers) in enumerate(zip(boxes, quad_statuses, quad_centers), start=1):
        # ... วาด bbox, วงกลม, เส้นแกนเอียง เหมือนที่เราแก้ก่อนหน้า ...

        # วางมาร์คและป้าย "ชื่อพิกัด" แทน TL/TR/BL/BR
        for is_blocked, (px, py) in zip([bool(x) for x in statuses], centers):
            color = (0, 0, 255) if is_blocked else (60, 200, 60)  # แดง=blocked, เขียว=clear
            cv2.circle(vis, (int(px), int(py)), 6, color, -1, lineType=cv2.LINE_AA)

            label = _coord_label(px, py)  # <<== ใช้พิกัดเป็นชื่อ
            _put_label_with_bg(
                vis, label, (int(px) + 8, int(py) - 8),
                font_scale=0.5, thickness=1, text_color=(255,255,255),
                bg_color=(0,0,180) if is_blocked else (0,100,0)
            )

            if is_blocked:
                blocked_points.append((int(px), int(py)))


    # บันทึกภาพ / แสดงหน้าต่าง เท่าเดิม
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        cv2.imwrite(save_path, vis)
    if show:
        vis_show = _resize_for_display(vis, max_w=1000, max_h=900)
        cv2.imshow("developerTest", vis_show)
        cv2.waitKey(wait_ms)

    return blocked_points




# ============================
# === CLI EXAMPLE (optional) ==
# ============================

if __name__ == "__main__":
    folder = r"C:\Project\nozzleScan\pictures\direction"
    outdir = None  # set to a folder to dump raw quadrant ROIs per image
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    pictures = [f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(exts)]
    pictures.sort()
    if not pictures:
        print("No images found.")
        raise SystemExit(0)

    try:
        idx = 0
        while True:
            pic = pictures[idx]
            fname = os.path.basename(pic)

            _ = developerTest(
                pic, weights_path=None,  # auto-find best.pt
                imgsz=1280, conf=0.9, iou=0.5, pad_ratio=0.05,
                show=True, wait_ms=1,
                save_path=None,          # e.g., "out/out.jpg"
                save_rois_dir=outdir
            )
            print(f"[{idx+1}/{len(pictures)}] {fname}")

            k = cv2.waitKey(0) & 0xFF
            if k in (ord('q'), ord('Q'), 27):   # Q or ESC → exit
                break
            elif k in (ord('d'), ord('D')):     # D → next image
                idx = (idx + 1) % len(pictures)
            elif k in (ord('a'), ord('A')):     # A → previous image
                idx = (idx - 1) % len(pictures)
            else:
                pass

    except InvalidInputImageError as e:
        print("Invalid image:", e)
    except FileNotFoundError as e:
        print(e)
    finally:
        cv2.destroyAllWindows()
