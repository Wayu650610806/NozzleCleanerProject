# blocked_orchestrator.py
"""
Blocked Nozzle Orchestrator — circle-ROI quadrant check (4×4 grid)
------------------------------------------------------------------
Purpose:
    Read an image → detect 16 nozzles → build per-nozzle circular ROIs →
    split into TL/TR/BL/BR quadrants → run `isBlockedHole` → return blocked names.

Quick use:
    from blocked_orchestrator import detect_blocked_nozzles, developerTest

    # เรียกใช้งานแบบง่ายสุด: ส่งแค่ path รูป
    names = detect_blocked_nozzles(
        image_path,
        imgsz=1280, conf=0.25, iou=0.5, pad_ratio=0.05,
        save_rois_dir=r"C:\out_rois"  # << บันทึก ROI ดิบ (ที่ส่งเข้า isBlockedHole) ของทุกควอดแรนท์
    )

    # โหมดนักพัฒนา (แสดงภาพ annotate) + บันทึก ROI ดิบเหมือนกัน
    dev = developerTest(
        image_path, weights_path=None,  # None = auto-find best.pt
        imgsz=1280, conf=0.25, iou=0.5, pad_ratio=0.05,
        show=True, wait_ms=1, save_path=None,
        save_rois_dir=r"C:\out_rois"    # << บันทึก ROI ดิบเหมือน detect_blocked_nozzles
    )

Workflow:
    1) NozzleDetector.detect_16(img)  → 16 boxes (row-major, with cx, cy, R)
    2) Circle ROI per box (+ pad_ratio) → split 2×2 (TL, TR, BL, BR)
    3) `isBlockedHole(roi)` per quadrant → collect blocked labels
    4) (optional) save ROIs (exactly the masked patches passed to isBlockedHole)

Key params:
    image_path (str)
    imgsz (int), conf (float), iou (float), pad_ratio (float, circle padding)

Returns:
    - detect_blocked_nozzles: List[str] (e.g., ['nozzle3TopRight', ...])
    - developerTest: same list; also shows/saves annotated preview

Raises:
    FileNotFoundError, InvalidInputImageError
"""

from __future__ import annotations

from typing import List, Tuple
import os

import cv2
import numpy as np

from detector import NozzleDetector, InvalidInputImageError
from holecheckAI import isBlockedHole
from nozzle_types import NozzleBox

# Quadrant name order for building result labels
_QUAD_NAMES = ("TopLeft", "TopRight", "BottomLeft", "BottomRight")


# ---------------------------- helpers ----------------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _auto_find_weights(image_path: str, filename: str = "best.pt") -> str:
    """
    Find 'best.pt' automatically in (priority):
        1) The folder of this Python module
        2) The folder of the image_path
        3) Current Working Directory
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir  = os.path.dirname(os.path.abspath(image_path))
    cwd        = os.getcwd()

    candidates = [
        os.path.join(module_dir, filename),
        os.path.join(image_dir, filename),
        os.path.join(cwd, filename),
    ]

    seen = set(); ordered = []
    for c in candidates:
        if c not in seen:
            ordered.append(c); seen.add(c)

    for p in ordered:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Could not find '{filename}'. Looked in:\n- " + "\n- ".join(ordered)
    )

def _save_rois(quads: List[np.ndarray], out_dir: str, stem: str, nozzle_num: int):
    """
    Save 4 quadrant ROIs (exact masked patches) with consistent names.
    No watermark/overlay. Pure inputs as sent to isBlockedHole.
    """
    _ensure_dir(out_dir)
    for q_idx, q in enumerate(quads):
        fname = f"{stem}_nozzle{nozzle_num}_{_QUAD_NAMES[q_idx]}.png"
        cv2.imwrite(os.path.join(out_dir, fname), q)


# --------------------------- core logic ---------------------------

def _analyze_nozzles(
    image_path: str,
    weights_path: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    pad_ratio: float,
    # NEW: โฟลเดอร์สำหรับบันทึก ROI ดิบของทุกควอดแรนท์
    save_rois_dir: str | None = None,
):
    """
    Read image → detect_16 → build 'circular' ROIs → return (img, boxes, quad_statuses)

    quad_statuses: List[List[bool]] with the same length as `boxes`,
                   each item is [TL, TR, BL, BR] booleans.

    If save_rois_dir is not None, dumps the exact ROI patches passed to
    isBlockedHole() under:
        save_rois_dir/<image_stem>/<image_stem>_nozzleN_Quadrant.png
    (ทุกควอดแรนท์จะถูกบันทึก ไม่แยกตาม label)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    det = NozzleDetector(weights_path, imgsz=imgsz, conf=conf, iou=iou)
    boxes = det.detect_16(img)  # may raise InvalidInputImageError

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(save_rois_dir, stem) if save_rois_dir else None
    if out_dir:
        _ensure_dir(out_dir)

    quad_statuses: List[List[bool]] = []
    for box in boxes:
        # ทำ ROI แยก 4 ส่วนแบบมาส์กวงกลม (เหมือน input จริงของ isBlockedHole)
        quads, _ = _crop_circle_quadrants(img, box, pad_ratio=pad_ratio)  # [TL, TR, BL, BR]

        # บันทึก ROI ดิบของทุกควอดแรนท์ (ถ้าระบุ save_rois_dir)
        if out_dir:
            nozzle_num = int(box.grid_index) + 1 if box.grid_index is not None else 0
            _save_rois(quads, out_dir, stem, nozzle_num)

        # เรียกใช้ isBlockedHole ตามเดิมเพื่อได้สถานะกลับ
        statuses = []
        for roi in quads:
            try:
                # for AI 
                statuses.append(bool(isBlockedHole(roi)))
                #  #For Texture 
                # is_blocked, _ = isBlockedHole(roi) 
                # statuses.append(bool(is_blocked)) 

                # # statuses.append(False)
            except Exception:
                statuses.append(False)

        quad_statuses.append(statuses)

    return img, boxes, quad_statuses


def _crop_circle_quadrants(
    img: np.ndarray,
    box: NozzleBox,
    pad_ratio: float = 0.05
) -> Tuple[List[np.ndarray], Tuple[int, int, int, int]]:
    """
    Return 4 ROIs by referencing the nozzle 'circle':
      - Build a bounding square around the circle (use cx, cy, R or estimate from bbox)
      - Split into 2×2 as TL, TR, BL, BR
      - Apply a 'circular mask' to each piece (pixels outside the circle are blacked out)

    Returns: ([TL, TR, BL, BR], (sx, sy, ex, ey))
             The second tuple is the square bounds around the circle for drawing guide lines.
    """
    H, W = img.shape[:2]

    # Use cx, cy, R if available; otherwise estimate from bbox
    x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
    w = x2 - x1
    h = y2 - y1
    cx = int((x1 + x2) / 2) if box.cx is None else int(box.cx)
    cy = int((y1 + y2) / 2) if box.cy is None else int(box.cy)
    R = int(0.4 * min(w, h)) if (box.R is None or box.R <= 0) else int(box.R)
    R = max(8, R)

    # Padding based on radius (not bbox)
    pr = int(R * pad_ratio)
    r_pad = R + pr

    sx = max(0, cx - r_pad)
    ex = min(W, cx + r_pad)
    sy = max(0, cy - r_pad)
    ey = min(H, cy + r_pad)
    if ex <= sx or ey <= sy:
        # Fallback: use bbox
        sx, sy, ex, ey = x1, y1, x2, y2

    patch = img[sy:ey, sx:ex].copy()
    if patch.size == 0:
        return [np.zeros((1, 1, 3), dtype=np.uint8)] * 4, (sx, sy, ex, ey)

    # Circle center in patch coordinates
    pcx = cx - sx
    pcy = cy - sy

    # Split patch into 4 quadrants
    H2, W2 = patch.shape[:2]
    mx = W2 // 2
    my = H2 // 2
    quads_rect = [
        (0, 0, mx, my),      # TL
        (mx, 0, W2, my),     # TR
        (0, my, mx, H2),     # BL
        (mx, my, W2, H2),    # BR
    ]

    out_quads: List[np.ndarray] = []
    # Apply circular mask for each quadrant
    for (qx1, qy1, qx2, qy2) in quads_rect:
        quad = patch[qy1:qy2, qx1:qx2].copy()
        qH, qW = quad.shape[:2]
        mask = np.zeros((qH, qW), dtype=np.uint8)

        # Draw circle on mask — shift center relative to the sub-quad
        local_cx = pcx - qx1
        local_cy = pcy - qy1
        # Keep using global R; overflow beyond quad bounds is naturally clipped
        cv2.circle(mask, (int(local_cx), int(local_cy)), int(R), 255, thickness=-1)

        # Apply mask (bitwise_and) → outside circle becomes black
        quad_masked = cv2.bitwise_and(quad, quad, mask=mask)
        out_quads.append(quad_masked)

    return out_quads, (sx, sy, ex, ey)


def detect_blocked_nozzles(
    image_path: str,
    *,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.5,
    pad_ratio: float = 0.05,
    # NEW: โฟลเดอร์เอาไว้บันทึก ROI ดิบของทุกควอดแรนท์
    save_rois_dir: str | None = None,
) -> List[str]:
    """
    Use 'circular' ROIs → return blocked labels, e.g., ["nozzle1TopLeft", ...]
    - Automatically finds 'best.pt' near the code/image/CWD.
    - If save_rois_dir is set, dumps the exact patches used by isBlockedHole (all quadrants).
    """
    weights_path = _auto_find_weights(image_path, filename="best.pt")

    img, boxes, quad_statuses = _analyze_nozzles(
        image_path, weights_path,
        imgsz=imgsz, conf=conf, iou=iou, pad_ratio=pad_ratio,
        save_rois_dir=save_rois_dir
    )

    blocked_names: List[str] = []
    for box, statuses in zip(boxes, quad_statuses):
        nozzle_num = int(box.grid_index) + 1 if box.grid_index is not None else 0
        for q_idx, is_blocked in enumerate(statuses):
            if bool(is_blocked):
                blocked_names.append(f"nozzle{nozzle_num}{_QUAD_NAMES[q_idx]}")
    return blocked_names


# ------------------------- developer utils -------------------------

def _resize_for_display(img: np.ndarray, max_w: int = 1600, max_h: int = 900) -> np.ndarray:
    H, W = img.shape[:2]
    scale = min(max_w / W, max_h / H, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    return img

def _blend_rect(img, x1, y1, x2, y2, color, alpha=0.28):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def developerTest(
    image_path: str,
    weights_path: str | None = None,   # None = auto-find 'best.pt'
    *,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.5,
    pad_ratio: float = 0.05,
    show: bool = True,
    wait_ms: int = 0,
    save_path: str | None = None,
    # NEW: เซฟ ROI ดิบ (ไม่มีตัวหนังสือ) — เหมือน detect_blocked_nozzles
    save_rois_dir: str | None = None,
    # NOTE: ตัด feature เซฟ debug ROI แบบมีตัวหนังสือออกเพื่อไม่สับสน
) -> List[str]:
    """
    Draw bbox, circle, split lines, color by status, and return the blocked labels.
    - If save_rois_dir is set → dumps the exact patches used by isBlockedHole (clean, all quadrants).
    """
    if weights_path is None:
        weights_path = _auto_find_weights(image_path, "best.pt")

    # วิเคราะห์จริง + (ถ้ามี) เซฟ ROI ดิบทั้งหมด
    img, boxes, quad_statuses = _analyze_nozzles(
        image_path, weights_path,
        imgsz=imgsz, conf=conf, iou=iou, pad_ratio=pad_ratio,
        save_rois_dir=save_rois_dir
    )

    vis = img.copy()
    green = (60, 200, 60)
    red   = (0, 0, 255)
    cyan  = (255, 200, 0)
    yellow= (0, 220, 255)
    white = (255, 255, 255)

    blocked_names: List[str] = []

    for box, statuses in zip(boxes, quad_statuses):
        x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
        nozzle_num = int(box.grid_index) + 1 if box.grid_index is not None else 0

        # วาดกรอบ YOLO + label
        cv2.rectangle(vis, (x1, y1), (x2, y2), cyan, 2)
        cv2.putText(
            vis, f"#{nozzle_num}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cyan, 2, cv2.LINE_AA
        )

        # วาดวงกลมอ้างอิง
        w = x2 - x1; h = y2 - y1
        cx = int((x1 + x2) / 2) if box.cx is None else int(box.cx)
        cy = int((y1 + y2) / 2) if box.cy is None else int(box.cy)
        R  = int(0.4 * min(w, h)) if (box.R is None or box.R <= 0) else int(box.R)
        R  = max(8, R)
        cv2.circle(vis, (cx, cy), R, yellow, 2)

        # กรอบแบ่ง 4 ส่วน (เพื่อทับสี)
        pr = int(R * pad_ratio)
        r_pad = R + pr
        sx, sy = max(0, cx - r_pad), max(0, cy - r_pad)
        ex, ey = min(vis.shape[1], cx + r_pad), min(vis.shape[0], cy + r_pad)
        mx, my = (sx + ex) // 2, (sy + ey) // 2
        cv2.rectangle(vis, (sx, sy), (ex, ey), (160, 160, 160), 1)
        cv2.line(vis, (mx, sy), (mx, ey), white, 1)
        cv2.line(vis, (sx, my), (ex, my), white, 1)

        # ทับสี overlay ตามสถานะ
        s0 = bool(statuses[0]) if len(statuses) > 0 else False
        s1 = bool(statuses[1]) if len(statuses) > 1 else False
        s2 = bool(statuses[2]) if len(statuses) > 2 else False
        s3 = bool(statuses[3]) if len(statuses) > 3 else False

        _blend_rect(vis, sx, sy, mx, my, red if s0 else green)  # TL
        _blend_rect(vis, mx, sy, ex, my, red if s1 else green)  # TR
        _blend_rect(vis, sx, my, mx, ey, red if s2 else green)  # BL
        _blend_rect(vis, mx, my, ex, ey, red if s3 else green)  # BR

        # เก็บชื่อควอดแรนท์ที่ตัน
        for q_idx, is_blocked in enumerate((s0, s1, s2, s3)):
            if is_blocked:
                blocked_names.append(f"nozzle{nozzle_num}{_QUAD_NAMES[q_idx]}")

    # บันทึกภาพ annotate ถ้าระบุ
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        cv2.imwrite(save_path, vis)

    # แสดงบนหน้าจอถ้าต้องการ
    if show:
        vis_show = _resize_for_display(vis, max_w=1000, max_h=900)
        cv2.imshow("developerTest", vis_show)
        cv2.waitKey(wait_ms)

    return blocked_names


# --------------------------- CLI example ---------------------------

if __name__ == "__main__":
    import glob

    folder  = r"C:\Project\nozzleScan\NozzleCleanerProject\dataset\clean"
    # outdir  = r"C:\Project\nozzleScan\ClassifierDataset\dirty"
    outdir  = None  # << โฟลเดอร์ปลายทางสำหรับบันทึก ROI ดิบ

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

            # แสดงผล + บันทึก ROI ดิบทั้งหมดของรูปนี้ไว้ใน outdir/<image_stem>/
            _ = developerTest(
                pic, weights_path=None,           # auto-find best.pt
                imgsz=1280, conf=0.25, iou=0.5, pad_ratio=0.05,
                show=True, wait_ms=1,
                save_path=None,                   # หรือ "out/out.jpg"
                save_rois_dir=outdir              # << สำคัญ: บันทึก ROI ดิบ
            )
            print(f"[{idx+1}/{len(pictures)}] {fname}")

            # ควบคุมการเลื่อนรูป
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
