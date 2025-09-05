# blocked_orchestrator.py
"""
Blocked Nozzle Orchestrator — circle-ROI quadrant check (4×4 grid)
------------------------------------------------------------------
Purpose:
    Read an image → detect 16 nozzles → build per-nozzle circular ROIs →
    split into TL/TR/BL/BR quadrants → run `isBlockedHole` → return blocked names.

Quick use:
    from blocked_orchestrator import detect_blocked_nozzles, developerTest
    names = detect_blocked_nozzles(image_path, weights_path,
                                   imgsz=1280, conf=0.25, iou=0.5, pad_ratio=0.05)
    # → ['nozzle1TopLeft', 'nozzle7BottomRight', ...]
    # For visualization:
    dev = developerTest(image_path, weights_path, show=True, save_path=None)

Workflow:
    1) NozzleDetector.detect_16(img)  → 16 boxes (row-major, with cx, cy, R)
    2) Circle ROI per box (+ pad_ratio) → split 2×2 (TL, TR, BL, BR)
    3) `isBlockedHole(roi)` per quadrant → collect blocked labels
    4) `developerTest` draws bbox/circle/grid & overlays red/green by status

Key params:
    image_path (str), weights_path (str),
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
from holecheck import isBlockedHole
from nozzle_types import NozzleBox

# Quadrant name order for building result labels
_QUAD_NAMES = ("TopLeft", "TopRight", "BottomLeft", "BottomRight")


def _analyze_nozzles(
    image_path: str,
    weights_path: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    pad_ratio: float,
):
    """
    Read image → detect_16 → build 'circular' ROIs → return (img, boxes, quad_statuses)

    quad_statuses: List[List[bool]] with the same length as `boxes`,
                   each item is [TL, TR, BL, BR] booleans.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    det = NozzleDetector(weights_path, imgsz=imgsz, conf=conf, iou=iou)
    boxes = det.detect_16(img)  # may raise InvalidInputImageError

    quad_statuses: List[List[bool]] = []
    for box in boxes:
        quads, _ = _crop_circle_quadrants(img, box, pad_ratio=pad_ratio)  # [TL, TR, BL, BR]
        statuses = []
        for roi in quads:
            try:
                statuses.append(bool(isBlockedHole(roi)))
                # statuses.append(False)
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
    weights_path: str,
    *,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.5,
    pad_ratio: float = 0.05,
) -> List[str]:
    """
    Use 'circular' ROIs → return blocked labels, e.g., ["nozzle1TopLeft", ...]
    """
    img, boxes, quad_statuses = _analyze_nozzles(
        image_path, weights_path, imgsz=imgsz, conf=conf, iou=iou, pad_ratio=pad_ratio
    )

    blocked_names: List[str] = []
    for box, statuses in zip(boxes, quad_statuses):
        nozzle_num = int(box.grid_index) + 1
        for q_idx, is_blocked in enumerate(statuses):
            if is_blocked:
                blocked_names.append(f"nozzle{nozzle_num}{_QUAD_NAMES[q_idx]}")
    return blocked_names


# ===== Developer visual test =====
# NOTE: This developer section is intentionally kept commented and unused in production.
#       Comments have been translated to English; behavior remains as in your original notes.

# def _resize_for_display(img: np.ndarray, max_w: int = 1600, max_h: int = 900) -> np.ndarray:
#     H, W = img.shape[:2]
#     scale = min(max_w / W, max_h / H, 1.0)
#     if scale < 1.0:
#         img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
#     return img

# def _blend_rect(img, x1, y1, x2, y2, color, alpha=0.28):
#     overlay = img.copy()
#     cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
#     cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# def developerTest(
#     image_path: str,
#     weights_path: str,
#     *,
#     imgsz: int = 1280,
#     conf: float = 0.25,
#     iou: float = 0.5,
#     pad_ratio: float = 0.05,
#     show: bool = True,
#     wait_ms: int = 0,
#     save_path: str | None = None,
#     save_rois: bool = True   # <<<< added option to save ROIs
# ) -> List[str]:
#     """
#     Draw bbox, circle, split lines, color by status, and return the blocked labels.
#     If save_rois=True, save each quadrant ROI under roi/<image_stem>/.
#     """
#     img, boxes, quad_statuses = _analyze_nozzles(
#         image_path, weights_path, imgsz=imgsz, conf=conf, iou=iou, pad_ratio=pad_ratio
#     )
#
#     vis = img.copy()
#     green = (60, 200, 60)
#     red   = (0, 0, 255)
#     cyan  = (255, 200, 0)
#     yellow= (0, 220, 255)
#     white = (255, 255, 255)
#
#     blocked_names: List[str] = []
#
#     # Prepare ROI output folder
#     roi_root = os.path.join("roi", os.path.splitext(os.path.basename(image_path))[0])
#     if save_rois:
#         os.makedirs(roi_root, exist_ok=True)
#
#     for box, statuses in zip(boxes, quad_statuses):
#         x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
#         nozzle_num = int(box.grid_index) + 1
#
#         # Draw bbox + label
#         cv2.rectangle(vis, (x1, y1), (x2, y2), cyan, 2)
#         cv2.putText(
#             vis, f"#{nozzle_num}", (x1, max(0, y1 - 6)),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, cyan, 2, cv2.LINE_AA
#         )
#
#         # Circle
#         w = x2 - x1
#         h = y2 - y1
#         cx = int((x1 + x2) / 2) if box.cx is None else int(box.cx)
#         cy = int((y1 + y2) / 2) if box.cy is None else int(box.cy)
#         R  = int(0.4 * min(w, h)) if (box.R is None or box.R <= 0) else int(box.R)
#         R  = max(8, R)
#         cv2.circle(vis, (cx, cy), R, yellow, 2)
#
#         pr = int(R * pad_ratio)
#         r_pad = R + pr
#         sx, sy = max(0, cx - r_pad), max(0, cy - r_pad)
#         ex, ey = min(vis.shape[1], cx + r_pad), min(vis.shape[0], cy + r_pad)
#         mx, my = (sx + ex) // 2, (sy + ey) // 2
#         cv2.rectangle(vis, (sx, sy), (ex, ey), (160, 160, 160), 1)
#         cv2.line(vis, (mx, sy), (mx, ey), white, 1)
#         cv2.line(vis, (sx, my), (ex, my), white, 1)
#
#         # Crop ROIs per quadrant + optionally save
#         quads = [
#             img[sy:my, sx:mx],  # TL
#             img[sy:my, mx:ex],  # TR
#             img[my:ey, sx:mx],  # BL
#             img[my:ey, mx:ex],  # BR
#         ]
#         if save_rois:
#             for q_idx, q in enumerate(quads):
#                 q_name = f"nozzle{nozzle_num}_{_QUAD_NAMES[q_idx]}.png"
#                 cv2.imwrite(os.path.join(roi_root, q_name), q)
#
#         # Overlay result colors
#         _blend_rect(vis, sx, sy, mx, my, red if statuses[0] else green)
#         _blend_rect(vis, mx, sy, ex, my, red if statuses[1] else green)
#         _blend_rect(vis, sx, my, mx, ey, red if statuses[2] else green)
#         _blend_rect(vis, mx, my, ex, ey, red if statuses[3] else green)
#
#         # Collect blocked labels
#         for q_idx, is_blocked in enumerate(statuses):
#             if is_blocked:
#                 blocked_names.append(f"nozzle{nozzle_num}{_QUAD_NAMES[q_idx]}")
#
#     # Save annotated image if requested
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         cv2.imwrite(save_path, vis)
#     if show:
#         vis_show = _resize_for_display(vis, max_w=1000, max_h=900)
#         cv2.imshow("developerTest", vis_show)
#         cv2.waitKey(wait_ms)
#
#     return blocked_names
#
#
# # ===== Example usage from main (call developerTest instead of detect_blocked_nozzles) =====
# if __name__ == "__main__":
#     import glob
#
#     weights = r"C:\Project\nozzleScan\NozzleCleanerProject\best.pt"
#     folder  = r"C:\Project\nozzleScan\NozzleCleanerProject\pictures"
#
#     exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
#     pictures = [f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(exts)]
#     pictures.sort()
#     if not pictures:
#         print("No images found.")
#         raise SystemExit(0)
#
#     try:
#         idx = 0
#         while True:
#             pic = pictures[idx]
#             fname = os.path.basename(pic)
#
#             # Show annotated image (the function shows quickly and returns)
#             _ = developerTest(
#                 pic, weights,
#                 imgsz=1280, conf=0.25, iou=0.5, pad_ratio=0.05,
#                 show=True, wait_ms=1,   # IMPORTANT: do not block for key inside the function
#                 save_rois=False
#             )
#             print(f"[{idx+1}/{len(pictures)}] {fname}")
#
#             # Handle key here instead to control navigation
#             k = cv2.waitKey(0) & 0xFF
#             if k in (ord('q'), ord('Q'), 27):   # Q or ESC → exit
#                 break
#             elif k in (ord('d'), ord('D')):     # D → next image
#                 idx = (idx + 1) % len(pictures)
#             elif k in (ord('a'), ord('A')):     # A → previous image
#                 idx = (idx - 1) % len(pictures)
#             else:
#                 # Other keys: do nothing (keep current image)
#                 pass
#
#     except InvalidInputImageError as e:
#         print("Invalid image:", e)
#     except FileNotFoundError as e:
#         print(e)
#     finally:
#         cv2.destroyAllWindows()
