# blocked_orchestrator.py
"""
Blocked Nozzle Orchestrator — circle-ROI quadrant check (4×4 grid)
------------------------------------------------------------------
Purpose (for users):
    Read an image → detect 16 nozzles → build per-nozzle *circular* ROI →
    split into 4 quadrants (TL/TR/BL/BR) → call `isBlockedHole(roi)` →
    return names of blocked quadrants, e.g. ["nozzle3TopRight", ...].

Quick Use:
    from blocked_orchestrator import detect_blocked_nozzles, developerTest

    names = detect_blocked_nozzles(
        image_path,
        imgsz=1280, conf=0.25, iou=0.5, pad_ratio=0.05,
        save_rois_dir=r"C:\out_rois"  # saves the exact patches passed into isBlockedHole
    )

    dev = developerTest(
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
    detect_blocked_nozzles(...) -> List[str]
    developerTest(...)          -> List[str] (and shows/saves an annotated preview if requested)

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
from wrotatePicture import auto_rotate_by_aruco


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


# ===========================
# === CORE (INTERNAL API) ===
# ===========================

def _crop_circle_quadrants(
    img: np.ndarray,
    box: NozzleBox,
    pad_ratio: float = 0.05
) -> Tuple[List[np.ndarray], Tuple[int, int, int, int]]:
    """
    Build 4 quadrant ROIs (TL/TR/BL/BR) from a circular reference inside a YOLO box:
      1) Build a square around the circle (radius + padding)
      2) Split into 2×2 rectangles
      3) Apply a *circular mask* on each sub-rectangle (outside-circle pixels become black)

    Returns:
        (quads, (sx, sy, ex, ey))
        quads: [TL, TR, BL, BR] masked BGR patches
        (sx, sy, ex, ey): the square bounds used for guide lines
    """
    H, W = img.shape[:2]
    cx, cy, R = _circle_from_box(box)

    # Radius padding
    pr = int(R * pad_ratio)
    r_pad = R + pr

    sx, ex = max(0, cx - r_pad), min(W, cx + r_pad)
    sy, ey = max(0, cy - r_pad), min(H, cy + r_pad)

    # Fallback to raw bbox if computed region is invalid
    if ex <= sx or ey <= sy:
        x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
        sx, sy, ex, ey = x1, y1, x2, y2

    patch = img[sy:ey, sx:ex].copy()
    if patch.size == 0:
        return [np.zeros((1, 1, 3), dtype=np.uint8)] * 4, (sx, sy, ex, ey)

    # Circle center in patch coordinates
    pcx, pcy = cx - sx, cy - sy

    # Split into 4 rectangles
    H2, W2 = patch.shape[:2]
    mx, my = W2 // 2, H2 // 2
    rects = [
        (0, 0, mx, my),        # TL
        (mx, 0, W2, my),       # TR
        (0, my, mx, H2),       # BL
        (mx, my, W2, H2),      # BR
    ]

    out_quads: List[np.ndarray] = []
    for (qx1, qy1, qx2, qy2) in rects:
        quad = patch[qy1:qy2, qx1:qx2].copy()
        qH, qW = quad.shape[:2]
        mask = np.zeros((qH, qW), dtype=np.uint8)

        # Circle center relative to the sub-quad
        local_cx = pcx - qx1
        local_cy = pcy - qy1
        cv2.circle(mask, (int(local_cx), int(local_cy)), int(R), 255, thickness=-1)

        # Apply mask (outside the circle becomes black)
        quad_masked = cv2.bitwise_and(quad, quad, mask=mask)
        out_quads.append(quad_masked)

    return out_quads, (sx, sy, ex, ey)


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
) -> Tuple[np.ndarray, List[NozzleBox], List[List[bool]]]:
    """
    Read image → YOLO.detect_16 → build circular 4-quadrant ROIs →
    call isBlockedHole on each ROI → return (img, boxes, quad_statuses)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    #rotate camera
    img_rot, rot_angle = auto_rotate_by_aruco(img, prefer_id=None, upscale=1.3)
    img = img_rot

    det = NozzleDetector(weights_path, imgsz=imgsz, conf=conf, iou=iou)
    boxes = det.detect_16(img)  # may raise InvalidInputImageError

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(save_rois_dir, stem) if save_rois_dir else None
    if out_dir:
        _ensure_dir(out_dir)

    quad_statuses: List[List[bool]] = []
    for box in boxes:
        # Create masked quadrant ROIs
        quads, _ = _crop_circle_quadrants(img, box, pad_ratio=pad_ratio)

        # Save raw quadrant ROIs (if requested)
        if out_dir:
            nozzle_num = int(box.grid_index) + 1 if box.grid_index is not None else 0
            _save_rois(quads, out_dir, stem, nozzle_num)

        # Query the blocker for each quadrant
        statuses: List[bool] = []
        for roi in quads:
            try:
                # If `isBlockedHole` returns bool directly
                statuses.append(bool(isBlockedHole(roi,modelName = modelClassifyname)))

                # If your implementation returns (is_blocked, info), switch to:
                # is_blocked, _ = isBlockedHole(roi)
                # statuses.append(bool(is_blocked))
            except Exception:
                # Fail-safe: treat as not blocked if the checker throws
                statuses.append(False)

        quad_statuses.append(statuses)

    return img, boxes, quad_statuses


# ===========================
# === MAIN (PUBLIC  API)  ===
# ===========================

def detect_blocked_nozzles(
    image_path: str,
    *,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.5,
    pad_ratio: float = 0.05,
    save_rois_dir: str | None = None,
    detectionModel : str = "bestNozzleV8.pt",
    classifyModel : str = "bestBlockV8.pt"
) -> List[str]:
    """
    User-facing API: build circular ROIs and return blocked labels, e.g. ["nozzle1TopLeft", ...]
    - Automatically finds 'best.pt' near the code/image/CWD if weights_path is not provided.
    - If save_rois_dir is set, dumps the exact masked quadrant patches used by isBlockedHole.
    """
    weights_path = _auto_find_weights(image_path = image_path,filename = detectionModel)

    _, boxes, quad_statuses = _analyze_nozzles(
        image_path, weights_path,
        imgsz=imgsz, conf=conf, iou=iou, pad_ratio=pad_ratio,
        save_rois_dir=save_rois_dir,
        modelClassifyname = classifyModel
    )

    def _remap_nozzle_num(n: int) -> int:
        # 5→8, 6→7, 7→6, 8→5
        if 5 <= n <= 8:
            return 13 - n
        # 13→16, 14→15, 15→14, 16→13
        if 13 <= n <= 16:
            return 29 - n
        return n

    blocked_names: List[str] = []
    for box, statuses in zip(boxes, quad_statuses):
        nozzle_num = int(box.grid_index) + 1 if box.grid_index is not None else 0
        nozzle_num = _remap_nozzle_num(nozzle_num)  # <-- แปลงเลขที่นี่
        for q_idx, is_blocked in enumerate(statuses):
            if bool(is_blocked):
                blocked_names.append(f"nozzle{nozzle_num}{_QUAD_NAMES[q_idx]}")
    return blocked_names



# # ================================
# # === DEVELOPER / VISUAL UTILS ===
# # ================================

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
    weights_path: str | None = None,   # None = auto-find 'best.pt'
    *,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.5,
    pad_ratio: float = 0.05,
    show: bool = True,
    wait_ms: int = 0,
    save_path: str | None = None,
    save_rois_dir: str | None = None,  # saves raw ROIs, same as detect_blocked_nozzles
) -> List[str]:
    """
    Developer utility:
        - Runs the same analysis as `detect_blocked_nozzles`
        - Overlays: YOLO boxes, circle, split lines
        - Colors each quadrant (green=clear, red=blocked)
        - Optionally dumps the *raw* ROI patches (no overlays)
    """
    if weights_path is None:
        weights_path = _auto_find_weights(image_path,filename = "bestNozzleV12.pt")

    img, boxes, quad_statuses = _analyze_nozzles(
        image_path, weights_path,
        imgsz=imgsz, conf=conf, iou=iou, pad_ratio=pad_ratio,
        save_rois_dir=save_rois_dir,
        modelClassifyname = "bestBlockV12.pt"
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

        # Draw YOLO bbox and label
        cv2.rectangle(vis, (x1, y1), (x2, y2), cyan, 2)
        cv2.putText(
            vis, f"#{nozzle_num}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cyan, 2, cv2.LINE_AA
        )

        # Draw circle reference
        cx, cy, R = _circle_from_box(box)
        cv2.circle(vis, (cx, cy), R, yellow, 2)

        # Draw the 2×2 split square used for coloring
        pr = int(R * pad_ratio)
        r_pad = R + pr
        sx, sy = max(0, cx - r_pad), max(0, cy - r_pad)
        ex, ey = min(vis.shape[1], cx + r_pad), min(vis.shape[0], cy + r_pad)
        mx, my = (sx + ex) // 2, (sy + ey) // 2
        cv2.rectangle(vis, (sx, sy), (ex, ey), (160, 160, 160), 1)
        cv2.line(vis, (mx, sy), (mx, ey), white, 1)
        cv2.line(vis, (sx, my), (ex, my), white, 1)

        # Overlay colors by status
        s0 = bool(statuses[0]) if len(statuses) > 0 else False
        s1 = bool(statuses[1]) if len(statuses) > 1 else False
        s2 = bool(statuses[2]) if len(statuses) > 2 else False
        s3 = bool(statuses[3]) if len(statuses) > 3 else False

        _blend_rect(vis, sx, sy, mx, my, red if s0 else green)  # TL
        _blend_rect(vis, mx, sy, ex, my, red if s1 else green)  # TR
        _blend_rect(vis, sx, my, mx, ey, red if s2 else green)  # BL
        _blend_rect(vis, mx, my, ex, ey, red if s3 else green)  # BR

        # Collect blocked quadrant names
        for q_idx, is_blocked in enumerate((s0, s1, s2, s3)):
            if is_blocked:
                blocked_names.append(f"nozzle{nozzle_num}{_QUAD_NAMES[q_idx]}")

    # Save annotated image if requested
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        cv2.imwrite(save_path, vis)

    # Show live window if requested
    if show:
        vis_show = _resize_for_display(vis, max_w=1000, max_h=900)
        cv2.imshow("developerTest", vis_show)
        cv2.waitKey(wait_ms)

    return blocked_names


# ============================
# === CLI EXAMPLE (optional) ==
# ============================

if __name__ == "__main__":
    folder = r"C:\Project\nozzleScan\pictures"
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
                imgsz=1280, conf=0.25, iou=0.5, pad_ratio=0.05,
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
