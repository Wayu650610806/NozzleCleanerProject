# holecheckAI.py
"""
Simple ROI classifier for "blocked hole" detection (YOLOv8-Classification)
---------------------------------------------------------------------------
- Lazily loads a YOLOv8 classification model on first use.
- Preprocesses ROI to 64×64 BGR.
- Returns a boolean: True if predicted "blocked" probability >= threshold.

Public API:
    isBlockedHole(roi_bgr, weights_path=..., p_block_thresh=0.6) -> bool

Developer utilities:
    predict_block_prob(roi_bgr, weights_path=...) -> float
    developer_view_folder(folder, weights_path=..., p_block_thresh=0.6)
        - Open images from a folder
        - Keys: a (prev), d (next), q/ESC (quit)
        - Overlays status and p_block on the image
"""

from __future__ import annotations
import os
from typing import Optional, Iterable, List
import cv2
import numpy as np
from pathlib import Path


# ===== Configuration =====
_AI_IMGSZ: int = 64
_AI_BLOCKED_CLASS_INDEX: int = 1
_AI_MODEL: Optional["YOLO"] = None
_AI_BACKEND: Optional[str] = None

try:
    from ultralytics import YOLO
    _ULTRA_OK = True
except ImportError:
    _ULTRA_OK = False


# =================
# === Preprocess ===
# =================
def _preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    if roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
        roi_bgr = np.zeros((_AI_IMGSZ, _AI_IMGSZ, 3), np.uint8)
    if roi_bgr.ndim == 2 or (roi_bgr.ndim == 3 and roi_bgr.shape[2] == 1):
        roi_bgr = cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)
    roi_resized = cv2.resize(roi_bgr, (_AI_IMGSZ, _AI_IMGSZ), interpolation=cv2.INTER_AREA)
    return roi_resized


# ==========================
# === Model load (once)  ===
# ==========================
def _load_model(weights_path: str) -> None:
    global _AI_MODEL, _AI_BACKEND

    ext = os.path.splitext(weights_path)[1].lower()
    if ext not in (".pt", ".pth"):
        raise ValueError(f"Unsupported model format: {ext}")

    if not _ULTRA_OK:
        raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics")

    _AI_MODEL = YOLO(weights_path)
    _AI_BACKEND = "ultralytics"


# ============================
# === Public classification ===
# ============================
def isBlockedHole(
    roi_bgr: np.ndarray,
    weights_path: str = None,
    p_block_thresh: float = 0.6,
    modelName: str = "bestBlockV8.pt"
) -> bool:
    # หาตำแหน่งโฟลเดอร์ไฟล์ปัจจุบัน
    CURRENT_DIR = Path(__file__).resolve().parent

    # bestBlockV1.pt อยู่โฟลเดอร์เดียวกับไฟล์นี้
    weights_path = CURRENT_DIR / modelName

    # ถ้าจะให้ชัวร์ก่อนโหลด
    if not weights_path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {weights_path}")
    
    p_block = predict_block_prob(roi_bgr, weights_path=weights_path)
    return p_block >= p_block_thresh


def predict_block_prob(
    roi_bgr: np.ndarray,
    weights_path: str = None
) -> float:
    global _AI_MODEL
    if _AI_MODEL is None:
        _load_model(weights_path)

    roi_in = _preprocess_roi(roi_bgr)
    preds = _AI_MODEL.predict(source=roi_in, imgsz=_AI_IMGSZ, verbose=False)

    p = preds[0].probs.data.cpu().numpy().astype(np.float32)
    if p.min() < 0.0 or p.max() > 1.0 or not np.isclose(p.sum(), 1.0, atol=1e-3):
        e = np.exp(p - p.max())
        p = e / (e.sum() + 1e-6)

    idx = min(_AI_BLOCKED_CLASS_INDEX, p.shape[-1] - 1)
    return float(p[idx])


# =========================
# === Developer viewer  ===
# =========================
# def _list_images(folder: str, exts: Iterable[str]) -> List[str]:
#     exts = tuple([e.lower() for e in exts])
#     return sorted([os.path.join(folder, f) for f in os.listdir(folder)
#                    if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(exts)])

# # --- add this helper near the top (with other helpers) ---
# def _status_canvas(is_blocked: bool, p_block: float,
#                    size: tuple[int,int]=(220, 70)) -> np.ndarray:
#     """Create a small status image to show in a separate window."""
#     w, h = size
#     canvas = np.zeros((h, w, 3), dtype=np.uint8)          # black background
#     status = "BLOCKED" if is_blocked else "CLEAR"
#     color  = (0, 0, 255) if is_blocked else (60, 200, 60) # red / green

#     # title
#     cv2.putText(canvas, status, (10, 26),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
#     # prob
#     cv2.putText(canvas, f"p={p_block:.3f}", (10, 54),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
#     return canvas

# def developer_view_folder(
#     folder: str,
#     *,
#     weights_path: str = r"C:\Project\nozzleScan\NozzleCleanerProject\bestBlockV1.pt",
#     p_block_thresh: float = 0.6,
#     exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
#     window_name: str = "holecheckAI: developer_view_folder",
#     wait_ms_first: int = 1
# ) -> None:
#     if not os.path.isdir(folder):
#         raise FileNotFoundError(f"Folder not found: {folder}")

#     files = _list_images(folder, exts)
#     if not files:
#         print("No images found in folder.")
#         return

#     idx = 0
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#     try:
#         while True:
#             path = files[idx]
#             img = cv2.imread(path)
#             if img is None:
#                 idx = (idx + 1) % len(files)
#                 continue

#             p_block = predict_block_prob(img, weights_path=weights_path)
#             is_blocked = p_block >= p_block_thresh


#             # --- show main image (original size, no overlay) ---
#             vis = img  # no resize
#             cv2.imshow(window_name, vis)

#             # --- show status in a separate small window ---
#             status_img = _status_canvas(is_blocked, p_block, size=(220, 70))
#             status_win = f"{window_name} [status]"
#             cv2.imshow(status_win, status_img)

#             # (optional) place the status window near the main window
#             try:
#                 cv2.moveWindow(status_win, 50, 50)
#             except Exception:
#                 pass




#             k = cv2.waitKey(wait_ms_first if wait_ms_first is not None else 0) & 0xFF
#             wait_ms_first = None
#             if k in (ord('q'), ord('Q'), 27):
#                 break
#             elif k in (ord('d'), ord('D')):
#                 idx = (idx + 1) % len(files)
#             elif k in (ord('a'), ord('A')):
#                 idx = (idx - 1) % len(files)

#     finally:
#         cv2.destroyWindow(window_name)


# # =================
# # === Main run  ===
# # =================
# if __name__ == "__main__":
#     folder = r"C:\Project\nozzleScan\pictures\roi\1"
#     weights = r"C:\Project\nozzleScan\NozzleCleanerProject\bestBlockV1.pt"
#     developer_view_folder(folder, weights_path=weights, p_block_thresh=0.6)
