import cv2
import numpy as np
import os

# ===== CONFIG =====
_AI_IMGSZ = 48
_AI_BLOCKED_CLASS_INDEX = 1
_AI_MODEL = None
_AI_BACKEND = None

# ----- Ultralytics (YOLOv8 classification) -----
try:
    from ultralytics import YOLO
    _ULTRA_OK = True
except ImportError:
    _ULTRA_OK = False


# ===== Preprocess =====
def _preprocess_roi(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        roi_bgr = np.zeros((64,64,3), np.uint8)
    roi_resized = cv2.resize(roi_bgr, (_AI_IMGSZ, _AI_IMGSZ), interpolation=cv2.INTER_AREA)
    return roi_resized


# ===== Load model once =====
def _load_model(weights_path: str):
    global _AI_MODEL, _AI_BACKEND
    ext = os.path.splitext(weights_path)[1].lower()

    if ext in (".pt", ".pth"):
        if not _ULTRA_OK:
            raise RuntimeError("Ultralytics not installed: pip install ultralytics")
        _AI_MODEL = YOLO(weights_path)
        _AI_BACKEND = "ultralytics"
        print(f"[AI] YOLOv8 classification model loaded: {weights_path}")
    else:
        raise ValueError(f"Unsupported model format: {ext}")


# ===== Public function: isBlockedHole =====
# ใช้ raw string ป้องกัน \n \t ปัญหา path
def isBlockedHole(
    roi_bgr,
    weights_path=r"C:\Project\nozzleScan\NozzleCleanerProject\bestBlock.pt",
    p_block_thresh=0.6
):
    global _AI_MODEL
    if _AI_MODEL is None:
        _load_model(weights_path)

    roi48 = _preprocess_roi(roi_bgr)
    preds = _AI_MODEL.predict(source=roi48, imgsz=_AI_IMGSZ, verbose=False)

    p = preds[0].probs.data.cpu().numpy().astype(np.float32)
    if p.min() < 0 or p.max() > 1.0:  # softmax ถ้าจำเป็น
        e = np.exp(p - p.max())
        p = e / (e.sum() + 1e-6)

    p_block = float(p[min(_AI_BLOCKED_CLASS_INDEX, p.shape[-1] - 1)])
    return p_block >= p_block_thresh