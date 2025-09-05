# holecheck.py (same behavior, English comments/prints)

from dataclasses import dataclass
import cv2
import numpy as np
import os, glob, math
from typing import Tuple, Dict, Union


def _refine_roi_center(roi_bgr: np.ndarray, max_shift: int = 8):
    """
    Refine the center inside a ROI by searching for a point that is 'dark + has sharp edges'.
    """
    # ---- FIX #1: guard against None/empty ----
    if roi_bgr is None:
        return (0, 0)
    if roi_bgr.size == 0:
        return (0, 0)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = w // 2, h // 2  # initial center (geometric)

    best_score = -1e9
    best_xy = (cx, cy)

    # Precompute sobel (faster)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    patch_r = 6  # half patch size → patch is approx (2*patch_r) x (2*patch_r)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            x = np.clip(cx + dx, 0, w - 1)
            y = np.clip(cy + dy, 0, h - 1)
            y1, y2 = max(0, y - patch_r), min(h, y + patch_r)
            x1, x2 = max(0, x - patch_r), min(w, x + patch_r)

            patch = gray[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            dark = 255.0 - float(np.mean(patch))  # lower brightness → darker → better
            edge = float(np.mean(mag[y1:y2, x1:x2]))
            score = dark + 0.5 * edge

            if score > best_score:
                best_score = score
                best_xy = (int(x), int(y))
    return best_xy


# ---------- 2) Data structure passed to subsequent stages ----------
@dataclass
class RefinedROI:
    roi_centered: np.ndarray                 # patch with the hole truly centered
    cx: int                                  # center x in the patch
    cy: int                                  # center y in the patch
    r_in: int                                # inner radius
    r_out: int                               # outer radius of the ring
    mask_in: np.ndarray                      # inner mask (0/255)
    mask_ring: np.ndarray                    # ring mask (0/255)
    quads: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # TL, TR, BL, BR
    focus: float                             # patch sharpness (variance of Laplacian)
    meta: Dict                               # extra info for debugging/visualization


# ---------- 3) helpers: build masks & quads ----------
def _make_masks(H: int, W: int, cx: int, cy: int, r_in: int, r_out: int):
    mask_in = np.zeros((H, W), np.uint8)
    mask_out = np.zeros((H, W), np.uint8)
    cv2.circle(mask_in, (cx, cy), r_in, 255, -1)
    cv2.circle(mask_out, (cx, cy), r_out, 255, -1)
    ring = cv2.subtract(mask_out, mask_in)
    return mask_in, ring


def _split_quadrants(img: np.ndarray, cx: int, cy: int):
    H, W = img.shape[:2]
    TL = img[0:cy, 0:cx]
    TR = img[0:cy, cx:W]
    BL = img[cy:H, 0:cx]
    BR = img[cy:H, cx:W]
    return (TL, TR, BL, BR)


# ---------- 4) DEV function that returns a RefinedROI pack ----------
def _refine_and_pack_dev(
    img: np.ndarray,
    *,
    radius_ratio: float = 0.4,   # initial ROI on full image (simulated)
    r_in_ratio: float = 0.18,
    ring_w_ratio: float = 0.08,
    max_shift: int = 8,
) -> RefinedROI:
    """
    1) Build an initial ROI (centered on full image)
    2) Find (rx, ry) within the ROI → recenter the patch
    3) Compute r_in/r_out + masks + quads + focus
    4) Return a RefinedROI + meta for visualization
    """
    Hfull, Wfull = img.shape[:2]
    cxf, cyf = Wfull // 2, Hfull // 2
    R = int(min(Wfull, Hfull) * radius_ratio)

    # Initial ROI on full image — for before/after comparison
    x1, y1 = max(0, cxf - R), max(0, cyf - R)
    x2, y2 = min(Wfull, cxf + R), min(Hfull, cyf + R)
    roi0 = img[y1:y2, x1:x2].copy()

    # refine center inside the initial ROI
    rx, ry = _refine_roi_center(roi0, max_shift=max_shift)

    # recenter: crop again so that (rx, ry) becomes the true center of the new patch
    H0, W0 = roi0.shape[:2]
    min_dim = min(H0, W0)
    r_in = max(3, int(min_dim * r_in_ratio))
    r_out = r_in + max(2, int(min_dim * ring_w_ratio))
    pad = int(0.35 * r_out)

    side = int(2 * (r_out + pad))
    nx1 = max(0, rx - side // 2)
    ny1 = max(0, ry - side // 2)
    nx2 = min(W0, rx + side // 2)
    ny2 = min(H0, ry + side // 2)
    roi_c = roi0[ny1:ny2, nx1:nx2].copy()
    if roi_c.size == 0:
        roi_c = roi0.copy()
        cx, cy = rx, ry
    else:
        cx, cy = rx - nx1, ry - ny1

    Hc, Wc = roi_c.shape[:2]
    r_in_c = max(3, min(r_in, min(Hc, Wc) // 2 - 2))
    r_out_c = max(r_in_c + 1, min(r_out, min(Hc, Wc) // 2 - 2))

    mask_in, mask_ring = _make_masks(Hc, Wc, cx, cy, r_in_c, r_out_c)
    quads = _split_quadrants(roi_c, cx, cy)

    # focus on the re-centered patch
    gray_c = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)
    focus = float(cv2.Laplacian(gray_c, cv2.CV_64F).var())

    shift_px = math.hypot(rx - W0 / 2.0, ry - H0 / 2.0)

    meta = dict(
        full_rect=(x1, y1, x2, y2),
        refined_on_full=(x1 + rx, y1 + ry),
        rx_ry=(int(rx), int(ry)),
        cx_cy=(int(cx), int(cy)),
        r_in=r_in_c, r_out=r_out_c,
        shift_px=float(shift_px),
        roi0_size=(int(W0), int(H0)),
        roi_c_size=(int(Wc), int(Hc)),
        params=dict(
            radius_ratio=radius_ratio,
            r_in_ratio=r_in_ratio,
            ring_w_ratio=ring_w_ratio,
            max_shift=max_shift,
        ),
        focus=focus,
    )

    return RefinedROI(
        roi_centered=roi_c,
        cx=int(cx), cy=int(cy),
        r_in=int(r_in_c), r_out=int(r_out_c),
        mask_in=mask_in, mask_ring=mask_ring,
        quads=quads,
        focus=focus,
        meta=meta,
    )


# ========= Homomorphic & helpers =========
def _feather_mask(mask: np.ndarray, ksize: int = 11) -> np.ndarray:
    """Feather the mask edge (0..1) for blending the donut region only."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    blur = cv2.GaussianBlur(mask, (ksize | 1, ksize | 1), 0)
    return (blur / 255.0).clip(0, 1).astype(np.float32)


def homomorphic_filter(
    gray: np.ndarray,
    area_mask: np.ndarray,
    *,
    sigma: float = 10.0,
    gain: float = 1.2
) -> np.ndarray:
    """log -> lowpass -> (logI - low)*gain -> exp (apply only on donut area and blend back)."""
    g = gray.astype(np.float32) + 1.0
    logI = np.log(g)

    low = cv2.GaussianBlur(logI, (0, 0), sigmaX=sigma, sigmaY=sigma)
    hp = (logI - low) * gain
    expI = np.exp(hp)
    expI = (expI / (expI.max() + 1e-6) * 255.0).astype(np.uint8)

    alpha = _feather_mask(area_mask, ksize=11)
    out = (alpha * expI + (1 - alpha) * gray).astype(np.uint8)
    return out


def illuminate_patch_homomorphic(
    pack: "RefinedROI", *, sigma: float = 10.0, gain: float = 1.2
) -> np.ndarray:
    """Return gray image processed with Homomorphic filtering only on (inner ∪ ring)."""
    gray = cv2.cvtColor(pack.roi_centered, cv2.COLOR_BGR2GRAY)
    area_mask = cv2.bitwise_or(pack.mask_in, pack.mask_ring)
    return homomorphic_filter(gray, area_mask, sigma=sigma, gain=gain)


# ============ METRICS & SCORE (Donutness, Edge, Focus) ============
def _safe_mean(img: np.ndarray, mask: np.ndarray) -> float:
    """Masked mean (0/255 mask). If empty, return 0."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    s = cv2.mean(img, mask)[0]
    return float(s) if not np.isnan(s) else 0.0


def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    y = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, y)))


def compute_metrics_from_pack(
    pack: "RefinedROI",
    *,
    use_homomorphic: bool = True,
    sigma: float = 10.0,
    gain: float = 1.2,
    donut_boost: float = 1.0,     # Adjust Donutness strength
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Return (metrics_dict, gray_used)
    - metrics = { 'donut': [0..1], 'edge': [0..1], 'focus': [0..1] }
    - gray_used = the gray image used for measuring (raw or homomorphic)
    """
    # 1) Prepare gray and gradient magnitude
    if use_homomorphic:
        gray = illuminate_patch_homomorphic(pack, sigma=sigma, gain=gain)
    else:
        gray = cv2.cvtColor(pack.roi_centered, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # 2) Donutness = (ring brightness - inner brightness) → normalize → boost
    mean_in = _safe_mean(gray, pack.mask_in)
    mean_ring = _safe_mean(gray, pack.mask_ring)
    raw_donut = (mean_ring - mean_in)  # bright ring and dark center → positive
    donut = _norm01(raw_donut, lo=-40.0, hi=40.0)  # heuristic 80-level span
    donut = max(0.0, min(1.0, donut * donut_boost))

    # 3) Edge strength = average gradient magnitude on the ring → normalize
    mean_edge = _safe_mean(mag, pack.mask_ring)
    edge = _norm01(mean_edge, lo=5.0, hi=60.0)  # tune to camera/lighting

    # 4) Focus = Var(Laplacian) on (inner ∪ ring)
    area_mask = cv2.bitwise_or(pack.mask_in, pack.mask_ring)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    vals = lap[area_mask.astype(bool)]
    if vals.size == 0:
        focus_var = 0.0
    else:
        focus_var = float(vals.var())
    # Normalize focus (tunable range): 0..800 → 0..1
    focus = _norm01(focus_var, lo=50.0, hi=800.0)

    return {'donut': donut, 'edge': edge, 'focus': focus}, gray


def _erode(mask: np.ndarray, r: int = 2) -> np.ndarray:
    if r <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(mask, k)


def _masked_vals(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return gray[mask.astype(bool)]


def compute_inner_dark_and_uniform(
    pack: "RefinedROI",
    gray: np.ndarray,      # gray used (raw or homomorphic)
    erode_px: int = 2
) -> dict:
    """
    Return dict: {'inner_darkpct','inner_uniform'} in [0..1]
    - inner_darkpct: fraction of dark pixels in the inner region (Otsu within mask)
    - inner_uniform: smoothness of the inner region (lower std → higher score)
    """
    mask_in = _erode(pack.mask_in, erode_px)
    vals = _masked_vals(gray, mask_in)
    if vals.size == 0:
        return dict(inner_darkpct=0.0, inner_uniform=0.0)

    # 1) Darkness %
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    darkpct = float((vals < thr).mean())  # 0..1

    # 2) Uniformity (lower std → better)
    std_in = float(vals.std())
    inner_uniform = max(0.0, min(1.0, 1.0 - (std_in / 25.0)))  # 25 is a tunable span

    return dict(inner_darkpct=darkpct, inner_uniform=inner_uniform)


# ============== RUNTIME PARAM HELPERS + SCORER ==============
def _ensure_pack(obj: Union["RefinedROI", np.ndarray]) -> "RefinedROI":
    """
    - If already a RefinedROI: return it as is
    - If a BGR image (np.ndarray): run refine+pack first (assumes it's already a hole ROI)
    """
    if isinstance(obj, RefinedROI):
        return obj
    if isinstance(obj, np.ndarray):
        # Expect the ROI of a hole → recenter/improve masks
        return _refine_and_pack_dev(
            obj,
            radius_ratio=0.45,  # tune to match incoming ROI size
            r_in_ratio=0.18,
            ring_w_ratio=0.08,
            max_shift=8,
        )
    raise TypeError("isBlockedHole expects RefinedROI or BGR image (np.ndarray).")


_DEFAULTS = {
    "sigma": 6,          # same as your defaults
    "gain": 0.8,         # gain_x10 = 8 -> 0.8
    "w1": 1.5,           # w1_donut_x10 = 15 -> 1.5
    "w2": 1.0,           # w2_edge_x10  = 10 -> 1.0
    "w3": 1.3,           # w3_dark_x10  = 13 -> 1.3
    "w4": 1.2,           # w4_uniform_x10 = 12 -> 1.2
    "donut_boost": 1.0,  # donut_boost_x10 = 10 -> 1.0
}


def _get_trackbar_or_default(win: str, name: str, default_int: int) -> int:
    try:
        v = cv2.getTrackbarPos(name, win)
        return int(v) if v >= 0 else int(default_int)
    except cv2.error:
        return int(default_int)


def _fetch_runtime_params():
    sigma = _get_trackbar_or_default("Homomorphic params", "sigma", _DEFAULTS["sigma"])
    gainx = _get_trackbar_or_default("Homomorphic params", "gain_x10", int(_DEFAULTS["gain"] * 10))
    gain = gainx / 10.0

    w1x = _get_trackbar_or_default("Score params", "w1_donut_x10", int(_DEFAULTS["w1"] * 10))
    w2x = _get_trackbar_or_default("Score params", "w2_edge_x10", int(_DEFAULTS["w2"] * 10))
    w3x = _get_trackbar_or_default("Score params", "w3_dark_x10", int(_DEFAULTS["w3"] * 10))
    w4x = _get_trackbar_or_default("Score params", "w4_uniform_x10", int(_DEFAULTS["w4"] * 10))
    dbx = _get_trackbar_or_default("Score params", "donut_boost_x10", int(_DEFAULTS["donut_boost"] * 10))

    return {
        "sigma": int(max(1, sigma)),
        "gain": float(gain),
        "w1": w1x / 10.0,
        "w2": w2x / 10.0,
        "w3": w3x / 10.0,
        "w4": w4x / 10.0,
        "donut_boost": dbx / 10.0,
    }


def compute_hole_score(pack: "RefinedROI") -> Tuple[int, dict]:
    """
    Compute overall hole score: returns (S_x100, detail).
    Pipeline:
      - Use homomorphic (log -> lowpass -> gain -> exp) only on inner/ring area
      - Metrics: Donutness, Edge, Focus (focus measured but not used in final S)
      - Inner stats: inner darkness (%) + inner uniformity
      - Combine with weights (read from trackbars if available, else defaults)
    """
    # 1) Read runtime parameters (trackbars if available; fallback to defaults)
    p = _fetch_runtime_params()

    # 2) Prepare gray via homomorphic
    gray_homo = illuminate_patch_homomorphic(pack, sigma=p["sigma"], gain=p["gain"])

    # 3) Metrics (donut/edge/focus)
    metrics, _ = compute_metrics_from_pack(
        pack,
        use_homomorphic=True,
        sigma=p["sigma"],
        gain=p["gain"],
        donut_boost=p["donut_boost"],
    )
    donut = metrics["donut"]
    edge = metrics["edge"]
    # focus = metrics["focus"]  # not used in S yet

    # 4) Inner stats on inner mask (with erosion to avoid rim leakage)
    inner = compute_inner_dark_and_uniform(pack, gray_homo, erode_px=2)
    idark = inner["inner_darkpct"]
    iuni = inner["inner_uniform"]

    # 5) Combine score
    S = p["w1"] * donut + p["w2"] * edge + p["w3"] * idark + p["w4"] * iuni
    S_x100 = int(round(S * 100))

    detail = {
        "params_used": p,
        "components": {
            "donut": float(donut),
            "edge": float(edge),
            "inner_darkpct": float(idark),
            "inner_uniform": float(iuni),
        },
        "S": float(S),
        "S_x100": int(S_x100),
    }
    return S_x100, detail


def isBlockedHole(
    obj: Union["RefinedROI", np.ndarray],
    threshold_Sx100: int = 300,
    return_detail: bool = False
):
    """
    Decide 'blocked?' from S_x100.

      - Input: BGR ROI (np.ndarray) or RefinedROI
      - Steps: refine ROI → homomorphic → scoring
      - Rule: S_x100 < threshold → BLOCKED (True) | S_x100 >= threshold → CLEAN (False)

    Parameters
    ----------
    obj : Union[RefinedROI, np.ndarray]
        BGR ROI or a pre-refined RefinedROI.
    threshold_Sx100 : int
        Decision threshold (tune per camera/lighting/deployment).
    return_detail : bool
        If True, return (bool, details) instead of bool.

    Returns
    -------
    bool | Tuple[bool, dict]
        Decision "blocked?" or (blocked?, scoring details).
    """
    # 1) Ensure we have a RefinedROI
    pack = _ensure_pack(obj)

    # 2) Compute score
    S_x100, detail = compute_hole_score(pack)

    # 3) Decide
    is_blocked = bool(S_x100 < int(threshold_Sx100))

    if return_detail:
        detail_out = dict(detail)
        detail_out["decision"] = "BLOCKED" if is_blocked else "CLEAN"
        detail_out["threshold_Sx100"] = int(threshold_Sx100)
        detail_out["margin"] = int(S_x100 - int(threshold_Sx100))  # + = above threshold (safer)
        return is_blocked, detail_out

    return is_blocked


# ================= DEV ONLY: ROI Refinement & Visualization =================
# The entire block below is dev-only. It stays commented out; runtime behavior is unchanged.

# TRACKBARS_CREATED = False
# _SCORE_TRACKBARS_READY = False
#
# def _ensure_score_trackbars():
#     global _SCORE_TRACKBARS_READY
#     if _SCORE_TRACKBARS_READY:
#         return
#     win = "Score params"
#     cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
#     # Weights (×10): 0..30 → 0.0..3.0
#     cv2.createTrackbar("w1_donut_x10",   win, 15, 30, lambda v: None)
#     cv2.createTrackbar("w2_edge_x10",    win, 10, 30, lambda v: None)
#     cv2.createTrackbar("w3_dark_x10",    win, 13, 30, lambda v: None)
#     cv2.createTrackbar("w4_uniform_x10", win, 12, 30, lambda v: None)
#     # Donutness pre-boost (×10): 1.0..5.0
#     cv2.createTrackbar("donut_boost_x10", win, 10, 50, lambda v: None)
#     # Read-only score indicator (S ×100)
#     cv2.createTrackbar("S_x100", win, 0, 600, lambda v: None)  # supports S up to 6.00
#     _SCORE_TRACKBARS_READY = True
#
# def _read_score_params():
#     win = "Score params"
#     w1 = cv2.getTrackbarPos("w1_donut_x10",   win) / 10.0
#     w2 = cv2.getTrackbarPos("w2_edge_x10",    win) / 10.0
#     w3 = cv2.getTrackbarPos("w3_dark_x10",    win) / 10.0
#     w4 = cv2.getTrackbarPos("w4_uniform_x10", win) / 10.0
#     db = cv2.getTrackbarPos("donut_boost_x10", win) / 10.0
#     return w1, w2, w3, w4, db
#
# def _set_S_indicator(S: float):
#     win = "Score params"
#     S_clamped = max(0.0, min(6.0, S))
#     cv2.setTrackbarPos("S_x100", win, int(round(S_clamped * 100)))
#
# def _ensure_homo_trackbars():
#     global TRACKBARS_CREATED
#     if TRACKBARS_CREATED:
#         return
#     cv2.namedWindow("Homomorphic params", cv2.WINDOW_AUTOSIZE)
#     # sigma: 1..50 | gain: 0.5..3.0 (×10 as int)
#     cv2.createTrackbar("sigma",    "Homomorphic params", 6, 50, lambda v: None)
#     cv2.createTrackbar("gain_x10", "Homomorphic params", 8, 30, lambda v: None)
#     TRACKBARS_CREATED = True
#
# def _read_homo_params():
#     sigma = max(1, cv2.getTrackbarPos("sigma", "Homomorphic params"))
#     gain  = cv2.getTrackbarPos("gain_x10", "Homomorphic params") / 10.0
#     return float(sigma), float(gain)
#
# def dev_browse_roi_folder(folder_path: str, *, radius_ratio: float = 0.4,
#                           r_in_ratio: float = 0.18, ring_w_ratio: float = 0.08, max_shift: int = 8):
#     """
#     Read all images in a folder → refine & pack → visualize:
#       - Full image + initial ROI box + refined point over full image
#       - Initial ROI (before recenter)
#       - Centered ROI (final) with r_in/r_out and crosshair
#     Hotkeys: [A]=prev, [D]=next, [Q]=quit
#     """
#     exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
#     files = [f for f in glob.glob(os.path.join(folder_path, "*")) if f.lower().endswith(exts)]
#     files.sort()
#     if not files:
#         print("No image files found in folder:", folder_path)
#         return
#
#     idx = 0
#     quit_all = False
#
#     while True:
#         path = files[idx]
#         img = cv2.imread(path)
#         if img is None:
#             print("Failed to read image:", path)
#             idx = (idx + 1) % len(files)
#             continue
#
#         pack = _refine_and_pack_dev(
#             img,
#             radius_ratio=radius_ratio,
#             r_in_ratio=r_in_ratio,
#             ring_w_ratio=ring_w_ratio,
#             max_shift=max_shift,
#         )
#
#         # Prepare static visuals per image
#         vis_full = img.copy()
#         (x1, y1, x2, y2) = pack.meta["full_rect"]
#         (rx_full, ry_full) = pack.meta["refined_on_full"]
#         cv2.rectangle(vis_full, (x1, y1), (x2, y2), (0, 255, 255), 1)
#         cv2.circle(vis_full, (rx_full, ry_full), 3, (0, 0, 255), -1)
#
#         roi_before = img[y1:y2, x1:x2].copy()
#         roi_after  = pack.roi_centered.copy()
#         cx, cy = pack.cx, pack.cy
#         cv2.circle(roi_after, (cx, cy), pack.r_in,  (0, 255, 255), 1)
#         cv2.circle(roi_after, (cx, cy), pack.r_out, (0, 255, 0),   1)
#         cv2.rectangle(roi_after, (0, 0), (roi_after.shape[1]-1, roi_after.shape[0]-1), (200, 200, 200), 1)
#
#         # Ensure trackbars and read initial values
#         _ensure_homo_trackbars()
#         _ensure_score_trackbars()
#         prev_sigma = prev_gain = None
#         prev_w1 = prev_w2 = prev_w3 = prev_w4 = prev_db = None
#         gray_raw = cv2.cvtColor(pack.roi_centered, cv2.COLOR_BGR2GRAY)
#
#         # Realtime loop
#         while True:
#             sigma, gain = _read_homo_params()
#             w1, w2, w3, w4, db = _read_score_params()
#
#             if (sigma != prev_sigma) or (gain != prev_gain) or \
#                (w1 != prev_w1) or (w2 != prev_w2) or (w3 != prev_w3) or (w4 != prev_w4) or (db != prev_db):
#                 prev_sigma, prev_gain = sigma, gain
#                 prev_w1, prev_w2, prev_w3, prev_w4, prev_db = w1, w2, w3, w4, db
#
#                 # 1) Homomorphic
#                 gray_homo = illuminate_patch_homomorphic(pack, sigma=sigma, gain=gain)
#                 vis_homo = cv2.cvtColor(gray_homo, cv2.COLOR_GRAY2BGR)
#                 cv2.rectangle(vis_homo, (0, 0), (vis_homo.shape[1]-1, vis_homo.shape[0]-1), (200, 200, 200), 1)
#
#                 # 2) Metrics + Inner
#                 metrics, _ = compute_metrics_from_pack(pack, use_homomorphic=True, sigma=sigma, gain=gain, donut_boost=db)
#                 donut, edge = metrics['donut'], metrics['edge']
#
#                 inner = compute_inner_dark_and_uniform(pack, gray_homo, erode_px=2)
#                 idark  = inner['inner_darkpct']
#                 iuni   = inner['inner_uniform']
#
#                 # 3) Combined score
#                 S = w1 * donut + w2 * edge + w3 * idark + w4 * iuni
#                 _set_S_indicator(S)
#
#                 # 4) Print to terminal instead of overlaying on image
#                 blocked, info = isBlockedHole(pack, threshold_Sx100=330, return_detail=True)
#                 decision = info["decision"]
#                 S_val = info["S"]
#                 print(f"S={S_val:.2f} | {decision}")
#
#                 # Show windows
#                 cv2.imshow("Image (full)", vis_full);               cv2.moveWindow("Image (full)", 250, 250)
#                 cv2.imshow("ROI before recenter", roi_before);      cv2.moveWindow("ROI before recenter", 350 + vis_full.shape[1], 250)
#                 cv2.imshow("ROI centered (final pack)", roi_after); cv2.moveWindow("ROI centered (final pack)", 600 + vis_full.shape[1], 250)
#                 cv2.imshow("gray_raw (ROI)", gray_raw);             cv2.moveWindow("gray_raw (ROI)", 250, 500)
#                 cv2.imshow("gray_homomorphic (ROI)", vis_homo);     cv2.moveWindow("gray_homomorphic (ROI)", 500, 500)
#
#             k = cv2.waitKey(20) & 0xFF  # ~50 FPS refresh
#             if k in (ord('q'), ord('Q'), 27):
#                 quit_all = True
#                 break
#             elif k in (ord('d'), ord('D')):
#                 idx = (idx + 1) % len(files)
#                 break
#             elif k in (ord('a'), ord('A')):
#                 idx = (idx - 1) % len(files)
#                 break
#
#         if quit_all:
#             break
#
#     cv2.destroyAllWindows()
#
# # ================== MAIN TEST (dev-only) ==================
# if __name__ == "__main__":
#     folder = r"C:\Project\nozzleScan\NozzleCleanerProject\roi\1"
#     # Tweak values here (does not affect production functions)
#     dev_browse_roi_folder(
#         folder_path=folder,
#         radius_ratio=0.4,
#         r_in_ratio=0.18,
#         ring_w_ratio=0.08,
#         max_shift=13,
#     )
