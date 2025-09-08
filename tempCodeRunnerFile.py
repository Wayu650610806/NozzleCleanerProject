
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