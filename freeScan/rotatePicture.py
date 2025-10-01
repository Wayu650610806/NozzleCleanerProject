import cv2
import math
import numpy as np



def auto_rotate_by_aruco(image, prefer_id=None, upscale=1.3):
    aruco = cv2.aruco
    dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(dict_aruco, params)

    orig_h, orig_w = image.shape[:2]
    if upscale and upscale != 1.0:
        image = cv2.resize(image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(corners) == 0:
        # ไม่เจอ marker → คืนภาพเดิม (ขนาดเดิม)
        return cv2.resize(image, (orig_w, orig_h)), 0.0

    # เลือก marker
    if prefer_id is not None and prefer_id in ids.flatten():
        k = list(ids.flatten()).index(prefer_id)
    else:
        perims = [cv2.arcLength(c, True) for c in [x[0] for x in corners]]
        k = int(np.argmax(perims))

    c = corners[k][0]
    dx, dy = c[1][0] - c[0][0], c[1][1] - c[0][1]
    angle = math.degrees(math.atan2(dy, dx))
    rot_angle = angle+180

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # ย่อกลับขนาดเดิมก่อน return
    rotated = cv2.resize(rotated, (orig_w, orig_h))
    return rotated, rot_angle

# =====================
# MAIN (แสดงก่อน/หลัง)
# =====================
# if __name__ == "__main__":
#     img = cv2.imread(r"C:\Project\nozzleScan\pictures\direction\12.jpg")
#     if img is None:
#         print("ไม่พบไฟล์ภาพ")
#         raise SystemExit

#     rotated, angle_used = auto_rotate_by_aruco(img, prefer_id=None, upscale=1.3)
#     print(f"[Rotate] Applied rotation: {angle_used:.2f} deg")

#     # แสดง Before/After แบบย่อ
#     h1, w1 = img.shape[:2]
#     h2, w2 = rotated.shape[:2]
#     H = min(900, max(h1, h2))  # กำหนดความสูงเป้าหมาย
#     left  = cv2.resize(img,      (int(w1 * H / h1), H))
#     right = cv2.resize(rotated,  (int(w2 * H / h2), H))
#     disp  = np.hstack([left, right])

#     display_scale = 0.35
#     disp = cv2.resize(disp, (int(disp.shape[1]*display_scale), int(disp.shape[0]*display_scale)))

#     cv2.imshow("Before (Left) | After (Right)", disp)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
