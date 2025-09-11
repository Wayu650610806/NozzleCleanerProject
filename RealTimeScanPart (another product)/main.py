import cv2
import numpy as np
import os, time
from ultralytics import YOLO

# ==========================
# CONFIG (ปรับตรงนี้)
# ==========================
IMAGE_PATH   = r"C:\Project\nozzleScan\pictures\fullNozzle.jpg"
MODEL_PATH   = r"C:\Project\nozzleScan\NozzleCleanerProject-1\bestNozzleV1.pt"

CANVAS_W, CANVAS_H = 750, 850
INIT_Z = 180
Z_STEP = 5
Z_MIN, Z_MAX = 170, 400
STEP_PX = 10

YOLO_CONF, YOLO_IOU, YOLO_IMGSZ = 0.25, 0.5, 512

COLOR_CAMBOX = (0, 255, 255)
COLOR_DET    = (0, 255, 0)
COLOR_HOUGH  = (0, 0, 255)
COLOR_TEXT   = (255, 255, 255)

LOG_PATH = "hough_tuning_log.csv"

# ==========================
# โหลดรูป
# ==========================
img0 = cv2.imread(IMAGE_PATH)
if img0 is None:
    raise FileNotFoundError("ไม่พบไฟล์ภาพ")

# ==========================
# โหลด YOLO
# ==========================
yolo_ok = True
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[WARN] โหลดโมเดลไม่ได้: {e}")
    yolo_ok = False
    model = None

# ==========================
# ฟังก์ชันช่วย
# ==========================
def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def run_nozzle_detection(roi):
    """รัน YOLO ใน ROI (Camera View) คืน list กล่อง [(x1,y1,x2,y2,conf,cls)] ในพิกัด ROI"""
    out = []
    if not yolo_ok or roi is None or roi.size == 0:
        return out
    try:
        results = model(roi, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
        if len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
            conf = boxes.conf.cpu().numpy()   if hasattr(boxes, "conf") else None
            cls  = boxes.cls.cpu().numpy()    if hasattr(boxes, "cls")  else None
            if xyxy is not None:
                for b, c, k in zip(xyxy, conf, cls):
                    x1,y1,x2,y2 = b.astype(int)
                    out.append((x1,y1,x2,y2,float(c),int(k)))
    except Exception:
        pass
    return out

def nothing(x):  # callback ว่างสำหรับ trackbar
    pass

# ==========================
# Trackbar (HoughCtrl)
# ==========================
cv2.namedWindow("HoughCtrl", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HoughCtrl", 420, 300)
# ค่าตั้งต้น (แนะนำเริ่มเล็กๆ แล้วค่อยไล่)
cv2.createTrackbar("dp_x10",   "HoughCtrl", 12, 50, nothing)  # 1.2
cv2.createTrackbar("param1",   "HoughCtrl", 100, 300, nothing)
cv2.createTrackbar("param2",   "HoughCtrl", 12, 100, nothing)
cv2.createTrackbar("rmin",     "HoughCtrl", 3, 150, nothing)
cv2.createTrackbar("rmax",     "HoughCtrl", 25, 200, nothing)
cv2.createTrackbar("minDist",  "HoughCtrl", 20, 200, nothing)
cv2.createTrackbar("blur_k",   "HoughCtrl", 3, 9, nothing)    # ต้องเป็นเลขคี่
cv2.createTrackbar("clahe",    "HoughCtrl", 1, 1, nothing)    # 0/1

cv2.createTrackbar("keep_in_%", "HoughCtrl", 80, 95, nothing)  # เก็บเฉพาะด้านในของจาน (%R)


# ==========================
# สถานะเริ่มต้น
# ==========================
x, y = 0, 0
z = clamp(INIT_Z, Z_MIN, Z_MAX)

cv2.namedWindow("Scene", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Scene", 900, 900)
cv2.resizeWindow("Camera View", 300, 300)

print("Controls: WASD=move  E=z+  Q=z-  C=log params  ESC=quit")

# เตรียมไฟล์ log (ใส่ header ถ้ายังไม่มี)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("timestamp,z,dp,param1,param2,rmin,rmax,minDist,blur_k,clahe,count,avg_r_px\n")

while True:
    # รีเฟรชภาพใหญ่ทุกเฟรม
    canvas = cv2.resize(img0, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)

    # กันค่าให้อยู่ในช่วง
    z = clamp(z, Z_MIN, Z_MAX)
    x = clamp(x, 0, CANVAS_W - z)
    y = clamp(y, 0, CANVAS_H - z)

    frame = canvas.copy()
    roi = frame[y:y+z, x:x+z]

    # อ่านค่าจาก Trackbar
    dp       = max(0.1, cv2.getTrackbarPos("dp_x10",  "HoughCtrl") / 10.0)
    param1   = max(1,   cv2.getTrackbarPos("param1",  "HoughCtrl"))
    param2   = max(1,   cv2.getTrackbarPos("param2",  "HoughCtrl"))
    rmin     = cv2.getTrackbarPos("rmin",    "HoughCtrl")
    rmax     = cv2.getTrackbarPos("rmax",    "HoughCtrl")
    minDist  = max(1,   cv2.getTrackbarPos("minDist", "HoughCtrl"))
    blur_k   = cv2.getTrackbarPos("blur_k",  "HoughCtrl")
    clahe_on = cv2.getTrackbarPos("clahe",   "HoughCtrl") > 0

    # ปรับ rmin<=rmax
    if rmax < rmin:
        rmax = rmin

    # ให้ blur_k เป็นเลขคี่ >=1
    if blur_k < 1:
        blur_k = 1
    if blur_k % 2 == 0:
        blur_k += 1

    # ตรวจจับ nozzle ใน ROI (Camera View)
    dets = run_nozzle_detection(roi)

    # วาดกล่องกล้อง
    cv2.rectangle(frame, (x, y), (x+z, y+z), COLOR_CAMBOX, 2)
    cv2.putText(frame, f"ROI {z}x{z}px", (x+4, y+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CAMBOX, 1, cv2.LINE_AA)

    # วาดกล่อง nozzle (พิกัดภาพใหญ่) และเลือกกล่องแรกสำหรับ Hough
    nozzle_bbox_global = None
    if len(dets) > 0:
        x1r, y1r, x2r, y2r, conf, cls_id = dets[0]  # ใช้กล่องแรก
        X1, Y1 = x + x1r, y + y1r
        X2, Y2 = x + x2r, y + y2r
        nozzle_bbox_global = (X1, Y1, X2, Y2)
        # cv2.rectangle(frame, (X1, Y1), (X2, Y2), COLOR_DET, 2)
        # cv2.putText(frame, f"{conf:.2f}", (X1, max(12, Y1-4)),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DET, 1, cv2.LINE_AA)

    # รัน HoughCircles เฉพาะในกรอบ nozzle ที่เจอ
    circles = []
    avg_r_px = 0.0

    if nozzle_bbox_global is not None:
        NX1, NY1, NX2, NY2 = nozzle_bbox_global
        NX1 = clamp(NX1, 0, frame.shape[1]-1)
        NY1 = clamp(NY1, 0, frame.shape[0]-1)
        NX2 = clamp(NX2, 0, frame.shape[1]-1)
        NY2 = clamp(NY2, 0, frame.shape[0]-1)

        noz_roi = frame[NY1:NY2, NX1:NX2]
        if noz_roi.size > 0:
            gray = cv2.cvtColor(noz_roi, cv2.COLOR_BGR2GRAY)
            # --- ตัดขอบรอยเชื่อม: ใช้ inner mask ---
            h, w = gray.shape
            cxr, cyr = w // 2, h // 2
            R = int(0.5 * min(w, h))

            keep_pct = cv2.getTrackbarPos("keep_in_%", "HoughCtrl") / 100.0  # เช่น 0.80
            inner_R = max(2, int(R * keep_pct))

            mask = np.zeros_like(gray, np.uint8)
            cv2.circle(mask, (cxr, cyr), inner_R, 255, -1)
            gray = cv2.bitwise_and(gray, mask)

            # (ออปชัน) วาดวง inner_R ให้เห็นขอบเขตที่ใช้จริงบนภาพใหญ่
            # cv2.circle(frame, ((NX1+NX2)//2, (NY1+NY2)//2), inner_R, (255,255,0), 1, cv2.LINE_AA)

            if clahe_on:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)

            if blur_k > 1:
                gray = cv2.medianBlur(gray, blur_k)

            # Hough
            try:
                cir = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                    param1=param1, param2=param2,
                    minRadius=rmin, maxRadius=rmax
                )
            except Exception:
                cir = None

            if cir is not None and len(cir) > 0:
                cir = np.round(cir[0]).astype(int)
                radii = []
                for (cxr, cyr, rr) in cir:
                    # map กลับสู่พิกัดภาพใหญ่
                    GX = NX1 + cxr
                    GY = NY1 + cyr
                    circles.append((GX, GY, rr))
                    radii.append(rr)

                if len(radii) > 0:
                    avg_r_px = float(np.mean(radii))

    # วาดวงกลมที่เจอในภาพใหญ่
    for (cxg, cyg, rg) in circles:
        cv2.circle(frame, (cxg, cyg), rg, COLOR_HOUGH, 2)
        cv2.circle(frame, (cxg, cyg), 2, COLOR_HOUGH, -1)

    # แสดงตำแหน่ง x,y,z และพารามิเตอร์ Hough (ที่มุมขวาบน)
    overlay = f"x={x} y={y} z={z} | dp={dp:.2f} p1={param1} p2={param2} r=[{rmin},{rmax}] d={minDist} blur={blur_k} clahe={int(clahe_on)} | holes={len(circles)} avg_r={avg_r_px:.1f}px"
    (tw, th), _ = cv2.getTextSize(overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, overlay, (max(10, CANVAS_W - tw - 10), 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 2, cv2.LINE_AA)

    # แสดงภาพ
    cv2.imshow("Scene", frame)
    cv2.imshow("Camera View", roi)
    cv2.resizeWindow("Scene", 900, 900)
    cv2.resizeWindow("Camera View", 300, 300)

    # คีย์บอร์ด
    key = cv2.waitKey(1) & 0xFF
    if key == 27:          # ESC
        break
    elif key == ord('w'):  y -= STEP_PX
    elif key == ord('s'):  y += STEP_PX
    elif key == ord('a'):  x -= STEP_PX
    elif key == ord('d'):  x += STEP_PX
    elif key == ord('e'):  z += Z_STEP
    elif key == ord('q'):  z -= Z_STEP
    elif key == ord('c'):  # บันทึกค่า+ผลลง CSV
        ts = int(time.time())
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{ts},{z},{dp:.3f},{param1},{param2},{rmin},{rmax},{minDist},{blur_k},{int(clahe_on)},{len(circles)},{avg_r_px:.3f}\n")
        print(f"[LOG] saved -> {LOG_PATH}")

cv2.destroyAllWindows()
