# holecheck.py
import cv2
import numpy as np

def isBlockedHole(roi_bgr) -> bool:
    """คืน True ถ้ารูตัน, False ถ้ารูไม่ตัน (ตัวอย่าง threshold ง่ายๆ)"""
    if roi_bgr is None or roi_bgr.size == 0:
        return False

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # ใช้ adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # หา contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ถ้ามี blob ดำใหญ่เกิน 40% ของพื้นที่ ถือว่าตัน
    area_total = roi_bgr.shape[0] * roi_bgr.shape[1]
    area_black = sum(cv2.contourArea(c) for c in contours)
    black_ratio = area_black / (area_total + 1e-5)

    return black_ratio > 0.4
