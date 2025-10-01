# วางมาร์คและป้ายชื่อ TL/TR/BL/BR ตามสถานะ
        # หมายเหตุ: centers มาจาก _quadrant_centers(...) ที่ใช้ angle_deg แล้ว
        for (name, is_blocked, (px, py)) in zip(_QUAD_NAMES, [bool(x) for x in statuses], centers):
            color = red if is_blocked else green
            cv2.circle(vis, (int(px), int(py)), 6, color, -1, lineType=cv2.LINE_AA)
            _put_label_with_bg(
                vis, f"{name}", (int(px) + 8, int(py) - 8),
                font_scale=0.5, thickness=1, text_color=(255,255,255),
                bg_color=(0,0,180) if is_blocked else (0,100,0)
            )
            if is_blocked:
                blocked_points.append((int(px), int(py)))