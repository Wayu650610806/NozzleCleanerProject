# nozzle_specs_mm.py
# ------------------
# เก็บสเปก nozzle ในหน่วย "มิลลิเมตร" และยูทิลิตี้สำหรับคำนวณตำแหน่งรู
# ใช้กับระบบตรวจจับ/จำลอง โดยไม่ผูกกับพิกเซล (px) เพื่อความชัดเจนด้านหน่วย
#
# ตัวอย่างการใช้งาน:
#   from nozzle_specs_mm import NOZZLE_4, NOZZLE_2, SPECS
#
#   # รับมุมรูทั้ง 4 ของจาน 4 รู เมื่อมีมุมหมุนเริ่มต้น θ = 15°
#   angles = NOZZLE_4.hole_angles_deg(theta0_deg=15)
#
#   # รับพิกัด (x,y) ของรูทั้ง 4 โดยสมมติศูนย์กลางจานอยู่ที่ (100, 50) mm
#   xy = NOZZLE_4.hole_positions_xy_mm(center_xy_mm=(100, 50), theta0_deg=15)
#
# หมายเหตุ: หากต้องการแปลง mm <-> px ให้คาลิเบรตจากภาพ (หาเส้นผ่านศูนย์กลางจานในพิกเซล)
#           แล้วใช้ตัวช่วยด้านล่าง (mm_per_px_from_radius / px_per_mm_from_radius)

from dataclasses import dataclass
import math
from typing import List, Tuple, Dict

@dataclass(frozen=True)
class NozzleSpec:
    name: str
    hole_count: int                  # จำนวนรู
    hole_diameter_mm: float          # เส้นผ่านศูนย์กลาง "รู" (mm)
    hole_center_radius_mm: float     # ระยะจากศูนย์กลางจาน -> ศูนย์กลางรู (mm)
    angular_spacing_deg: float       # มุมห่างระหว่างรู (deg)
    nozzle_diameter_mm: float        # เส้นผ่านศูนย์กลาง "จาน" (mm)

    @property
    def nozzle_radius_mm(self) -> float:
        return self.nozzle_diameter_mm / 2.0

    @property
    def hole_radius_mm(self) -> float:
        # alias ให้ชื่ออ่านง่าย (เท่ากับ hole_center_radius_mm)
        return self.hole_center_radius_mm

    def hole_angles_deg(self, theta0_deg: float = 0.0) -> List[float]:
        """
        คืนลิสต์มุมของรูทุกอัน (หน่วย deg) โดยออฟเซ็ตเริ่มต้นด้วย theta0_deg
        สำหรับ 4 รู -> [θ, θ+90, θ+180, θ+270]
        สำหรับ 2 รู -> [θ, θ+180]
        """
        return [ (theta0_deg + i * self.angular_spacing_deg) % 360.0
                 for i in range(self.hole_count) ]

    def hole_positions_xy_mm(
        self,
        center_xy_mm: Tuple[float, float] = (0.0, 0.0),
        theta0_deg: float = 0.0
    ) -> List[Tuple[float, float]]:
        """
        คืนตำแหน่ง (x_mm, y_mm) ของศูนย์กลางรูแต่ละรู เมื่อกำหนด:
          - center_xy_mm : พิกัดศูนย์กลาง "จาน" (mm)
          - theta0_deg   : มุมหมุนเริ่มต้นของแพทเทิร์นรู (deg)
        หมายเหตุ: ใช้ระบบแกน x ขวา+, y ขึ้น+ ตามคณิตมาตรฐาน
        """
        cx, cy = center_xy_mm
        r = self.hole_center_radius_mm
        xy: List[Tuple[float, float]] = []
        for a in self.hole_angles_deg(theta0_deg):
            rad = math.radians(a)
            x = cx + r * math.cos(rad)
            y = cy + r * math.sin(rad)
            xy.append((x, y))
        return xy


# ----------------------------
# สเปกที่ผู้ใช้กำหนด (ทั้งหมดเป็น mm)
# ----------------------------

NOZZLE_4 = NozzleSpec(
    name="nozzle_4_hole",
    hole_count=4,
    hole_diameter_mm=4.0,
    hole_center_radius_mm=20.0,      # ระยะศูนย์จาน -> ศูนย์รู
    angular_spacing_deg=90.0,        # ห่างกัน 90°
    nozzle_diameter_mm=73.0
)

NOZZLE_2 = NozzleSpec(
    name="nozzle_2_hole",
    hole_count=2,
    hole_diameter_mm=4.0,
    hole_center_radius_mm=7.5,       # ระยะศูนย์จาน -> ศูนย์รู
    angular_spacing_deg=180.0,       # ห่างกัน 180°
    nozzle_diameter_mm=41.301
)

# ดิกชันนารีให้เรียกใช้ง่าย
SPECS: Dict[str, NozzleSpec] = {
    "4hole": NOZZLE_4,
    "2hole": NOZZLE_2,
}

__all__ = [
    "NozzleSpec",
    "NOZZLE_4",
    "NOZZLE_2",
    "SPECS",
]


# -------------------------------------------------------------
# (ทางเลือก) ตัวช่วยแปลงหน่วย mm ↔ px เมื่อมีการคาลิเบรตจากภาพจริง
# -------------------------------------------------------------
def mm_per_px_from_radius(nozzle: NozzleSpec, radius_px: float) -> float:
    """
    คำนวณ mm/px จาก "รัศมีจานในภาพ (px)" และ "เส้นผ่านศูนย์กลางจริง (mm)"
    mm_per_px = (เส้นผ่านศูนย์กลางจริง mm) / (เส้นผ่านศูนย์กลางภาพ px)
               = nozzle_diameter_mm / (2 * radius_px)
    """
    return nozzle.nozzle_diameter_mm / (2.0 * max(1e-9, radius_px))

def px_per_mm_from_radius(nozzle: NozzleSpec, radius_px: float) -> float:
    """
    ผกผันของฟังก์ชันข้างบน: px/mm
    px_per_mm = (เส้นผ่านศูนย์กลางภาพ px) / (เส้นผ่านศูนย์กลางจริง mm)
              = (2 * radius_px) / nozzle_diameter_mm
    """
    return (2.0 * max(0.0, radius_px)) / max(1e-9, nozzle.nozzle_diameter_mm)
