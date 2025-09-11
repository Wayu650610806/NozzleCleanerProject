# scale_model.py
# ----------------
# สร้าง/ใช้สเกล px/mm จากการคาลิเบรตที่ความสูงอ้างอิง (inverse-height model)

from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class InverseHeightScale:
    """ px/mm ~ k / H  (H = ความสูงกล้องหน่วย mm) """
    k_pxmm_mul_mm: float   # k = (px/mm)_ref * H_ref

    def px_per_mm(self, height_mm: float) -> float:
        height_mm = max(1e-6, float(height_mm))
        return self.k_pxmm_mul_mm / height_mm

    def mm_per_px(self, height_mm: float) -> float:
        return 1.0 / self.px_per_mm(height_mm)

    def px_of_mm(self, length_mm: float, height_mm: float) -> float:
        return length_mm * self.px_per_mm(height_mm)

    def radius_px_range(
        self,
        diameter_mm: float,
        height_mm: float,
        tol_ratio: Tuple[float, float] = (0.7, 1.3),
    ) -> Tuple[int, int]:
        """
        ให้ช่วงรัศมีพิกเซลสำหรับ Hough ตามขนาดจริงของรู (mm) และ tolerance (เช่น ±30%)
        """
        r_px = 0.5 * self.px_of_mm(diameter_mm, height_mm)
        r_min = max(1, int(r_px * tol_ratio[0]))
        r_max = max(r_min + 1, int(r_px * tol_ratio[1]))
        return r_min, r_max


def build_scale_from_reference(
    ref_height_mm: float,
    known_size_mm: float,
    observed_size_px: float,
) -> InverseHeightScale:
    """
    คาลิเบรตจากข้อมูลอ้างอิงครั้งเดียว:
    - ref_height_mm: ความสูงตอนคาลิเบรต (mm)
    - known_size_mm: ขนาดจริงที่รู้ (mm) เช่น เส้นผ่านศูนย์กลาง nozzle = 73 mm
    - observed_size_px: ขนาดพิกเซลที่วัดได้จากภาพ ณ ref_height_mm
    คืนค่า InverseHeightScale สำหรับใช้คำนวณ px/mm ที่ความสูงอื่น
    """
    px_per_mm_ref = observed_size_px / max(1e-9, known_size_mm)
    k = px_per_mm_ref * ref_height_mm
    return InverseHeightScale(k_pxmm_mul_mm=k)
