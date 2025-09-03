# nozzle_types.py
"""
Nozzle Types — Data Structures for Nozzle Detection & Blockage
--------------------------------------------------------------
Purpose:
    Defines simple data containers for nozzle bounding boxes and blocked
    quadrants, shared across detector/orchestrator modules.

Types:
    - NozzleBox:
        Stores bbox coords (x1,y1,x2,y2), YOLO confidence, optional circle
        geometry (cx,cy,R), and grid index [0..15] for 4×4 grid layout.

    - BlockedLabel:
        Holds a nozzle ID [1..16] and a list of blocked quadrants
        (e.g., ["TopLeft","BottomRight"]).

Usage:
    from nozzle_types import NozzleBox, BlockedLabel

    b = NozzleBox(10,20,110,120,conf=0.85,cx=60,cy=70,R=30,grid_index=5)
    blocked = BlockedLabel(nozzle_id=1, blocked_quadrants=["TopLeft"])
"""

from dataclasses import dataclass
from typing import List, Optional

Quadrant = str  # "top-left" | "top-right" | "bottom-left" | "bottom-right"

@dataclass
class NozzleBox:
    x1: int; y1: int; x2: int; y2: int
    conf: float
    cx: Optional[int] = None
    cy: Optional[int] = None
    R:  Optional[int] = None
    grid_index: Optional[int] = None  # 0..15

@dataclass
class BlockedLabel:
    nozzle_id: int                    # 1..16
    blocked_quadrants: List[Quadrant] # รายชื่อควอดแรนท์ที่ตัน
