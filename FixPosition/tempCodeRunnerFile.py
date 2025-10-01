ef _save_rois(quads: List[np.ndarray], out_dir: str, stem: str, nozzle_num: int) -> None:
    """
    Save 4 raw quadrant ROIs (TL/TR/BL/BR) using consistent filenames.
    These are the exact masked inputs passed into isBlockedHole.
    """