from pathlib import Path
from typing import List
import cv2
import numpy as np


class HardNegativeMiner:
    """
    When YOLO finds borderline detections (e.g. 0.25-0.44 confidence) on an S3 .tif tile,
    this saves the 3-band RGB image as a high-quality .jpg alongside a YOLO .txt pre-label
    so you can drag-and-drop or upload the folder straight into Roboflow.
    """

    def __init__(self, out_dir: str = "./runs/hard_negatives", max_saved: int = 500):
        self.root = Path(out_dir).resolve()
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.max_saved = max_saved
        self.saved_count = len(list(self.images_dir.glob("*.jpg")))

    def save_candidate(
        self, h3_index: str, county: str, img_bgr: np.ndarray, hard_neg_boxes: List[dict]
    ) -> bool:
        if not hard_neg_boxes or self.saved_count >= self.max_saved:
            return False

        stem = f"{county.lower()}_{h3_index}"
        jpg_path = self.images_dir / f"{stem}.jpg"
        txt_path = self.labels_dir / f"{stem}.txt"

        if jpg_path.exists():
            return False

        # Save .tif RGB bands as 95%-quality .jpg for Roboflow
        cv2.imwrite(str(jpg_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Write pre-annotations in standard YOLO format: <class_id> <x_center> <y_center> <width> <height>
        lines = []
        for b in hard_neg_boxes:
            nx, ny, nw, nh = b["xywhn"]
            lines.append(f"{b['cls_id']} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
        txt_path.write_text("\n".join(lines) + "\n")

        self.saved_count += 1
        return True