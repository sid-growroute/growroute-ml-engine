from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detectors.pool_canopy import PoolSpaCanopyDetector, TileGeoContext


CANONICAL_DETECTION_KEYS = {
    "h3_index_res9",
    "lat",
    "lng",
    "ai_pool_confidence",
    "ai_pool_condition",
    "ai_pool_sqft",
    "ai_has_spa",
    "ai_tree_canopy_pct",
    "ai_tree_overhang",
}


class _FakeTensor:
    def __init__(self, vals):
        self._vals = vals

    def item(self):
        return self._vals[0]

    def tolist(self):
        return list(self._vals)


class _FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy: list[float], xywhn: list[float]):
        self.cls = [_FakeTensor([cls_id])]
        self.conf = [_FakeTensor([conf])]
        self.xyxy = [_FakeTensor(xyxy)]
        self.xywhn = [_FakeTensor(xywhn)]


class _FakePredResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_analyze_tile_emits_strictly_canonical_ai_keys():
    fake_img_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    fake_img_bgr[:, :, 0] = 220  # Blue water in BGR
    fake_nir = np.full((200, 200), 180, dtype=np.uint8)
    fake_geo_ctx = TileGeoContext(
        width_px=200,
        height_px=200,
        meters_per_px=0.6,
        fallback_bounds=(-97.152, 32.908, -97.148, 32.912),
    )

    pool_box = _FakeBox(
        cls_id=0,
        conf=0.812,
        xyxy=[50.0, 50.0, 75.0, 70.0],
        xywhn=[0.3125, 0.30, 0.125, 0.10],
    )
    spa_box = _FakeBox(
        cls_id=1,
        conf=0.654,
        xyxy=[76.0, 52.0, 82.0, 58.0],
        xywhn=[0.395, 0.275, 0.03, 0.03],
    )

    with patch("src.detectors.pool_canopy.YOLO") as mock_yolo, patch(
        "src.detectors.pool_canopy.load_geotiff_or_image",
        return_value=(fake_img_bgr, fake_nir, fake_geo_ctx),
    ):
        mock_model_inst = MagicMock()
        mock_model_inst.predict.return_value = [_FakePredResult([pool_box, spa_box])]
        mock_yolo.return_value = mock_model_inst

        detector = PoolSpaCanopyDetector(
            weights_path="models/pool_spa_dfw_v2.onnx",
            conf_threshold=0.45,
            hard_neg_range=(0.25, 0.44),
        )
        pools, hard_negs, _ = detector.analyze_tile(
            Path("tx_tarrant_hex_8926c86810fffff.tif"),
            "8926c86810fffff",
            buffer_meters=15.0,
        )

    assert len(pools) == 1
    assert len(hard_negs) == 0
    det = pools[0]

    assert set(det.keys()) == CANONICAL_DETECTION_KEYS
    assert det["h3_index_res9"] == "8926c86810fffff"
    assert det["ai_pool_confidence"] == 0.812
    assert det["ai_pool_condition"] in ("clean", "green")
    assert 150 <= det["ai_pool_sqft"] <= 2500
    assert det["ai_has_spa"] is True
    assert isinstance(det["ai_tree_canopy_pct"], float)
    assert det["ai_tree_overhang"] is (det["ai_tree_canopy_pct"] >= 35.0)


class TestCanonicalDetectionKeys(unittest.TestCase):
    def test_canonical_keys(self):
        test_analyze_tile_emits_strictly_canonical_ai_keys()


if __name__ == "__main__":
    unittest.main()
