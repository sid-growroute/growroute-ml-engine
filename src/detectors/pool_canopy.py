from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import h3
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from ultralytics import YOLO


@dataclass
class TileGeoContext:
    width_px: int
    height_px: int
    meters_per_px: float
    crs: Optional[object] = None
    affine: Optional[object] = None
    fallback_bounds: Optional[Tuple[float, float, float, float]] = None  # (west, south, east, north)

    def pixel_to_lnglat(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Converts image pixel (x, y) to EPSG:4326 (lng, lat) using GeoTIFF affine transform + CRS."""
        if self.affine is not None and self.crs is not None:
            # rasterio.transform.xy takes (row=y, col=x)
            x, y = rasterio.transform.xy(self.affine, y_px, x_px)
            if str(self.crs).upper() in ("EPSG:4326", "WGS84"):
                return float(x), float(y)
            lngs, lats = warp_transform(self.crs, "EPSG:4326", [float(x)], [float(y)])
            return float(lngs[0]), float(lats[0])

        west, south, east, north = self.fallback_bounds
        lng = west + (x_px / max(self.width_px, 1)) * (east - west)
        lat = north - (y_px / max(self.height_px, 1)) * (north - south)
        return float(lng), float(lat)


def load_geotiff_or_image(
    tile_path: Path, h3_index: str
) -> Tuple[np.ndarray, Optional[np.ndarray], TileGeoContext]:
    """
    Loads a .tif (3-band RGB or 4-band RGB+NIR) or fallback .jpg/.png.
    Returns:
      - img_bgr: 3-channel uint8 BGR array (matches Roboflow .jpg input for YOLO)
      - nir_band: optional 4th Near-Infrared band (for true NDVI tree canopy calculation)
      - geo_ctx: TileGeoContext for exact pixel-to-(lng, lat) projection
    """
    if tile_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(tile_path) as src:
            rgb = src.read([1, 2, 3])
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            img_rgb = np.transpose(rgb, (1, 2, 0))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            nir_band = src.read(4).astype(np.float32) if src.count >= 4 else None

            # Compute accurate meters_per_px from GeoTIFF resolution
            res_x = abs(src.transform.a)
            if src.crs and src.crs.is_projected:
                meters_per_px = float(res_x)  # UTM / Web Mercator is already in meters
            else:
                mid_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                meters_per_px = float(res_x * 111_320.0 * np.cos(np.radians(mid_lat)))

            geo_ctx = TileGeoContext(
                width_px=src.width,
                height_px=src.height,
                meters_per_px=max(meters_per_px, 0.1),
                crs=src.crs,
                affine=src.transform,
            )
            return img_bgr, nir_band, geo_ctx

    # Fallback for .jpg / .png tiles
    img_bgr = cv2.imread(str(tile_path))
    if img_bgr is None:
        raise ValueError(f"Failed to decode image: {tile_path}")
    h_px, w_px = img_bgr.shape[:2]
    boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
    lngs = [pt[0] for pt in boundary]
    lats = [pt[1] for pt in boundary]
    west, south, east, north = min(lngs), min(lats), max(lngs), max(lats)
    mid_lat_rad = np.radians((north + south) / 2.0)
    meters_wide = (east - west) * 111_320.0 * np.cos(mid_lat_rad)

    geo_ctx = TileGeoContext(
        width_px=w_px,
        height_px=h_px,
        meters_per_px=max(meters_wide / max(w_px, 1), 0.1),
        fallback_bounds=(west, south, east, north),
    )
    return img_bgr, None, geo_ctx


def estimate_canopy_pct_around_pool(
    img_bgr: np.ndarray,
    nir_band: Optional[np.ndarray],
    bbox_xyxy: Tuple[int, int, int, int],
    meters_per_px: float,
    buffer_meters: float = 15.0,
) -> float:
    """
    Calculates tree canopy % in a 15m ring around the pool.
    Uses 4-band NAIP NDVI + darkness threshold if NIR band exists, else RGB Excess Green + HSV.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    pad_px = int(round(buffer_meters / meters_per_px))

    rx1, ry1 = max(0, x1 - pad_px), max(0, y1 - pad_px)
    rx2, ry2 = min(w, x2 + pad_px), min(h, y2 + pad_px)
    roi_bgr = img_bgr[ry1:ry2, rx1:rx2]
    if roi_bgr.size == 0:
        return 0.0

    ring_mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
    ring_mask[(y1 - ry1):(y2 - ry1), (x1 - rx1):(x2 - rx1)] = False
    ring_bgr = roi_bgr[ring_mask]
    if ring_bgr.size == 0:
        return 0.0

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[ring_mask]
    val = roi_hsv[:, 2]

    if nir_band is not None:
        roi_nir = nir_band[ry1:ry2, rx1:rx2][ring_mask]
        red = ring_bgr[:, 2].astype(np.float32)
        ndvi = (roi_nir - red) / (roi_nir + red + 1e-5)
        # Trees have high NDVI (> 0.18) and darker canopy texture/shade than bright open lawn
        is_canopy = (ndvi >= 0.18) & (val >= 18) & (val <= 155)
        return round(float(np.mean(is_canopy) * 100.0), 2)

    b_ch = ring_bgr[:, 0].astype(np.int16)
    g_ch = ring_bgr[:, 1].astype(np.int16)
    r_ch = ring_bgr[:, 2].astype(np.int16)
    exg = (2 * g_ch) - r_ch - b_ch
    hue, sat = roi_hsv[:, 0], roi_hsv[:, 1]
    is_canopy = (hue >= 30) & (hue <= 95) & (sat >= 35) & (val >= 20) & (val <= 155) & (exg > 12)
    return round(float(np.mean(is_canopy) * 100.0), 2)


def classify_pool_water_condition(img_bgr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    pw, ph = x2 - x1, y2 - y1
    cx1, cy1 = int(x1 + 0.25 * pw), int(y1 + 0.25 * ph)
    cx2, cy2 = int(x2 - 0.25 * pw), int(y2 - 0.25 * ph)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "clean"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_hue = float(np.mean(hsv[:, :, 0]))
    mean_sat = float(np.mean(hsv[:, :, 1]))
    if 35.0 <= mean_hue <= 82.0 and mean_sat >= 45.0:
        return "green"
    return "clean"


class PoolSpaCanopyDetector:
    def __init__(
        self,
        weights_path: str,
        device: str = "mps",
        conf_threshold: float = 0.45,
        hard_neg_range: Tuple[float, float] = (0.25, 0.44),
    ):
        self.model = YOLO(weights_path, task="segment")
        self.device = device
        self.conf_threshold = conf_threshold
        self.hard_neg_min, self.hard_neg_max = hard_neg_range

    def analyze_tile(
        self, tile_path: Path, h3_index: str, buffer_meters: float = 15.0
    ) -> Tuple[List[dict], List[dict], np.ndarray]:
        """
        Runs YOLO at low threshold (hard_neg_min) once:
        - Detections >= conf_threshold go to PostGIS.
        - Borderline detections in [hard_neg_min, hard_neg_max] are returned for Roboflow .jpg export.
        """
        img_bgr, nir_band, geo_ctx = load_geotiff_or_image(tile_path, h3_index)
        preds = self.model.predict(
            source=img_bgr,
            conf=self.hard_neg_min,
            device=self.device,
            verbose=False,
        )[0]

        if preds.boxes is None or len(preds.boxes) == 0:
            return [], [], img_bgr

        m_per_px = geo_ctx.meters_per_px
        sqft_per_px2 = (m_per_px ** 2) * 10.7639

        pools = []
        spa_centers = []
        hard_neg_boxes = []

        for box in preds.boxes:
            cls_id = int(box.cls[0].item())
            conf = round(min(max(float(box.conf[0].item()), 0.0), 1.0), 3)
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]

            # Borderline detection -> queue for Roboflow hard-negative .jpg export
            if conf < self.conf_threshold:
                if self.hard_neg_min <= conf <= self.hard_neg_max and cls_id in (0, 1):
                    nx, ny, nw, nh = [round(float(v), 6) for v in box.xywhn[0].tolist()]
                    hard_neg_boxes.append({
                        "cls_id": cls_id,
                        "conf": conf,
                        "xywhn": (nx, ny, nw, nh),
                    })
                continue

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if cls_id == 1:  # Spa
                spa_centers.append((cx, cy))
                continue
            if cls_id != 0:  # Pool is class 0
                continue

            lng, lat = geo_ctx.pixel_to_lnglat(cx, cy)
            area_px = max((x2 - x1) * (y2 - y1) * 0.78, 1.0)
            ai_pool_sqft = int(round(area_px * sqft_per_px2))
            canopy_pct = estimate_canopy_pct_around_pool(
                img_bgr, nir_band, (x1, y1, x2, y2), m_per_px, buffer_meters
            )
            condition = classify_pool_water_condition(img_bgr, (x1, y1, x2, y2))

            pools.append({
                "h3_index_res9": h3_index,
                "lat": round(lat, 7),
                "lng": round(lng, 7),
                "ai_pool_confidence": conf,
                "ai_pool_condition": condition,
                "ai_pool_sqft": max(min(ai_pool_sqft, 2500), 150),
                "ai_has_spa": False,
                "ai_tree_canopy_pct": canopy_pct,
                "ai_tree_overhang": bool(canopy_pct >= 35.0),
                "_center_px": (cx, cy),
            })

        max_spa_dist_px = 20.0 / m_per_px
        for p in pools:
            px, py = p.pop("_center_px")
            for sx, sy in spa_centers:
                if np.hypot(px - sx, py - sy) <= max_spa_dist_px:
                    p["ai_has_spa"] = True
                    break

        return pools, hard_neg_boxes, img_bgr