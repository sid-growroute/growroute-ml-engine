from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import h3
import numpy as np
from PIL import Image
from pyproj import Transformer
import rasterio
from shapely.geometry import Polygon
from ultralytics import YOLO


@dataclass
class TileGeoContext:
    width_px: int
    height_px: int
    meters_per_px: float
    crs: Optional[object] = None
    affine: Optional[object] = None
    fallback_bounds: Optional[Tuple[float, float, float, float]] = None  # (west, south, east, north)
    _transformer: Optional[Transformer] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.crs is not None and str(self.crs).upper() not in ("EPSG:4326", "WGS84", "OGC:CRS84"):
            self._transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    def pixel_to_lnglat(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """
        Converts continuous image pixel (x_col, y_row) to EPSG:4326 (lng, lat).
        Uses `affine * (x_col, y_row)` directly in pixel-corner coordinates (matching YOLO/OpenCV
        continuous coordinates without rasterio.transform.xy's extra +0.5px center shift) and
        projects via cached pyproj.Transformer(..., always_xy=True).
        """
        if self.affine is not None and self.crs is not None:
            x_crs, y_crs = self.affine * (float(x_px), float(y_px))
            if self._transformer is None:
                return float(x_crs), float(y_crs)
            lng, lat = self._transformer.transform(float(x_crs), float(y_crs))
            return float(lng), float(lat)

        west, south, east, north = self.fallback_bounds
        lng = west + (float(x_px) / max(self.width_px, 1)) * (east - west)
        lat = north - (float(y_px) / max(self.height_px, 1)) * (north - south)
        return float(lng), float(lat)


def load_geotiff_or_image(
    tile_path: Path, h3_index: str
) -> Tuple[np.ndarray, Optional[np.ndarray], TileGeoContext]:
    """
    Loads a .tif (3-band RGB or 4-band RGB+NIR) or fallback .jpg/.png.
    Returns:
      - img_rgb: canonical 3-channel uint8 RGB array (H, W, 3)
      - nir_band: optional 4th Near-Infrared band (float32, for NDWI water verification & NDVI canopy)
      - geo_ctx: TileGeoContext for exact pixel-to-(lng, lat) projection
    """
    if tile_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(tile_path) as src:
            rgb = src.read([1, 2, 3])
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            img_rgb = np.ascontiguousarray(np.transpose(rgb, (1, 2, 0)))

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
            return img_rgb, nir_band, geo_ctx

    # Fallback for .jpg / .png tiles (OpenCV decodes BGR -> convert immediately to canonical RGB)
    img_bgr = cv2.imread(str(tile_path))
    if img_bgr is None:
        raise ValueError(f"Failed to decode image: {tile_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_px, w_px = img_rgb.shape[:2]
    boundary = (
        [(float(pt[0]), float(pt[1])) for pt in h3.h3_to_geo_boundary(h3_index, geo_json=True)]
        if hasattr(h3, "h3_to_geo_boundary")
        else [(float(latlng[1]), float(latlng[0])) for latlng in h3.cell_to_boundary(h3_index)]
    )
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
    return img_rgb, None, geo_ctx


def _extract_interior_mask_pixels(
    img_rgb: np.ndarray,
    nir_band: Optional[np.ndarray],
    bbox_xyxy: Tuple[int, int, int, int],
    poly_px: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extracts RGB (and optional NIR) pixels strictly inside the detected pool water surface,
    eroding 1px off the segmentation mask boundary to avoid concrete coping / deck stones.
    """
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [
        max(0, min(w, int(bbox_xyxy[0]))),
        max(0, min(h, int(bbox_xyxy[1]))),
        max(0, min(w, int(bbox_xyxy[2]))),
        max(0, min(h, int(bbox_xyxy[3]))),
    ]

    if poly_px is not None and len(poly_px) >= 3:
        rx1 = max(0, int(np.floor(np.min(poly_px[:, 0]))) - 1)
        ry1 = max(0, int(np.floor(np.min(poly_px[:, 1]))) - 1)
        rx2 = min(w, int(np.ceil(np.max(poly_px[:, 0]))) + 2)
        ry2 = min(h, int(np.ceil(np.max(poly_px[:, 1]))) + 2)
        if rx2 > rx1 and ry2 > ry1:
            local_mask = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
            shifted = np.round(poly_px - np.array([[rx1, ry1]], dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(local_mask, [shifted], 1)
            eroded = cv2.erode(local_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            active = (eroded if int(eroded.sum()) >= 8 else local_mask).astype(bool)
            if np.any(active):
                px_rgb = img_rgb[ry1:ry2, rx1:rx2][active]
                px_nir = nir_band[ry1:ry2, rx1:rx2][active] if nir_band is not None else None
                return px_rgb, px_nir

    pw, ph = max(x2 - x1, 1), max(y2 - y1, 1)
    cx1, cy1 = int(x1 + 0.25 * pw), int(y1 + 0.25 * ph)
    cx2, cy2 = max(cx1 + 1, int(x2 - 0.25 * pw)), max(cy1 + 1, int(y2 - 0.25 * ph))
    crop_rgb = img_rgb[cy1:cy2, cx1:cx2].reshape(-1, 3)
    crop_nir = nir_band[cy1:cy2, cx1:cx2].reshape(-1) if nir_band is not None else None
    return crop_rgb, crop_nir


def is_spectrally_valid_pool(
    img_rgb: np.ndarray,
    nir_band: Optional[np.ndarray],
    bbox_xyxy: Tuple[int, int, int, int],
    conf: float,
    poly_px: Optional[np.ndarray] = None,
) -> bool:
    """
    Rejects non-water false positives (wooden pergolas/decks, asphalt roofs, driveways, tree shadows):
      - When 4-band NAIP NIR is present: liquid water (blue, dark-plaster, or green algae) strongly
        absorbs NIR relative to Green (`NDWI = (G - NIR) / (G + NIR) >= 0.05`), whereas roofs, decks,
        concrete, and shaded vegetation reflect NIR (`NDWI < 0.05`). Also rejects warm brown/tan
        patios (`B < R` with weak `NDWI < 0.16`) below high confidence (`0.70`).
      - When 3-band RGB only: rejects warm red/brown surfaces (`R > B + 10` and `R > G + 6`).
    """
    px_rgb, px_nir = _extract_interior_mask_pixels(img_rgb, nir_band, bbox_xyxy, poly_px)
    if px_rgb.size == 0:
        return False

    rgb_f = px_rgb.astype(np.float32)
    r_mean = float(np.mean(rgb_f[:, 0]))
    g_mean = float(np.mean(rgb_f[:, 1]))
    b_mean = float(np.mean(rgb_f[:, 2]))

    if px_nir is not None and px_nir.size == px_rgb.shape[0]:
        ndwi = float(np.mean((rgb_f[:, 1] - px_nir) / (rgb_f[:, 1] + px_nir + 1e-5)))
        if ndwi < 0.05:
            return False
        if b_mean < r_mean and ndwi < 0.16 and conf < 0.70:
            return False
        return True

    if r_mean > b_mean + 10.0 and r_mean > g_mean + 6.0:
        return False
    return True


def estimate_canopy_pct_around_pool(
    img_rgb: np.ndarray,
    nir_band: Optional[np.ndarray],
    bbox_xyxy: Tuple[int, int, int, int],
    meters_per_px: float,
    buffer_meters: float = 15.0,
) -> float:
    """
    Calculates tree canopy % in a 15m ring around the pool.
    Uses 4-band NAIP NDVI + darkness threshold if NIR band exists, else RGB Excess Green + HSV.
    """
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    pad_px = int(round(buffer_meters / meters_per_px))

    rx1, ry1 = max(0, x1 - pad_px), max(0, y1 - pad_px)
    rx2, ry2 = min(w, x2 + pad_px), min(h, y2 + pad_px)
    roi_rgb = img_rgb[ry1:ry2, rx1:rx2]
    if roi_rgb.size == 0:
        return 0.0

    ring_mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
    ring_mask[max(0, y1 - ry1) : max(0, y2 - ry1), max(0, x1 - rx1) : max(0, x2 - rx1)] = False
    ring_rgb = roi_rgb[ring_mask]
    if ring_rgb.size == 0:
        return 0.0

    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)[ring_mask]
    val = roi_hsv[:, 2]

    if nir_band is not None:
        roi_nir = nir_band[ry1:ry2, rx1:rx2][ring_mask]
        red = ring_rgb[:, 0].astype(np.float32)
        ndvi = (roi_nir - red) / (roi_nir + red + 1e-5)
        # Trees have high NDVI (> 0.18) and darker canopy texture/shade than bright open lawn
        is_canopy = (ndvi >= 0.18) & (val >= 18) & (val <= 155)
        return round(float(np.mean(is_canopy) * 100.0), 2)

    r_ch = ring_rgb[:, 0].astype(np.int16)
    g_ch = ring_rgb[:, 1].astype(np.int16)
    b_ch = ring_rgb[:, 2].astype(np.int16)
    exg = (2 * g_ch) - r_ch - b_ch
    hue, sat = roi_hsv[:, 0], roi_hsv[:, 1]
    is_canopy = (hue >= 30) & (hue <= 95) & (sat >= 35) & (val >= 20) & (val <= 155) & (exg > 12)
    return round(float(np.mean(is_canopy) * 100.0), 2)


def classify_pool_water_condition(
    img_rgb: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    poly_px: Optional[np.ndarray] = None,
) -> str:
    px_rgb, _ = _extract_interior_mask_pixels(img_rgb, None, bbox_xyxy, poly_px)
    if px_rgb.size == 0:
        return "clean"

    hsv = cv2.cvtColor(px_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    mean_hue = float(np.mean(hsv[:, 0, 0]))
    mean_sat = float(np.mean(hsv[:, 0, 1]))
    if 35.0 <= mean_hue <= 82.0 and mean_sat >= 42.0:
        return "green"
    return "clean"


def _compute_mask_interior_point_and_area(
    poly_px: Optional[np.ndarray],
    bbox_xyxy_f: Tuple[float, float, float, float],
) -> Tuple[float, float, float]:
    """
    Computes an interior point `(cx, cy)` guaranteed strictly inside the segmented pool polygon
    (using Polygon.representative_point() when L-shaped/kidney centroid falls outside water)
    and exact polygon pixel area. Falls back to float bbox center if mask is unavailable.
    """
    x1_f, y1_f, x2_f, y2_f = bbox_xyxy_f
    if poly_px is not None and len(poly_px) >= 3:
        try:
            poly = Polygon(poly_px)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty and poly.area >= 4.0:
                c = poly.centroid
                interior = c if poly.contains(c) else poly.representative_point()
                return float(interior.x), float(interior.y), float(poly.area)
        except Exception:
            pass

    cx = (x1_f + x2_f) / 2.0
    cy = (y1_f + y2_f) / 2.0
    area_px = max((x2_f - x1_f) * (y2_f - y1_f) * 0.78, 1.0)
    return cx, cy, area_px


class PoolSpaCanopyDetector:
    def __init__(
        self,
        weights_path: str,
        device: str = "mps",
        conf_threshold: float = 0.55,
        hard_neg_range: Tuple[float, float] = (0.25, 0.54),
        imgsz: int = 640,
    ):
        self.model = YOLO(weights_path, task="segment")
        self.device = device
        self.conf_threshold = conf_threshold
        self.hard_neg_min, self.hard_neg_max = hard_neg_range
        self.imgsz = imgsz

    def analyze_tile(
        self,
        tile_path: Path,
        h3_index: str,
        buffer_meters: float = 15.0,
        include_pixel_geometry: bool = False,
    ) -> Tuple[List[dict], List[dict], np.ndarray]:
        """
        Runs YOLO segmentation at low threshold (hard_neg_min) once using explicit RGB PIL input:
        - Detections >= conf_threshold passing 4-band NDWI / chromatic water check go to PostGIS.
        - Borderline detections in [hard_neg_min, hard_neg_max] are returned for Roboflow .jpg export.
        Returns (pools, hard_neg_boxes, img_bgr) where img_bgr is ready for OpenCV cv2.imwrite.
        """
        img_rgb, nir_band, geo_ctx = load_geotiff_or_image(tile_path, h3_index)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Pass explicit RGB PIL.Image so Ultralytics LoadPilAndNumpy never misinterprets RGB/BGR
        preds = self.model.predict(
            source=Image.fromarray(img_rgb),
            imgsz=self.imgsz,
            conf=self.hard_neg_min,
            device=self.device,
            verbose=False,
        )[0]

        if preds.boxes is None or len(preds.boxes) == 0:
            return [], [], img_bgr

        m_per_px = geo_ctx.meters_per_px
        sqft_per_px2 = (m_per_px ** 2) * 10.7639
        mask_polys = preds.masks.xy if preds.masks is not None else None

        pools = []
        spa_centers = []
        hard_neg_boxes = []

        for idx, box in enumerate(preds.boxes):
            cls_id = int(box.cls[0].item())
            conf = round(min(max(float(box.conf[0].item()), 0.0), 1.0), 3)
            x1_f, y1_f, x2_f, y2_f = [float(v) for v in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = int(round(x1_f)), int(round(y1_f)), int(round(x2_f)), int(round(y2_f))
            poly_px = (
                mask_polys[idx]
                if mask_polys is not None and idx < len(mask_polys) and len(mask_polys[idx]) >= 3
                else None
            )

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

            cx, cy, area_px = _compute_mask_interior_point_and_area(poly_px, (x1_f, y1_f, x2_f, y2_f))
            if cls_id == 1:  # Spa
                spa_centers.append((cx, cy))
                continue
            if cls_id != 0:  # Pool is class 0
                continue

            if not is_spectrally_valid_pool(img_rgb, nir_band, (x1, y1, x2, y2), conf, poly_px=poly_px):
                if self.hard_neg_min <= conf <= 0.70:
                    nx, ny, nw, nh = [round(float(v), 6) for v in box.xywhn[0].tolist()]
                    hard_neg_boxes.append({
                        "cls_id": cls_id,
                        "conf": conf,
                        "xywhn": (nx, ny, nw, nh),
                    })
                continue

            lng, lat = geo_ctx.pixel_to_lnglat(cx, cy)
            ai_pool_sqft = int(round(max(area_px, 1.0) * sqft_per_px2))
            canopy_pct = estimate_canopy_pct_around_pool(
                img_rgb, nir_band, (x1, y1, x2, y2), m_per_px, buffer_meters
            )
            condition = classify_pool_water_condition(img_rgb, (x1, y1, x2, y2), poly_px=poly_px)

            pool_record = {
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
            }
            if include_pixel_geometry:
                pool_record["center_px"] = (round(cx, 2), round(cy, 2))
                pool_record["bbox_px"] = (x1, y1, x2, y2)
                pool_record["mask_xy"] = (
                    [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in poly_px]
                    if poly_px is not None
                    else None
                )
            pools.append(pool_record)

        max_spa_dist_px = 20.0 / m_per_px
        for p in pools:
            px, py = p.pop("_center_px")
            for sx, sy in spa_centers:
                if np.hypot(px - sx, py - sy) <= max_spa_dist_px:
                    p["ai_has_spa"] = True
                    break

        return pools, hard_neg_boxes, img_bgr
