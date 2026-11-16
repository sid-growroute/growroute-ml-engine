from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import cv2
import h3
import numpy as np
from PIL import Image
from pyproj import Transformer
import rasterio
from shapely.geometry import Polygon
from ultralytics import YOLO

ABOVE_GROUND_RECOVERY_MIN_CONF = 0.38


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

            res_x = abs(src.transform.a)
            if src.crs and src.crs.is_projected:
                meters_per_px = float(res_x)
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


def _build_local_water_and_ring_masks(
    img_shape: Tuple[int, int],
    bbox_xyxy: Tuple[int, int, int, int],
    poly_px: Optional[np.ndarray] = None,
    pad_px: int = 6,
) -> Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs aligned boolean masks on a local padded ROI `(wx1, wy1, wx2, wy2)`:
      - `water_mask`: full segmented pool footprint (or inscribed ellipse fallback)
      - `interior_mask`: 1px-eroded interior core (when >= 6px survive) to isolate pure water
      - `rim_ring`: 1–3px immediate perimeter ring (captures above-ground wall shadow arc)
      - `apron_ring`: 2–4px outer ring (`~1.2m–2.4m` on NAIP) for turf vs. concrete coping check
    """
    h, w = img_shape
    x1, y1, x2, y2 = [
        max(0, min(w, int(bbox_xyxy[0]))),
        max(0, min(h, int(bbox_xyxy[1]))),
        max(0, min(w, int(bbox_xyxy[2]))),
        max(0, min(h, int(bbox_xyxy[3]))),
    ]
    if poly_px is not None and len(poly_px) >= 3:
        px_min_x = int(np.floor(np.min(poly_px[:, 0])))
        px_min_y = int(np.floor(np.min(poly_px[:, 1])))
        px_max_x = int(np.ceil(np.max(poly_px[:, 0])))
        px_max_y = int(np.ceil(np.max(poly_px[:, 1])))
        x1, y1 = min(x1, max(0, px_min_x)), min(y1, max(0, px_min_y))
        x2, y2 = max(x2, min(w, px_max_x)), max(y2, min(h, px_max_y))

    wx1 = max(0, x1 - pad_px)
    wy1 = max(0, y1 - pad_px)
    wx2 = min(w, max(x2 + pad_px + 1, wx1 + 3))
    wy2 = min(h, max(y2 + pad_px + 1, wy1 + 3))

    roi_h, roi_w = max(wy2 - wy1, 1), max(wx2 - wx1, 1)
    water_u8 = np.zeros((roi_h, roi_w), dtype=np.uint8)

    if poly_px is not None and len(poly_px) >= 3:
        shifted = np.round(poly_px - np.array([[wx1, wy1]], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(water_u8, [shifted], 1)

    if int(water_u8.sum()) < 4:
        cx_l = int(round(((x1 + x2) / 2.0) - wx1))
        cy_l = int(round(((y1 + y2) / 2.0) - wy1))
        ax_x = max(int(round((x2 - x1) / 2.0)), 1)
        ax_y = max(int(round((y2 - y1) / 2.0)), 1)
        cv2.ellipse(water_u8, (cx_l, cy_l), (ax_x, ax_y), 0, 0, 360, 1, -1)
        if int(water_u8.sum()) == 0:
            water_u8[max(0, y1 - wy1) : max(1, y2 - wy1), max(0, x1 - wx1) : max(1, x2 - wx1)] = 1

    eroded_u8 = cv2.erode(water_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
    interior_mask = (eroded_u8 if int(eroded_u8.sum()) >= 6 else water_u8).astype(bool)
    water_mask = water_u8.astype(bool)

    dil_1 = cv2.dilate(water_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    dil_3 = cv2.dilate(water_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    dil_4 = cv2.dilate(water_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    rim_ring = (dil_3 > 0) & (~water_mask)
    apron_ring = (dil_4 > 0) & (dil_1 == 0)
    if int(apron_ring.sum()) < 8:
        apron_ring = rim_ring

    return (wx1, wy1, wx2, wy2), water_mask, interior_mask, rim_ring, apron_ring


def _extract_interior_mask_pixels(
    img_rgb: np.ndarray,
    nir_band: Optional[np.ndarray],
    bbox_xyxy: Tuple[int, int, int, int],
    poly_px: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extracts RGB (and optional NIR) pixels strictly inside the detected pool water surface,
    eroding 1px off the segmentation mask boundary when >= 6 interior pixels remain.
    """
    (wx1, wy1, wx2, wy2), _water_mask, interior_mask, _rim, _apron = _build_local_water_and_ring_masks(
        img_rgb.shape[:2], bbox_xyxy, poly_px=poly_px, pad_px=3
    )
    if np.any(interior_mask):
        px_rgb = img_rgb[wy1:wy2, wx1:wx2][interior_mask]
        px_nir = nir_band[wy1:wy2, wx1:wx2][interior_mask] if nir_band is not None else None
        return px_rgb, px_nir
    return np.empty((0, 3), dtype=np.uint8), None


def evaluate_pool_water_and_context(
    img_rgb: np.ndarray,
    nir_band: Optional[np.ndarray],
    bbox_xyxy: Tuple[int, int, int, int],
    poly_px: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes interior water spectral metrics (4-band NDWI + RGB/HSV cyan-blue & green-water ratios)
    and 2–4px outer ring context metrics (turf NDVI/green ratio, dark wall-shadow rim ratio,
    and bright low-NDVI concrete apron ratio).
    """
    (wx1, wy1, wx2, wy2), _water_mask, interior_mask, rim_ring, apron_ring = (
        _build_local_water_and_ring_masks(img_rgb.shape[:2], bbox_xyxy, poly_px=poly_px, pad_px=6)
    )
    roi_rgb = img_rgb[wy1:wy2, wx1:wx2]
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    roi_nir = nir_band[wy1:wy2, wx1:wx2] if nir_band is not None else None

    px_rgb_u8 = roi_rgb[interior_mask]
    if px_rgb_u8.size == 0:
        return {"valid": 0.0}

    px_rgb = px_rgb_u8.astype(np.float32)
    px_hsv = roi_hsv[interior_mask]
    r_arr, g_arr, b_arr = px_rgb[:, 0], px_rgb[:, 1], px_rgb[:, 2]
    h_arr, s_arr, v_arr = px_hsv[:, 0], px_hsv[:, 1], px_hsv[:, 2]

    blue_cyan_px = (
        ((h_arr >= 72) & (h_arr <= 138) & (s_arr >= 18) & (v_arr >= 35))
        | ((b_arr >= r_arr + 6.0) & (g_arr >= r_arr + 3.0) & (b_arr >= 52.0))
    )
    green_water_hsv_px = (h_arr >= 35) & (h_arr <= 82) & (s_arr >= 42) & (v_arr >= 28)

    r_mean = float(np.mean(r_arr))
    g_mean = float(np.mean(g_arr))
    b_mean = float(np.mean(b_arr))
    blue_cyan_ratio = float(np.mean(blue_cyan_px))

    has_nir = roi_nir is not None
    if has_nir:
        px_nir = roi_nir[interior_mask].astype(np.float32)
        ndwi_arr = (g_arr - px_nir) / (g_arr + px_nir + 1e-6)
        ndwi_mean = float(np.mean(ndwi_arr))
        ndwi_p75 = float(np.percentile(ndwi_arr, 75))
        water_px_ratio = float(np.mean(blue_cyan_px | (green_water_hsv_px & (ndwi_arr >= 0.02))))
    else:
        ndwi_mean = 0.0
        ndwi_p75 = 0.0
        water_px_ratio = float(np.mean(blue_cyan_px | green_water_hsv_px))

    # Outer 1–3px rim shadow arc (normalized V < 0.28 -> uint8 V < 72)
    if np.any(rim_ring):
        rim_v_norm = roi_hsv[rim_ring][:, 2].astype(np.float32) / 255.0
        shadow_rim_ratio = float(np.mean(rim_v_norm < 0.28))
    else:
        shadow_rim_ratio = 0.0

    # Outer 2–4px apron ring: grass/turf vs. bright concrete/stone patio coping
    apron_ndvi_med = 0.0
    if np.any(apron_ring):
        ap_rgb = roi_rgb[apron_ring].astype(np.float32)
        ap_hsv = roi_hsv[apron_ring]
        ap_h = ap_hsv[:, 0]
        ap_s_norm = ap_hsv[:, 1].astype(np.float32) / 255.0
        ap_v_norm = ap_hsv[:, 2].astype(np.float32) / 255.0
        if has_nir:
            ap_nir = roi_nir[apron_ring].astype(np.float32)
            ap_ndvi = (ap_nir - ap_rgb[:, 0]) / (ap_nir + ap_rgb[:, 0] + 1e-6)
            apron_ndvi_med = float(np.median(ap_ndvi))
            # Green irrigated turf OR summer-dormant golden-tan lawn (warm R > B, non-negative/near-zero NDVI)
            turf_px = (
                (ap_ndvi >= 0.18)
                | ((ap_ndvi >= 0.08) & (ap_rgb[:, 1] >= ap_rgb[:, 0] - 12.0))
                | (
                    (ap_ndvi >= -0.04)
                    & (ap_rgb[:, 0] > ap_rgb[:, 2] + 10.0)
                    & (ap_s_norm >= 0.12)
                    & (ap_v_norm < 0.66)
                )
            )
            # Bright concrete/travertine patio OR low-saturation gray stone/aggregate coping (negative NDVI)
            concrete_px = (
                ((ap_ndvi < 0.12) & (ap_v_norm >= 0.60) & (ap_s_norm <= 0.24))
                | ((ap_ndvi < -0.06) & ((ap_v_norm >= 0.50) | (ap_s_norm <= 0.15)))
            )
        else:
            turf_px = (
                ((ap_h >= 28) & (ap_h <= 95) & (ap_s_norm >= 0.12) & (ap_rgb[:, 1] >= ap_rgb[:, 0] - 8.0))
                | (
                    (ap_h >= 12)
                    & (ap_h < 32)
                    & (ap_s_norm >= 0.15)
                    & (ap_v_norm < 0.65)
                    & (ap_rgb[:, 0] > ap_rgb[:, 2] + 12.0)
                )
            )
            concrete_px = (
                (ap_v_norm >= 0.62)
                & (ap_s_norm <= 0.20)
                & (np.abs(ap_rgb[:, 0] - ap_rgb[:, 2]) <= 28.0)
            )
        turf_ratio = float(np.mean(turf_px))
        bright_concrete_ratio = float(np.mean(concrete_px))
    else:
        turf_ratio = 0.0
        bright_concrete_ratio = 0.0

    return {
        "valid": 1.0,
        "has_nir": 1.0 if has_nir else 0.0,
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "ndwi_mean": ndwi_mean,
        "ndwi_p75": ndwi_p75,
        "blue_cyan_ratio": blue_cyan_ratio,
        "water_px_ratio": water_px_ratio,
        "apron_ndvi_med": apron_ndvi_med,
        "turf_ratio": turf_ratio,
        "shadow_rim_ratio": shadow_rim_ratio,
        "bright_concrete_ratio": bright_concrete_ratio,
    }


def is_spectrally_valid_pool(
    img_rgb: np.ndarray,
    nir_band: Optional[np.ndarray],
    bbox_xyxy: Tuple[int, int, int, int],
    conf: float,
    poly_px: Optional[np.ndarray] = None,
    stats: Optional[Dict[str, float]] = None,
) -> bool:
    """
    Rejects non-water false positives (wooden pergolas/decks, shade sails/tarps, asphalt roofs,
    trampolines, driveways, tree shadows) using 4-band NDWI + RGB/HSV water-pixel checks.
    """
    s = stats or evaluate_pool_water_and_context(img_rgb, nir_band, bbox_xyxy, poly_px=poly_px)
    if not s.get("valid"):
        return False

    r_mean = s["r_mean"]
    g_mean = s["g_mean"]
    b_mean = s["b_mean"]
    blue_cyan_ratio = s["blue_cyan_ratio"]
    water_px_ratio = s["water_px_ratio"]

    if s["has_nir"] > 0.5:
        ndwi = s["ndwi_mean"]
        ndwi_p75 = s["ndwi_p75"]
        # Non-water NIR-reflective surfaces (tarps, shade sails, trampolines, roofs, pergolas)
        if conf < 0.72 and water_px_ratio < 0.08 and ndwi < 0.05:
            return False
        if ndwi < 0.05 and not (ndwi > 0.01 and ndwi_p75 >= 0.06 and blue_cyan_ratio >= 0.45):
            return False
        if b_mean < r_mean and ndwi < 0.16 and conf < 0.72:
            return False
        return True

    if conf < 0.72 and water_px_ratio < 0.05:
        return False
    if r_mean > b_mean + 10.0 and r_mean > g_mean + 6.0:
        return False
    return True


def classify_pool_type(
    raw_pool_sqft: float,
    circularity: float,
    aspect_ratio: float,
    has_adjacent_spa: bool,
    stats: Dict[str, float],
    solidity: float = 1.0,
    forced_native_type: Optional[str] = None,
) -> str:
    """
    Classifies a verified pool detection as `'above_ground'` vs `'inground'`:
      1. Direct YOLO native class mapping if model includes an `above_ground*` class.
      2. Otherwise `'above_ground'` when ALL hold:
         - No `spa` overlaps/adjoins this pool, `100 <= pool_sqft <= 625`, and `solidity >= 0.95`.
         - Compact circular/oval or small-frame geometry (`circularity >= 0.76`, OR
           `0.80 <= w/h <= 1.25` when `pool_sqft <= 575`, OR small oval/frame `<= 340 sqft`).
         - Lack of wide concrete/stone coping apron (`apron_ndvi_med >= -0.05` when 4-band NIR
           exists, plus `turf_ratio >= 0.50`, OR `shadow_rim_ratio >= 0.20` with
           `bright_concrete_ratio < 0.35`, OR compact `<= 320 sqft` pool with partial wood/turf border).
      3. Otherwise `'inground'`.
    """
    if forced_native_type in ("above_ground", "inground"):
        return forced_native_type

    if has_adjacent_spa or not (100.0 <= raw_pool_sqft <= 625.0) or solidity < 0.95:
        return "inground"

    compact_geom = (
        (circularity >= 0.78 and 0.78 <= aspect_ratio <= 1.28)
        or (0.80 <= aspect_ratio <= 1.25 and raw_pool_sqft <= 575.0 and circularity >= 0.75)
        or (0.55 <= aspect_ratio <= 1.82 and raw_pool_sqft <= 360.0 and circularity >= 0.74)
    )
    if not compact_geom:
        return "inground"

    if stats.get("has_nir", 0.0) > 0.5 and stats.get("apron_ndvi_med", -1.0) < -0.05:
        return "inground"

    turf_ratio = stats.get("turf_ratio", 0.0)
    shadow_rim = stats.get("shadow_rim_ratio", 0.0)
    concrete_ratio = stats.get("bright_concrete_ratio", 1.0)

    lacks_concrete_apron = (
        (turf_ratio >= 0.50 and concrete_ratio < 0.45)
        or (shadow_rim >= 0.20 and concrete_ratio < 0.35)
        or (raw_pool_sqft <= 320.0 and turf_ratio >= 0.28 and concrete_ratio < 0.55)
    )
    if lacks_concrete_apron:
        return "above_ground"

    return "inground"


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


def _compute_mask_interior_point_area_and_circularity(
    poly_px: Optional[np.ndarray],
    bbox_xyxy_f: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float, float]:
    """
    Computes an interior point `(cx, cy)` guaranteed strictly inside the segmented pool polygon,
    exact polygon pixel area, Isoperimetric circularity `4 * pi * area / perimeter^2`, and
    convex-hull solidity `area / convex_hull.area`.
    """
    x1_f, y1_f, x2_f, y2_f = bbox_xyxy_f
    bw = max(x2_f - x1_f, 1.0)
    bh = max(y2_f - y1_f, 1.0)

    if poly_px is not None and len(poly_px) >= 3:
        try:
            poly = Polygon(poly_px)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty and poly.area >= 2.5:
                c = poly.centroid
                interior = c if poly.contains(c) else poly.representative_point()
                perim = max(float(poly.length), 1e-3)
                circ = float(min(max((4.0 * np.pi * float(poly.area)) / (perim * perim), 0.0), 1.0))
                hull_area = max(float(poly.convex_hull.area), float(poly.area), 1e-3)
                solidity = float(min(max(float(poly.area) / hull_area, 0.0), 1.0))
                return float(interior.x), float(interior.y), float(poly.area), circ, solidity
        except Exception:
            pass

    cx = (x1_f + x2_f) / 2.0
    cy = (y1_f + y2_f) / 2.0
    area_px = max(bw * bh * 0.785, 1.0)
    a, b = bw / 2.0, bh / 2.0
    approx_perim = np.pi * (3.0 * (a + b) - np.sqrt(max((3.0 * a + b) * (a + 3.0 * b), 1e-6)))
    circ = float(min(max((4.0 * np.pi * area_px) / (approx_perim ** 2 + 1e-6), 0.0), 1.0))
    return cx, cy, area_px, circ, 1.0


def _resolve_model_class_roles(names_obj: object) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Inspects YOLO class names (`dict` or `list`) and returns `(pool_cls_ids, above_ground_cls_ids, spa_cls_ids)`.
    Automatically supports future fine-tuned models with native `above_ground` / `above_ground_pool` classes.
    """
    if isinstance(names_obj, dict):
        name_map = {int(k): str(v) for k, v in names_obj.items()}
    elif isinstance(names_obj, (list, tuple)):
        name_map = {idx: str(v) for idx, v in enumerate(names_obj)}
    else:
        name_map = {0: "pool", 1: "spa"}

    pool_ids: Set[int] = set()
    above_ground_ids: Set[int] = set()
    spa_ids: Set[int] = set()

    for cid, raw_name in name_map.items():
        norm = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
        if "spa" in norm or "hot_tub" in norm or "jacuzzi" in norm:
            spa_ids.add(cid)
        elif "above" in norm:
            above_ground_ids.add(cid)
            pool_ids.add(cid)
        elif "pool" in norm or "inground" in norm or "in_ground" in norm:
            pool_ids.add(cid)

    if not pool_ids and 0 in name_map and 0 not in spa_ids:
        pool_ids.add(0)
    if not spa_ids and 1 in name_map and 1 not in pool_ids:
        spa_ids.add(1)

    return pool_ids, above_ground_ids, spa_ids


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
        self.pool_cls_ids, self.above_ground_cls_ids, self.spa_cls_ids = _resolve_model_class_roles(
            getattr(self.model, "names", None)
        )

    def analyze_tile(
        self,
        tile_path: Path,
        h3_index: str,
        buffer_meters: float = 15.0,
        include_pixel_geometry: bool = False,
    ) -> Tuple[List[dict], List[dict], np.ndarray]:
        """
        Runs YOLO segmentation at low threshold once using explicit RGB PIL input:
        - Detections >= conf_threshold passing 4-band NDWI / chromatic water check go to PostGIS.
        - Small circular above-ground pools in [0.38, conf_threshold) are recovered when 4-band NDWI
          and blue/cyan water ratios confirm liquid water.
        - Classifies every confirmed pool as `'inground'` vs `'above_ground'` (hybrid native YOLO
          class + 4-band spectral/geometric ring classifier).
        """
        img_rgb, nir_band, geo_ctx = load_geotiff_or_image(tile_path, h3_index)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        predict_conf = min(float(self.hard_neg_min), ABOVE_GROUND_RECOVERY_MIN_CONF, float(self.conf_threshold))
        preds = self.model.predict(
            source=Image.fromarray(img_rgb),
            imgsz=self.imgsz,
            conf=predict_conf,
            device=self.device,
            verbose=False,
        )[0]

        if preds.boxes is None or len(preds.boxes) == 0:
            return [], [], img_bgr

        pool_cls_ids, above_ground_cls_ids, spa_cls_ids = (
            self.pool_cls_ids,
            self.above_ground_cls_ids,
            self.spa_cls_ids,
        )
        if not pool_cls_ids and getattr(preds, "names", None):
            pool_cls_ids, above_ground_cls_ids, spa_cls_ids = _resolve_model_class_roles(preds.names)

        m_per_px = geo_ctx.meters_per_px
        sqft_per_px2 = (m_per_px ** 2) * 10.7639
        mask_polys = preds.masks.xy if preds.masks is not None else None

        pools: List[dict] = []
        spa_centers: List[Tuple[float, float]] = []
        hard_neg_boxes: List[dict] = []

        parsed_items = []
        for idx, box in enumerate(preds.boxes):
            cls_id = int(box.cls[0].item())
            conf = round(min(max(float(box.conf[0].item()), 0.0), 1.0), 3)
            x1_f, y1_f, x2_f, y2_f = [float(v) for v in box.xyxy[0].tolist()]
            poly_px = (
                mask_polys[idx]
                if mask_polys is not None and idx < len(mask_polys) and len(mask_polys[idx]) >= 3
                else None
            )
            xywhn = tuple(round(float(v), 6) for v in box.xywhn[0].tolist())
            parsed_items.append((cls_id, conf, (x1_f, y1_f, x2_f, y2_f), poly_px, xywhn))

        # Pass 1: Collect spa detections first so adjoining spas inform pool_type
        spa_min_conf = min(float(self.conf_threshold), 0.45)
        for cls_id, conf, bbox_f, poly_px, xywhn in parsed_items:
            if cls_id not in spa_cls_ids:
                continue
            if conf >= spa_min_conf:
                sx, sy, _s_area, _s_circ, _s_sol = _compute_mask_interior_point_area_and_circularity(poly_px, bbox_f)
                spa_centers.append((sx, sy))
            elif self.hard_neg_min <= conf <= self.hard_neg_max:
                hard_neg_boxes.append({"cls_id": cls_id, "conf": conf, "xywhn": xywhn})

        max_parcel_spa_dist_px = 20.0 / m_per_px
        adjoining_spa_dist_px = 12.0 / m_per_px
        dedup_pool_dist_px = 2.5 / m_per_px
        accepted_pool_centers: List[Tuple[float, float]] = []

        # Pass 2: Evaluate pool candidates (including [0.38, conf_threshold) above-ground recovery)
        for cls_id, conf, bbox_f, poly_px, xywhn in parsed_items:
            if cls_id not in pool_cls_ids:
                continue

            x1_f, y1_f, x2_f, y2_f = bbox_f
            x1, y1, x2, y2 = int(round(x1_f)), int(round(y1_f)), int(round(x2_f)), int(round(y2_f))
            bw_f = max(x2_f - x1_f, 1.0)
            bh_f = max(y2_f - y1_f, 1.0)
            aspect_ratio = bw_f / bh_f

            cx, cy, area_px, circularity, solidity = _compute_mask_interior_point_area_and_circularity(poly_px, bbox_f)
            if any(np.hypot(cx - ax, cy - ay) <= dedup_pool_dist_px for ax, ay in accepted_pool_centers):
                continue
            raw_pool_sqft = max(area_px, 1.0) * sqft_per_px2

            has_adjoining_spa = any(
                np.hypot(cx - sx, cy - sy) <= adjoining_spa_dist_px for sx, sy in spa_centers
            )
            has_parcel_spa = any(
                np.hypot(cx - sx, cy - sy) <= max_parcel_spa_dist_px for sx, sy in spa_centers
            )

            stats = evaluate_pool_water_and_context(img_rgb, nir_band, (x1, y1, x2, y2), poly_px=poly_px)
            native_type = "above_ground" if cls_id in above_ground_cls_ids else None

            if conf < self.conf_threshold:
                # Task 2.1.b: Recover small circular above-ground pools in [0.38, conf_threshold)
                nir_water_ok = (stats.get("ndwi_mean", 0.0) >= 0.10) if stats.get("has_nir", 0.0) > 0.5 else True
                recovered_ag = (
                    conf >= ABOVE_GROUND_RECOVERY_MIN_CONF
                    and stats.get("valid", 0.0) > 0.5
                    and 100.0 <= raw_pool_sqft <= 600.0
                    and 0.78 <= aspect_ratio <= 1.28
                    and nir_water_ok
                    and stats.get("blue_cyan_ratio", 0.0) >= 0.55
                )
                if recovered_ag:
                    candidate_type = classify_pool_type(
                        raw_pool_sqft=raw_pool_sqft,
                        circularity=circularity,
                        aspect_ratio=aspect_ratio,
                        has_adjacent_spa=has_adjoining_spa,
                        stats=stats,
                        solidity=solidity,
                        forced_native_type=native_type,
                    )
                    if candidate_type != "above_ground":
                        recovered_ag = False

                if not recovered_ag:
                    if self.hard_neg_min <= conf <= self.hard_neg_max:
                        hard_neg_boxes.append({"cls_id": cls_id, "conf": conf, "xywhn": xywhn})
                    continue
                pool_type = "above_ground"
            else:
                if not is_spectrally_valid_pool(
                    img_rgb, nir_band, (x1, y1, x2, y2), conf, poly_px=poly_px, stats=stats
                ):
                    if self.hard_neg_min <= conf <= 0.72:
                        hard_neg_boxes.append({"cls_id": cls_id, "conf": conf, "xywhn": xywhn})
                    continue

                pool_type = classify_pool_type(
                    raw_pool_sqft=raw_pool_sqft,
                    circularity=circularity,
                    aspect_ratio=aspect_ratio,
                    has_adjacent_spa=has_adjoining_spa,
                    stats=stats,
                    solidity=solidity,
                    forced_native_type=native_type,
                )

            accepted_pool_centers.append((cx, cy))
            lng, lat = geo_ctx.pixel_to_lnglat(cx, cy)
            ai_pool_sqft = int(round(raw_pool_sqft))
            canopy_pct = estimate_canopy_pct_around_pool(
                img_rgb, nir_band, (x1, y1, x2, y2), m_per_px, buffer_meters
            )
            condition = classify_pool_water_condition(img_rgb, (x1, y1, x2, y2), poly_px=poly_px)

            pool_record = {
                "h3_index_res9": h3_index,
                "lat": round(lat, 7),
                "lng": round(lng, 7),
                "pool_type": pool_type,
                "ai_pool_confidence": conf,
                "ai_pool_condition": condition,
                "ai_pool_sqft": max(min(ai_pool_sqft, 2500), 100),
                "ai_has_spa": bool(has_parcel_spa),
                "ai_tree_canopy_pct": canopy_pct,
                "ai_tree_overhang": bool(canopy_pct >= 35.0),
            }
            if include_pixel_geometry:
                pool_record["center_px"] = (round(cx, 2), round(cy, 2))
                pool_record["bbox_px"] = (x1, y1, x2, y2)
                pool_record["circularity"] = round(circularity, 3)
                pool_record["aspect_ratio"] = round(aspect_ratio, 3)
                pool_record["ndwi_mean"] = round(stats.get("ndwi_mean", 0.0), 3)
                pool_record["blue_cyan_ratio"] = round(stats.get("blue_cyan_ratio", 0.0), 3)
                pool_record["turf_ratio"] = round(stats.get("turf_ratio", 0.0), 3)
                pool_record["shadow_rim_ratio"] = round(stats.get("shadow_rim_ratio", 0.0), 3)
                pool_record["bright_concrete_ratio"] = round(stats.get("bright_concrete_ratio", 0.0), 3)
                pool_record["mask_xy"] = (
                    [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in poly_px]
                    if poly_px is not None
                    else None
                )
            pools.append(pool_record)

        return pools, hard_neg_boxes, img_bgr
