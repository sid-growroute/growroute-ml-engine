#!/usr/bin/env python3
"""
Ground-truth verification script for Above-Ground vs. In-Ground pool detection & classification
across 5 verified Tarrant County coordinates:
  Known Above-Ground Pools (must detect & classify `above_ground`):
    1. (32.9134911, -97.3268845)
    2. (32.9127418, -97.3279366)
    3. (32.5819864, -97.1639825)
  Known In-Ground Control Pools (must detect & classify `inground`):
    4. (32.96585, -97.26835)   (13008 Berrywood Tr area)
    5. (32.96615, -97.26865)   (13013 Berrywood Tr area)

Converts each (lat, lon) to h3_index_res9, fetches/caches 4-band NAIP GeoTIFFs via S3TileLoader,
runs CPU .onnx inference (with .pt CPU fallback when onnxruntime is unavailable on host OS),
saves annotated PNGs to /tmp/growroute_ag_check/, and prints (lat, lon), pool_type, confidence,
pool_sqft, and clickable Google Maps links.
"""

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import boto3
import cv2
from dotenv import load_dotenv
import h3
import numpy as np
from pyproj import Transformer
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from src.detectors.pool_canopy import PoolSpaCanopyDetector  # noqa: E402
from src.io.s3_tile_loader import S3TileLoader  # noqa: E402


@dataclass(frozen=True)
class GroundTruthCoordinate:
    label: str
    lat: float
    lon: float
    expected_pool_type: str
    county_slug: str = "tarrant"
    max_match_m: Optional[float] = None


GROUND_TRUTH_COORDINATES: List[GroundTruthCoordinate] = [
    GroundTruthCoordinate("Tarrant AG #1", 32.9134911, -97.3268845, "above_ground"),
    GroundTruthCoordinate("Tarrant AG #2", 32.9127418, -97.3279366, "above_ground"),
    GroundTruthCoordinate("Tarrant AG #3", 32.5819864, -97.1639825, "above_ground"),
    GroundTruthCoordinate("Tarrant IG Control #4 (13008 Berrywood Tr)", 32.96585, -97.26835, "inground", max_match_m=80.0),
    GroundTruthCoordinate("Tarrant IG Control #5 (13013 Berrywood Tr)", 32.96615, -97.26865, "inground", max_match_m=80.0),
]


def latlon_to_h3_res9(lat: float, lon: float) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return str(h3.latlng_to_cell(lat, lon, 9))
    return str(h3.geo_to_h3(lat, lon, 9))


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_m = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r_m * math.asin(math.sqrt(min(max(a, 0.0), 1.0)))


def resolve_weights_path(
    requested_model: str,
    s3_data_bucket: str,
    aws_region: str,
) -> Path:
    model_path = Path(requested_model)
    if not model_path.is_absolute():
        model_path = (REPO_ROOT / model_path).resolve()

    has_onnxruntime = importlib.util.find_spec("onnxruntime") is not None

    if model_path.suffix.lower() == ".onnx":
        if not model_path.exists() and s3_data_bucket:
            try:
                s3 = boto3.client("s3", region_name=aws_region)
                s3_key = f"models/{model_path.name}"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(s3_data_bucket, s3_key, str(model_path))
            except Exception:
                pass

        if model_path.exists() and has_onnxruntime:
            return model_path

        for pt_name in ("pool_spa_dfw_v2.pt", "pool-spa-dfw-v2.pt"):
            pt_candidate = REPO_ROOT / "models" / pt_name
            if pt_candidate.exists():
                reason = "onnxruntime not installed locally" if not has_onnxruntime else f"{model_path.name} missing"
                print(f"[Model] {reason}; using local PyTorch checkpoint {pt_candidate.name} on CPU.")
                return pt_candidate

    if model_path.exists():
        return model_path

    raise FileNotFoundError(f"Model weights not found at {model_path}")


def lnglat_to_pixel(tile_path: Path, lon: float, lat: float) -> Optional[Tuple[int, int]]:
    if tile_path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    try:
        with rasterio.open(tile_path) as src:
            if src.crs is None:
                return None
            if str(src.crs).upper() in ("EPSG:4326", "WGS84", "OGC:CRS84"):
                x_crs, y_crs = lon, lat
            else:
                to_crs = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x_crs, y_crs = to_crs.transform(float(lon), float(lat))
            col_x, row_y = (~src.transform) * (x_crs, y_crs)
            return int(round(col_x)), int(round(row_y))
    except Exception:
        return None


def annotate_tile(
    img_bgr: np.ndarray,
    tile_path: Path,
    h3_cell: str,
    pools: List[dict],
    targets_in_cell: List[GroundTruthCoordinate],
) -> np.ndarray:
    canvas = img_bgr.copy()
    overlay = img_bgr.copy()

    # Draw H3 boundary in gold
    try:
        with rasterio.open(tile_path) as src:
            if src.crs is not None:
                to_crs = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                inv_affine = ~src.transform
                boundary_lnglat = (
                    [(float(pt[0]), float(pt[1])) for pt in h3.h3_to_geo_boundary(h3_cell, geo_json=True)]
                    if hasattr(h3, "h3_to_geo_boundary")
                    else [(float(latlng[1]), float(latlng[0])) for latlng in h3.cell_to_boundary(h3_cell)]
                )
                pts_px = []
                for lng_b, lat_b in boundary_lnglat:
                    x_c, y_c = to_crs.transform(lng_b, lat_b)
                    cx_b, cy_b = inv_affine * (x_c, y_c)
                    pts_px.append([int(round(cx_b)), int(round(cy_b))])
                if len(pts_px) >= 3:
                    arr = np.array(pts_px, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [arr], True, (0, 215, 255), 2, cv2.LINE_AA)
                    cv2.polylines(overlay, [arr], True, (0, 215, 255), 2, cv2.LINE_AA)
    except Exception:
        pass

    # Draw pool detections: magenta/orange for above_ground, cyan-blue for inground
    for idx, det in enumerate(pools, start=1):
        p_type = det.get("pool_type", "inground")
        color = (0, 140, 255) if p_type == "above_ground" else (255, 195, 0)
        mask_xy = det.get("mask_xy")
        bbox_px = det.get("bbox_px")
        center_px = det.get("center_px")

        if mask_xy and len(mask_xy) >= 3:
            pts = np.round(np.array(mask_xy, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color=color)
            cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        elif bbox_px:
            x1, y1, x2, y2 = bbox_px
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        if center_px is not None:
            cx_i, cy_i = int(round(center_px[0])), int(round(center_px[1]))
            cv2.circle(canvas, (cx_i, cy_i), 3, (0, 0, 255), -1, cv2.LINE_AA)
            short_tag = "AG" if p_type == "above_ground" else "IG"
            label = f"{short_tag} {det['ai_pool_confidence']:.2f} ({det['ai_pool_sqft']}sf)"
            tx = min(max(cx_i + 6, 4), max(canvas.shape[1] - 135, 4))
            ty = min(max(cy_i - 6, 16), canvas.shape[0] - 6)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Mark target ground-truth coordinate pins in bright lime green ring
    for gt in targets_in_cell:
        px = lnglat_to_pixel(tile_path, gt.lon, gt.lat)
        if px is not None:
            cv2.circle(canvas, px, 11, (80, 255, 80), 2, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0.0)


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Verify Above-Ground vs. In-Ground pool detection & classification on 5 Tarrant ground-truth coordinates"
    )
    parser.add_argument(
        "--model",
        default="models/pool_spa_dfw_v2.onnx",
        help="Path to .onnx (preferred) or .pt weights (default: models/pool_spa_dfw_v2.onnx)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.55,
        help="Standard pool confidence threshold (default: 0.55; [0.38, 0.55) AG recovery enabled)",
    )
    parser.add_argument(
        "--match-radius-m",
        type=float,
        default=22.0,
        help="Max distance (meters) from ground-truth coordinate to match a detected pool (default: 22m)",
    )
    parser.add_argument(
        "--tile-bucket",
        default=os.getenv("S3_TILE_BUCKET", "growroute-naip-tiles"),
        help="S3 bucket for cached H3 GeoTIFF tiles (default: growroute-naip-tiles)",
    )
    parser.add_argument(
        "--data-bucket",
        default=os.getenv("S3_DATA_BUCKET", "growroute-stage-data"),
        help="S3 data bucket for fallback ONNX download (default: growroute-stage-data)",
    )
    parser.add_argument(
        "--cache-dir",
        default="/tmp/growroute_gt_tiles",
        help="Local cache directory for fetched .tif tiles (default: /tmp/growroute_gt_tiles)",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/growroute_ag_check",
        help="Directory for annotated PNGs & JSON report (default: /tmp/growroute_ag_check)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any of the 5 ground-truth coordinates is missed or misclassified",
    )
    args = parser.parse_args()

    aws_region = os.getenv("AWS_REGION", "us-west-2").strip() or "us-west-2"
    weights_path = resolve_weights_path(args.model, args.data_bucket, aws_region)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    detector = PoolSpaCanopyDetector(
        weights_path=str(weights_path),
        device=args.device,
        conf_threshold=args.conf,
    )
    loader = S3TileLoader(
        bucket=args.tile_bucket,
        s3_prefix="tx/h3_res9",
        cache_dir=str(cache_dir),
        file_ext=".tif",
        write_through_s3=False,
        enable_public_naip_fallback=True,
    )

    cell_to_targets: Dict[str, List[GroundTruthCoordinate]] = {}
    for gt in GROUND_TRUTH_COORDINATES:
        cell = latlon_to_h3_res9(gt.lat, gt.lon)
        cell_to_targets.setdefault(cell, []).append(gt)

    unique_cells = list(cell_to_targets.keys())
    fetched_tiles = loader.fetch_batch(unique_cells, county="tarrant")

    cell_detections: Dict[str, List[dict]] = {}
    cell_pngs: Dict[str, str] = {}

    print(
        f"\n=== GrowRoute Above-Ground vs. In-Ground Pool Verification ===\n"
        f"Model  : {weights_path}\n"
        f"Device : {args.device} | conf_threshold={args.conf} (AG recovery >= 0.38)\n"
        f"Output : {out_dir}\n"
    )

    for h3_cell in unique_cells:
        tile_path = fetched_tiles.get(h3_cell)
        if tile_path is None:
            print(f"[ERROR] Failed to fetch NAIP tile for H3 cell {h3_cell}")
            cell_detections[h3_cell] = []
            continue

        pools, _hard_negs, img_bgr = detector.analyze_tile(
            tile_path=tile_path,
            h3_index=h3_cell,
            include_pixel_geometry=True,
        )
        pools_sorted = sorted(pools, key=lambda d: d["ai_pool_confidence"], reverse=True)
        cell_detections[h3_cell] = pools_sorted

        annotated = annotate_tile(img_bgr, tile_path, h3_cell, pools_sorted, cell_to_targets[h3_cell])
        png_path = out_dir / f"tarrant_{h3_cell}_ag_check.png"
        cv2.imwrite(str(png_path), annotated)
        cell_pngs[h3_cell] = str(png_path)

    results = []
    passed_count = 0
    used_det_indices: Dict[str, set] = {}

    print("-" * 112)
    for idx, gt in enumerate(GROUND_TRUTH_COORDINATES, start=1):
        h3_cell = latlon_to_h3_res9(gt.lat, gt.lon)
        candidates = cell_detections.get(h3_cell, [])
        claimed = used_det_indices.setdefault(h3_cell, set())
        best_det: Optional[dict] = None
        best_det_idx: Optional[int] = None
        best_dist_m = float("inf")

        for det_i, det in enumerate(candidates):
            if det_i in claimed:
                continue
            dist_m = haversine_meters(gt.lat, gt.lon, float(det["lat"]), float(det["lng"]))
            if dist_m < best_dist_m:
                best_dist_m = dist_m
                best_det = det
                best_det_idx = det_i

        allowed_radius_m = float(gt.max_match_m) if gt.max_match_m is not None else float(args.match_radius_m)
        matched = best_det is not None and best_dist_m <= allowed_radius_m
        if matched and best_det_idx is not None:
            claimed.add(best_det_idx)
        actual_type = best_det.get("pool_type") if matched else None
        conf = float(best_det["ai_pool_confidence"]) if matched else 0.0
        sqft = int(best_det["ai_pool_sqft"]) if matched else 0
        det_lat = float(best_det["lat"]) if matched else gt.lat
        det_lon = float(best_det["lng"]) if matched else gt.lon
        gmaps_url = f"https://www.google.com/maps?q={det_lat:.7f},{det_lon:.7f}"

        ok = bool(matched and actual_type == gt.expected_pool_type)
        if ok:
            passed_count += 1
        status_str = "PASS" if ok else "FAIL"

        print(
            f"[{status_str}] #{idx} {gt.label}\n"
            f"       target=({gt.lat:.7f}, {gt.lon:.7f}) | h3={h3_cell} | expected={gt.expected_pool_type}\n"
            f"       detected=({det_lat:.7f}, {det_lon:.7f}) | dist={best_dist_m if matched else -1.0:.1f}m | "
            f"pool_type={actual_type} | conf={conf:.3f} | pool_sqft={sqft} | {gmaps_url}"
        )
        if matched:
            print(
                f"       metrics: circ={best_det.get('circularity')} aspect={best_det.get('aspect_ratio')} "
                f"ndwi={best_det.get('ndwi_mean')} blue_cyan={best_det.get('blue_cyan_ratio')} "
                f"turf={best_det.get('turf_ratio')} shadow_rim={best_det.get('shadow_rim_ratio')} "
                f"concrete={best_det.get('bright_concrete_ratio')} | png={cell_pngs.get(h3_cell)}"
            )

        results.append({
            "index": idx,
            "label": gt.label,
            "target_lat": gt.lat,
            "target_lon": gt.lon,
            "h3_index_res9": h3_cell,
            "expected_pool_type": gt.expected_pool_type,
            "matched": matched,
            "distance_m": round(best_dist_m, 2) if matched else None,
            "pool_type": actual_type,
            "confidence": conf,
            "pool_sqft": sqft,
            "detected_lat": det_lat if matched else None,
            "detected_lon": det_lon if matched else None,
            "google_maps_url": gmaps_url,
            "annotated_png": cell_pngs.get(h3_cell),
            "passed": ok,
        })

    print("-" * 112)
    report_path = out_dir / "above_ground_verification.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        f"Summary: {passed_count}/{len(GROUND_TRUTH_COORDINATES)} passed | "
        f"PNGs & JSON saved to {out_dir}\n"
    )

    if args.strict and passed_count != len(GROUND_TRUTH_COORDINATES):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
