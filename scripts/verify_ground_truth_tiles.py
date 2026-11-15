#!/usr/bin/env python3
"""
CLI verification script against known Stage ground-truth H3 res-9 NAIP tiles:
  - 8926c8698cfffff (Tarrant - 7824 Hidden Oaks Dr vs 7820 Hidden Oaks Dr)
  - 8926c8689cbffff (Tarrant - 2908 Scarborough Ln W vs 2910 Scarborough Ln W)
  - 8926c878967ffff (Denton  - 3601 Sandhurst Dr pergola/patio check)

Pulls GeoTIFFs from s3://growroute-naip-tiles (falling back to NAIP STAC via S3TileLoader
if a tile has not yet been cached in S3), runs PoolSpaCanopyDetector (.onnx on EC2/Docker
or .pt fallback when onnxruntime is unavailable locally), saves annotated PNGs, and prints
Google Maps pin links + confidence scores for every detection.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple

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

from src.detectors.pool_canopy import PoolSpaCanopyDetector  # noqa: E402
from src.io.s3_tile_loader import S3TileLoader  # noqa: E402

DEFAULT_GROUND_TRUTH_TILES: List[Tuple[str, str, str]] = [
    ("tarrant", "8926c8698cfffff", "Tarrant — 7824 Hidden Oaks Dr (pool) vs 7820 Hidden Oaks Dr"),
    ("tarrant", "8926c8689cbffff", "Tarrant — 2908 Scarborough Ln W (pool) vs 2910 Scarborough Ln W"),
    ("denton", "8926c878967ffff", "Denton  — 3601 Sandhurst Dr (pergola/patio false-positive check)"),
]


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
            s3 = boto3.client("s3", region_name=aws_region)
            s3_key = f"models/{model_path.name}"
            try:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[Model] Downloading s3://{s3_data_bucket}/{s3_key} -> {model_path}...")
                s3.download_file(s3_data_bucket, s3_key, str(model_path))
            except Exception as exc:
                print(f"[Model Warn] Could not download {s3_key} from S3: {exc}")

        if model_path.exists() and has_onnxruntime:
            return model_path

        for pt_name in ("pool_spa_dfw_v2.pt", "pool-spa-dfw-v2.pt"):
            pt_candidate = REPO_ROOT / "models" / pt_name
            if pt_candidate.exists():
                reason = "onnxruntime not installed locally" if not has_onnxruntime else f"{model_path.name} missing"
                print(f"[Model] {reason}; using local PyTorch checkpoint {pt_candidate.name} (identical RGB/LetterBox pipeline).")
                return pt_candidate

    if model_path.exists():
        return model_path

    raise FileNotFoundError(f"Model weights not found at {model_path}")


def parse_tile_arg(raw: str) -> Tuple[str, str, str]:
    if ":" in raw:
        county, cell = raw.split(":", 1)
        return county.strip().lower(), cell.strip(), f"{county.strip().title()} — {cell.strip()}"
    return "dfw", raw.strip(), f"Custom — {raw.strip()}"


def draw_h3_boundary_on_geotiff(img_bgr: np.ndarray, tile_path: Path, h3_cell: str) -> None:
    if tile_path.suffix.lower() not in {".tif", ".tiff"}:
        return
    try:
        with rasterio.open(tile_path) as src:
            if src.crs is None:
                return
            to_crs = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            inv_affine = ~src.transform
            boundary_lnglat = (
                [(float(pt[0]), float(pt[1])) for pt in h3.h3_to_geo_boundary(h3_cell, geo_json=True)]
                if hasattr(h3, "h3_to_geo_boundary")
                else [(float(latlng[1]), float(latlng[0])) for latlng in h3.cell_to_boundary(h3_cell)]
            )
            pts_px = []
            for lng, lat in boundary_lnglat:
                x_crs, y_crs = to_crs.transform(float(lng), float(lat))
                col_x, row_y = inv_affine * (x_crs, y_crs)
                pts_px.append([int(round(col_x)), int(round(row_y))])
            if len(pts_px) >= 3:
                arr = np.array(pts_px, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_bgr, [arr], isClosed=True, color=(0, 215, 255), thickness=2, lineType=cv2.LINE_AA)
    except Exception:
        pass


def annotate_detections(
    img_bgr: np.ndarray,
    tile_path: Path,
    h3_cell: str,
    pools: List[dict],
) -> np.ndarray:
    canvas = img_bgr.copy()
    overlay = img_bgr.copy()

    draw_h3_boundary_on_geotiff(canvas, tile_path, h3_cell)
    draw_h3_boundary_on_geotiff(overlay, tile_path, h3_cell)

    for idx, det in enumerate(pools, start=1):
        is_green = det.get("ai_pool_condition") == "green"
        mask_color = (40, 210, 90) if is_green else (255, 195, 0)  # BGR: green vs cyan-blue
        pin_color = (0, 0, 255)  # crisp red pin crosshair

        mask_xy = det.get("mask_xy")
        bbox_px = det.get("bbox_px")
        center_px: Optional[Tuple[float, float]] = det.get("center_px")

        if mask_xy and len(mask_xy) >= 3:
            pts = np.round(np.array(mask_xy, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color=mask_color)
            cv2.polylines(canvas, [pts], isClosed=True, color=mask_color, thickness=2, lineType=cv2.LINE_AA)
        elif bbox_px:
            x1, y1, x2, y2 = bbox_px
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=mask_color, thickness=2)

        if center_px is not None:
            cx_i, cy_i = int(round(center_px[0])), int(round(center_px[1]))
            cv2.circle(canvas, (cx_i, cy_i), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (cx_i, cy_i), 3, pin_color, -1, lineType=cv2.LINE_AA)
            cv2.drawMarker(
                canvas,
                (cx_i, cy_i),
                pin_color,
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=1,
                line_type=cv2.LINE_AA,
            )

            label = f"#{idx} {det['ai_pool_confidence']:.2f} {det['ai_pool_condition'][0].upper()}"
            tx = min(max(cx_i + 7, 4), canvas.shape[1] - 95)
            ty = min(max(cy_i - 7, 16), canvas.shape[0] - 6)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0.0)
    return blended


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Verify fixed pool detector + georeferencing against known Stage ground-truth H3 tiles"
    )
    parser.add_argument(
        "--model",
        default="models/pool_spa_dfw_v2.onnx",
        help="Path to .onnx or .pt model weights (default: models/pool_spa_dfw_v2.onnx)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device: cpu, mps, or 0 (default: cpu)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.55,
        help="Pool confidence threshold (default: 0.55)",
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
        help="Local directory to cache fetched .tif tiles (default: /tmp/growroute_gt_tiles)",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/growroute_gt_verify",
        help="Directory to write annotated PNGs and summary JSON (default: /tmp/growroute_gt_verify)",
    )
    parser.add_argument(
        "--tiles",
        nargs="*",
        default=None,
        help="Optional override list of '<county>:<h3_cell>' or '<h3_cell>' items",
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

    targets = (
        [parse_tile_arg(t) for t in args.tiles]
        if args.tiles
        else DEFAULT_GROUND_TRUTH_TILES
    )

    summary_report = []

    print(
        f"\n=== GrowRoute Ground-Truth Tile Verification ===\n"
        f"Model  : {weights_path}\n"
        f"Device : {args.device} | conf_threshold={args.conf}\n"
        f"Bucket : s3://{args.tile_bucket}/tx/h3_res9/\n"
        f"Output : {out_dir}\n"
    )

    for county_slug, h3_cell, label in targets:
        fetched = loader.fetch_batch([h3_cell], county=county_slug)
        tile_path = fetched.get(h3_cell)
        if tile_path is None:
            print(f"[ERROR] Could not fetch tile {h3_cell} ({label}) from S3 or NAIP STAC.")
            continue

        pools, hard_negs, img_bgr = detector.analyze_tile(
            tile_path=tile_path,
            h3_index=h3_cell,
            include_pixel_geometry=True,
        )
        pools_sorted = sorted(pools, key=lambda d: d["ai_pool_confidence"], reverse=True)

        annotated = annotate_detections(img_bgr, tile_path, h3_cell, pools_sorted)
        png_path = out_dir / f"{county_slug}_{h3_cell}_verified.png"
        cv2.imwrite(str(png_path), annotated)

        print(f"------------------------------------------------------------------------")
        print(f"Tile   : {h3_cell} ({label})")
        print(f"Source : {tile_path} ({img_bgr.shape[1]}x{img_bgr.shape[0]} px)")
        print(f"PNG    : {png_path}")
        print(f"Found  : {len(pools_sorted)} confirmed pool(s) (>= {args.conf:.2f}) | {len(hard_negs)} rejected/borderline")

        tile_detections_out = []
        for idx, det in enumerate(pools_sorted, start=1):
            lat = det["lat"]
            lng = det["lng"]
            gmaps_url = f"https://www.google.com/maps?q={lat:.7f},{lng:.7f}"
            print(
                f"  #{idx:02d}  conf={det['ai_pool_confidence']:.3f}  "
                f"cond={det['ai_pool_condition']:<5}  sqft={det['ai_pool_sqft']:>4}  "
                f"canopy={det['ai_tree_canopy_pct']:>5.1f}%  ->  {gmaps_url}"
            )
            tile_detections_out.append({
                "rank": idx,
                "lat": lat,
                "lng": lng,
                "google_maps_url": gmaps_url,
                "ai_pool_confidence": det["ai_pool_confidence"],
                "ai_pool_condition": det["ai_pool_condition"],
                "ai_pool_sqft": det["ai_pool_sqft"],
                "ai_has_spa": det["ai_has_spa"],
                "ai_tree_canopy_pct": det["ai_tree_canopy_pct"],
            })

        summary_report.append({
            "county": county_slug,
            "h3_index_res9": h3_cell,
            "label": label,
            "annotated_png": str(png_path),
            "pool_count": len(pools_sorted),
            "detections": tile_detections_out,
        })

    json_path = out_dir / "ground_truth_verification.json"
    json_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print(f"------------------------------------------------------------------------")
    print(f"Saved JSON report: {json_path}\n")


if __name__ == "__main__":
    main()
