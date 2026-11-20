from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from training.config import TrainingConfig
from training.naip_fetcher import ExtractedTileRecord, create_boto_session

MANIFEST_PARQUET_SCHEMA = pa.schema([
    ("tile_id", pa.string()),
    ("county", pa.string()),
    ("split", pa.string()),
    ("h3_index", pa.string()),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("naip_year", pa.int32()),
    ("source_cog", pa.string()),
    ("crs_epsg", pa.int32()),
    ("gsd_m", pa.float64()),
    ("window_transform_6", pa.list_(pa.float64())),
    ("s3_image_key", pa.string()),
    ("parcels_px", pa.string()),
])


def list_existing_s3_image_keys(
    cfg: TrainingConfig,
    run_id: str,
) -> set[str]:
    """
    Lists existing S3 object keys under `s3://{bucket}/{prefix}/{run_id}/images/` so
    interrupted `fetch-naip` runs resume idempotently without re-uploading JPEGs.
    """
    session = create_boto_session(cfg)
    s3 = session.client("s3", region_name=cfg.aws_region)
    prefix = f"{cfg.training_s3_prefix}/{run_id}/images/"
    found: set[str] = set()
    continuation: Optional[str] = None

    while True:
        kwargs: Dict[str, Any] = {
            "Bucket": cfg.training_s3_bucket,
            "Prefix": prefix,
        }
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = str(obj.get("Key") or "")
            if key and int(obj.get("Size", 0)) > 0:
                found.add(key)
        if not resp.get("IsTruncated"):
            break
        continuation = resp.get("NextContinuationToken")

    return found


def load_cached_local_manifest(cache_dir: Path) -> Dict[str, Dict[str, Any]]:
    manifest_path = cache_dir / "manifest.parquet"
    if not manifest_path.is_file():
        return {}
    try:
        tbl = pq.read_table(str(manifest_path))
        return {str(r["tile_id"]): r for r in tbl.to_pylist() if r.get("tile_id")}
    except Exception:
        return {}


def encode_rgb_jpeg_bytes(img_rgb: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_rgb, mode="RGB").save(
        buf,
        format="JPEG",
        quality=int(quality),
        subsampling=0,
    )
    return buf.getvalue()


def render_qa_overlay_image(
    img_rgb: np.ndarray,
    parcels_px: Sequence[Dict[str, Any]],
    header_label: Optional[str] = None,
) -> Image.Image:
    """
    Renders parcel boundary overlays on top of a `640x640` NAIP RGB chip:
      - Negative lots (`cad_has_pool=false`, `cad_has_spa=false`): thin gray `(165, 175, 185)`
      - CAD pool lots (`cad_has_pool=true`): bright lime `(150, 255, 40)`
      - CAD spa lots (`cad_has_spa=true`): bright cyan `(0, 235, 255)` (inner cyan + outer lime if both)
    """
    canvas = Image.fromarray(img_rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    neg_parcels = [
        p for p in parcels_px if not p.get("cad_has_pool") and not p.get("cad_has_spa")
    ]
    pool_parcels = [p for p in parcels_px if p.get("cad_has_pool")]
    spa_only_parcels = [
        p for p in parcels_px if p.get("cad_has_spa") and not p.get("cad_has_pool")
    ]

    # 1. Negative lots first (thin gray outline)
    for p in neg_parcels:
        for ring in p.get("pixel_rings") or []:
            if len(ring) >= 3:
                pts = [(float(pt[0]), float(pt[1])) for pt in ring]
                draw.line(pts, fill=(170, 180, 190, 165), width=1)

    # 2. CAD spa-only lots (cyan outline)
    for p in spa_only_parcels:
        for ring in p.get("pixel_rings") or []:
            if len(ring) >= 3:
                pts = [(float(pt[0]), float(pt[1])) for pt in ring]
                draw.polygon(pts, fill=(0, 235, 255, 22))
                draw.line(pts, fill=(0, 235, 255, 245), width=2)

    # 3. CAD pool lots on top (lime outline; dual lime+cyan stroke when cad_has_spa is also true)
    for p in pool_parcels:
        has_spa = bool(p.get("cad_has_spa"))
        for ring in p.get("pixel_rings") or []:
            if len(ring) >= 3:
                pts = [(float(pt[0]), float(pt[1])) for pt in ring]
                draw.polygon(pts, fill=(150, 255, 40, 26))
                if has_spa:
                    draw.line(pts, fill=(150, 255, 40, 250), width=3)
                    draw.line(pts, fill=(0, 235, 255, 250), width=1)
                else:
                    draw.line(pts, fill=(150, 255, 40, 245), width=2)

    if header_label:
        bar_w = min(canvas.width - 8, max(260, len(header_label) * 6 + 14))
        draw.rectangle([(4, 4), (4 + bar_w, 22)], fill=(11, 15, 23, 210))
        draw.text((9, 7), header_label, fill=(240, 248, 255, 255))

    composited = Image.alpha_composite(canvas, overlay).convert("RGB")
    return composited


def _select_diverse_qa_records(
    records: Sequence[ExtractedTileRecord],
    limit: int,
) -> List[ExtractedTileRecord]:
    """
    Selects up to `limit` tiles round-robin across `(split, county)` groups (prioritizing tiles
    that have freshly fetched `img_rgb` bytes) so QA overlays cover `val`, `test`, and diverse
    `train` counties.
    """
    if limit <= 0:
        return []

    by_group: Dict[Tuple[int, str, str], List[ExtractedTileRecord]] = defaultdict(list)
    split_priority = {"val": 0, "test": 1, "train": 2}

    for r in records:
        if r.img_rgb is None:
            continue
        grp = (split_priority.get(r.split, 3), r.split, r.county)
        by_group[grp].append(r)

    ordered_groups = [by_group[k] for k in sorted(by_group.keys())]
    selected: List[ExtractedTileRecord] = []
    idx = 0
    while len(selected) < limit:
        progressed = False
        for grp_list in ordered_groups:
            if idx < len(grp_list):
                selected.append(grp_list[idx])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        idx += 1

    return selected


def export_dataset_to_s3(
    cfg: TrainingConfig,
    run_id: str,
    records: Sequence[ExtractedTileRecord],
    existing_s3_keys: Optional[set[str]] = None,
    qa_overlay_limit: int = 50,
    upload_workers: int = 16,
) -> Dict[str, Any]:
    """
    Phase 2b:
      1. Uploads clean RGB JPEGs (`quality=92`) to `s3://{bucket}/{prefix}/{run_id}/images/...`,
         skipping any keys already present on S3.
      2. Renders up to `qa_overlay_limit` QA debug overlays locally and uploads them to
         `s3://{bucket}/{prefix}/{run_id}/qa_overlays/...`.
      3. Writes `manifest.parquet` locally (`.cache/training/{run_id}/manifest.parquet`) and
         uploads it to `s3://{bucket}/{prefix}/{run_id}/manifest.parquet`.
    """
    cache_dir = cfg.run_cache_dir(run_id)
    qa_local_dir = cache_dir / "qa_overlays"
    qa_local_dir.mkdir(parents=True, exist_ok=True)

    already_uploaded = existing_s3_keys or set()
    session = create_boto_session(cfg)
    bucket = cfg.training_s3_bucket
    prefix = cfg.training_s3_prefix

    to_upload_images = [
        r for r in records if r.img_rgb is not None and r.s3_image_key not in already_uploaded
    ]
    skipped_existing = len(records) - len(to_upload_images)

    def _upload_image(rec: ExtractedTileRecord) -> str:
        assert rec.img_rgb is not None
        s3 = session.client("s3", region_name=cfg.aws_region)
        jpg_bytes = encode_rgb_jpeg_bytes(rec.img_rgb, quality=92)
        s3.put_object(
            Bucket=bucket,
            Key=rec.s3_image_key,
            Body=jpg_bytes,
            ContentType="image/jpeg",
        )
        return rec.s3_image_key

    uploaded_count = 0
    if to_upload_images:
        pool_size = max(1, min(int(upload_workers), len(to_upload_images)))
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            futs = [pool.submit(_upload_image, r) for r in to_upload_images]
            for fut in as_completed(futs):
                fut.result()
                uploaded_count += 1

    # Render & upload QA debug overlays
    qa_targets = _select_diverse_qa_records(records, limit=max(0, int(qa_overlay_limit)))
    local_qa_paths: List[Path] = []
    s3_qa_keys: List[str] = []

    def _render_and_upload_qa(rec: ExtractedTileRecord) -> Tuple[Path, str]:
        assert rec.img_rgb is not None
        n_pool = sum(1 for p in rec.parcels_px_list if p.get("cad_has_pool"))
        n_spa = sum(1 for p in rec.parcels_px_list if p.get("cad_has_spa"))
        n_neg = len(rec.parcels_px_list) - n_pool
        label = (
            f"{rec.county} [{rec.split}] {rec.h3_index} ({rec.naip_year}) | "
            f"pool={n_pool} spa={n_spa} neg={n_neg}"
        )
        qa_img = render_qa_overlay_image(rec.img_rgb, rec.parcels_px_list, header_label=label)
        local_qa_path = qa_local_dir / f"{rec.tile_id}_qa.jpg"
        buf = io.BytesIO()
        qa_img.save(buf, format="JPEG", quality=92, subsampling=0)
        jpg_bytes = buf.getvalue()
        local_qa_path.write_bytes(jpg_bytes)

        s3_qa_key = f"{prefix}/{run_id}/qa_overlays/{rec.tile_id}_qa.jpg"
        s3 = session.client("s3", region_name=cfg.aws_region)
        s3.put_object(
            Bucket=bucket,
            Key=s3_qa_key,
            Body=jpg_bytes,
            ContentType="image/jpeg",
        )
        return local_qa_path, s3_qa_key

    if qa_targets:
        pool_size = max(1, min(int(upload_workers), len(qa_targets)))
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            futs = [pool.submit(_render_and_upload_qa, r) for r in qa_targets]
            for fut in as_completed(futs):
                lp, sk = fut.result()
                local_qa_paths.append(lp)
                s3_qa_keys.append(sk)

    local_qa_paths.sort()
    s3_qa_keys.sort()

    # Build & write manifest.parquet locally and to S3
    ordered_records = sorted(records, key=lambda r: (r.split, r.county, r.h3_index))
    manifest_cols: Dict[str, List[Any]] = {
        "tile_id": [r.tile_id for r in ordered_records],
        "county": [r.county for r in ordered_records],
        "split": [r.split for r in ordered_records],
        "h3_index": [r.h3_index for r in ordered_records],
        "lat": [float(r.lat) for r in ordered_records],
        "lon": [float(r.lon) for r in ordered_records],
        "naip_year": [int(r.naip_year) for r in ordered_records],
        "source_cog": [r.source_cog for r in ordered_records],
        "crs_epsg": [int(r.crs_epsg) for r in ordered_records],
        "gsd_m": [float(r.gsd_m) for r in ordered_records],
        "window_transform_6": [list(r.window_transform_6) for r in ordered_records],
        "s3_image_key": [r.s3_image_key for r in ordered_records],
        "parcels_px": [r.parcels_px_json for r in ordered_records],
    }

    manifest_table = pa.Table.from_pydict(manifest_cols, schema=MANIFEST_PARQUET_SCHEMA)
    local_manifest_path = cache_dir / "manifest.parquet"
    pq.write_table(manifest_table, str(local_manifest_path), compression="zstd")

    s3_manifest_key = f"{prefix}/{run_id}/manifest.parquet"
    s3_main = session.client("s3", region_name=cfg.aws_region)
    s3_main.upload_file(str(local_manifest_path), bucket, s3_manifest_key)

    return {
        "bucket": bucket,
        "s3_images_prefix_uri": f"s3://{bucket}/{prefix}/{run_id}/images/",
        "s3_qa_overlays_prefix_uri": f"s3://{bucket}/{prefix}/{run_id}/qa_overlays/",
        "s3_manifest_uri": f"s3://{bucket}/{s3_manifest_key}",
        "local_manifest_path": str(local_manifest_path),
        "uploaded_images_count": uploaded_count,
        "resumed_existing_images_count": skipped_existing,
        "qa_overlays_rendered": len(local_qa_paths),
        "local_qa_overlay_paths": [str(p) for p in local_qa_paths],
        "total_manifest_rows": len(ordered_records),
    }
