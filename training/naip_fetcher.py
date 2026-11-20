from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from affine import Affine
import boto3
import numpy as np
import pyarrow.parquet as pq
from pyproj import CRS, Transformer
import rasterio
from rasterio.session import AWSSession
from rasterio.windows import Window
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely import wkb as shapely_wkb
from tqdm import tqdm

from training.config import (
    NAIP_AWS_REGION,
    NAIP_BUCKET,
    TILE_PX,
    TrainingConfig,
)

NAIP_COG_KEY_RE = re.compile(
    r"(?:^|/)m_(\d{7})_(nw|ne|sw|se)_(\d{2})_(\d{3})_(\d{8})\.tif$",
    re.IGNORECASE,
)


def create_boto_session(cfg: TrainingConfig, region_name: Optional[str] = None) -> boto3.Session:
    kwargs: Dict[str, Any] = {
        "region_name": region_name or cfg.aws_region or NAIP_AWS_REGION,
    }
    if cfg.aws_access_key_id and cfg.aws_secret_access_key:
        kwargs["aws_access_key_id"] = cfg.aws_access_key_id
        kwargs["aws_secret_access_key"] = cfg.aws_secret_access_key
        if cfg.aws_session_token:
            kwargs["aws_session_token"] = cfg.aws_session_token
    return boto3.Session(**kwargs)


def latlon_to_usgs_quarter_quad(lat: float, lon: float) -> Tuple[str, str]:
    """
    Maps WGS84 `(lat, lon)` in Northern/Western hemisphere (CONUS) to its USGS:
      - 1-degree SE reference block `DDddd` (e.g. `'25097'` for 25..26N, -98..-97W)
      - 3.75-minute quarter-quad ID `DDdddQQ_qq` (e.g. `'2509703_ne'`)

    USGS 7.5-minute quadrangle grid inside a 1x1 degree cell:
      - 8 rows x 8 cols = 64 quads (`01`..`64`), each 0.125 deg x 0.125 deg.
      - Row 1 is Northernmost (`lat` in `[deg_lat + 0.875, deg_lat + 1.0)`), Row 8 is Southernmost.
      - Col 1 is Westernmost (`lon` in `[-(deg_lon + 1.0), -(deg_lon + 0.875))`), Col 8 is Easternmost.
      - Each 7.5-minute quad is split into 4 quarter-quads (`nw`, `ne`, `sw`, `se`) of 0.0625 deg.
    """
    if not (0.0 < lat < 89.0) or not (-179.9 < lon < 0.0):
        raise ValueError(f"Expected CONUS (lat > 0, lon < 0); received ({lat}, {lon})")

    deg_lat = int(math.floor(lat))
    west_pos = -lon
    deg_lon = int(math.floor(west_pos))
    # Handle exact integer western degree edge cleanly
    west_edge = -float(deg_lon + 1)
    south_edge = float(deg_lat)

    u_east = (lon - west_edge) * 8.0
    v_north = (lat - south_edge) * 8.0

    col_from_west = min(7, max(0, int(math.floor(u_east))))
    south_idx = min(7, max(0, int(math.floor(v_north))))
    row_from_north = 7 - south_idx

    quad_num = row_from_north * 8 + col_from_west + 1  # 1..64

    sub_east = u_east - col_from_west
    sub_north = v_north - south_idx
    ns_char = "n" if sub_north >= 0.5 else "s"
    ew_char = "e" if sub_east >= 0.5 else "w"

    block_ddddd = f"{deg_lat:02d}{deg_lon:03d}"
    qquad_id = f"{block_ddddd}{quad_num:02d}_{ns_char}{ew_char}"
    return block_ddddd, qquad_id


def nominal_utm_zone_for_lon(lon: float) -> int:
    return int(math.floor((lon + 180.0) / 6.0)) + 1


def discover_latest_naip_year(
    s3_client: Any,
    state: str = "tx",
    bucket: str = NAIP_BUCKET,
) -> int:
    """
    Auto-discovers the newest available 4-digit year under `s3://naip-visualization/{state}/{year}/60cm/rgb/`
    in `us-west-2` (`RequestPayer='requester'`).
    """
    st = state.strip().lower()
    resp = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=f"{st}/",
        Delimiter="/",
        RequestPayer="requester",
    )
    years: List[int] = []
    for item in resp.get("CommonPrefixes", []):
        prefix = str(item.get("Prefix") or "").strip("/")
        parts = prefix.split("/")
        if len(parts) == 2 and len(parts[1]) == 4 and parts[1].isdigit():
            years.append(int(parts[1]))

    if not years:
        raise RuntimeError(f"No NAIP year prefixes discovered under s3://{bucket}/{st}/")

    for yr in sorted(years, reverse=True):
        chk = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{st}/{yr}/60cm/rgb/",
            MaxKeys=1,
            RequestPayer="requester",
        )
        if chk.get("KeyCount", 0) > 0 or chk.get("Contents"):
            return yr

    return max(years)


def index_naip_quarter_quads_for_blocks(
    s3_client: Any,
    state: str,
    naip_year: int,
    blocks_ddddd: Iterable[str],
    bucket: str = NAIP_BUCKET,
    max_workers: int = 8,
) -> Dict[str, List[Tuple[str, int, str]]]:
    """
    Lists `s3://naip-visualization/{state}/{year}/60cm/rgb/{DDddd}/` ONLY for the needed 1-degree
    blocks and returns a map:
      `qquad_id -> [(acq_yyyymmdd, utm_zone, cog_s3_key), ...]` sorted newest-first.
    """
    st = state.strip().lower()
    unique_blocks = sorted({b.strip() for b in blocks_ddddd if b and b.strip()})
    qquad_map: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)

    def _list_one_block(block: str) -> List[Tuple[str, str, int, str]]:
        prefix = f"{st}/{naip_year}/60cm/rgb/{block}/"
        entries: List[Tuple[str, str, int, str]] = []
        continuation: Optional[str] = None
        while True:
            kwargs: Dict[str, Any] = {
                "Bucket": bucket,
                "Prefix": prefix,
                "RequestPayer": "requester",
            }
            if continuation:
                kwargs["ContinuationToken"] = continuation
            resp = s3_client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                key = str(obj.get("Key") or "")
                m = NAIP_COG_KEY_RE.search(key)
                if not m:
                    continue
                qq_id = f"{m.group(1)}_{m.group(2).lower()}"
                utm_zone = int(m.group(3))
                acq_date = m.group(5)
                entries.append((qq_id, acq_date, utm_zone, key))
            if not resp.get("IsTruncated"):
                break
            continuation = resp.get("NextContinuationToken")
        return entries

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, max(1, len(unique_blocks))))) as pool:
        futures = [pool.submit(_list_one_block, blk) for blk in unique_blocks]
        for fut in as_completed(futures):
            for qq_id, acq_date, utm_zone, key in fut.result():
                qquad_map[qq_id].append((acq_date, utm_zone, key))

    for qq_id in qquad_map:
        qquad_map[qq_id].sort(key=lambda t: (t[0], t[2]), reverse=True)

    return dict(qquad_map)


def select_best_cog_for_candidate(
    lat: float,
    lon: float,
    qquad_index: Dict[str, List[Tuple[str, int, str]]],
) -> Optional[str]:
    """
    Picks the newest acquisition-date COG key for `(lat, lon)`'s quarter-quad (preferring the
    nominal UTM zone when duplicate same-date cross-zone scenes exist).
    """
    _blk, qq_id = latlon_to_usgs_quarter_quad(lat, lon)
    matches = qquad_index.get(qq_id)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][2]

    newest_date = matches[0][0]
    same_date = [m for m in matches if m[0] == newest_date]
    if len(same_date) == 1:
        return same_date[0][2]

    pref_zone = nominal_utm_zone_for_lon(lon)
    same_date.sort(key=lambda m: (0 if m[1] == pref_zone else 1, -int(m[0])))
    return same_date[0][2]


def is_chip_exposure_valid(
    img_rgb: np.ndarray,
    min_valid_fraction: float = 0.98,
) -> Tuple[bool, float]:
    """
    Verifies that at least `min_valid_fraction` (98%) of pixels satisfy `15 < R + G + B < 755`
    (rejecting quarter-quad black/white nodata borders and blown-out/corrupt chips).
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] < 3 or img_rgb.size == 0:
        return False, 0.0
    rgb_sum = (
        img_rgb[:, :, 0].astype(np.int32)
        + img_rgb[:, :, 1].astype(np.int32)
        + img_rgb[:, :, 2].astype(np.int32)
    )
    valid_mask = (rgb_sum > 15) & (rgb_sum < 755)
    frac = float(np.mean(valid_mask))
    return frac >= min_valid_fraction, frac


def _extract_polygons(geom: Any) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: List[Polygon] = []
        for sub in geom.geoms:
            out.extend(_extract_polygons(sub))
        return out
    return []


def project_parcels_to_tile_pixels(
    parcels_wkb: Sequence[Dict[str, Any]],
    dst_crs: Any,
    win_transform: Affine,
    tile_px: int = TILE_PX,
    min_area_px: float = 4.0,
    transformer_cache: Optional[Dict[str, Transformer]] = None,
) -> List[Dict[str, Any]]:
    """
    Projects EPSG:4326 parcel WKB geometries -> `dst_crs` -> exact continuous tile pixel `(col, row)`
    via `inv_transform = ~win_transform`, clips to `box(0, 0, tile_px, tile_px)`, and drops
    boundary slivers smaller than `min_area_px` (4.0 sq px).
    """
    if not parcels_wkb:
        return []

    crs_obj = CRS.from_user_input(dst_crs)
    crs_key = crs_obj.to_string()
    if transformer_cache is not None and crs_key in transformer_cache:
        to_crs = transformer_cache[crs_key]
    else:
        to_crs = Transformer.from_crs("EPSG:4326", crs_obj, always_xy=True)
        if transformer_cache is not None:
            transformer_cache[crs_key] = to_crs

    inv_transform = ~win_transform
    ia, ib, ic, id_, ie, if_ = (
        float(inv_transform.a),
        float(inv_transform.b),
        float(inv_transform.c),
        float(inv_transform.d),
        float(inv_transform.e),
        float(inv_transform.f),
    )
    clip_box = box(0.0, 0.0, float(tile_px), float(tile_px))

    def _coord_transform(coords: np.ndarray) -> np.ndarray:
        xs_crs, ys_crs = to_crs.transform(coords[:, 0], coords[:, 1])
        xs_arr = np.asarray(xs_crs, dtype=np.float64)
        ys_arr = np.asarray(ys_crs, dtype=np.float64)
        cols = ia * xs_arr + ib * ys_arr + ic
        rows = id_ * xs_arr + ie * ys_arr + if_
        return np.column_stack([cols, rows])

    projected_out: List[Dict[str, Any]] = []

    for p in parcels_wkb:
        raw_wkb = p.get("geom_wkb")
        if not raw_wkb:
            continue
        try:
            geom_wgs84 = shapely_wkb.loads(bytes(raw_wkb))
        except Exception:
            continue
        if geom_wgs84.is_empty:
            continue

        geom_px = shapely.transform(geom_wgs84, _coord_transform)
        if not geom_px.is_valid:
            geom_px = geom_px.buffer(0)
        if geom_px.is_empty or not geom_px.intersects(clip_box):
            continue

        clipped = geom_px.intersection(clip_box)
        if clipped.is_empty:
            continue

        rings_px: List[List[List[float]]] = []
        surviving_area = 0.0
        for poly in _extract_polygons(clipped):
            if poly.is_empty or float(poly.area) < min_area_px:
                continue
            ext_coords = np.asarray(poly.exterior.coords, dtype=np.float64)
            if len(ext_coords) < 4:
                continue
            surviving_area += float(poly.area)
            ring = [
                [round(float(pt[0]), 2), round(float(pt[1]), 2)]
                for pt in ext_coords
            ]
            rings_px.append(ring)

        if not rings_px or surviving_area < min_area_px:
            continue

        projected_out.append({
            "parcel_id": str(p.get("parcel_id") or ""),
            "cad_has_pool": bool(p.get("cad_has_pool", False)),
            "cad_has_spa": bool(p.get("cad_has_spa", False)),
            "is_residential": bool(p.get("is_residential", True)),
            "normalized_use_type": str(p.get("normalized_use_type") or ""),
            "area_px": round(surviving_area, 2),
            "pixel_rings": rings_px,
        })

    return projected_out


def compute_centroid_tile_window(
    src_transform: Affine,
    src_crs: Any,
    lat: float,
    lon: float,
    tile_px: int = TILE_PX,
    transformer_cache: Optional[Dict[str, Transformer]] = None,
) -> Window:
    """
    Computes the exact `tile_px x tile_px` (`640x640` at native 0.60m GSD = `384m x 384m`)
    rasterio Window centered on H3 centroid `(lat, lon)`.
    """
    crs_obj = CRS.from_user_input(src_crs)
    crs_key = crs_obj.to_string()
    if transformer_cache is not None and crs_key in transformer_cache:
        to_crs = transformer_cache[crs_key]
    else:
        to_crs = Transformer.from_crs("EPSG:4326", crs_obj, always_xy=True)
        if transformer_cache is not None:
            transformer_cache[crs_key] = to_crs

    cx_crs, cy_crs = to_crs.transform(float(lon), float(lat))
    inv_src = ~src_transform
    center_col = float(inv_src.a) * float(cx_crs) + float(inv_src.b) * float(cy_crs) + float(inv_src.c)
    center_row = float(inv_src.d) * float(cx_crs) + float(inv_src.e) * float(cy_crs) + float(inv_src.f)

    half = float(tile_px) / 2.0
    col_off = int(round(center_col - half))
    row_off = int(round(center_row - half))
    return Window(col_off=col_off, row_off=row_off, width=int(tile_px), height=int(tile_px))


@dataclass
class ExtractedTileRecord:
    tile_id: str
    county: str
    split: str
    h3_index: str
    lat: float
    lon: float
    naip_year: int
    source_cog: str
    crs_epsg: int
    gsd_m: float
    window_transform_6: List[float]
    s3_image_key: str
    parcels_px_json: str
    parcels_px_list: List[Dict[str, Any]]
    img_rgb: Optional[np.ndarray]  # None when resumed from already-cached manifest + S3 image
    valid_px_fraction: float


class _WorkerThreadContext(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.aws_session: Optional[AWSSession] = None
        self.transformers: Dict[str, Transformer] = {}


def fetch_and_project_naip_tiles(
    cfg: TrainingConfig,
    run_id: str,
    candidates_parquet: Optional[Path] = None,
    workers: int = 16,
    naip_year_override: Optional[int] = None,
    existing_s3_keys: Optional[set[str]] = None,
    cached_manifest_by_tile_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[ExtractedTileRecord], Dict[str, Any]]:
    """
    Phase 2a:
      1. Reads `candidates.parquet` and discovers the latest NAIP 60cm RGB year in `us-west-2`.
      2. Indexes only needed USGS `DDddd` 1-degree blocks and maps each candidate to its newest
         quarter-quad COG (`DDdddQQ_qq`).
      3. Groups primary candidates by parent `cog_key` so each COG is opened once per worker thread,
         extracts `640x640` RGB chips, enforces the `98%` `(15 < R+G+B < 755)` nodata guard, and
         projects all intersecting 290m parcels to pixel coordinates via `~src.window_transform(win)`.
      4. Automatically backfills any rejected chips from the county's `1.15x` oversampled pool until
         exact per-county target quotas are filled.
    """
    cache_dir = cfg.run_cache_dir(run_id)
    cand_path = candidates_parquet or (cache_dir / "candidates.parquet")
    if not cand_path.is_file():
        raise FileNotFoundError(
            f"Missing candidates file {cand_path}. Run `sample-stage --run-id {run_id}` first."
        )

    table = pq.read_table(str(cand_path))
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"No candidate rows found in {cand_path}")

    session = create_boto_session(cfg, region_name=NAIP_AWS_REGION)
    s3_naip = session.client("s3", region_name=NAIP_AWS_REGION)

    naip_year = (
        int(naip_year_override)
        if naip_year_override
        else discover_latest_naip_year(s3_naip, state=cfg.naip_state, bucket=cfg.naip_bucket)
    )
    print(
        f"[NAIP Fetcher] Using s3://{cfg.naip_bucket}/{cfg.naip_state}/{naip_year}/60cm/rgb/ "
        f"(RequestPayer='requester', region={NAIP_AWS_REGION})",
        flush=True,
    )

    # Collect unique USGS 1-degree SE blocks across all candidates
    needed_blocks: set[str] = set()
    for r in rows:
        blk, _qq = latlon_to_usgs_quarter_quad(float(r["lat"]), float(r["lon"]))
        needed_blocks.add(blk)

    qquad_index = index_naip_quarter_quads_for_blocks(
        s3_client=s3_naip,
        state=cfg.naip_state,
        naip_year=naip_year,
        blocks_ddddd=needed_blocks,
        bucket=cfg.naip_bucket,
        max_workers=min(12, max(1, workers)),
    )
    print(
        f"[NAIP Fetcher] Indexed {len(qquad_index):,} unique quarter-quads across "
        f"{len(needed_blocks)} USGS 1-degree blocks.",
        flush=True,
    )

    # Organize candidates per (split, county) in deterministic candidate_rank order
    candidates_by_county: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    quota_by_county: Dict[Tuple[str, str], int] = {}

    for r in rows:
        key = (str(r["split"]), str(r["county"]))
        r["cog_key"] = select_best_cog_for_candidate(
            float(r["lat"]), float(r["lon"]), qquad_index
        )
        tile_id = f"{r['county']}_{r['h3_index']}_{naip_year}"
        r["tile_id"] = tile_id
        r["s3_image_key"] = f"{cfg.training_s3_prefix}/{run_id}/images/{tile_id}.jpg"
        candidates_by_county[key].append(r)
        quota_by_county[key] = int(r["target_quota"])

    for key in candidates_by_county:
        candidates_by_county[key].sort(key=lambda x: int(x["candidate_rank"]))

    existing_keys = existing_s3_keys or set()
    cached_manifest = cached_manifest_by_tile_id or {}
    tls = _WorkerThreadContext()

    def _get_thread_aws_session() -> AWSSession:
        if tls.aws_session is None:
            boto_sess = create_boto_session(cfg, region_name=NAIP_AWS_REGION)
            tls.aws_session = AWSSession(boto_sess, requester_pays=True)
        return tls.aws_session

    def _process_cog_group(
        cog_key: str,
        group_candidates: List[Dict[str, Any]],
    ) -> Tuple[List[ExtractedTileRecord], List[Tuple[str, str]]]:
        accepted_group: List[ExtractedTileRecord] = []
        rejected_group: List[Tuple[str, str]] = []  # (tile_id, reason)

        # Fast-path for tiles already uploaded AND already recorded in local manifest cache
        to_read: List[Dict[str, Any]] = []
        for cand in group_candidates:
            tid = str(cand["tile_id"])
            s3_img_key = str(cand["s3_image_key"])
            if s3_img_key in existing_keys and tid in cached_manifest:
                m_row = cached_manifest[tid]
                parcels_decoded = json.loads(m_row["parcels_px"])
                accepted_group.append(
                    ExtractedTileRecord(
                        tile_id=tid,
                        county=str(cand["county"]),
                        split=str(cand["split"]),
                        h3_index=str(cand["h3_index"]),
                        lat=float(cand["lat"]),
                        lon=float(cand["lon"]),
                        naip_year=int(naip_year),
                        source_cog=str(m_row["source_cog"]),
                        crs_epsg=int(m_row["crs_epsg"]),
                        gsd_m=float(m_row["gsd_m"]),
                        window_transform_6=[float(v) for v in m_row["window_transform_6"]],
                        s3_image_key=s3_img_key,
                        parcels_px_json=str(m_row["parcels_px"]),
                        parcels_px_list=parcels_decoded,
                        img_rgb=None,
                        valid_px_fraction=1.0,
                    )
                )
            else:
                to_read.append(cand)

        if not to_read:
            return accepted_group, rejected_group

        vsis3_uri = f"/vsis3/{cfg.naip_bucket}/{cog_key}"
        source_cog_uri = f"s3://{cfg.naip_bucket}/{cog_key}"
        aws_sess = _get_thread_aws_session()

        env_opts = {
            "AWS_REQUEST_PAYER": "requester",
            "AWS_REGION": NAIP_AWS_REGION,
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "VSI_CACHE": "TRUE",
            "VSI_CACHE_SIZE": "16777216",
        }

        try:
            with rasterio.Env(aws_sess, **env_opts):
                with rasterio.open(vsis3_uri) as src:
                    crs_epsg = int(src.crs.to_epsg() or 0) if src.crs else 0

                    for cand in to_read:
                        tid = str(cand["tile_id"])
                        lat = float(cand["lat"])
                        lon = float(cand["lon"])

                        win = compute_centroid_tile_window(
                            src_transform=src.transform,
                            src_crs=src.crs,
                            lat=lat,
                            lon=lon,
                            tile_px=cfg.tile_px,
                            transformer_cache=tls.transformers,
                        )

                        # Strict bounds check + boundless read so border overedge works cleanly
                        rgb_chw = src.read(
                            [1, 2, 3],
                            window=win,
                            boundless=True,
                            fill_value=0,
                        )
                        if rgb_chw.dtype != np.uint8:
                            rgb_chw = np.clip(rgb_chw, 0, 255).astype(np.uint8)
                        img_rgb = np.ascontiguousarray(np.transpose(rgb_chw, (1, 2, 0)))

                        is_ok, valid_frac = is_chip_exposure_valid(img_rgb, min_valid_fraction=0.98)
                        if not is_ok:
                            rejected_group.append((tid, f"nodata_or_glare({valid_frac:.4f})"))
                            continue

                        win_transform = src.window_transform(win)
                        gsd_m = round(float(abs(win_transform.a)), 4)
                        wt6 = [
                            float(win_transform.a),
                            float(win_transform.b),
                            float(win_transform.c),
                            float(win_transform.d),
                            float(win_transform.e),
                            float(win_transform.f),
                        ]

                        projected_parcels = project_parcels_to_tile_pixels(
                            parcels_wkb=cand.get("parcels_wkb") or [],
                            dst_crs=src.crs,
                            win_transform=win_transform,
                            tile_px=cfg.tile_px,
                            min_area_px=4.0,
                            transformer_cache=tls.transformers,
                        )
                        parcels_json = json.dumps(projected_parcels, separators=(",", ":"))

                        accepted_group.append(
                            ExtractedTileRecord(
                                tile_id=tid,
                                county=str(cand["county"]),
                                split=str(cand["split"]),
                                h3_index=str(cand["h3_index"]),
                                lat=lat,
                                lon=lon,
                                naip_year=int(naip_year),
                                source_cog=source_cog_uri,
                                crs_epsg=crs_epsg,
                                gsd_m=gsd_m,
                                window_transform_6=wt6,
                                s3_image_key=str(cand["s3_image_key"]),
                                parcels_px_json=parcels_json,
                                parcels_px_list=projected_parcels,
                                img_rgb=img_rgb,
                                valid_px_fraction=valid_frac,
                            )
                        )
        except Exception as exc:
            for cand in to_read:
                rejected_group.append((str(cand["tile_id"]), f"cog_open_error:{exc}"))

        return accepted_group, rejected_group

    accepted_by_county: Dict[Tuple[str, str], List[ExtractedTileRecord]] = defaultdict(list)
    next_cursor_by_county: Dict[Tuple[str, str], int] = {}
    rejected_details: List[Tuple[str, str]] = []
    missing_cog_count = 0

    # Initialize Pass 1 cursors to target_quota for each (split, county)
    for key, cand_list in candidates_by_county.items():
        next_cursor_by_county[key] = 0

    round_num = 0
    total_quota = sum(quota_by_county.values())
    pbar = tqdm(total=total_quota, desc="Streaming NAIP 640x640 COGs", unit="tile")

    try:
        while True:
            round_num += 1
            batch_to_fetch: List[Dict[str, Any]] = []

            for key, cand_list in candidates_by_county.items():
                needed = quota_by_county[key] - len(accepted_by_county[key])
                if needed <= 0:
                    continue

                cursor = next_cursor_by_county[key]
                picked = 0
                while cursor < len(cand_list) and picked < needed:
                    cand = cand_list[cursor]
                    cursor += 1
                    if not cand.get("cog_key"):
                        missing_cog_count += 1
                        rejected_details.append((str(cand["tile_id"]), "missing_quarter_quad_cog"))
                        continue
                    batch_to_fetch.append(cand)
                    picked += 1
                next_cursor_by_county[key] = cursor

            if not batch_to_fetch:
                break

            cog_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for cand in batch_to_fetch:
                cog_groups[str(cand["cog_key"])].append(cand)

            max_pool = max(1, min(int(workers), len(cog_groups)))
            with ThreadPoolExecutor(max_workers=max_pool) as pool:
                future_map = {
                    pool.submit(_process_cog_group, c_key, c_list): c_key
                    for c_key, c_list in cog_groups.items()
                }
                for fut in as_completed(future_map):
                    acc_list, rej_list = fut.result()
                    rejected_details.extend(rej_list)
                    for rec in acc_list:
                        ckey = (rec.split, rec.county)
                        if len(accepted_by_county[ckey]) < quota_by_county[ckey]:
                            accepted_by_county[ckey].append(rec)
                            pbar.update(1)
    finally:
        pbar.close()

    # Deterministic final ordering by (split, county, h3_index)
    all_accepted: List[ExtractedTileRecord] = []
    for key in sorted(accepted_by_county.keys()):
        recs = sorted(accepted_by_county[key], key=lambda r: r.h3_index)
        all_accepted.extend(recs[: quota_by_county[key]])

    stats = {
        "naip_year": naip_year,
        "target_quota_total": total_quota,
        "accepted_total": len(all_accepted),
        "rejected_total": len(rejected_details),
        "missing_cog_count": missing_cog_count,
        "rounds_executed": round_num,
        "rejected_samples": rejected_details[:10],
    }
    return all_accepted, stats
