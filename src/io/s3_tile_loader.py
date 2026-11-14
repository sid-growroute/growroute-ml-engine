from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import threading
import time
from typing import Dict, List, Optional
import boto3
import h3
import rasterio
from rasterio.session import AWSSession
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import requests


class S3TileLoader:
    """
    3-tier read-through NAIP tile loader:
      1. Local / ephemeral cache (NAS cache locally or /tmp/tiles_cache on AWS EC2)
      2. Private S3 tile cache bucket (S3_TILE_BUCKET, default growroute-naip-tiles in us-west-2)
      3. Direct us-west-2 AWS Open Data NAIP (when AWS_REQUEST_PAYER=requester) +
         Public USDA NAIP STAC fallback (Planetary Computer COG windowed crop)
    """

    AWS_EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
    STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    STAC_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"

    def __init__(
        self,
        bucket: Optional[str] = None,
        s3_prefix: str = "tx/h3_res9",
        cache_dir: str = "./tiles_cache",
        file_ext: str = ".tif",
        max_workers: Optional[int] = None,
        enable_public_naip_fallback: bool = True,
        ephemeral_cache: bool = False,
        write_through_s3: Optional[bool] = None,
    ):
        resolved_bucket = (
            bucket
            if bucket is not None
            else (
                os.getenv("S3_TILE_BUCKET")
                or os.getenv("S3_TILES_BUCKET")
                or "growroute-naip-tiles"
            )
        )
        self.bucket = resolved_bucket.strip()
        self.aws_region = os.getenv("AWS_REGION", "us-west-2").strip() or "us-west-2"
        self.requester_pays = os.getenv("AWS_REQUEST_PAYER", "").strip().lower() == "requester"
        self.s3_prefix = s3_prefix.strip("/")
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.file_ext = file_ext if file_ext.startswith(".") else f".{file_ext}"
        if max_workers is None:
            max_workers = int(os.getenv("TILE_LOADER_WORKERS", "4"))
        self.max_workers = max(1, int(max_workers))
        self.enable_public_naip_fallback = enable_public_naip_fallback
        self.ephemeral_cache = ephemeral_cache

        if write_through_s3 is None:
            self.write_through_s3 = (
                os.getenv("S3_TILE_WRITE_THROUGH", "true").strip().lower() != "false"
            )
        else:
            self.write_through_s3 = write_through_s3

        self._boto_session = boto3.Session(region_name=self.aws_region)
        self.s3_client = self._boto_session.client("s3", region_name=self.aws_region)
        self._lock = threading.Lock()
        self._gdal_lock = threading.Lock()
        self._sas_token: Optional[str] = None
        self._sas_expiry: float = 0.0
        self._s3_upload_disabled = False

        self._local_index: Dict[str, Path] = {}
        if not self.ephemeral_cache:
            self._refresh_local_index()

    def _refresh_local_index(self) -> None:
        for ext in (".tif", ".tiff", ".jpg", ".png"):
            try:
                for path in self.cache_dir.rglob(f"*{ext}"):
                    if path.name.startswith("."):
                        continue
                    h3_key = path.stem.split("_")[-1]
                    self._local_index[h3_key] = path
            except OSError:
                continue

    def _find_cached_tile(self, h3_index: str) -> Optional[Path]:
        with self._lock:
            path = self._local_index.get(h3_index)
        if path and path.exists() and path.stat().st_size > 0:
            return path
        return None

    def _get_naip_sas_token(self, force_refresh: bool = False) -> str:
        now = time.time()
        with self._lock:
            if not force_refresh and self._sas_token and now < self._sas_expiry:
                return self._sas_token
            resp = requests.get(self.STAC_TOKEN_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self._sas_token = data["token"]
            self._sas_expiry = now + 1500  # Refresh every 25 minutes
            return self._sas_token

    @staticmethod
    def _to_s3_uri(href: str) -> str:
        if href.startswith("s3://"):
            return href
        match = re.match(r"^https?://([^./]+)\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com/(.+)$", href)
        if match:
            return f"s3://{match.group(1)}/{match.group(2)}"
        return href

    def _crop_cog_window(
        self,
        cog_uri: str,
        bbox_wgs84: List[float],
        local_path: Path,
        requester_pays: bool = False,
    ) -> bool:
        env_kwargs = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_USE_HEAD": "NO",
            "GDAL_HTTP_MAX_RETRY": "3",
            "GDAL_HTTP_RETRY_DELAY": "1",
            "GDAL_HTTP_TIMEOUT": "20",
            "GDAL_CACHEMAX": 64,
            "VSI_CACHE": False,
        }
        if requester_pays:
            env_kwargs["AWS_REQUEST_PAYER"] = "requester"
            env_kwargs["AWS_REGION"] = "us-west-2"

        aws_session = (
            AWSSession(boto3.Session(region_name="us-west-2"), requester_pays=True)
            if requester_pays
            else None
        )

        with self._gdal_lock:
            env_ctx = (
                rasterio.Env(aws_session, **env_kwargs)
                if aws_session is not None
                else rasterio.Env(**env_kwargs)
            )
            with env_ctx:
                with rasterio.open(cog_uri) as src:
                    left, bottom, right, top = transform_bounds(
                        "EPSG:4326", src.crs, *bbox_wgs84, densify_pts=21
                    )
                    window = from_bounds(left, bottom, right, top, transform=src.transform)
                    window = window.round_offsets().round_lengths()

                    data = src.read(window=window, boundless=True, fill_value=0)
                    win_transform = src.window_transform(window)

                    profile = src.profile.copy()
                    profile.update(
                        driver="GTiff",
                        height=int(window.height),
                        width=int(window.width),
                        count=src.count,
                        transform=win_transform,
                        compress="deflate",
                        tiled=False,
                    )

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(local_path, "w", **profile) as dst:
            dst.write(data)

        return local_path.exists() and local_path.stat().st_size > 0

    def _pull_from_aws_open_data_naip(
        self, bbox_wgs84: List[float], local_path: Path
    ) -> Optional[Path]:
        """Direct us-west-2 AWS Open Data NAIP COG crop when AWS_REQUEST_PAYER=requester."""
        try:
            resp = requests.post(
                self.AWS_EARTH_SEARCH_URL,
                json={
                    "collections": ["naip"],
                    "bbox": bbox_wgs84,
                    "limit": 1,
                    "sortby": [{"field": "properties.datetime", "direction": "desc"}],
                },
                timeout=12,
            )
            if not resp.ok:
                return None
            features = resp.json().get("features", [])
            if not features:
                return None

            assets = features[0].get("assets", {})
            asset = assets.get("image") or assets.get("visual") or next(iter(assets.values()), None)
            if not asset or not asset.get("href"):
                return None

            cog_uri = self._to_s3_uri(str(asset["href"]))
            if self._crop_cog_window(cog_uri, bbox_wgs84, local_path, requester_pays=True):
                return local_path
        except Exception:
            return None
        return None

    def _pull_from_public_naip_to_cache(self, h3_index: str, local_path: Path) -> Optional[Path]:
        """
        Queries NAIP STAC for the H3 cell's bounding box, crops the 4-band RGB+NIR GeoTIFF window,
        trying direct us-west-2 AWS Open Data first (if AWS_REQUEST_PAYER=requester) before
        falling back to Planetary Computer STAC.
        """
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        lngs = [pt[0] for pt in boundary]
        lats = [pt[1] for pt in boundary]
        pad = 0.00035
        bbox_wgs84 = [min(lngs) - pad, min(lats) - pad, max(lngs) + pad, max(lats) + pad]

        # Option A: Direct AWS Open Data NAIP in us-west-2 (zero cross-cloud egress on EC2)
        if self.requester_pays:
            aws_cropped = self._pull_from_aws_open_data_naip(bbox_wgs84, local_path)
            if aws_cropped is not None:
                with self._lock:
                    self._local_index[h3_index] = aws_cropped
                return aws_cropped

        # Option B: Planetary Computer USDA NAIP STAC fallback
        for attempt in range(3):
            try:
                search_resp = requests.post(
                    self.STAC_SEARCH_URL,
                    json={
                        "collections": ["naip"],
                        "bbox": bbox_wgs84,
                        "limit": 1,
                        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
                    },
                    timeout=15,
                )
                search_resp.raise_for_status()
                features = search_resp.json().get("features", [])
                if not features:
                    return None

                href = features[0]["assets"]["image"]["href"]
                token = self._get_naip_sas_token(force_refresh=(attempt > 0))
                signed_url = f"{href}?{token}"

                if self._crop_cog_window(signed_url, bbox_wgs84, local_path, requester_pays=False):
                    with self._lock:
                        self._local_index[h3_index] = local_path
                    return local_path

            except Exception as exc:
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    print(f"[NAIP Pull Warn] Failed to fetch {h3_index}: {exc}", flush=True)

        return None

    def _upload_tile_to_s3_cache(self, local_path: Path, s3_key: str) -> None:
        if not self.bucket or not self.write_through_s3 or self._s3_upload_disabled:
            return
        try:
            self.s3_client.upload_file(str(local_path), self.bucket, s3_key)
        except Exception as exc:
            err_str = str(exc)
            if "NoSuchBucket" in err_str or "AccessDenied" in err_str:
                self._s3_upload_disabled = True

    def _download_single(self, h3_index: str, county: Optional[str] = None) -> Optional[Path]:
        # Tier 1: Check local / ephemeral disk cache first
        cached = self._find_cached_tile(h3_index)
        if cached is not None:
            return cached

        county_slug = re.sub(r"[^a-z0-9]+", "_", (county or "dfw").strip().lower()).strip("_") or "dfw"
        target_dir = self.cache_dir / county_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = target_dir / f"tx_{county_slug}_hex_{h3_index}.tif"
        s3_key = f"{self.s3_prefix}/{county_slug}/{h3_index}{self.file_ext}"

        # Tier 2: Check private S3 tile bucket (S3_TILE_BUCKET)
        if self.bucket:
            try:
                self.s3_client.download_file(self.bucket, s3_key, str(local_path))
                if local_path.exists() and local_path.stat().st_size > 0:
                    with self._lock:
                        self._local_index[h3_index] = local_path
                    return local_path
            except Exception:
                pass

        # Tier 3: Pull & crop from AWS Open Data NAIP / Planetary Computer STAC
        if self.enable_public_naip_fallback:
            pulled = self._pull_from_public_naip_to_cache(h3_index, local_path)
            if pulled is not None:
                self._upload_tile_to_s3_cache(pulled, s3_key)
                return pulled

        return None

    def fetch_batch(self, h3_cells: List[str], county: Optional[str] = None) -> Dict[str, Path]:
        resolved: Dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_map = {
                pool.submit(self._download_single, h3_idx, county): h3_idx
                for h3_idx in h3_cells
            }
            for future in as_completed(future_map):
                h3_idx = future_map[future]
                path = future.result()
                if path is not None:
                    resolved[h3_idx] = path
        return resolved

    def cleanup_ephemeral(self, paths: Dict[str, Path]) -> None:
        with self._lock:
            for h3_idx in paths.keys():
                self._local_index.pop(h3_idx, None)
        for p in paths.values():
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
