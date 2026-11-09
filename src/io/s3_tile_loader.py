from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional
import boto3
import h3
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import requests


class S3TileLoader:
    """
    3-tier read-through NAIP tile loader:
      1. Local Synology NAS cache (/app/tiles_cache -> /Volumes/GrowrouteLLC/growroute-data/ml/naip_h3_tiles)
      2. Private S3 bucket cache (growroute-naip-tiles)
      3. Public USDA NAIP Cloud-Optimized GeoTIFF (STAC windowed crop saved directly to NAS)
    """

    STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    STAC_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"

    def __init__(
        self,
        bucket: str,
        s3_prefix: str = "tx/h3_res9",
        cache_dir: str = "./tiles_cache",
        file_ext: str = ".tif",
        max_workers: Optional[int] = None,
        enable_public_naip_fallback: bool = True,
    ):
        self.bucket = bucket
        self.s3_prefix = s3_prefix.strip("/")
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.file_ext = file_ext if file_ext.startswith(".") else f".{file_ext}"
        if max_workers is None:
            max_workers = int(os.getenv("TILE_LOADER_WORKERS", "4"))
        self.max_workers = max(1, int(max_workers))
        self.enable_public_naip_fallback = enable_public_naip_fallback

        self.s3_client = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        self._lock = threading.Lock()
        self._gdal_lock = threading.Lock()
        self._sas_token: Optional[str] = None
        self._sas_expiry: float = 0.0

        self._local_index: Dict[str, Path] = {}
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

    def _pull_from_public_naip_to_nas(self, h3_index: str, local_path: Path) -> Optional[Path]:
        """
        Queries the public USDA NAIP STAC catalog for the H3 cell's bounding box,
        crops the 4-band RGB+NIR GeoTIFF window, and saves it directly to the NAS.
        """
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        lngs = [pt[0] for pt in boundary]
        lats = [pt[1] for pt in boundary]
        pad = 0.00035
        bbox_wgs84 = [min(lngs) - pad, min(lats) - pad, max(lngs) + pad, max(lats) + pad]

        for attempt in range(3):
            try:
                # Find the latest Texas NAIP COG covering this H3 cell
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

                # Stream only the H3 window from the remote COG and write it to the Synology NAS.
                # Serialize GDAL /vsicurl/ reads per process and disable sidecar probing to prevent
                # Azure Blob 403s and libgdal multi-thread curl state corruption.
                with self._gdal_lock:
                    with rasterio.Env(
                        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                        CPL_VSIL_CURL_USE_HEAD="NO",
                        GDAL_HTTP_MAX_RETRY="3",
                        GDAL_HTTP_RETRY_DELAY="1",
                        GDAL_HTTP_TIMEOUT="20",
                        GDAL_CACHEMAX=64,
                        VSI_CACHE=False,
                    ):
                        with rasterio.open(signed_url) as src:
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

                if local_path.exists() and local_path.stat().st_size > 0:
                    with self._lock:
                        self._local_index[h3_index] = local_path
                    return local_path

            except Exception as exc:
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    print(f"[NAIP Pull Warn] Failed to fetch {h3_index}: {exc}", flush=True)

        return None

    def _download_single(self, h3_index: str, county: Optional[str] = None) -> Optional[Path]:
        # Tier 1: Check Synology NAS cache first
        cached = self._find_cached_tile(h3_index)
        if cached is not None:
            return cached

        county_slug = county.lower() if county else "dfw"
        target_dir = self.cache_dir / county_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = target_dir / f"tx_{county_slug}_hex_{h3_index}.tif"

        # Tier 2: Check private S3 bucket (if configured)
        if self.bucket:
            s3_key = f"{self.s3_prefix}/{county_slug}/{h3_index}{self.file_ext}"
            try:
                self.s3_client.download_file(self.bucket, s3_key, str(local_path))
                if local_path.exists() and local_path.stat().st_size > 0:
                    with self._lock:
                        self._local_index[h3_index] = local_path
                    return local_path
            except Exception:
                pass

        # Tier 3: Pull & crop directly from public USDA NAIP COG and save to NAS
        if self.enable_public_naip_fallback:
            return self._pull_from_public_naip_to_nas(h3_index, local_path)

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

    @staticmethod
    def cleanup_ephemeral(paths: Dict[str, Path]) -> None:
        for p in paths.values():
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass