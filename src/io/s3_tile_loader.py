from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError


class S3TileLoader:
    """
    Read-through tile loader:
    1. Checks local SSD cache (./tiles_cache) first.
    2. Downloads missing H3 tiles from S3 in parallel threads.
    3. Supports ephemeral cleanup when running on small-disk AWS EC2/Batch instances.
    """

    def __init__(
        self,
        bucket: str,
        s3_prefix: str = "tx/h3_res9",
        cache_dir: str = "./tiles_cache",
        file_ext: str = ".tif",
        max_workers: int = 16,
    ):
        self.bucket = bucket
        self.s3_prefix = s3_prefix.strip("/")
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.file_ext = file_ext if file_ext.startswith(".") else f".{file_ext}"
        self.max_workers = max_workers
        self.s3_client = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )

    def _find_cached_tile(self, h3_index: str, county: Optional[str] = None) -> Optional[Path]:
        """Checks SSD cache for existing tile (.tif, .tiff, .png, .jpg) flat or under county subfolder."""
        search_dirs = [self.cache_dir]
        if county:
            search_dirs.insert(0, self.cache_dir / county.lower())

        for folder in search_dirs:
            for ext in (self.file_ext, ".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                candidate = folder / f"{h3_index}{ext}"
                if candidate.exists() and candidate.stat().st_size > 0:
                    return candidate
        return None

    def _download_single(self, h3_index: str, county: Optional[str] = None) -> Optional[Path]:
        # 1. Return immediately if already on local SSD
        cached = self._find_cached_tile(h3_index, county)
        if cached is not None:
            return cached

        # 2. Otherwise pull from S3 into local cache
        target_dir = self.cache_dir / county.lower() if county else self.cache_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = target_dir / f"{h3_index}{self.file_ext}"

        candidate_keys = []
        if county:
            candidate_keys.append(f"{self.s3_prefix}/{county.lower()}/{h3_index}{self.file_ext}")
        candidate_keys.append(f"{self.s3_prefix}/{h3_index}{self.file_ext}")

        for s3_key in candidate_keys:
            try:
                self.s3_client.download_file(self.bucket, s3_key, str(local_path))
                return local_path
            except ClientError:
                continue

        return None

    def fetch_batch(self, h3_cells: List[str], county: Optional[str] = None) -> Dict[str, Path]:
        """Returns {h3_index: local_path} for a batch of H3 cells, downloading misses in parallel."""
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
        """Deletes temporary downloaded tiles when running on stateless AWS EC2/Batch."""
        for p in paths.values():
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass