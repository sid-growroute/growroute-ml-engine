from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq
from supabase import Client, create_client
import torch
from ultralytics import YOLO
import yaml

from src.detectors.pool_canopy import PoolSpaCanopyDetector
from src.io.s3_tile_loader import S3TileLoader

load_dotenv()

VISIBILITY_TIMEOUT_SEC = int(os.getenv("ML_SQS_VISIBILITY_SEC", "900"))
HEARTBEAT_INTERVAL_SEC = int(os.getenv("ML_SQS_HEARTBEAT_SEC", "300"))
SQS_WAIT_SECONDS = 20
POSTGREST_PAGE_SIZE = 1000

DETECTION_PARQUET_SCHEMA = pa.schema([
    ("h3_cell", pa.string()),
    ("lat", pa.float64()),
    ("lng", pa.float64()),
    ("ai_pool_confidence", pa.float64()),
    ("ai_pool_sqft", pa.int32()),
    ("ai_pool_condition", pa.string()),
    ("ai_has_spa", pa.bool_()),
    ("ai_tree_canopy_pct", pa.float64()),
    ("ai_tree_overhang", pa.bool_()),
    ("run_id", pa.string()),
])


def slugify_county(county_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", county_name.strip().lower()).strip("_")


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


class VisibilityHeartbeat:
    """Extends SQS message VisibilityTimeout every 5 minutes while a shard or plan runs."""

    def __init__(self, sqs_client: Any, queue_url: str, receipt_handle: str):
        self.sqs_client = sqs_client
        self.queue_url = queue_url
        self.receipt_handle = receipt_handle
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.wait(HEARTBEAT_INTERVAL_SEC):
            try:
                self.sqs_client.change_message_visibility(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=self.receipt_handle,
                    VisibilityTimeout=VISIBILITY_TIMEOUT_SEC,
                )
            except Exception as exc:
                print(f"[ML Heartbeat Warn] Failed extending visibility: {exc}", flush=True)


def _s3_object_exists(s3_client: Any, bucket: str, key: str) -> bool:
    if not bucket:
        return False
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def _find_local_pt_weights(models_dir: Path) -> Optional[Path]:
    for candidate_name in ("pool_spa_dfw_v2.pt", "pool-spa-dfw-v2.pt"):
        candidate = models_dir / candidate_name
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _download_pt_from_s3(s3_client: Any, bucket: str, models_dir: Path) -> Optional[Path]:
    if not bucket:
        return None
    dest = models_dir / "pool_spa_dfw_v2.pt"
    for s3_key in ("models/pool_spa_dfw_v2.pt", "models/pool-spa-dfw-v2.pt"):
        if _s3_object_exists(s3_client, bucket, s3_key):
            models_dir.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket, s3_key, str(dest))
            if dest.exists() and dest.stat().st_size > 0:
                return dest
    return None


def bootstrap_model_and_device(
    s3_client: Any,
    s3_data_bucket: str,
    models_dir: Path,
    imgsz: int = 640,
) -> Tuple[Path, str]:
    """
    Auto-selects device ('0' on CUDA GPU, else 'cpu') and ensures model weights exist locally:
      - On 'cpu': downloads s3://<S3_DATA_BUCKET>/models/pool_spa_dfw_v2.onnx if available;
        otherwise exports from local/S3 .pt once via YOLO(...).export(format='onnx', imgsz=640)
        and uploads the .onnx back to s3://<S3_DATA_BUCKET>/models/pool_spa_dfw_v2.onnx.
      - On '0' (GPU): pulls pool_spa_dfw_v2.pt from local/S3 for native CUDA inference.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    explicit_device = os.getenv("ML_DEVICE", "").strip()
    if explicit_device and explicit_device.lower() != "auto":
        device = explicit_device
    else:
        device = "0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        onnx_path = models_dir / "pool_spa_dfw_v2.onnx"
        s3_onnx_key = "models/pool_spa_dfw_v2.onnx"

        if onnx_path.exists() and onnx_path.stat().st_size > 0:
            print(f"[ML Bootstrap] Using local CPU ONNX model: {onnx_path}", flush=True)
            return onnx_path, device

        if _s3_object_exists(s3_client, s3_data_bucket, s3_onnx_key):
            print(
                f"[ML Bootstrap] Downloading s3://{s3_data_bucket}/{s3_onnx_key} -> {onnx_path}...",
                flush=True,
            )
            s3_client.download_file(s3_data_bucket, s3_onnx_key, str(onnx_path))
            if onnx_path.exists() and onnx_path.stat().st_size > 0:
                return onnx_path, device

        pt_path = _find_local_pt_weights(models_dir) or _download_pt_from_s3(
            s3_client, s3_data_bucket, models_dir
        )
        if pt_path is None:
            raise FileNotFoundError(
                f"Neither pool_spa_dfw_v2.onnx nor .pt weights found in {models_dir} "
                f"or s3://{s3_data_bucket}/models/"
            )

        print(
            f"[ML Bootstrap] Exporting {pt_path} -> ONNX (imgsz={imgsz}) for CPU inference...",
            flush=True,
        )
        exported = Path(YOLO(str(pt_path), task="segment").export(format="onnx", imgsz=imgsz))
        if exported.resolve() != onnx_path.resolve():
            shutil.move(str(exported), str(onnx_path))

        if s3_data_bucket:
            try:
                s3_client.upload_file(str(onnx_path), s3_data_bucket, s3_onnx_key)
                print(
                    f"[ML Bootstrap] Uploaded exported ONNX model to s3://{s3_data_bucket}/{s3_onnx_key}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[ML Bootstrap Warn] Could not upload ONNX to S3: {exc}", flush=True)

        return onnx_path, device

    # GPU ('0') or MPS path -> native PyTorch .pt weights
    pt_path = _find_local_pt_weights(models_dir) or _download_pt_from_s3(
        s3_client, s3_data_bucket, models_dir
    )
    if pt_path is None:
        raise FileNotFoundError(
            f"PyTorch weights not found in {models_dir} or s3://{s3_data_bucket}/models/"
        )
    print(f"[ML Bootstrap] Using native PyTorch weights {pt_path} on device={device}", flush=True)
    return pt_path, device


def fetch_all_unscanned_h3_cells(
    supabase: Client,
    county_name: str,
    ml_max_tiles: Optional[int],
) -> List[str]:
    """
    Fetches unscanned H3 cells via get_unscanned_h3_cells RPC, paginating cleanly across
    PostgREST's 1,000-row response limit when ml_max_tiles is > 1000 or unlimited (0 / NULL).
    """
    unlimited = ml_max_tiles is None or int(ml_max_tiles) <= 0
    rpc_limit = 0 if unlimited else int(ml_max_tiles)

    collected: List[str] = []
    offset = 0

    while True:
        remaining = POSTGREST_PAGE_SIZE if unlimited else min(POSTGREST_PAGE_SIZE, rpc_limit - len(collected))
        if remaining <= 0:
            break

        resp = (
            supabase.rpc(
                "get_unscanned_h3_cells",
                {
                    "p_county_name": county_name.upper(),
                    "p_limit": rpc_limit,
                },
            )
            .range(offset, offset + remaining - 1)
            .execute()
        )
        rows = resp.data or []
        batch_cells = [row["h3_index_res9"] for row in rows if row.get("h3_index_res9")]
        if not batch_cells:
            break

        collected.extend(batch_cells)
        if len(batch_cells) < remaining:
            break
        offset += len(batch_cells)

    # Preserve deterministic order while deduplicating any boundary shifts
    return list(dict.fromkeys(collected))


def enqueue_sqs_shards(
    sqs_client: Any,
    queue_url: str,
    shard_messages: List[Dict[str, Any]],
) -> None:
    is_fifo = queue_url.endswith(".fifo")
    for batch_idx, batch in enumerate(chunk_list(shard_messages, 10)):
        entries = []
        for idx, msg_body in enumerate(batch):
            entry: Dict[str, Any] = {
                "Id": f"s{batch_idx}_{idx}",
                "MessageBody": json.dumps(msg_body),
            }
            if is_fifo:
                group_id = f"ml-{msg_body['state_code']}-{msg_body['county_fips']}"
                entry["MessageGroupId"] = group_id
                entry["MessageDeduplicationId"] = f"{msg_body['run_id']}-{msg_body['shard_index']}"
            entries.append(entry)

        pending = entries
        for attempt in range(4):
            res = sqs_client.send_message_batch(QueueUrl=queue_url, Entries=pending)
            failed = res.get("Failed", [])
            if not failed:
                break
            if attempt == 3:
                raise RuntimeError(f"SQS send_message_batch failed entries: {failed}")
            failed_ids = {f["Id"] for f in failed}
            pending = [e for e in pending if e["Id"] in failed_ids]
            time.sleep(1.0 * (attempt + 1))


def handle_plan_county(
    payload: Dict[str, Any],
    supabase: Client,
    sqs_client: Any,
    s3_client: Any,
    ml_sqs_url: str,
    s3_data_bucket: str,
    worker_id: str,
    shard_size: int,
) -> None:
    state_code = str(payload.get("state_code") or "TX").strip().upper()
    county_fips = str(payload.get("county_fips") or "").strip()
    county_name = str(payload.get("county_name") or "").strip()
    ml_max_tiles = payload.get("ml_max_tiles")

    if not county_fips or not county_name:
        lookup = (
            supabase.table("county_lookup")
            .select("state_code, county_fips, county_name, ml_max_tiles")
            .eq("county_fips", county_fips)
            .maybe_single()
            .execute()
        )
        if lookup.data:
            state_code = lookup.data.get("state_code") or state_code
            county_name = lookup.data.get("county_name") or county_name
            if ml_max_tiles is None:
                ml_max_tiles = lookup.data.get("ml_max_tiles")

    if ml_max_tiles is None:
        lookup_tiles = (
            supabase.table("county_lookup")
            .select("ml_max_tiles")
            .eq("state_code", state_code)
            .eq("county_fips", county_fips)
            .maybe_single()
            .execute()
        )
        if lookup_tiles.data:
            ml_max_tiles = lookup_tiles.data.get("ml_max_tiles")

    max_tiles_int = int(ml_max_tiles) if ml_max_tiles is not None else 0
    county_slug = slugify_county(county_name)

    try:
        h3_cells = fetch_all_unscanned_h3_cells(supabase, county_name, max_tiles_int)
        total_cells = len(h3_cells)

        if total_cells == 0:
            print(
                f"[ML Planner] 0 unscanned H3 cells remaining for {county_name} ({state_code}-{county_fips}); marking completed.",
                flush=True,
            )
            supabase.table("county_lookup").update({
                "ml_status": "completed",
                "last_ml_run_at": datetime.now(timezone.utc).isoformat(),
                "active_worker_id": None,
                "last_pipeline_error": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("state_code", state_code).eq("county_fips", county_fips).execute()

            supabase.rpc(
                "refresh_market_intelligence_metrics",
                {"p_state_code": state_code, "p_county_fips": county_fips},
            ).execute()
            return

        run_id = (
            f"ml-{state_code.lower()}-{county_slug}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
        )
        shards = chunk_list(h3_cells, max(1, shard_size))
        total_shards = len(shards)

        manifest = {
            "run_id": run_id,
            "pipeline": "ml",
            "state_code": state_code,
            "county_fips": county_fips,
            "county_name": county_name,
            "ml_max_tiles": max_tiles_int,
            "shard_size": shard_size,
            "total_h3_cells": total_cells,
            "total_shards": total_shards,
            "planned_at": datetime.now(timezone.utc).isoformat(),
            "planner_worker_id": worker_id,
            "shards": [
                {"shard_index": idx, "cell_count": len(cells)}
                for idx, cells in enumerate(shards)
            ],
        }
        manifest_key = f"manifests/{state_code.lower()}/{county_slug}/{run_id}.json"
        s3_client.put_object(
            Bucket=s3_data_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        supabase.table("county_lookup").update({
            "ml_status": "running",
            "total_h3_cells": total_cells,
            "scanned_h3_cells": 0,
            "active_worker_id": worker_id,
            "last_pipeline_error": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("state_code", state_code).eq("county_fips", county_fips).execute()

        shard_messages = [
            {
                "mode": "run_shard",
                "run_id": run_id,
                "shard_index": idx,
                "total_shards": total_shards,
                "state_code": state_code,
                "county_fips": county_fips,
                "county_name": county_name,
                "h3_cells": cells,
            }
            for idx, cells in enumerate(shards)
        ]
        enqueue_sqs_shards(sqs_client, ml_sqs_url, shard_messages)
        print(
            f"[ML Planner] Planned {county_name} ({state_code}-{county_fips}): "
            f"{total_cells} H3 cells across {total_shards} shard(s) (run_id={run_id}).",
            flush=True,
        )

    except Exception as exc:
        supabase.table("county_lookup").update({
            "ml_status": "failed",
            "active_worker_id": None,
            "last_pipeline_error": str(exc)[:1500],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("state_code", state_code).eq("county_fips", county_fips).execute()
        raise


def write_shard_detections_parquet_to_s3(
    s3_client: Any,
    bucket: str,
    state_code: str,
    county_slug: str,
    run_id: str,
    shard_index: int,
    detections: List[Dict[str, Any]],
) -> str:
    rows = {
        "h3_cell": [str(d.get("h3_index_res9") or "") for d in detections],
        "lat": [float(d["lat"]) for d in detections],
        "lng": [float(d["lng"]) for d in detections],
        "ai_pool_confidence": [float(d.get("ai_pool_confidence", 0.0)) for d in detections],
        "ai_pool_sqft": [int(d.get("ai_pool_sqft", 0)) for d in detections],
        "ai_pool_condition": [str(d.get("ai_pool_condition") or "clean") for d in detections],
        "ai_has_spa": [bool(d.get("ai_has_spa", False)) for d in detections],
        "ai_tree_canopy_pct": [float(d.get("ai_tree_canopy_pct", 0.0)) for d in detections],
        "ai_tree_overhang": [bool(d.get("ai_tree_overhang", False)) for d in detections],
        "run_id": [run_id for _ in detections],
    }
    table = pa.Table.from_pydict(rows, schema=DETECTION_PARQUET_SCHEMA)

    tmp_dir = Path("/tmp/ml_parquet")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_parquet = tmp_dir / f"{run_id}_shard_{shard_index}.parquet"
    pq.write_table(table, str(local_parquet), compression="zstd")

    s3_key = (
        f"detections/{state_code.lower()}/{county_slug}/"
        f"run_id={run_id}/shard_{shard_index}.parquet"
    )
    try:
        s3_client.upload_file(str(local_parquet), bucket, s3_key)
    finally:
        local_parquet.unlink(missing_ok=True)

    return f"s3://{bucket}/{s3_key}"


def handle_run_shard(
    payload: Dict[str, Any],
    detector: PoolSpaCanopyDetector,
    s3_loader: S3TileLoader,
    supabase: Client,
    s3_client: Any,
    s3_data_bucket: str,
    buffer_meters: float,
) -> None:
    run_id = str(payload["run_id"])
    shard_index = int(payload["shard_index"])
    total_shards = int(payload.get("total_shards", 1))
    state_code = str(payload.get("state_code") or "TX").strip().upper()
    county_fips = str(payload["county_fips"]).strip()
    county_name = str(payload["county_name"]).strip()
    h3_cells: List[str] = list(payload.get("h3_cells") or [])
    county_slug = slugify_county(county_name)

    if not h3_cells:
        return

    tile_map: Dict[str, Path] = {}
    try:
        tile_map = s3_loader.fetch_batch(h3_cells, county=county_name)

        scanned_h3: List[str] = []
        detections: List[Dict[str, Any]] = []

        for h3_idx, tile_path in tile_map.items():
            try:
                pools, _hard_negs, _img_bgr = detector.analyze_tile(
                    tile_path=tile_path,
                    h3_index=h3_idx,
                    buffer_meters=buffer_meters,
                )
                scanned_h3.append(h3_idx)
                detections.extend(pools)
            except Exception as tile_exc:
                print(
                    f"[ML Shard Warn] Skipping unreadable tile {h3_idx} in shard {shard_index}: {tile_exc}",
                    flush=True,
                )

        # 1. S3 Persistence First: write shard Parquet before mutating PostGIS
        parquet_uri = write_shard_detections_parquet_to_s3(
            s3_client=s3_client,
            bucket=s3_data_bucket,
            state_code=state_code,
            county_slug=county_slug,
            run_id=run_id,
            shard_index=shard_index,
            detections=detections,
        )

        # 2. Spatially join detections and stamp scanned H3 cells in public.parcels
        matched_parcels = 0
        if scanned_h3:
            ingest_res = supabase.rpc(
                "ingest_ml_detections",
                {
                    "p_scanned_h3_cells": scanned_h3,
                    "p_detections": detections,
                    "p_run_id": run_id,
                },
            ).execute()
            if isinstance(ingest_res.data, int):
                matched_parcels = ingest_res.data

        # 3. Atomically increment county_lookup.scanned_h3_cells (+ ai_pool_count) & finalize on final shard
        progress_res = supabase.rpc(
            "increment_county_ml_progress",
            {
                "p_state_code": state_code,
                "p_county_fips": county_fips,
                "p_scanned_delta": len(h3_cells),
            },
        ).execute()
        progress_data = progress_res.data if isinstance(progress_res.data, dict) else {}
        scanned_now = int(progress_data.get("scanned_h3_cells", 0))
        total_planned = int(progress_data.get("total_h3_cells", 0))
        completed_now = bool(progress_data.get("completed_now", False))

        print(
            f"[ML Shard {shard_index + 1}/{total_shards}] {county_name}: "
            f"tiles={len(scanned_h3)}/{len(h3_cells)} detections={len(detections)} "
            f"matched={matched_parcels} progress={scanned_now}/{total_planned} s3={parquet_uri}",
            flush=True,
        )

        if completed_now or (total_planned > 0 and scanned_now >= total_planned):
            supabase.table("county_lookup").update({
                "ml_status": "completed",
                "last_ml_run_at": datetime.now(timezone.utc).isoformat(),
                "active_worker_id": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("state_code", state_code).eq("county_fips", county_fips).execute()

            if completed_now:
                print(
                    f"[ML Worker] Final shard completed for {county_name} ({state_code}-{county_fips}); "
                    "refreshing market_intelligence_metrics...",
                    flush=True,
                )
                supabase.rpc(
                    "refresh_market_intelligence_metrics",
                    {"p_state_code": state_code, "p_county_fips": county_fips},
                ).execute()

    except Exception as exc:
        supabase.table("county_lookup").update({
            "last_pipeline_error": f"Shard {shard_index} error: {str(exc)[:1400]}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("state_code", state_code).eq("county_fips", county_fips).execute()
        raise
    finally:
        if tile_map:
            s3_loader.cleanup_ephemeral(tile_map)


def main() -> None:
    with open("configs/pipeline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    aws_region = os.getenv("AWS_REGION", "us-west-2").strip() or "us-west-2"
    ml_sqs_url = os.getenv("ML_SQS_URL", "").strip()
    s3_data_bucket = os.getenv("S3_DATA_BUCKET", "").strip()
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

    if not ml_sqs_url:
        raise ValueError("Missing required environment variable: ML_SQS_URL")
    if not s3_data_bucket:
        raise ValueError("Missing required environment variable: S3_DATA_BUCKET")
    if not supabase_url or not supabase_service_key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

    shard_size = int(os.getenv("ML_SHARD_SIZE", "150"))
    models_dir = Path(os.getenv("ML_MODELS_DIR", "/app/models" if Path("/app").exists() else "./models"))
    cache_dir = os.getenv("ML_EPHEMERAL_CACHE_DIR", "/tmp/tiles_cache")
    imgsz = int(cfg.get("model", {}).get("imgsz", 640))
    conf_threshold = float(cfg.get("model", {}).get("conf_threshold", 0.45))
    hard_neg_range = tuple(cfg.get("model", {}).get("hard_negative_range", [0.25, 0.44]))
    buffer_meters = float(cfg.get("canopy", {}).get("buffer_meters", 15.0))

    sqs_client = boto3.client("sqs", region_name=aws_region)
    s3_client = boto3.client("s3", region_name=aws_region)
    supabase = create_client(supabase_url, supabase_service_key)
    worker_id = os.getenv("WORKER_ID") or socket.gethostname()

    weights_path, device = bootstrap_model_and_device(
        s3_client=s3_client,
        s3_data_bucket=s3_data_bucket,
        models_dir=models_dir,
        imgsz=imgsz,
    )

    detector = PoolSpaCanopyDetector(
        weights_path=str(weights_path),
        device=device,
        conf_threshold=conf_threshold,
        hard_neg_range=(float(hard_neg_range[0]), float(hard_neg_range[1])),
    )

    s3_loader = S3TileLoader(
        bucket=os.getenv("S3_TILE_BUCKET") or os.getenv("S3_TILES_BUCKET") or "growroute-naip-tiles",
        s3_prefix=os.getenv("S3_TILES_PREFIX", cfg.get("s3", {}).get("prefix", "tx/h3_res9")),
        cache_dir=cache_dir,
        file_ext=".tif",
        ephemeral_cache=True,
    )

    stop_requested = False

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        print(f"[ML Worker] Received signal {signum}; shutting down after current message...", flush=True)
        stop_requested = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print(
        f"[ML Worker] Ready worker={worker_id} device={device} weights={weights_path.name} "
        f"shard_size={shard_size} queue={ml_sqs_url}",
        flush=True,
    )

    while not stop_requested:
        resp = sqs_client.receive_message(
            QueueUrl=ml_sqs_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=SQS_WAIT_SECONDS,
            VisibilityTimeout=VISIBILITY_TIMEOUT_SEC,
        )
        messages = resp.get("Messages", [])
        if not messages:
            continue

        msg = messages[0]
        receipt_handle = msg["ReceiptHandle"]
        payload = json.loads(msg["Body"])
        mode = payload.get("mode", "")

        heartbeat = VisibilityHeartbeat(sqs_client, ml_sqs_url, receipt_handle)
        heartbeat.start()
        try:
            if mode == "plan_county":
                handle_plan_county(
                    payload=payload,
                    supabase=supabase,
                    sqs_client=sqs_client,
                    s3_client=s3_client,
                    ml_sqs_url=ml_sqs_url,
                    s3_data_bucket=s3_data_bucket,
                    worker_id=worker_id,
                    shard_size=shard_size,
                )
            elif mode == "run_shard":
                handle_run_shard(
                    payload=payload,
                    detector=detector,
                    s3_loader=s3_loader,
                    supabase=supabase,
                    s3_client=s3_client,
                    s3_data_bucket=s3_data_bucket,
                    buffer_meters=buffer_meters,
                )
            else:
                print(f"[ML Worker Warn] Unknown message mode={mode}", flush=True)

            sqs_client.delete_message(
                QueueUrl=ml_sqs_url,
                ReceiptHandle=receipt_handle,
            )
        finally:
            heartbeat.stop()


if __name__ == "__main__":
    main()
