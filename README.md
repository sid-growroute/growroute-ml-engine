# growroute-ml-engine 🛰️🏊

Spatial computer vision, 4-band USDA NAIP GeoTIFF caching, above-ground vs. in-ground pool classification, and distributed SQS shard inference engine for **GrowRoute**.

`growroute-ml-engine` scans residential **Uber H3 Resolution-9** hex cells across US counties, runs **YOLOv11s-seg** segmentation (`PoolSpaCanopyDetector`) over 4-band RGB+NIR NAIP GeoTIFFs, classifies `ai_pool_type` (`inground` vs `above_ground`) and `ai_pool_condition` (`clean` vs `green`), measures surrounding **15m tree canopy overhang** (`>= 35%`), persists shard-level Zstd Parquet detections to S3, and spatially joins pool leads onto active `public.parcels` + `public.market_intelligence_metrics` in Supabase/PostGIS.

---

## 1. AWS SQS Shard Architecture & Compute Variants

```mermaid
flowchart TD
    DB["Supabase: public.county_lookup<br/>(SET run_ml = true, ml_max_tiles = 250 | 0)"] -->|pg_net AFTER UPDATE| EF["Edge Function:<br/>dispatch-county-pipeline"]
    EF -->|SigV4 StartInstances + plan_county| SQS["AWS SQS (us-west-2)<br/>growroute-{env}-ml-jobs"]
    SQS -->|mode = plan_county| P["Planner (src.aws_ml_worker)<br/>Paginates get_unscanned_h3_cells (is_active=true)<br/>Writes S3 manifest + 150-tile shards"]
    P -->|send_message_batch (run_shard)| SQS
    SQS -->|mode = run_shard (1..N workers)| W["Shard Worker<br/>Stage: CPU ONNX (:stage-cpu)<br/>Prod: NVIDIA L4 PyTorch (:prod-gpu)"]
    W --> T["3-Tier S3TileLoader (src/io/s3_tile_loader.py)<br/>/tmp -> s3://growroute-naip-tiles -> NAIP 4-band COGs"]
    T --> D["PoolSpaCanopyDetector (YOLOv11s-seg)<br/>UTM->WGS84 + NDWI Water + pool_type + Canopy %"]
    D -->|1. S3 Parquet First| S3P["s3://growroute-{env}-data/detections/tx/&lt;county&gt;/<br/>run_id=&lt;id&gt;/shard_&lt;idx&gt;.parquet"]
    S3P -->|2. PostGIS Spatial Join| RPC["ingest_ml_detections (p_model_version, pool_type)<br/>+ increment_county_ml_progress"]
    RPC -->|Final shard completes| M["refresh_market_intelligence_metrics(state, fips)"]
```

### Execution Lifecycle & Cooperative Controls

1. **Two-Phase SQS Fan-Out (`src/aws_ml_worker.py`):**
   - **`plan_county`:** Paginates `public.get_unscanned_h3_cells` across PostgREST's 1,000-row page cap (`is_active = true` parcels only, prioritized by 30–250 parcel suburban density), writes `manifests/tx/<county_slug>/<run_id>.json` to S3, and enqueues `150`-tile `run_shard` batches (`ML_SHARD_SIZE=150`) with a 5-minute SQS `VisibilityHeartbeat`.
   - **`run_shard`:** Pulls 4-band `.tif` tiles via `S3TileLoader` (unlinking `/tmp/tiles_cache` in `finally`), writes Zstd Parquet (`h3_cell, lat, lng, pool_type, ai_pool_confidence, ai_pool_sqft, ai_pool_condition, ai_has_spa, ai_tree_canopy_pct, ai_tree_overhang, run_id`) to S3 **first**, calls `public.ingest_ml_detections(..., p_model_version)`, and atomically advances `public.increment_county_ml_progress`. The worker completing the final shard marks `ml_status = 'completed'` and refreshes `market_intelligence_metrics`.
2. **CPU ONNX (`:stage-cpu`) vs. GPU PyTorch (`:prod-gpu`):**
   - **Stage (`Dockerfile.cpu` -> `:stage-cpu` on `c7i.2xlarge`):** Runs ONNX Runtime (`pool_spa_dfw_v2.onnx`, auto-exported from `.pt` and cached to `s3://<S3_DATA_BUCKET>/models/` if missing) with `OMP_NUM_THREADS=2` across `--scale ml-worker=4`.
   - **Prod (`Dockerfile.gpu` -> `:prod-gpu` on `g6.xlarge` NVIDIA L4):** Runs CUDA 12.1 + PyTorch 2.2.2 (`pool_spa_dfw_v2.pt`, `ML_DEVICE="0"`) at `--scale ml-worker=2` (`pyarrow==25.0.1` + `numpy==1.26.4` pinned in both images).
3. **Cooperative `paused` / `cancelled` Shard Draining:**
   - Before both `plan_county` and `run_shard`, workers check `county_lookup.ml_status`. When `paused` or `cancelled` (via `make pause` / `make cancel`), queued SQS shards are acknowledged and dropped cleanly without scanning or tripping DLQ `maxReceiveCount`, and in-flight shards never overwrite `paused`/`cancelled` back to `completed`. Resuming (`make resume`) picks up seamlessly from `ai_last_verified_at IS NULL` H3 cells.

---

## 2. Georeferencing & Computer Vision Pipeline (`src/detectors/pool_canopy.py`)

1. **Exact 4-Band NAIP Georeferencing (`EPSG:26914` UTM -> `EPSG:4326` WGS84):**
   - `S3TileLoader` crops 4-band (`R, G, B, NIR`) USDA NAIP COGs (`0.6 m/px`, `+0.00035°` / `~35m` collar around each H3 Res-9 hex so boundary pools are never clipped) in native projected UTM (`EPSG:26914` / `EPSG:26915`) and caches them in shared `s3://growroute-naip-tiles/tx/h3_res9/<county_slug>/<h3_cell>.tif`.
   - `TileGeoContext.pixel_to_lnglat` applies `src.transform * (x_col, y_row)` directly in continuous pixel-corner coordinates (avoiding `rasterio.transform.xy`'s `+0.5px` shift) and reprojects to `EPSG:4326` via a cached `pyproj.Transformer(..., always_xy=True)`.
   - Pool pins use `shapely.Polygon.representative_point()` when a kidney/L-shaped centroid falls outside water, guaranteeing strict `ST_Contains(parcels.geom, pt_geom)` parcel matching (zero 1–2 lot drift).
2. **Canonical RGB Input & Letterbox Unscaling:**
   - GeoTIFF bands `[1, 2, 3]` are transposed into a contiguous `uint8` RGB array and passed as `PIL.Image.fromarray(img_rgb)` into Ultralytics `YOLO.predict(imgsz=640)` so RGB/BGR channels are never inverted, while `preds.masks.xy` and `preds.boxes.xyxy` automatically unscale `640x640` letterbox padding back to exact native GeoTIFF pixel coordinates.
3. **4-Band Spectral Water Gate & `ai_pool_type` (`inground` vs. `above_ground`):**
   - **Water Verification & Small-Pool Recovery:** Uses 1px-eroded interior `NDWI = (G - NIR) / (G + NIR)` + HSV cyan-blue/aqua ratios to reject NIR-reflective non-water false positives (pergolas, shade sails, trampolines, roofs) while recovering small circular above-ground pools in the `[0.38, 0.55)` confidence band (`100–600 sqft`, `0.78 <= w/h <= 1.28`, `NDWI > 0.02`, blue/cyan water ratio `>= 0.45`).
   - **`ai_pool_type` Classification:** Maps native YOLO `above_ground*` classes directly when present; otherwise classifies a verified pool as `'above_ground'` when no `spa` adjoins it, `100 <= pool_sqft <= 625`, mask circularity `>= 0.76` (or compact round/frame aspect ratio), and its `2–4 px` (`1.2m–2.4m`) outer ring lacks a bright low-NDVI concrete coping apron (`>= 50%` turf `NDVI >= 0.20` or `>= 20%` dark raised-wall shadow arc `V < 0.28` with `< 35%` bright concrete). In `ingest_ml_detections`, any parcel with `cad_has_pool = true` is authoritatively stamped `'inground'` (TX CAD only assesses permanent in-ground pools).
4. **Water Condition & 15m Canopy Overhang:**
   - **`ai_pool_condition`:** Classifies interior water HSV (`35 <= H <= 82`, `S >= 42`) into `'green'` (algae recovery lead) vs. `'clean'`.
   - **`ai_tree_canopy_pct` & `ai_tree_overhang`:** Measures shaded high-NDVI (`>= 0.18`) tree cover within a `15m` buffer around the pool (`ai_tree_overhang = true` at `>= 35%`).

---

## 3. Ground-Truth Verification Scripts

```bash
cd /Users/karthik/src/growroute-llc/growroute-ml-engine

# 1. Verify lot-exact georeferencing & pergola rejection across 3 Stage H3 tiles
python3 scripts/verify_ground_truth_tiles.py

# 2. Verify Above-Ground (3 coords) vs. In-Ground (2 Berrywood Tr controls) in Tarrant County
#    Writes annotated PNGs + JSON report to /tmp/growroute_ag_check/
python3 scripts/verify_above_ground_pools.py --strict
```

---

## 4. Deployment, CI/CD Promotion & Operations Runbook

### CI/CD (`deploy-ecr.yml` — `main` -> Stage, `release` -> Prod)
- **Push to `main`:** Runs Python syntax gate and publishes `:stage-cpu`, `:stage`, and `:sha-<short>` via keyless AWS OIDC (`growroute-github-ecr-deploy-role`). Commits touching only `README.md`, `**/*.md`, `docs/**`, `scripts/**`, `training/**`, or `migrations/**` are ignored on `push`.
- **Push / PR merge to `release`:** Gated by GitHub Environment `production`, publishes `:prod-cpu`, `:prod`, and `:release-<YYYYMMDD>-<short_sha>`, and enforces the **Zero-Idle-Cost Deploy Rule** on `i-0f914b78bfe09b165` (never wakes a `stopped` `$0.00/hr` worker; restarts `growroute-workers.service` via SSM only if `running` with `0` active SQS jobs).
- **Day-to-Day Operations CLI:** Full Stage & Prod operational tooling (`make smoke-stage`, `make smoke-prod`, `make inspect-prod`, `make status-prod`, `make logs-prod`, `ops/pipeline_ctl.py`) lives in `../growroute-parcel-engine/ops/` — see [`growroute-parcel-engine/README.md`](../growroute-parcel-engine/README.md) for the complete Stage & Prod runbook.

### Handy `ml-worker`, Inspector & DLQ Debugging Commands

```bash
# 1. Run non-destructive Stage/Prod smoke test & deep ML shard inspector (works while EC2 is stopped)
make -C ../growroute-parcel-engine smoke-prod
python3 ../growroute-parcel-engine/ops/inspect_pipelines.py --env prod --pipeline ml --county Dallas
make -C ../growroute-parcel-engine logs-prod

# 2. Inspect CPU/RAM container stats (Stage) or NVIDIA L4 GPU utilization (Prod) via AWS SSM
CMD_ID=$(aws ssm send-command \
  --instance-ids "i-06e0d9784875b9c3c" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "docker ps --filter name=ml-worker",
    "docker stats --no-stream",
    "command -v nvidia-smi >/dev/null && nvidia-smi || lscpu | grep -E \"Model name|CPU\\(s\\):\""
  ]' \
  --region us-west-2 --query "Command.CommandId" --output text)

sleep 4 && aws ssm get-command-invocation \
  --command-id "$CMD_ID" --instance-id "i-06e0d9784875b9c3c" \
  --region us-west-2 --query "[StandardOutputContent, StandardErrorContent]" --output text

# 3. Check ML SQS shard backlog & peek ML Dead-Letter Queue (DLQ)
aws sqs get-queue-attributes \
  --queue-url "https://sqs.us-west-2.amazonaws.com/226259226741/growroute-stage-ml-jobs" \
  --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible \
  --region us-west-2

aws sqs receive-message \
  --queue-url "https://sqs.us-west-2.amazonaws.com/226259226741/growroute-stage-ml-jobs-dlq" \
  --max-number-of-messages 5 --visibility-timeout 15 --region us-west-2
```

```sql
-- 4. Inspect inground vs. above_ground breakdown in Supabase after an ML run
SELECT county_name, ai_pool_type, pool_detection_source, COUNT(*) AS parcels
FROM public.parcels
WHERE state_code = 'TX' AND is_active IS TRUE AND (ai_has_pool IS TRUE OR cad_has_pool IS TRUE)
GROUP BY 1, 2, 3
ORDER BY 1, 4 DESC;
```
