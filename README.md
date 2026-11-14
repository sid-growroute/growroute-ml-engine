# growroute-ml-engine 🛰️🏊

Spatial computer vision, 4-band USDA NAIP GeoTIFF caching, and distributed SQS shard inference engine for **GrowRoute**.

`growroute-ml-engine` scans residential **Uber H3 Resolution-9** hex cells across US counties, runs **YOLOv11s-seg** instance segmentation (`PoolSpaCanopyDetector`) to detect swimming pools and attached spas, measures surrounding **15m tree canopy overhang** (`>= 35%`) using 4-band Near-Infrared NDVI, classifies pool water condition (`clean` vs. `green`), persists shard-level Parquet detections to S3, and spatially joins pool leads onto `public.parcels` + `public.market_intelligence_metrics` in Supabase/PostGIS.

---

## 1. Overview & Spatial CV Architecture

### End-to-End Sharded SQS Architecture

```mermaid
flowchart TD
    DB["Supabase: public.county_lookup<br/>(SET run_ml = true, ml_max_tiles = 500 | 0)"] -->|pg_net AFTER UPDATE| EF["Edge Function:<br/>dispatch-county-pipeline"]
    EF -->|SigV4 StartInstances + plan_county| SQS["AWS SQS (us-west-2)<br/>growroute-{env}-ml-jobs"]
    SQS -->|mode = plan_county| P["Planner (src.aws_ml_worker)<br/>Paginates get_unscanned_h3_cells<br/>Writes S3 manifest + batches of 150 H3 cells"]
    P -->|send_message_batch (run_shard)| SQS
    SQS -->|mode = run_shard (1..N workers)| W["Shard Worker (CPU ONNX or GPU L4 CUDA)"]
    W --> T["3-Tier S3TileLoader (src/io/s3_tile_loader.py)"]
    T -->|Tier 1| T1["Ephemeral /tmp/tiles_cache<br/>(Auto-purged after every shard)"]
    T -->|Tier 2| T2["Shared S3 Cache (us-west-2)<br/>s3://growroute-naip-tiles/tx/h3_res9/&lt;county&gt;/"]
    T -->|Tier 3| T3["us-west-2 AWS Open Data NAIP COGs<br/>(VPC S3 Endpoint / Planetary Computer STAC)"]
    T1 & T2 & T3 --> D["PoolSpaCanopyDetector (YOLOv11s-seg)<br/>Pool + Spa + 15m NIR NDVI Canopy + Green Water"]
    D -->|1. S3 Persistence First| S3P["s3://growroute-{env}-data/detections/tx/&lt;county&gt;/<br/>run_id=&lt;id&gt;/shard_&lt;idx&gt;.parquet"]
    S3P -->|2. Spatial Join RPC| RPC["PostGIS: ingest_ml_detections<br/>+ increment_county_ml_progress"]
    RPC -->|Final shard completes| M["refresh_market_intelligence_metrics(state, fips)"]
```

### Core Architectural Design Principles

1. **H3 Resolution-9 Tile-Level Inference (Never Per-Parcel Imagery Downloads):**
   - Instead of fetching millions of individual property crops, `public.get_unscanned_h3_cells` groups unscanned parcels into **H3 Res-9** hex tiles (`~0.105 km²` each), prioritized by suburban single-family density (`30–250` parcels/hex) and known CAD pool counts. A single GeoTIFF covers `50–180` backyards in one pass.
2. **3-Tier Read-Through `S3TileLoader` (`src/io/s3_tile_loader.py`):**
   - **Tier 1 (`local /tmp/tiles_cache`):** Ephemeral per-shard working directory, automatically unlinked in a `finally` block (`cleanup_ephemeral`) so EC2 root volumes never fill up.
   - **Tier 2 (`s3://growroute-naip-tiles/tx/h3_res9/<county_slug>/<h3_cell>.tif`):** Shared private S3 cache in `us-west-2`. Newly cropped tiles write through (`S3_TILE_WRITE_THROUGH=true`) so Stage QA runs pre-warm tiles for Prod at zero re-cropping cost.
   - **Tier 3 (`us-west-2` USDA NAIP 4-Band RGB+NIR COGs):** Queries Element84 Earth Search STAC (`AWS_REQUEST_PAYER=requester`) to crop `EPSG:26914` / UTM 4-band windows (`+0.00035°` / `~35m` collar so boundary pools are never clipped) directly over the free `us-west-2` S3 VPC Gateway Endpoint, with automatic fallback to Microsoft Planetary Computer NAIP STAC.
3. **YOLOv11s-seg + Spectral Canopy & Water Analytics (`src/detectors/pool_canopy.py`):**
   - **Pool & Spa Segmentation:** Runs `YOLO(..., task="segment")` at `640x640` (`conf_threshold = 0.45`), reprojects pixel centroids via the GeoTIFF affine transform (`TileGeoContext.pixel_to_lnglat`) to `EPSG:4326` `(lng, lat)`, estimates `ai_pool_sqft` (`[150, 2500]`), and links any `spa` (class `1`) within `20m` (`ai_has_spa = true`).
   - **15m NDVI/HSV Tree Canopy Overhang:** Computes vegetation density in a `15m` ring outside each pool bbox using 4-band `NDVI = (NIR - Red) / (NIR + Red) >= 0.18` + shade filtering (or RGB Excess Green + HSV fallback), setting `ai_tree_overhang = true` when `ai_tree_canopy_pct >= 35.0%`.
   - **Pool Water Condition:** Classifies inner-pool HSV hue (`35–82`) & saturation (`>= 45`) into `"green"` (algae recovery lead) vs. `"clean"`.
4. **S3-First Shard Persistence & Race-Free PostGIS Rollup (`src/aws_ml_worker.py`):**
   - Every `run_shard` worker writes its Zstandard-compressed Parquet (`h3_cell, lat, lng, ai_pool_confidence, ai_pool_sqft, ai_pool_condition, ai_has_spa, ai_tree_canopy_pct, ai_tree_overhang, run_id`) to `s3://growroute-{env}-data/detections/tx/<county_slug>/run_id=<run_id>/shard_<idx>.parquet` **before** mutating Postgres (`pyarrow==25.0.1` pinned for NumPy `1.26.4` ABI safety).
   - Calls `public.ingest_ml_detections(p_scanned_h3_cells, p_detections, p_run_id)` to spatially join pool centroids against `public.parcels.geom` (`ST_Contains` prioritized with `~20m` `ST_DWithin` tolerance).
   - Calls `public.increment_county_ml_progress(p_state_code, p_county_fips, p_scanned_delta)` (`FOR UPDATE` row lock); the exact worker finishing the final shard (`scanned_h3_cells >= total_h3_cells`) marks `ml_status = 'completed'` and triggers `public.refresh_market_intelligence_metrics(state_code, county_fips)`.

### Repository Map

```text
growroute-ml-engine/
├── Dockerfile.cpu                       # Stage CPU image (Python 3.11, ONNX Runtime, pyarrow==25.0.1)
├── Dockerfile.gpu                       # Prod GPU image (CUDA 12.1 + cuDNN8, PyTorch 2.2.2, pyarrow==25.0.1)
├── requirements.txt                     # Local Python dependencies (pyarrow==25.0.1 pinned)
├── configs/
│   └── pipeline.yaml                    # Model thresholds (0.45), imgsz (640), 15m canopy buffer, S3 prefix
├── deploy/
│   └── aws/
│       ├── docker-compose.stage.yml     # Stage c7i.2xlarge: parcel-worker + scalable CPU ml-worker
│       └── docker-compose.prod-gpu.yml  # Prod g6.xlarge: NVIDIA L4 GPU ml-worker
├── models/                              # Local .pt / .onnx weights (optional; auto-bootstrapped from S3)
├── src/
│   ├── aws_ml_worker.py                 # Two-stage SQS worker (plan_county + run_shard) & S3 model bootstrapper
│   ├── detectors/
│   │   └── pool_canopy.py               # PoolSpaCanopyDetector: YOLOv11s-seg, CRS warp, NDVI canopy & algae check
│   ├── io/
│   │   └── s3_tile_loader.py            # 3-tier loader (/tmp -> s3://growroute-naip-tiles -> NAIP COGs)
│   ├── run_pipeline.py                  # Standalone local CLI runner (without SQS)
│   ├── orchestrator.py                  # Legacy Mac mini / NAS local Docker orchestrator
│   ├── dry_run_local.py                 # Offline tile inference smoke-test (zero DB writes)
│   └── utils/
│       └── hard_negatives.py            # Borderline [0.25, 0.44] Roboflow active-learning exporter
└── training/
    ├── pull_roboflow.py                 # Pulls labeled YOLOv11 segmentation datasets from Roboflow
    ├── tif_to_jpg.py                    # Converts 4-band NAIP GeoTIFFs to RGB .jpg for labeling
    └── train_yolo.py                    # Fine-tunes YOLOv11s-seg and exports .pt + .onnx
```

---

## 2. How to Trigger Single & Parallel ML Runs (Stage CPU & Prod GPU)

### Step 1: Trigger via SQL (`public.county_lookup`)

Updating `run_ml = true` fires `trg_county_lookup_dispatch_pipeline` -> `/functions/v1/dispatch-county-pipeline`, which wakes EC2 (if `STAGE_EC2_INSTANCE_ID` is set), marks `ml_status = 'queued'`, and posts `{ "mode": "plan_county", ... }` to `ML_SQS_URL`.

#### A. Stage QA Run (Capped at 500 H3 Tiles)
```sql
UPDATE public.county_lookup
SET ml_max_tiles = 500,
    run_ml = true,
    last_pipeline_error = NULL
WHERE state_code = 'TX'
  AND county_fips = '48085'; -- Collin County
```

#### B. Full Single-County Production Sweep (`ml_max_tiles = 0` = Unlimited)
Setting `ml_max_tiles = 0` (or `NULL`) instructs `aws_ml_worker.py` to paginate through **all** unscanned residential H3 cells in the county (`1,000` cells/page across PostgREST's row cap) and slice them into `150`-tile SQS shards:
```sql
UPDATE public.county_lookup
SET ml_max_tiles = 0,
    run_ml = true,
    last_pipeline_error = NULL
WHERE state_code = 'TX'
  AND county_fips = '48113'; -- Dallas County (full unscanned sweep)
```

#### C. Multi-County Parallel Sweep
```sql
UPDATE public.county_lookup
SET ml_max_tiles = 0,
    run_ml = true,
    last_pipeline_error = NULL
WHERE state_code = 'TX'
  AND county_fips IN ('48113', '48085', '48121', '48439', '48201'); -- DFW + Harris
```

### Step 2: Scale Worker Containers (Sequential -> Parallel CPU -> Prod GPU)

Because `plan_county` fans out independent `run_shard` messages (`ML_SHARD_SIZE=150`) onto `ML_SQS_URL` and shard progress updates use a row-locked SQL RPC (`increment_county_ml_progress`), you can scale from `1` container to `N` containers dynamically with zero coordination races:

#### 1. Stage Sequential CPU (`1` Worker)
```bash
cd ~/growroute-ml-engine
docker compose -f deploy/aws/docker-compose.stage.yml up -d --scale ml-worker=1
```

#### 2. Stage Parallel CPU (`4` Workers on `c7i.2xlarge`)
`docker-compose.stage.yml` pins `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and `TORCH_NUM_THREADS=2` per container so `4` parallel ONNX Runtime workers saturate all `8` vCPUs on `c7i.2xlarge` cleanly without OpenMP thread contention:
```bash
cd ~/growroute-ml-engine
docker compose -f deploy/aws/docker-compose.stage.yml up -d --scale ml-worker=4
```

#### 3. Prod NVIDIA L4 GPU (`g6.xlarge` via `docker-compose.prod-gpu.yml`)
`docker-compose.prod-gpu.yml` reserves the NVIDIA L4 GPU (`24 GiB` VRAM, `ML_DEVICE="0"`, `TILE_LOADER_WORKERS=8`). You can run `1` GPU worker or `--scale ml-worker=2` to overlap NAIP S3/STAC I/O with CUDA tensor inference on the same L4 GPU:
```bash
cd ~/growroute-ml-engine
docker compose -f deploy/aws/docker-compose.prod-gpu.yml up -d --scale ml-worker=2
```

### Step 3: Monitor Shard Progress & Verify Detections

```sql
SELECT
  county_fips,
  county_name,
  ml_status,
  scanned_h3_cells,
  total_h3_cells,
  CASE
    WHEN total_h3_cells > 0 THEN ROUND((scanned_h3_cells::NUMERIC / total_h3_cells) * 100, 1)
    ELSE 0
  END AS pct_complete,
  cad_pool_count,
  ai_pool_count,
  last_ml_run_at,
  last_pipeline_error
FROM public.county_lookup
WHERE state_code = 'TX'
  AND (ml_status IN ('queued', 'running', 'failed', 'completed') OR last_ml_run_at IS NOT NULL)
ORDER BY updated_at DESC;
```

```bash
# Watch live shard completions & S3 Parquet URIs on EC2
docker compose -f deploy/aws/docker-compose.stage.yml logs -f --tail=100 ml-worker

# Inspect persisted shard Parquet files in S3
aws s3 ls s3://growroute-stage-data/detections/tx/collin/ --recursive --human-readable --region us-west-2
```

---

## 3. Updating Code & Hot-Swapping Retrained YOLO Weights

### A. Model-Only Hot-Swap (Zero Docker Rebuild)

At startup, `bootstrap_model_and_device()` in `src/aws_ml_worker.py` resolves weights for the active device (`cpu` -> `pool_spa_dfw_v2.onnx`; `0` GPU -> `pool_spa_dfw_v2.pt`) from `ML_MODELS_DIR` and `s3://${S3_DATA_BUCKET}/models/` (and if only `.pt` exists on a CPU worker, it auto-exports `640x640` ONNX once and uploads `models/pool_spa_dfw_v2.onnx` back to S3).

1. **Export (if needed) and upload updated weights to S3:**
   ```bash
   export S3_DATA_BUCKET="growroute-stage-data" # or growroute-prod-data

   # Upload PyTorch checkpoint (used natively by Prod GPU workers & as CPU fallback)
   AWS_REQUEST_CHECKSUM_CALCULATION=when_required aws s3 cp \
     models/pool_spa_dfw_v2.pt \
     "s3://${S3_DATA_BUCKET}/models/pool_spa_dfw_v2.pt" \
     --region us-west-2

   # Upload simplified 640x640 ONNX weights (used by Stage CPU workers)
   AWS_REQUEST_CHECKSUM_CALCULATION=when_required aws s3 cp \
     models/pool_spa_dfw_v2.onnx \
     "s3://${S3_DATA_BUCKET}/models/pool_spa_dfw_v2.onnx" \
     --region us-west-2
   ```
2. **Restart `ml-worker` on EC2 to pull fresh S3 weights (zero image rebuild):**
   *(Set `ML_MODELS_DIR=/tmp/models` in `.env` — or keep local `models/` empty when building images — so container recreate always downloads the latest S3 object rather than reusing a baked-in file in `/app/models/`.)*
   ```bash
   # Stage CPU host
   cd ~/growroute-ml-engine
   ML_MODELS_DIR=/tmp/models docker compose -f deploy/aws/docker-compose.stage.yml up -d --force-recreate --scale ml-worker=4

   # Prod GPU host
   cd ~/growroute-ml-engine
   ML_MODELS_DIR=/tmp/models docker compose -f deploy/aws/docker-compose.prod-gpu.yml up -d --force-recreate --scale ml-worker=2
   ```

### B. Shipping Worker / Detector Code Updates (`Dockerfile.cpu` & `Dockerfile.gpu`)

```bash
export ECR_REGISTRY="226259226741.dkr.ecr.us-west-2.amazonaws.com"
export ECR_REPO="${ECR_REGISTRY}/growroute-ml-engine"

aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"

cd /Users/karthik/src/growroute-llc/growroute-ml-engine

# 1. Build & push Stage CPU image (linux/amd64)
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.cpu \
  -t "${ECR_REPO}:stage-cpu" \
  --push .

# 2. Build & push Prod CUDA GPU image (linux/amd64)
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.gpu \
  -t "${ECR_REPO}:prod-gpu" \
  --push .
```

Roll workers cleanly on EC2 (`aws_ml_worker.py` catches `SIGTERM` and completes its active `150`-tile shard before exiting):
```bash
# On Stage EC2 (CPU)
export ML_WORKER_CPU_IMAGE="226259226741.dkr.ecr.us-west-2.amazonaws.com/growroute-ml-engine:stage-cpu"
cd ~/growroute-ml-engine
docker compose -f deploy/aws/docker-compose.stage.yml pull ml-worker
docker compose -f deploy/aws/docker-compose.stage.yml stop -t 300 ml-worker
docker compose -f deploy/aws/docker-compose.stage.yml up -d --scale ml-worker=4

# On Prod EC2 (GPU)
export ML_WORKER_GPU_IMAGE="226259226741.dkr.ecr.us-west-2.amazonaws.com/growroute-ml-engine:prod-gpu"
cd ~/growroute-ml-engine
docker compose -f deploy/aws/docker-compose.prod-gpu.yml pull ml-worker
docker compose -f deploy/aws/docker-compose.prod-gpu.yml stop -t 300 ml-worker
docker compose -f deploy/aws/docker-compose.prod-gpu.yml up -d --scale ml-worker=2
```

---

## 4. Stage CPU -> Prod GPU Promotion Guide & Cost Operations

### 1. Verify AWS `G and VT` GPU vCPU Quota in `us-west-2`
`g6.xlarge` (`1x NVIDIA L4`, `4` vCPUs, `16 GiB` RAM) requires at least **`4` vCPUs** under Service Quota `L-DB2E81BA` (*Running On-Demand G and VT instances*) in `us-west-2`:
```bash
# Check current On-Demand G/VT vCPU quota in us-west-2
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --region us-west-2 \
  --query 'Quota.{Name:QuotaName,Value:Value}'

# Request quota increase to 8 vCPUs (allows up to 2x g6.xlarge) if Value < 4
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --desired-value 8 \
  --region us-west-2
```

### 2. Launch `g6.xlarge` (Ubuntu 24.04 Deep Learning Nvidia Driver AMI)
1. Resolve the latest **AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)** in `us-west-2` (ships with pre-installed NVIDIA drivers, Docker, and `nvidia-container-toolkit`):
   ```bash
   aws ec2 describe-images \
     --owners amazon \
     --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" \
               "Name=state,Values=available" \
     --query 'reverse(sort_by(Images, &CreationDate))[0].{ImageId:ImageId,Name:Name}' \
     --region us-west-2
   ```
2. Launch `g6.xlarge` in the `us-west-2` subnet attached to your **S3 Gateway VPC Endpoint** (`100 GB gp3` root volume + Prod EC2 IAM Instance Profile), SSH in, and verify GPU passthrough:
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```
3. Populate `~/growroute-ml-engine/.env` (`S3_DATA_BUCKET=growroute-prod-data`, `S3_TILE_BUCKET=growroute-naip-tiles`, `ML_SQS_URL=https://sqs.us-west-2.amazonaws.com/226259226741/growroute-prod-ml-jobs`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`) and start the GPU worker:
   ```bash
   export ML_WORKER_GPU_IMAGE="226259226741.dkr.ecr.us-west-2.amazonaws.com/growroute-ml-engine:prod-gpu"
   cd ~/growroute-ml-engine
   docker compose -f deploy/aws/docker-compose.prod-gpu.yml up -d --scale ml-worker=2
   ```

### 3. Shared `s3://growroute-naip-tiles` Cache & VPC Cost Controls
- **Zero Duplicate Imagery Work:** Keep `S3_TILE_BUCKET=growroute-naip-tiles` identical across Stage and Prod. Every H3 `.tif` tile cropped during Stage QA is immediately reused via Tier 2 (`s3.download_file`) during Prod runs in milliseconds, avoiding repeat STAC lookups and requester-pays NAIP COG reads.
- **$0 S3 Data Transfer:** Keep EC2 workers, `growroute-{env}-data`, `growroute-naip-tiles`, and AWS Open Data NAIP in `us-west-2` behind a free **VPC S3 Gateway Endpoint** so multi-terabyte GeoTIFF + Parquet traffic never traverses a paid NAT Gateway.

### 4. Auto-Wake & Stop Idle EC2 Compute (`$0/hr` Idle Compute)
Because Docker services use `restart: unless-stopped`, starting a stopped EC2 host automatically boots `parcel-worker` / `ml-worker` and begins draining SQS within ~35 seconds:
- **Auto-Wake on DB Trigger:** Set `STAGE_EC2_INSTANCE_ID` in `dispatch-county-pipeline` Edge Function secrets (in Stage, set your Stage `c7i.2xlarge` ID; in Prod's Supabase project, set your Prod EC2 instance ID or comma-separated `i-parcelprod,i-mlgpuprod` — `wakeStageEc2IfConfigured` in `dispatch-county-pipeline/index.ts` calls SigV4 `ec2:StartInstances` on every comma-separated ID idempotently).
- **Stop Idle Hosts After Queue Drains:** Once `county_lookup` shows `0` `queued`/`running` counties and SQS `ApproximateNumberOfMessages` + `ApproximateNumberOfMessagesNotVisible` both equal `0`, stop the EC2 instance so only EBS storage (`~$8/mo` per 100 GB `gp3`) is billed while idle:
  ```bash
  # Stop idle Stage or Prod EC2 worker host when batch completes
  aws ec2 stop-instances --instance-ids i-xxxxxxxxxxxxxxxxx --region us-west-2
  ```
