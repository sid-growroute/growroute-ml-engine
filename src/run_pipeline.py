import argparse
import os
from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm
import yaml

from src.detectors.pool_canopy import PoolSpaCanopyDetector
from src.io.s3_tile_loader import S3TileLoader
from src.utils.hard_negatives import HardNegativeMiner


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run GrowRoute S3 .tif -> YOLO + Canopy -> PostGIS pipeline")
    parser.add_argument("--county", required=True, help="County name (e.g. DALLAS, COLLIN, TARRANT, DENTON)")
    parser.add_argument("--device", default="mps", help="mps (Mac mini), cpu (ONNX EC2), or 0 (CUDA GPU)")
    parser.add_argument("--batch-tiles", type=int, default=100, help="H3 tiles per S3 prefetch & PostGIS RPC flush")
    parser.add_argument("--max-tiles", type=int, default=1200, help="Total H3 tiles to scan in this run")
    parser.add_argument("--ephemeral-cache", action="store_true", help="Delete downloaded .tif files after each batch (for AWS EC2/Batch)")
    args = parser.parse_args()

    with open("configs/pipeline.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )

    s3_loader = S3TileLoader(
        bucket=os.getenv("S3_TILES_BUCKET", cfg["s3"]["bucket"]),
        s3_prefix=os.getenv("S3_TILES_PREFIX", cfg["s3"]["prefix"]),
        cache_dir="./tiles_cache",
        file_ext=".tif",
    )

    weights = cfg["model"]["weights_onnx"] if args.device == "cpu" else cfg["model"]["weights_pt"]
    detector = PoolSpaCanopyDetector(
        weights_path=weights,
        device=args.device,
        conf_threshold=cfg["model"]["conf_threshold"],
        hard_neg_range=tuple(cfg["model"]["hard_negative_range"]),
    )
    miner = HardNegativeMiner(out_dir="./runs/hard_negatives", max_saved=500)

    # 1. Fetch unscanned residential H3 cells from Supabase
    resp = supabase.rpc(
        "get_unscanned_h3_cells",
        {"p_county_name": args.county.upper(), "p_limit": args.max_tiles},
    ).execute()
    h3_cells = [row["h3_index_res9"] for row in (resp.data or [])]
    if not h3_cells:
        print(f"No unscanned H3 cells remaining for {args.county.upper()}.")
        return

    total_matched = 0
    for h3_batch in tqdm(list(chunk_list(h3_cells, args.batch_tiles)), desc=f"Batches ({args.county.upper()})"):
        # 2. Read from SSD cache or parallel-download .tif files from S3
        tile_map = s3_loader.fetch_batch(h3_batch, county=args.county)
        if not tile_map:
            continue

        scanned_h3 = []
        detections = []

        # 3. Run YOLO + 15m Canopy + Green Water check on each GeoTIFF
        for h3_idx, tile_path in tile_map.items():
            try:
                pools, hard_negs, img_bgr = detector.analyze_tile(
                    tile_path=tile_path,
                    h3_index=h3_idx,
                    buffer_meters=cfg["canopy"]["buffer_meters"],
                )
                scanned_h3.append(h3_idx)
                detections.extend(pools)

                if hard_negs:
                    miner.save_candidate(h3_idx, args.county, img_bgr, hard_negs)
            except Exception as exc:
                print(f"Skipping corrupted tile {h3_idx}: {exc}")

        # 4. Flush batch to PostGIS ingest_ml_detections RPC
        if scanned_h3:
            rpc_res = supabase.rpc(
                "ingest_ml_detections",
                {
                    "p_scanned_h3_cells": scanned_h3,
                    "p_detections": detections,
                    "p_run_id": cfg["run_id"],
                },
            ).execute()
            if rpc_res.data and isinstance(rpc_res.data, dict):
                total_matched += rpc_res.data.get("matched_detections", 0)

        # 5. Optional cleanup when running on small-disk AWS instances
        if args.ephemeral_cache:
            s3_loader.cleanup_ephemeral(tile_map)

    # 6. Refresh county rollups in public.market_intelligence_metrics
    supabase.rpc(
        "refresh_market_intelligence_metrics",
        {"p_county_name": args.county.upper(), "p_run_id": cfg["run_id"]},
    ).execute()
    print(
        f"Done! Matched {total_matched} pools in {args.county.upper()} "
        f"and saved {miner.saved_count} Roboflow hard-negative .jpg tiles."
    )


if __name__ == "__main__":
    main()