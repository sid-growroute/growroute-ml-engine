import argparse
import json
from pathlib import Path
from src.detectors.pool_canopy import PoolSpaCanopyDetector


def main():
    parser = argparse.ArgumentParser(description="Dry-run YOLO + Canopy on 10 local tiles without Supabase")
    parser.add_argument("--tiles-dir", default="./tiles_cache", help="Folder with .tif or .jpg tiles")
    parser.add_argument("--model", default="models/pool_spa_dfw_v2.onnx", help="Path to .onnx or .pt model")
    parser.add_argument("--limit", type=int, default=10, help="Number of tiles to test")
    args = parser.parse_args()

    detector = PoolSpaCanopyDetector(weights_path=args.model, device="cpu", conf_threshold=0.45)

    tiles_dir = Path(args.tiles_dir).resolve()
    tiles = sorted(
        list(tiles_dir.rglob("*.tif"))
        + list(tiles_dir.rglob("*.tiff"))
        + list(tiles_dir.rglob("*.jpg"))
    )[: args.limit]

    if not tiles:
        print(f"No .tif or .jpg tiles found in {tiles_dir}")
        return

    all_detections = []
    for tile_path in tiles:
        h3_index = tile_path.stem.split("_")[-1]
        pools, hard_negs, _ = detector.analyze_tile(tile_path=tile_path, h3_index=h3_index)
        print(f"[{tile_path.name}] -> {len(pools)} pools detected, {len(hard_negs)} borderline hard-negatives")
        all_detections.extend(pools)

    print("\nSample PostGIS Payload Preview (First 3 Detections):")
    print(json.dumps(all_detections[:3], indent=2))


if __name__ == "__main__":
    main()