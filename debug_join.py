from pathlib import Path
import rasterio
from rasterio.warp import transform

# 1. Check the CRS and raw vs. EPSG:4326 coordinates of the cached Tarrant GeoTIFF
tif = Path("/Volumes/GrowrouteLLC/growroute-data/ml/naip_h3_tiles/tarrant/tx_tarrant_hex_8926c86810fffff.tif")
with rasterio.open(tif) as src:
    raw_x, raw_y = src.xy(src.height // 2, src.width // 2)
    lon, lat = transform(src.crs, "EPSG:4326", [raw_x], [raw_y])
    print("=== 1. TILE CRS & CENTER COORDINATES ===")
    print(f"Tile CRS         : {src.crs}")
    print(f"Raw src.xy()     : x={raw_x}, y={raw_y}")
    print(f"Reprojected 4326 : lng={lon[0]:.6f}, lat={lat[0]:.6f}")

# 2. Print all .py files in src/ so we can see how YOLO() and src.xy() are called
print("\n=== 2. SRC FILES ===")
for p in sorted(Path("src").rglob("*.py")):
    if p.name == "orchestrator.py" or p.name == "s3_tile_loader.py":
        continue
    print(f"\n--- {p} ---")
    print(p.read_text())
