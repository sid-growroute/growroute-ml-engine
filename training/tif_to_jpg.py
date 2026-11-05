import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import rasterio


def convert_one(args_tuple):
    tif_path, out_dir = args_tuple
    jpg_path = out_dir / f"{tif_path.stem}.jpg"
    if jpg_path.exists():
        return
    with rasterio.open(tif_path) as src:
        rgb = np.clip(src.read([1, 2, 3]), 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(np.transpose(rgb, (1, 2, 0)), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(jpg_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def main():
    parser = argparse.ArgumentParser(description="Batch convert S3 NAIP .tif tiles to .jpg for Roboflow labeling")
    parser.add_argument("--src-dir", required=True, help="Directory with .tif files")
    parser.add_argument("--out-dir", required=True, help="Output directory for .jpg files")
    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = list(src_dir.rglob("*.tif")) + list(src_dir.rglob("*.tiff"))
    with ProcessPoolExecutor() as pool:
        list(pool.map(convert_one, [(t, out_dir) for t in tifs]))
    print(f"Converted {len(tifs)} GeoTIFFs to .jpg in {out_dir}")


if __name__ == "__main__":
    main()