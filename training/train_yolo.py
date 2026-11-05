import argparse
from pathlib import Path
import shutil
import yaml
from ultralytics import YOLO


def resolve_roboflow_yaml(dataset_dir: Path) -> Path:
    """Rewrites Roboflow data.yaml 'path' to the resolved SSD absolute path."""
    yaml_path = dataset_dir.resolve() / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing data.yaml in {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    resolved_root = dataset_dir.resolve()
    cfg["path"] = str(resolved_root)
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images"
    if (resolved_root / "test" / "images").exists():
        cfg["test"] = "test/images"

    resolved_yaml = resolved_root / "data_resolved.yaml"
    with open(resolved_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return resolved_yaml


def main():
    parser = argparse.ArgumentParser(description="Train Pool & Spa YOLO on SSD + export ONNX")
    parser.add_argument("--dataset", required=True, help="Path inside ./data (e.g. data/pool_spa_dfw_v1)")
    parser.add_argument("--base-model", default="yolov8s.pt", help="Base YOLO checkpoint")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps", help="mps (Mac mini), 0 (CUDA GPU), or cpu")
    parser.add_argument("--export-name", default="pool_spa_dfw_v1", help="Output model name in ./models/")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    resolved_yaml = resolve_roboflow_yaml(dataset_dir)
    runs_dir = Path("./runs").resolve()

    model = YOLO(args.base_model)
    results = model.train(
        data=str(resolved_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(runs_dir),
        name=args.export_name,
        exist_ok=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)

    target_pt = models_dir / f"{args.export_name}.pt"
    shutil.copy2(best_pt, target_pt)
    print(f"Saved PyTorch weights -> {target_pt}")

    # Export ONNX for fast CPU EC2 staging
    best_model = YOLO(str(target_pt))
    onnx_path = best_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    target_onnx = models_dir / f"{args.export_name}.onnx"
    if Path(onnx_path).exists() and Path(onnx_path) != target_onnx:
        shutil.move(onnx_path, target_onnx)
    print(f"Saved ONNX weights -> {target_onnx}")


if __name__ == "__main__":
    main()