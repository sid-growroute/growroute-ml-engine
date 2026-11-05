import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Download labeled YOLO dataset from Roboflow to SSD ./data")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    parser.add_argument("--version", type=int, required=True, help="Dataset version number (e.g. 1)")
    parser.add_argument("--format", default="yolov8", help="Export format (yolov8 or yolov11)")
    parser.add_argument("--out-name", default=None, help="Folder name inside ./data (defaults to project_vN)")
    args = parser.parse_args()

    api_key = os.environ["ROBOFLOW_API_KEY"]
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    folder_name = args.out_name or f"{args.project}_v{args.version}"
    target_dir = Path("./data").resolve() / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    version.download(args.format, location=str(target_dir), overwrite=True)
    print(f"Downloaded Roboflow dataset to SSD -> {target_dir}")


if __name__ == "__main__":
    main()