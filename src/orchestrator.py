import argparse
import os
from pathlib import Path
import socket
import subprocess
import time
import boto3
from dotenv import load_dotenv
from supabase import create_client


def verify_nas_root(nas_root: Path) -> None:
    """Verifies the Synology SMB share (/Volumes/GrowrouteLLC/growroute-data) is mounted."""
    if not nas_root.exists():
        raise RuntimeError(
            f"Synology NAS share not mounted at {nas_root}. "
            "Mount smb://YOUR_NAS_IP/growroute-data in Finder first."
        )
    (nas_root / "ml" / "naip_h3_tiles").mkdir(parents=True, exist_ok=True)
    (nas_root / "ml" / "training_runs").mkdir(parents=True, exist_ok=True)
    (nas_root / "parcels").mkdir(parents=True, exist_ok=True)


def run_parcel_docker_container(
    county_name: str,
    county_fips: str,
    state_code: str,
    nas_root: Path,
    env_file: Path,
    docker_image: str = "growroute-parcel:latest",
) -> None:
    """Spins up a Docker container for a single county parcel ingestion run using NAS storage."""
    parcels_mount = f"{nas_root / 'parcels'}:/app/data_nas"
    cmd = [
        "docker", "run", "--rm",
        "--name", f"growroute-parcel-{county_name.lower()}",
        "--env-file", str(env_file),
        "-e", "OMP_NUM_THREADS=6",
        "-v", parcels_mount,
        docker_image,
        "--county", county_name.upper(),
        "--fips", county_fips,
        "--state", state_code.upper(),
    ]
    print(f"[Parcel Docker] Starting {county_name.upper()} ({county_fips})...")
    subprocess.run(cmd, check=True)


def run_ml_docker_container(
    county_name: str,
    max_tiles: int,
    nas_root: Path,
    env_file: Path,
    docker_image: str = "growroute-ml:cpu",
) -> None:
    """Spins up a Docker container for a single county ML scan using NAS storage."""
    tiles_mount = f"{nas_root / 'ml' / 'naip_h3_tiles'}:/app/tiles_cache"
    runs_mount = f"{nas_root / 'ml' / 'training_runs'}:/app/runs"
    cmd = [
        "docker", "run", "--rm",
        "--name", f"growroute-ml-{county_name.lower()}",
        "--env-file", str(env_file),
        "-e", "OMP_NUM_THREADS=6",
        "-v", tiles_mount,
        "-v", runs_mount,
        docker_image,
        "--county", county_name.upper(),
        "--device", "cpu",
        "--batch-tiles", "100",
        "--max-tiles", str(max_tiles),
    ]
    print(f"[ML Docker] Starting {county_name.upper()} (max_tiles={max_tiles})...")
    subprocess.run(cmd, check=True)


def submit_aws_batch_job(
    pipeline: str,
    county_name: str,
    county_fips: str,
    max_tiles: int,
) -> str:
    """Submits either a Parcel or ML county container job to AWS Batch."""
    batch = boto3.client("batch", region_name=os.getenv("AWS_REGION", "us-east-1"))
    if pipeline == "ml":
        queue = os.environ["AWS_BATCH_ML_QUEUE"]
        job_def = os.environ["AWS_BATCH_ML_JOB_DEF"]
        command = [
            "--county", county_name.upper(),
            "--device", "0",
            "--batch-tiles", "250",
            "--max-tiles", str(max_tiles),
            "--ephemeral-cache",
        ]
    else:
        queue = os.environ["AWS_BATCH_PARCEL_QUEUE"]
        job_def = os.environ["AWS_BATCH_PARCEL_JOB_DEF"]
        command = [
            "--county", county_name.upper(),
            "--fips", county_fips,
        ]

    resp = batch.submit_job(
        jobName=f"growroute-{pipeline}-{county_name.lower()}-{int(time.time())}",
        jobQueue=queue,
        jobDefinition=job_def,
        containerOverrides={"command": command},
    )
    return resp["jobId"]


def process_next_queued_job(
    supabase,
    pipeline: str,
    executor: str,
    worker_id: str,
    nas_root: Path,
    env_file: Path,
) -> bool:
    """Claims and executes 1 job for the given pipeline ('parcel' or 'ml'). Returns True if a job ran."""
    resp = supabase.rpc(
        "claim_next_county_job",
        {"p_pipeline": pipeline, "p_worker_id": worker_id},
    ).execute()
    jobs = resp.data or []
    if not jobs:
        return False

    job = jobs[0]
    county_fips = job["county_fips"]
    county_name = job["county_name"]
    state_code = job.get("state_code") or "TX"
    max_tiles = job.get("ml_max_tiles") or 10000

    try:
        if executor == "docker-local":
            verify_nas_root(nas_root)
            if pipeline == "parcel":
                run_parcel_docker_container(
                    county_name=county_name,
                    county_fips=county_fips,
                    state_code=state_code,
                    nas_root=nas_root,
                    env_file=env_file,
                )
            else:
                run_ml_docker_container(
                    county_name=county_name,
                    max_tiles=max_tiles,
                    nas_root=nas_root,
                    env_file=env_file,
                )
        else:
            job_id = submit_aws_batch_job(
                pipeline=pipeline,
                county_name=county_name,
                county_fips=county_fips,
                max_tiles=max_tiles,
            )
            print(f"[AWS Batch] Dispatched {pipeline.upper()} for {county_name} -> Job {job_id}")

        supabase.rpc(
            "complete_county_job",
            {
                "p_county_fips": county_fips,
                "p_pipeline": pipeline,
                "p_success": True,
            },
        ).execute()
        print(f"[Completed] {pipeline.upper()} finished for {county_name} ({county_fips}).")

    except Exception as exc:
        err_msg = str(exc)[:500]
        supabase.rpc(
            "complete_county_job",
            {
                "p_county_fips": county_fips,
                "p_pipeline": pipeline,
                "p_success": False,
                "p_error_msg": err_msg,
            },
        ).execute()
        print(f"[Failed] {pipeline.upper()} failed for {county_name} ({county_fips}): {err_msg}")

    return True


def main():
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    parser = argparse.ArgumentParser(description="GrowRoute County Pipeline Orchestrator")
    parser.add_argument(
        "--executor",
        choices=["docker-local", "aws-batch"],
        default="docker-local",
        help="Run sequentially in local Docker w/ Synology NAS or submit to AWS Batch",
    )
    parser.add_argument(
        "--nas-root",
        default="/Volumes/GrowrouteLLC/growroute-data",
        help="Root path of mounted Synology SMB share",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll county_lookup every 30 seconds",
    )
    args = parser.parse_args()

    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )
    worker_id = f"{socket.gethostname()}-{args.executor}"
    nas_root = Path(args.nas_root).resolve()
    env_file = (repo_root / ".env").resolve()

    while True:
        # 1. Always prioritize queued Parcel jobs before ML jobs
        ran_parcel = process_next_queued_job(
            supabase, "parcel", args.executor, worker_id, nas_root, env_file
        )
        if ran_parcel:
            continue

        # 2. Run queued ML jobs next
        ran_ml = process_next_queued_job(
            supabase, "ml", args.executor, worker_id, nas_root, env_file
        )
        if ran_ml:
            continue

        if not args.watch:
            print("All queued run_parcel and run_ml jobs in county_lookup are complete.")
            break

        time.sleep(30)


if __name__ == "__main__":
    main()