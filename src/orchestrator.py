import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import boto3
from dotenv import load_dotenv
from supabase import create_client

_PRINT_LOCK = threading.Lock()


def log_prefixed(county_name: str, county_fips: str, message: str) -> None:
    prefix = f"[{county_name.upper()} | {county_fips}]"
    with _PRINT_LOCK:
        print(f"{prefix} {message}", flush=True)


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


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _stream_docker_container(
    cmd: list[str],
    county_name: str,
    county_fips: str,
) -> None:
    """Runs a docker command and streams stdout/stderr line-by-line with county prefix."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.replace("\r", "\n").strip()
        if not line:
            continue
        for subline in line.splitlines():
            clean = subline.strip()
            if clean:
                log_prefixed(county_name, county_fips, clean)

    return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_parcel_docker_container(
    county_name: str,
    county_fips: str,
    state_code: str,
    nas_root: Path,
    env_file: Path,
    omp_threads: int = 2,
    docker_image: str = "growroute-parcel:latest",
) -> None:
    """Spins up a Docker container for a single county parcel ingestion run using NAS storage."""
    county_slug = _slugify(county_name)
    container_name = f"growroute-parcel-{county_slug}-{county_fips}"
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    parcels_mount = f"{nas_root / 'parcels'}:/app/data_nas"
    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--memory=2560m",
        "--env-file", str(env_file),
        "-e", f"OMP_NUM_THREADS={omp_threads}",
        "-v", parcels_mount,
        docker_image,
        "--county", county_name.upper(),
        "--fips", county_fips,
        "--state", state_code.upper(),
    ]
    log_prefixed(
        county_name,
        county_fips,
        f"Starting Parcel container {container_name} (OMP_NUM_THREADS={omp_threads}, mem=2560m)...",
    )
    _stream_docker_container(cmd, county_name, county_fips)


def run_ml_docker_container(
    county_name: str,
    county_fips: str,
    max_tiles: int,
    nas_root: Path,
    env_file: Path,
    omp_threads: int = 2,
    tile_loader_workers: int = 4,
    docker_image: str = "growroute-ml:cpu",
) -> None:
    """Spins up a Docker container for a single county ML scan using NAS storage."""
    county_slug = _slugify(county_name)
    container_name = f"growroute-ml-{county_slug}-{county_fips}"
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    tiles_mount = f"{nas_root / 'ml' / 'naip_h3_tiles'}:/app/tiles_cache"
    runs_mount = f"{nas_root / 'ml' / 'training_runs'}:/app/runs"
    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--memory=2560m",
        "--env-file", str(env_file),
        "-e", f"OMP_NUM_THREADS={omp_threads}",
        "-e", f"TILE_LOADER_WORKERS={tile_loader_workers}",
        "-v", tiles_mount,
        "-v", runs_mount,
        docker_image,
        "--county", county_name.upper(),
        "--device", "cpu",
        "--batch-tiles", "100",
        "--max-tiles", str(max_tiles),
    ]
    log_prefixed(
        county_name,
        county_fips,
        f"Starting ML container {container_name} (max_tiles={max_tiles}, "
        f"OMP_NUM_THREADS={omp_threads}, TILE_LOADER_WORKERS={tile_loader_workers}, mem=2560m)...",
    )
    _stream_docker_container(cmd, county_name, county_fips)


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
    omp_threads: int = 2,
    tile_loader_workers: int = 4,
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
            if pipeline == "parcel":
                run_parcel_docker_container(
                    county_name=county_name,
                    county_fips=county_fips,
                    state_code=state_code,
                    nas_root=nas_root,
                    env_file=env_file,
                    omp_threads=omp_threads,
                )
            else:
                run_ml_docker_container(
                    county_name=county_name,
                    county_fips=county_fips,
                    max_tiles=max_tiles,
                    nas_root=nas_root,
                    env_file=env_file,
                    omp_threads=omp_threads,
                    tile_loader_workers=tile_loader_workers,
                )
        else:
            job_id = submit_aws_batch_job(
                pipeline=pipeline,
                county_name=county_name,
                county_fips=county_fips,
                max_tiles=max_tiles,
            )
            log_prefixed(
                county_name,
                county_fips,
                f"[AWS Batch] Dispatched {pipeline.upper()} -> Job {job_id}",
            )

        supabase.rpc(
            "complete_county_job",
            {
                "p_county_fips": county_fips,
                "p_pipeline": pipeline,
                "p_success": True,
            },
        ).execute()
        log_prefixed(county_name, county_fips, f"[Completed] {pipeline.upper()} finished successfully.")

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
        log_prefixed(county_name, county_fips, f"[Failed] {pipeline.upper()} failed: {err_msg}")

    return True


def main():
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    parser = argparse.ArgumentParser(description="GrowRoute County Pipeline Orchestrator")
    parser.add_argument(
        "--executor",
        choices=["docker-local", "aws-batch"],
        default="docker-local",
        help="Run in local Docker w/ Synology NAS or submit to AWS Batch",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum parallel county containers (default: 3, configurable up to 5)",
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

    max_workers = max(1, min(args.max_workers, 5))
    omp_threads = max(1, 6 // max_workers)
    tile_loader_workers = int(os.getenv("TILE_LOADER_WORKERS", "4"))

    nas_root = Path(args.nas_root).resolve()
    env_file = (repo_root / ".env").resolve()

    if args.executor == "docker-local":
        verify_nas_root(nas_root)

    hostname = socket.gethostname()
    print(
        f"[Orchestrator] Executor={args.executor} | max_workers={max_workers} | "
        f"OMP_NUM_THREADS={omp_threads} | TILE_LOADER_WORKERS={tile_loader_workers} | "
        f"container_mem=2560m",
        flush=True,
    )

    def worker_loop(worker_idx: int) -> int:
        sb = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
        worker_id = f"{hostname}-{args.executor}-w{worker_idx}"
        jobs_ran = 0
        while True:
            # 1. Prioritize queued Parcel jobs before ML jobs
            ran_parcel = process_next_queued_job(
                supabase=sb,
                pipeline="parcel",
                executor=args.executor,
                worker_id=worker_id,
                nas_root=nas_root,
                env_file=env_file,
                omp_threads=omp_threads,
                tile_loader_workers=tile_loader_workers,
            )
            if ran_parcel:
                jobs_ran += 1
                continue

            # 2. Claim and run next queued ML county job
            ran_ml = process_next_queued_job(
                supabase=sb,
                pipeline="ml",
                executor=args.executor,
                worker_id=worker_id,
                nas_root=nas_root,
                env_file=env_file,
                omp_threads=omp_threads,
                tile_loader_workers=tile_loader_workers,
            )
            if ran_ml:
                jobs_ran += 1
                continue

            break
        return jobs_ran

    while True:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(worker_loop, idx + 1) for idx in range(max_workers)]
            for future in as_completed(futures):
                future.result()

        if not args.watch:
            print("All queued run_parcel and run_ml jobs in county_lookup are complete.", flush=True)
            break

        time.sleep(30)


if __name__ == "__main__":
    main()
