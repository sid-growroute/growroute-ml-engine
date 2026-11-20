from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import urllib.parse
from typing import Optional
import psycopg

PROD_PROJECT_REF = "dokbiswusogiophuuqjk"
STAGE_PROJECT_REF = "zpxyywgucrtekxyoumct"

# Exact Step 0 Inference Parity Constants:
# - H3 resolution 9 (`public.parcels.h3_index_res9`, `configs/pipeline.yaml` tx/h3_res9)
# - Model input resolution 640x640 (`configs/pipeline.yaml` imgsz: 640)
# - Centroid-centered 640x640 window at native 0.60 m/px NAIP GSD (384.0m x 384.0m UTM footprint;
#   half-diagonal = 271.53m < 290.0m PostGIS extraction radius)
TILE_PX: int = 640
H3_RES: int = 9
NAIP_GSD_M: float = 0.60
PARCEL_SEARCH_RADIUS_M: float = 290.0
DEFAULT_OVERSAMPLE_FACTOR: float = 1.15
RANDOM_SEED: int = 20261119

NAIP_BUCKET: str = "naip-visualization"
NAIP_AWS_REGION: str = "us-west-2"
DEFAULT_NAIP_STATE: str = "tx"
DEFAULT_TRAINING_S3_BUCKET: str = "growroute-stage-data"
DEFAULT_TRAINING_S3_PREFIX: str = "training"

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT: Path = REPO_ROOT / ".cache" / "training"

_URI_PASSWORD_RE = re.compile(r"(postgres(?:ql)?://[^:\s/@]+:)([^@\s]+)(@)", re.IGNORECASE)


def redact_dsn(dsn: str) -> str:
    """Redacts password tokens from Postgres connection strings before logging."""
    if not dsn:
        return ""
    if "://" in dsn and "@" in dsn:
        scheme, rest = dsn.split("://", 1)
        userinfo, hostpart = rest.rsplit("@", 1)
        user = userinfo.split(":", 1)[0] if ":" in userinfo else userinfo
        return f"{scheme}://{user}:***REDACTED***@{hostpart}"
    return _URI_PASSWORD_RE.sub(r"\1***REDACTED***\3", dsn)


def assert_stage_dsn_safe(raw_dsn: str) -> str:
    """
    Validates that `raw_dsn` is non-empty, does NOT point to Production (`dokbiswusogiophuuqjk`),
    and safely URL-encodes special characters in the password component.
    """
    cleaned = (raw_dsn or "").strip().strip("'\"")
    if not cleaned:
        raise ValueError(
            "Missing STAGE_DATABASE_URL. Export STAGE_DATABASE_URL (Stage project ref "
            f"'{STAGE_PROJECT_REF}') in .env / .envrc or environment."
        )

    lowered = cleaned.lower()
    if PROD_PROJECT_REF in lowered or "growroute-prod" in lowered:
        raise ValueError(
            f"Refusing to connect: DSN contains Production identifier ('{PROD_PROJECT_REF}'). "
            "Training sampler must strictly target Stage read-only."
        )

    if "://" not in cleaned or "@" not in cleaned:
        return cleaned

    scheme, rest = cleaned.split("://", 1)
    userinfo, hostpart = rest.rsplit("@", 1)
    if ":" in userinfo:
        user, raw_password = userinfo.split(":", 1)
        encoded_password = urllib.parse.quote(urllib.parse.unquote(raw_password), safe="")
        return f"{scheme}://{user}:{encoded_password}@{hostpart}"
    return cleaned


def _parse_dotenv_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'\"")
        if key and val:
            values[key] = val
    return values


def _collect_env_map(extra_env_files: Optional[list[Path]] = None) -> dict[str, str]:
    """
    Collects environment variables from local `.env` / `.envrc` files (and sibling Stage
    `.env.staging` when present, never `.env.production`), giving shell `os.environ` priority.
    """
    candidate_files: list[Path] = [
        REPO_ROOT / ".env",
        REPO_ROOT / ".env.local",
        REPO_ROOT / ".env.stage",
        REPO_ROOT / ".env.staging",
        REPO_ROOT / ".envrc",
        REPO_ROOT.parent / ".env",
        REPO_ROOT.parent / ".envrc",
        REPO_ROOT.parent / "growroute-parcel-engine" / ".env.staging",
        REPO_ROOT.parent / "growroute-parcel-engine" / "ops" / ".env.ops",
    ]
    if extra_env_files:
        candidate_files = list(extra_env_files) + candidate_files

    merged: dict[str, str] = {}
    for env_path in candidate_files:
        for k, v in _parse_dotenv_file(env_path).items():
            merged.setdefault(k, v)

    for k, v in os.environ.items():
        if v:
            merged[k] = v

    return merged


@dataclass(frozen=True)
class TrainingConfig:
    stage_database_url: str = field(repr=False)
    training_s3_bucket: str = DEFAULT_TRAINING_S3_BUCKET
    training_s3_prefix: str = DEFAULT_TRAINING_S3_PREFIX
    aws_region: str = NAIP_AWS_REGION
    aws_access_key_id: Optional[str] = field(default=None, repr=False)
    aws_secret_access_key: Optional[str] = field(default=None, repr=False)
    aws_session_token: Optional[str] = field(default=None, repr=False)
    naip_bucket: str = NAIP_BUCKET
    naip_state: str = DEFAULT_NAIP_STATE
    tile_px: int = TILE_PX
    h3_res: int = H3_RES
    naip_gsd_m: float = NAIP_GSD_M
    parcel_search_radius_m: float = PARCEL_SEARCH_RADIUS_M
    oversample_factor: float = DEFAULT_OVERSAMPLE_FACTOR
    random_seed: int = RANDOM_SEED
    cache_root: Path = DEFAULT_CACHE_ROOT

    def run_cache_dir(self, run_id: str) -> Path:
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id.strip())
        if not safe_run_id:
            raise ValueError("run_id must not be empty")
        out = self.cache_root / safe_run_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    @classmethod
    def from_env(
        cls,
        *,
        require_db: bool = True,
        extra_env_files: Optional[list[Path]] = None,
    ) -> "TrainingConfig":
        env = _collect_env_map(extra_env_files=extra_env_files)

        raw_dsn = (
            env.get("STAGE_DATABASE_URL")
            or env.get("STAGE_DB_URL")
            or env.get("STAGE_SUPABASE_DB_URL")
            or env.get("SUPABASE_DB_URL")
            or ""
        )
        supabase_url = env.get("STAGE_SUPABASE_URL") or env.get("SUPABASE_URL") or ""
        if PROD_PROJECT_REF in supabase_url.lower() and not (
            env.get("STAGE_DATABASE_URL") or env.get("STAGE_DB_URL") or env.get("STAGE_SUPABASE_DB_URL")
        ):
            raise ValueError(
                f"Refusing to use generic SUPABASE_DB_URL when SUPABASE_URL points to Prod ({PROD_PROJECT_REF}). "
                "Set explicit STAGE_DATABASE_URL."
            )

        validated_dsn = assert_stage_dsn_safe(raw_dsn) if (require_db or raw_dsn) else ""

        s3_bucket = (
            env.get("TRAINING_S3_BUCKET")
            or env.get("STAGE_S3_DATA_BUCKET")
            or env.get("S3_DATA_BUCKET")
            or DEFAULT_TRAINING_S3_BUCKET
        ).strip()
        if "prod" in s3_bucket.lower():
            raise ValueError(
                f"Refusing to write training artifacts to production bucket '{s3_bucket}'."
            )

        s3_prefix = (env.get("TRAINING_S3_PREFIX") or DEFAULT_TRAINING_S3_PREFIX).strip().strip("/")
        naip_state = (env.get("NAIP_STATE") or DEFAULT_NAIP_STATE).strip().lower()
        tile_px = int(env.get("TILE_PX") or TILE_PX)
        h3_res = int(env.get("H3_RES") or H3_RES)
        random_seed = int(env.get("RANDOM_SEED") or RANDOM_SEED)

        # Force us-west-2 for S3 / NAIP COG requester-pays unless TRAINING_AWS_REGION overrides
        aws_region = (env.get("TRAINING_AWS_REGION") or NAIP_AWS_REGION).strip() or NAIP_AWS_REGION
        cache_root = Path(env.get("TRAINING_CACHE_DIR") or DEFAULT_CACHE_ROOT).resolve()

        return cls(
            stage_database_url=validated_dsn,
            training_s3_bucket=s3_bucket,
            training_s3_prefix=s3_prefix,
            aws_region=aws_region,
            aws_access_key_id=env.get("AWS_ACCESS_KEY_ID") or None,
            aws_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY") or None,
            aws_session_token=env.get("AWS_SESSION_TOKEN") or None,
            naip_state=naip_state,
            tile_px=tile_px,
            h3_res=h3_res,
            random_seed=random_seed,
            cache_root=cache_root,
        )


def open_readonly_stage_connection(dsn: str, statement_timeout_sec: int = 120) -> psycopg.Connection:
    """
    Opens a PostgreSQL connection to Stage after verifying the Prod-DSN guard and immediately
    locks the session into `TRANSACTION READ ONLY` mode.
    """
    safe_dsn = assert_stage_dsn_safe(dsn)
    conn = psycopg.connect(safe_dsn, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;")
            cur.execute(f"SET statement_timeout = '{int(statement_timeout_sec)}s';")
            cur.execute("SHOW transaction_read_only;")
            row = cur.fetchone()
            if not row or str(row[0]).strip().lower() != "on":
                raise RuntimeError("Failed to enforce READ ONLY session on Stage connection!")
    except Exception:
        conn.close()
        raise
    return conn
