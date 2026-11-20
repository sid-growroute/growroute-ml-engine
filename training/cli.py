from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from typing import List, Optional

from training.config import (
    DEFAULT_OVERSAMPLE_FACTOR,
    RANDOM_SEED,
    TrainingConfig,
)
from training.naip_fetcher import fetch_and_project_naip_tiles
from training.s3_exporter import (
    export_dataset_to_s3,
    list_existing_s3_image_keys,
    load_cached_local_manifest,
)
from training.stage_sampler import (
    CountyStageSummary,
    inspect_stage_schema_and_counties,
    sample_stage_candidates,
)


def _cmd_inspect_stage(args: argparse.Namespace) -> int:
    cfg = TrainingConfig.from_env(require_db=True)
    report = inspect_stage_schema_and_counties(cfg, state_code=args.state)

    print("=== Stage PostGIS Schema & CAD Pool Audit ===")
    print(f"DSN (redacted)        : {report['dsn_redacted']}")
    print(f"transaction_read_only : {report['transaction_read_only']}")
    print(f"Parcel Table          : {report['table']} (PK={report['primary_key']})")
    print(
        f"Geometry Column       : {report['geometry']['column']} "
        f"({report['geometry']['type']}, SRID={report['geometry']['srid']})"
    )
    print(f"H3 Column             : {report['h3_column']} (res={cfg.h3_res}, tile_px={cfg.tile_px})")
    print(f"CAD Truth Columns     : {report['cad_ground_truth_columns']}")
    print(f"Excluded ML Columns   : {report['ml_inferred_columns_excluded']}")
    print("\nPool Source Breakdown (State=" + args.state.upper() + "):")
    for row in report["pool_source_breakdown"]:
        print(
            f"  source={row['pool_detection_source']:<8} "
            f"cad_has_pool={str(row['cad_has_pool']):<5} "
            f"ai_has_pool={str(row['ai_has_pool']):<5} -> {row['count']:,} parcels"
        )

    counties: List[CountyStageSummary] = report["counties"]
    pos_counties = report["cad_positive_counties"]
    print(
        f"\nEligible CAD-Positive Counties: {len(pos_counties)} / {len(counties)} active counties "
        f"({sum(c.cad_pool_parcels for c in pos_counties):,} CAD pools across "
        f"{sum(c.cad_positive_hexes for c in pos_counties):,} H3 res-9 hexes)"
    )
    print(
        f"{'County Slug':<16} {'DB Name':<16} {'FIPS':<7} "
        f"{'Total Lots':>11} {'CAD Pools':>10} {'CAD Spas':>9} {'CAD H3 Hexes':>13}"
    )
    print("-" * 88)
    for c in counties:
        print(
            f"{c.county_slug:<16} {c.county_name_db:<16} {c.county_fips:<7} "
            f"{c.total_parcels:>11,} {c.cad_pool_parcels:>10,} "
            f"{c.cad_spa_parcels:>9,} {c.cad_positive_hexes:>13,}"
        )
    return 0


def _cmd_sample_stage(args: argparse.Namespace) -> int:
    cfg = TrainingConfig.from_env(require_db=True)
    cand_path, plan, stats = sample_stage_candidates(
        cfg=cfg,
        run_id=args.run_id,
        total_count=args.count,
        val_county=args.val_county,
        test_county=args.test_county,
        oversample_factor=args.oversample,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    print("\n=== Phase 1 Stratified Multi-County Sampling Summary ===")
    print(f"Run ID             : {args.run_id}")
    print(f"Candidates Parquet : {cand_path}")
    print(f"Val Holdout County : {plan.val_county} (requested: {plan.requested_val_county})")
    print(f"Test Holdout County: {plan.test_county} (requested: {plan.requested_test_county})")
    print(
        f"Total Quotas       : target={plan.total_target_tiles:,} tiles | "
        f"oversampled_candidates={stats['candidate_tiles']:,} tiles"
    )
    print(
        f"290m Parcels Pulled: {stats['total_parcels_extracted']:,} total lots "
        f"({stats['total_cad_pool_parcels_extracted']:,} cad_has_pool=true)"
    )
    print(f"\n{'Split':<7} {'County':<16} {'Avail Hex':>10} {'Target':>8} {'Candidates':>11}")
    print("-" * 58)
    for cp in plan.county_plans:
        print(
            f"{cp.split:<7} {cp.county_slug:<16} {cp.available_hexes:>10,} "
            f"{cp.target_quota:>8,} {cp.candidate_quota:>11,}"
        )
    return 0


def _cmd_fetch_naip(args: argparse.Namespace) -> int:
    cfg = TrainingConfig.from_env(require_db=False)
    cache_dir = cfg.run_cache_dir(args.run_id)

    existing_s3 = list_existing_s3_image_keys(cfg, run_id=args.run_id)
    cached_manifest = load_cached_local_manifest(cache_dir)
    if existing_s3:
        print(
            f"[Resume Check] Found {len(existing_s3):,} existing S3 images under "
            f"s3://{cfg.training_s3_bucket}/{cfg.training_s3_prefix}/{args.run_id}/images/",
            flush=True,
        )

    naip_yr = None if str(args.naip_year).lower() == "auto" else int(args.naip_year)
    records, fetch_stats = fetch_and_project_naip_tiles(
        cfg=cfg,
        run_id=args.run_id,
        workers=args.workers,
        naip_year_override=naip_yr,
        existing_s3_keys=existing_s3,
        cached_manifest_by_tile_id=cached_manifest,
    )

    export_stats = export_dataset_to_s3(
        cfg=cfg,
        run_id=args.run_id,
        records=records,
        existing_s3_keys=existing_s3,
        qa_overlay_limit=args.qa_overlay_limit,
        upload_workers=args.workers,
    )

    split_counts = Counter(r.split for r in records)
    county_split_counts = Counter((r.split, r.county) for r in records)

    print("\n=== Phase 2 NAIP COG Stream & S3 Export Summary ===")
    print(f"Run ID            : {args.run_id}")
    print(f"NAIP Vintage      : {fetch_stats['naip_year']} (60cm RGB)")
    print(
        f"Accepted Tiles    : {fetch_stats['accepted_total']:,} / {fetch_stats['target_quota_total']:,} "
        f"(rejected_nodata_or_seam={fetch_stats['rejected_total']:,}, rounds={fetch_stats['rounds_executed']})"
    )
    print(
        f"Split Totals      : train={split_counts.get('train', 0):,} | "
        f"val={split_counts.get('val', 0):,} | test={split_counts.get('test', 0):,}"
    )
    print(f"S3 Images URI     : {export_stats['s3_images_prefix_uri']}")
    print(f"S3 QA Overlays URI: {export_stats['s3_qa_overlays_prefix_uri']}")
    print(f"S3 Manifest URI   : {export_stats['s3_manifest_uri']}")
    print(f"Local Manifest    : {export_stats['local_manifest_path']}")

    print(f"\n{'Split':<7} {'County':<16} {'Exported Tiles':>15}")
    print("-" * 42)
    for (sp, cty), cnt in sorted(county_split_counts.items()):
        print(f"{sp:<7} {cty:<16} {cnt:>15,}")

    qa_paths = export_stats["local_qa_overlay_paths"]
    if qa_paths:
        print(f"\nRendered {len(qa_paths)} QA Overlays (sample local paths):")
        for p in qa_paths[:5]:
            print(f"  {p}")

    if args.json_summary:
        print("\nJSON_SUMMARY=" + json.dumps({**fetch_stats, **export_stats}))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="growroute-training",
        description="GrowRoute Stage PostGIS + NAIP COG training dataset pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser(
        "inspect-stage",
        help="Read-only inspection of Stage parcel schema and per-county CAD pool counts",
    )
    p_inspect.add_argument("--state", default="TX", help="State code (default: TX)")
    p_inspect.set_defaults(func=_cmd_inspect_stage)

    p_sample = sub.add_parser(
        "sample-stage",
        help="Phase 1: Stratified multi-county H3 sampling & 290m PostGIS WKB parcel extraction",
    )
    p_sample.add_argument("--run-id", required=True, help="Unique dataset run identifier")
    p_sample.add_argument("--count", type=int, default=10_000, help="Total target valid tiles")
    p_sample.add_argument("--val-county", default="comal", help="Spatial validation holdout county")
    p_sample.add_argument("--test-county", default="fort_bend", help="Spatial test holdout county")
    p_sample.add_argument(
        "--oversample",
        type=float,
        default=DEFAULT_OVERSAMPLE_FACTOR,
        help="Candidate oversampling multiplier (default: 1.15)",
    )
    p_sample.add_argument("--seed", type=int, default=RANDOM_SEED, help="Deterministic RNG/hash seed")
    p_sample.add_argument("--batch-size", type=int, default=500, help="Hexes per PostGIS GIST batch")
    p_sample.set_defaults(func=_cmd_sample_stage)

    p_fetch = sub.add_parser(
        "fetch-naip",
        help="Phase 2: Stream 640x640 NAIP COGs, project parcel polygons to pixels & export to S3",
    )
    p_fetch.add_argument("--run-id", required=True, help="Dataset run identifier from sample-stage")
    p_fetch.add_argument("--workers", type=int, default=16, help="Parallel COG/S3 worker threads")
    p_fetch.add_argument(
        "--qa-overlay-limit",
        type=int,
        default=50,
        help="Number of visual QA parcel-overlay JPEGs to render (default: 50)",
    )
    p_fetch.add_argument(
        "--naip-year",
        default="auto",
        help="NAIP 4-digit year or 'auto' for newest available (default: auto)",
    )
    p_fetch.add_argument(
        "--json-summary",
        action="store_true",
        help="Print machine-readable JSON summary line at completion",
    )
    p_fetch.set_defaults(func=_cmd_fetch_naip)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
