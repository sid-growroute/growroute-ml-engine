from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h3
import pyarrow as pa
import pyarrow.parquet as pq

from training.config import (
    DEFAULT_OVERSAMPLE_FACTOR,
    PARCEL_SEARCH_RADIUS_M,
    RANDOM_SEED,
    TrainingConfig,
    open_readonly_stage_connection,
    redact_dsn,
)

# Canonical multi-word Texas county FIPS / compact-name normalization
TX_FIPS_TO_CANONICAL_SLUG: Dict[str, str] = {
    "48091": "comal",
    "48125": "dickens",
    "48117": "deaf_smith",
    "48131": "duval",
    "48141": "el_paso",
    "48157": "fort_bend",
    "48249": "jim_wells",
    "48283": "la_salle",
    "48297": "live_oak",
    "48385": "real",
    "48387": "red_river",
    "48407": "san_jacinto",
    "48409": "san_patricio",
    "48411": "san_saba",
    "48451": "tom_green",
    "48465": "val_verde",
    "48467": "van_zandt",
}

COMPACT_NAME_TO_CANONICAL_SLUG: Dict[str, str] = {
    "fortbend": "fort_bend",
    "elpaso": "el_paso",
    "sanpatricio": "san_patricio",
    "sanjacinto": "san_jacinto",
    "sansaba": "san_saba",
    "tomgreen": "tom_green",
    "valverde": "val_verde",
    "vanzandt": "van_zandt",
    "liveoak": "live_oak",
    "deafsmith": "deaf_smith",
    "lasalle": "la_salle",
    "jimwells": "jim_wells",
    "redriver": "red_river",
}

RESIDENTIAL_USE_TYPES = frozenset({
    "sfr",
    "condo_townhome",
    "multifamily_small",
    "mobile_home",
    "residential",
})

PARCEL_STRUCT_TYPE = pa.struct([
    ("parcel_id", pa.string()),
    ("county", pa.string()),
    ("cad_has_pool", pa.bool_()),
    ("cad_has_spa", pa.bool_()),
    ("is_residential", pa.bool_()),
    ("normalized_use_type", pa.string()),
    ("land_use_code", pa.string()),
    ("geom_wkb", pa.binary()),
])

CANDIDATES_PARQUET_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("county", pa.string()),
    ("county_name_db", pa.string()),
    ("county_fips", pa.string()),
    ("split", pa.string()),
    ("target_quota", pa.int32()),
    ("candidate_quota", pa.int32()),
    ("candidate_rank", pa.int32()),
    ("h3_index", pa.string()),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("cad_pool_parcel_count", pa.int32()),
    ("total_parcel_count", pa.int32()),
    ("parcels_wkb", pa.list_(PARCEL_STRUCT_TYPE)),
])


def compact_county_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())


def canonical_county_slug(county_name: str, county_fips: str = "") -> str:
    fips_clean = (county_fips or "").strip()
    if fips_clean in TX_FIPS_TO_CANONICAL_SLUG:
        return TX_FIPS_TO_CANONICAL_SLUG[fips_clean]
    compact = compact_county_key(county_name)
    if compact in COMPACT_NAME_TO_CANONICAL_SLUG:
        return COMPACT_NAME_TO_CANONICAL_SLUG[compact]
    slug = re.sub(r"[^a-z0-9]+", "_", (county_name or "").strip().lower()).strip("_")
    return slug or fips_clean or "unknown"


def h3_cell_to_latlon(h3_index: str) -> Tuple[float, float]:
    if hasattr(h3, "cell_to_latlng"):
        lat, lng = h3.cell_to_latlng(h3_index)
        return float(lat), float(lng)
    lat, lng = h3.h3_to_geo(h3_index)
    return float(lat), float(lng)


def deterministic_cell_sort_key(seed: int, county_slug: str, h3_index: str) -> str:
    payload = f"{seed}:{county_slug}:{h3_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CountyStageSummary:
    county_slug: str
    county_name_db: str
    county_fips: str
    total_parcels: int
    cad_pool_parcels: int
    cad_spa_parcels: int
    cad_positive_hexes: int


@dataclass(frozen=True)
class CountyQuotaPlan:
    county_slug: str
    split: str  # 'train' | 'val' | 'test'
    available_hexes: int
    target_quota: int
    candidate_quota: int


@dataclass(frozen=True)
class StratificationPlan:
    val_county: str
    test_county: str
    requested_val_county: str
    requested_test_county: str
    val_fallback_note: Optional[str]
    test_fallback_note: Optional[str]
    county_plans: List[CountyQuotaPlan]

    @property
    def total_target_tiles(self) -> int:
        return sum(p.target_quota for p in self.county_plans)

    @property
    def total_candidate_tiles(self) -> int:
        return sum(p.candidate_quota for p in self.county_plans)


def _match_county_in_available(
    requested: str,
    available_map: Dict[str, int],
) -> Optional[str]:
    req_compact = compact_county_key(requested)
    if not req_compact:
        return None
    for slug in available_map:
        if compact_county_key(slug) == req_compact:
            return slug
    # Also check FIPS code if user passed e.g. '48157'
    if requested.strip() in TX_FIPS_TO_CANONICAL_SLUG:
        mapped = TX_FIPS_TO_CANONICAL_SLUG[requested.strip()]
        if mapped in available_map:
            return mapped
    return None


def _select_fallback_holdout_county(
    available_map: Dict[str, int],
    desired_quota: int,
    exclude_slugs: set[str],
    preferred_fallbacks: Sequence[str],
) -> str:
    for pref in preferred_fallbacks:
        if pref in available_map and pref not in exclude_slugs and available_map[pref] >= min(desired_quota, 50):
            return pref
    eligible = [c for c in available_map if c not in exclude_slugs]
    if not eligible:
        raise ValueError("Insufficient CAD-positive counties in Stage to construct train/val/test split")
    # Prefer counties that can satisfy desired_quota, sorted deterministically by (can_satisfy, available_hexes, slug)
    eligible.sort(
        key=lambda c: (
            1 if available_map[c] >= desired_quota else 0,
            available_map[c],
            c,
        ),
        reverse=True,
    )
    return eligible[0]


def _allocate_sqrt_weighted_quotas(
    available_by_county: Dict[str, int],
    target_total: int,
    floor_per_county: int,
) -> Dict[str, int]:
    counties = sorted(available_by_county.keys())
    if not counties or target_total <= 0:
        return {c: 0 for c in counties}

    max_possible = sum(available_by_county.values())
    effective_total = min(target_total, max_possible)

    # Ensure the sum of per-county floors never exceeds effective_total
    effective_floor = floor_per_county
    while effective_floor > 0 and sum(min(effective_floor, available_by_county[c]) for c in counties) > effective_total:
        effective_floor -= 1

    alloc: Dict[str, int] = {
        c: min(effective_floor, available_by_county[c]) for c in counties
    }
    remaining = effective_total - sum(alloc.values())

    # Iteratively distribute remaining tiles proportional to sqrt(available_hexes),
    # capping at each county's headroom and redistributing any capped overflow.
    while remaining > 0:
        expandable = [c for c in counties if alloc[c] < available_by_county[c]]
        if not expandable:
            break

        weights = {c: math.sqrt(float(available_by_county[c])) for c in expandable}
        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            break

        raw_shares = {c: remaining * (weights[c] / weight_sum) for c in expandable}
        int_adds: Dict[str, int] = {}
        remainders: List[Tuple[float, str]] = []

        for c in expandable:
            headroom = available_by_county[c] - alloc[c]
            whole = min(headroom, int(math.floor(raw_shares[c])))
            int_adds[c] = whole
            if whole < headroom:
                remainders.append((raw_shares[c] - whole, c))

        added_sum = sum(int_adds.values())
        for c, add_val in int_adds.items():
            alloc[c] += add_val
        remaining -= added_sum

        if remaining <= 0:
            break

        # Largest Remainder tie-breaking (deterministic by remainder desc, available desc, slug asc)
        remainders.sort(key=lambda item: (-item[0], -available_by_county[item[1]], item[1]))
        progressed = False
        for _frac, c in remainders:
            if remaining <= 0:
                break
            if alloc[c] < available_by_county[c]:
                alloc[c] += 1
                remaining -= 1
                progressed = True

        if not progressed and added_sum == 0:
            break

    return alloc


def compute_stratified_county_quotas(
    county_hex_counts: Dict[str, int],
    total_target: int = 10_000,
    val_county: str = "comal",
    test_county: str = "fort_bend",
    oversample_factor: float = DEFAULT_OVERSAMPLE_FACTOR,
    base_train_floor: int = 200,
) -> StratificationPlan:
    """
    Computes zero-overlap spatial holdout (`val` + `test`) and proportional sqrt-weighted `train`
    quotas with `1.15x` candidate oversampling across CAD-positive counties.
    """
    if total_target <= 0:
        raise ValueError("total_target must be > 0")

    positive_map = {
        canonical_county_slug(k): int(v)
        for k, v in county_hex_counts.items()
        if int(v) > 0
    }
    if len(positive_map) < 3:
        raise ValueError(
            f"Need at least 3 CAD-positive counties for spatial train/val/test split; found {len(positive_map)}"
        )

    desired_val = 1_000 if total_target == 10_000 else max(1, int(round(total_target * 0.10)))
    desired_test = 500 if total_target == 10_000 else max(1, int(round(total_target * 0.05)))
    if desired_val + desired_test >= total_target and total_target >= 3:
        desired_val = 1
        desired_test = 1

    matched_test = _match_county_in_available(test_county, positive_map)
    test_note: Optional[str] = None
    if matched_test is None:
        matched_test = _select_fallback_holdout_county(
            positive_map,
            desired_quota=desired_test,
            exclude_slugs=set(),
            preferred_fallbacks=("fort_bend", "montgomery", "brazoria", "galveston"),
        )
        test_note = (
            f"Requested test_county='{test_county}' has 0 CAD-positive hexes in Stage; "
            f"fell back to '{matched_test}' ({positive_map[matched_test]} hexes)."
        )

    matched_val = _match_county_in_available(val_county, positive_map)
    val_note: Optional[str] = None
    if matched_val is None or matched_val == matched_test:
        matched_val = _select_fallback_holdout_county(
            positive_map,
            desired_quota=desired_val,
            exclude_slugs={matched_test},
            preferred_fallbacks=("williamson", "johnson", "nueces", "travis", "bell"),
        )
        val_note = (
            f"Requested val_county='{val_county}' has 0 CAD-positive hexes in Stage; "
            f"fell back to '{matched_val}' ({positive_map[matched_val]} hexes)."
        )

    val_target = min(desired_val, positive_map[matched_val])
    test_target = min(desired_test, positive_map[matched_test])
    train_target = max(0, total_target - val_target - test_target)

    train_available = {
        c: cnt
        for c, cnt in positive_map.items()
        if c not in {matched_val, matched_test}
    }
    scaled_floor = (
        base_train_floor
        if total_target >= 10_000
        else max(1, int(round(base_train_floor * (total_target / 10_000.0))))
    )
    train_alloc = _allocate_sqrt_weighted_quotas(
        available_by_county=train_available,
        target_total=train_target,
        floor_per_county=scaled_floor,
    )

    def _calc_candidate_quota(target_q: int, avail: int) -> int:
        if target_q <= 0:
            return 0
        raw_cand = int(math.ceil(target_q * oversample_factor - 1e-9))
        # Ensure at least +1 spare candidate when oversample_factor > 1.0 and headroom exists
        if oversample_factor > 1.0 and raw_cand == target_q and avail > target_q:
            raw_cand = target_q + 1
        return min(avail, max(target_q, raw_cand))

    plans: List[CountyQuotaPlan] = []
    for c in sorted(train_alloc.keys()):
        tq = train_alloc[c]
        if tq <= 0:
            continue
        avail = train_available[c]
        plans.append(
            CountyQuotaPlan(
                county_slug=c,
                split="train",
                available_hexes=avail,
                target_quota=tq,
                candidate_quota=_calc_candidate_quota(tq, avail),
            )
        )

    plans.append(
        CountyQuotaPlan(
            county_slug=matched_val,
            split="val",
            available_hexes=positive_map[matched_val],
            target_quota=val_target,
            candidate_quota=_calc_candidate_quota(val_target, positive_map[matched_val]),
        )
    )
    plans.append(
        CountyQuotaPlan(
            county_slug=matched_test,
            split="test",
            available_hexes=positive_map[matched_test],
            target_quota=test_target,
            candidate_quota=_calc_candidate_quota(test_target, positive_map[matched_test]),
        )
    )

    return StratificationPlan(
        val_county=matched_val,
        test_county=matched_test,
        requested_val_county=val_county,
        requested_test_county=test_county,
        val_fallback_note=val_note,
        test_fallback_note=test_note,
        county_plans=plans,
    )


def inspect_stage_schema_and_counties(
    cfg: TrainingConfig,
    state_code: str = "TX",
) -> Dict[str, Any]:
    """
    Connects to Stage in strict READ ONLY mode, inspects `public.parcels` columns/SRID,
    and aggregates CAD-confirmed (`cad_has_pool IS TRUE`) parcel & H3 res-9 counts per county.
    """
    st_upper = state_code.strip().upper()
    with open_readonly_stage_connection(cfg.stage_database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW transaction_read_only;")
            ro_mode = str(cur.fetchone()[0])

            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'parcels'
                ORDER BY ordinal_position;
                """
            )
            parcel_columns = {str(col): str(dtype) for col, dtype in cur.fetchall()}

            cur.execute(
                """
                SELECT f_geometry_column, srid, type
                FROM public.geometry_columns
                WHERE f_table_schema = 'public' AND f_table_name = 'parcels'
                LIMIT 1;
                """
            )
            geom_row = cur.fetchone()
            geom_info = {
                "column": str(geom_row[0]) if geom_row else "geom",
                "srid": int(geom_row[1]) if geom_row else 4326,
                "type": str(geom_row[2]) if geom_row else "MULTIPOLYGON",
            }

            cur.execute(
                """
                SELECT
                  COALESCE(county_name, '') AS county_name,
                  COALESCE(county_fips, '') AS county_fips,
                  COUNT(*) AS total_parcels,
                  COUNT(*) FILTER (WHERE cad_has_pool IS TRUE) AS cad_pool_parcels,
                  COUNT(*) FILTER (WHERE cad_has_spa IS TRUE) AS cad_spa_parcels,
                  COUNT(DISTINCT h3_index_res9) FILTER (
                    WHERE cad_has_pool IS TRUE
                      AND h3_index_res9 IS NOT NULL
                      AND btrim(h3_index_res9) <> ''
                  ) AS cad_positive_hexes
                FROM public.parcels
                WHERE state_code = %s
                  AND COALESCE(is_active, true) = true
                GROUP BY 1, 2
                ORDER BY cad_pool_parcels DESC, total_parcels DESC;
                """,
                (st_upper,),
            )
            counties: List[CountyStageSummary] = []
            for cname, cfips, tot_p, cad_p, cad_s, cad_h3 in cur.fetchall():
                counties.append(
                    CountyStageSummary(
                        county_slug=canonical_county_slug(str(cname), str(cfips)),
                        county_name_db=str(cname),
                        county_fips=str(cfips),
                        total_parcels=int(tot_p),
                        cad_pool_parcels=int(cad_p),
                        cad_spa_parcels=int(cad_s),
                        cad_positive_hexes=int(cad_h3),
                    )
                )

            cur.execute(
                """
                SELECT
                  COALESCE(pool_detection_source, 'none') AS pool_source,
                  COALESCE(cad_has_pool, false) AS cad_has_pool,
                  COALESCE(ai_has_pool, false) AS ai_has_pool,
                  COUNT(*) AS cnt
                FROM public.parcels
                WHERE state_code = %s
                  AND (cad_has_pool IS TRUE OR ai_has_pool IS TRUE)
                GROUP BY 1, 2, 3
                ORDER BY cnt DESC;
                """,
                (st_upper,),
            )
            source_breakdown = [
                {
                    "pool_detection_source": str(r[0]),
                    "cad_has_pool": bool(r[1]),
                    "ai_has_pool": bool(r[2]),
                    "count": int(r[3]),
                }
                for r in cur.fetchall()
            ]

    return {
        "dsn_redacted": redact_dsn(cfg.stage_database_url),
        "transaction_read_only": ro_mode,
        "table": "public.parcels",
        "primary_key": ["id", "state_code"],
        "geometry": geom_info,
        "h3_column": "h3_index_res9",
        "cad_ground_truth_columns": ["cad_has_pool", "cad_has_spa", "cad_pool_sqft"],
        "ml_inferred_columns_excluded": ["ai_has_pool", "ai_has_spa", "ai_pool_confidence", "pool_detection_source"],
        "parcel_columns": parcel_columns,
        "counties": counties,
        "cad_positive_counties": [c for c in counties if c.cad_positive_hexes > 0],
        "pool_source_breakdown": source_breakdown,
    }


def _chunk_seq(seq: Sequence[Any], size: int) -> List[Sequence[Any]]:
    return [seq[idx : idx + size] for idx in range(0, len(seq), size)]


def sample_stage_candidates(
    cfg: TrainingConfig,
    run_id: str,
    total_count: int = 10_000,
    val_county: str = "comal",
    test_county: str = "fort_bend",
    oversample_factor: float = DEFAULT_OVERSAMPLE_FACTOR,
    seed: Optional[int] = None,
    batch_size: int = 500,
) -> Tuple[Path, StratificationPlan, Dict[str, Any]]:
    """
    Phase 1:
      1. Audits CAD-positive counties in Stage (`cad_has_pool IS TRUE` strictly).
      2. Computes spatial holdout (`val` + `test`) and proportional sqrt-weighted `train` quotas
         with `1.15x` deterministic candidate oversampling.
      3. Queries Stage PostGIS in 500-hex GIST-indexed batches (`ST_DWithin(geom::geography, centroid::geography, 290)`)
         capturing BOTH `cad_has_pool=true` and neighboring `cad_has_pool=false` lots as EPSG:4326 WKB.
      4. Writes `.cache/training/{run_id}/candidates.parquet`.
    """
    effective_seed = int(seed if seed is not None else cfg.random_seed)
    st_upper = cfg.naip_state.strip().upper()
    radius_m = float(cfg.parcel_search_radius_m or PARCEL_SEARCH_RADIUS_M)
    # 0.0036 deg envelope (~398m N-S, >=320m E-W across TX) guarantees lossless GIST index pre-filter for 290m
    deg_envelope = max(0.0036, (radius_m / 110_540.0) * 1.38)

    audit = inspect_stage_schema_and_counties(cfg, state_code=st_upper)
    cad_counties: List[CountyStageSummary] = audit["cad_positive_counties"]
    county_meta_by_slug: Dict[str, CountyStageSummary] = {
        c.county_slug: c for c in cad_counties
    }
    hex_counts_by_slug = {c.county_slug: c.cad_positive_hexes for c in cad_counties}

    plan = compute_stratified_county_quotas(
        county_hex_counts=hex_counts_by_slug,
        total_target=total_count,
        val_county=val_county,
        test_county=test_county,
        oversample_factor=oversample_factor,
    )

    if plan.val_fallback_note:
        print(f"[Stage Sampler] NOTE: {plan.val_fallback_note}", flush=True)
    if plan.test_fallback_note:
        print(f"[Stage Sampler] NOTE: {plan.test_fallback_note}", flush=True)

    selected_candidates: List[Dict[str, Any]] = []

    with open_readonly_stage_connection(cfg.stage_database_url, statement_timeout_sec=180) as conn:
        with conn.cursor() as cur:
            # Step 1: Deterministic candidate H3 sampling per county (strictly cad_has_pool IS TRUE)
            for cplan in plan.county_plans:
                meta = county_meta_by_slug[cplan.county_slug]
                cur.execute(
                    """
                    SELECT DISTINCT p.h3_index_res9
                    FROM public.parcels p
                    WHERE p.state_code = %s
                      AND p.county_fips = %s
                      AND COALESCE(p.is_active, true) = true
                      AND p.cad_has_pool IS TRUE
                      AND p.h3_index_res9 IS NOT NULL
                      AND btrim(p.h3_index_res9) <> '';
                    """,
                    (st_upper, meta.county_fips),
                )
                all_cells = [str(r[0]).strip() for r in cur.fetchall() if r[0]]
                all_cells.sort(
                    key=lambda cell: deterministic_cell_sort_key(
                        effective_seed, cplan.county_slug, cell
                    )
                )
                chosen_cells = all_cells[: cplan.candidate_quota]

                for rank_idx, h3_idx in enumerate(chosen_cells):
                    lat, lon = h3_cell_to_latlon(h3_idx)
                    selected_candidates.append({
                        "run_id": run_id,
                        "county": cplan.county_slug,
                        "county_name_db": meta.county_name_db,
                        "county_fips": meta.county_fips,
                        "split": cplan.split,
                        "target_quota": cplan.target_quota,
                        "candidate_quota": len(chosen_cells),
                        "candidate_rank": rank_idx,
                        "h3_index": h3_idx,
                        "lat": lat,
                        "lon": lon,
                    })

            # Step 2: 500-hex GIST-indexed 290m full-tile parcel WKB extraction
            parcels_by_h3: Dict[str, List[Dict[str, Any]]] = {
                item["h3_index"]: [] for item in selected_candidates
            }

            batches = _chunk_seq(selected_candidates, max(1, int(batch_size)))
            for b_idx, batch in enumerate(batches, start=1):
                h3_list = [c["h3_index"] for c in batch]
                lat_list = [float(c["lat"]) for c in batch]
                lon_list = [float(c["lon"]) for c in batch]

                cur.execute(
                    """
                    WITH targets AS (
                      SELECT
                        u.h3_index,
                        ST_SetSRID(ST_MakePoint(u.clon, u.clat), 4326) AS centroid_geom
                      FROM UNNEST(%s::text[], %s::float8[], %s::float8[]) AS u(h3_index, clat, clon)
                    )
                    SELECT
                      t.h3_index,
                      p.id::text AS parcel_id,
                      COALESCE(p.county_name, '') AS county_name,
                      COALESCE(p.county_fips, '') AS county_fips,
                      COALESCE(p.cad_has_pool, false) AS cad_has_pool,
                      COALESCE(p.cad_has_spa, false) AS cad_has_spa,
                      COALESCE(p.land_use_code, '') AS land_use_code,
                      COALESCE(p.normalized_use_type, '') AS normalized_use_type,
                      ST_AsBinary(ST_Transform(p.geom, 4326)) AS geom_wkb
                    FROM targets t
                    JOIN public.parcels p
                      ON p.state_code = %s
                     AND COALESCE(p.is_active, true) = true
                     AND p.geom IS NOT NULL
                     AND p.geom && ST_Expand(t.centroid_geom, %s)
                     AND ST_DWithin(p.geom::geography, t.centroid_geom::geography, %s)
                    ORDER BY t.h3_index, p.id;
                    """,
                    (h3_list, lat_list, lon_list, st_upper, deg_envelope, radius_m),
                )

                for (
                    t_h3,
                    parcel_id,
                    p_cname,
                    p_cfips,
                    cad_pool,
                    cad_spa,
                    lu_code,
                    norm_use,
                    geom_wkb,
                ) in cur.fetchall():
                    if geom_wkb is None:
                        continue
                    norm_use_str = str(norm_use or "").strip().lower()
                    is_res = (norm_use_str in RESIDENTIAL_USE_TYPES) or (norm_use_str == "")
                    wkb_bytes = bytes(geom_wkb)
                    parcels_by_h3[str(t_h3)].append({
                        "parcel_id": str(parcel_id),
                        "county": canonical_county_slug(str(p_cname), str(p_cfips)),
                        "cad_has_pool": bool(cad_pool),
                        "cad_has_spa": bool(cad_spa),
                        "is_residential": bool(is_res),
                        "normalized_use_type": norm_use_str,
                        "land_use_code": str(lu_code or ""),
                        "geom_wkb": wkb_bytes,
                    })

                print(
                    f"[Stage Sampler] Extracted 290m parcels batch {b_idx}/{len(batches)} "
                    f"({len(batch)} candidate hexes)",
                    flush=True,
                )

    # Assemble columnar PyArrow table & write candidates.parquet
    out_dir = cfg.run_cache_dir(run_id)
    candidates_path = out_dir / "candidates.parquet"

    col_data: Dict[str, List[Any]] = {name: [] for name in CANDIDATES_PARQUET_SCHEMA.names}
    total_parcels_extracted = 0
    total_cad_pools_extracted = 0

    for item in selected_candidates:
        p_list = parcels_by_h3.get(item["h3_index"], [])
        cad_cnt = sum(1 for p in p_list if p["cad_has_pool"])
        total_parcels_extracted += len(p_list)
        total_cad_pools_extracted += cad_cnt

        col_data["run_id"].append(item["run_id"])
        col_data["county"].append(item["county"])
        col_data["county_name_db"].append(item["county_name_db"])
        col_data["county_fips"].append(item["county_fips"])
        col_data["split"].append(item["split"])
        col_data["target_quota"].append(int(item["target_quota"]))
        col_data["candidate_quota"].append(int(item["candidate_quota"]))
        col_data["candidate_rank"].append(int(item["candidate_rank"]))
        col_data["h3_index"].append(item["h3_index"])
        col_data["lat"].append(float(item["lat"]))
        col_data["lon"].append(float(item["lon"]))
        col_data["cad_pool_parcel_count"].append(int(cad_cnt))
        col_data["total_parcel_count"].append(int(len(p_list)))
        col_data["parcels_wkb"].append(p_list)

    table = pa.Table.from_pydict(col_data, schema=CANDIDATES_PARQUET_SCHEMA)
    pq.write_table(table, str(candidates_path), compression="zstd")

    stats = {
        "run_id": run_id,
        "candidates_parquet": str(candidates_path),
        "target_tiles": plan.total_target_tiles,
        "candidate_tiles": len(selected_candidates),
        "total_parcels_extracted": total_parcels_extracted,
        "total_cad_pool_parcels_extracted": total_cad_pools_extracted,
        "val_county": plan.val_county,
        "test_county": plan.test_county,
    }
    return candidates_path, plan, stats
