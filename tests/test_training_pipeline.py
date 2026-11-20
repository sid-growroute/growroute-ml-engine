from __future__ import annotations

from affine import Affine
import numpy as np
from pyproj import Transformer
import pytest
from shapely.geometry import Polygon
from shapely import wkb as shapely_wkb

from training.config import PROD_PROJECT_REF, STAGE_PROJECT_REF, assert_stage_dsn_safe, redact_dsn
from training.naip_fetcher import (
    compute_centroid_tile_window,
    is_chip_exposure_valid,
    latlon_to_usgs_quarter_quad,
    project_parcels_to_tile_pixels,
    select_best_cog_for_candidate,
)
from training.stage_sampler import (
    canonical_county_slug,
    compute_stratified_county_quotas,
)


class TestProdDsnGuard:
    def test_rejects_prod_project_ref_in_dsn(self) -> None:
        prod_dsn = (
            f"postgresql://postgres.{PROD_PROJECT_REF}:secret123"
            "@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
        )
        with pytest.raises(ValueError, match=PROD_PROJECT_REF):
            assert_stage_dsn_safe(prod_dsn)

        with pytest.raises(ValueError):
            assert_stage_dsn_safe("")

    def test_accepts_stage_dsn_and_encodes_special_password_chars(self) -> None:
        raw_stage = (
            f"postgresql://postgres.{STAGE_PROJECT_REF}:p@ss[w0rd]#1"
            "@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
        )
        safe = assert_stage_dsn_safe(raw_stage)
        assert STAGE_PROJECT_REF in safe
        assert "p%40ss%5Bw0rd%5D%231@" in safe
        redacted = redact_dsn(safe)
        assert "p%40ss" not in redacted
        assert "***REDACTED***" in redacted


class TestUsgsQuarterQuadMath:
    def test_brownsville_verified_naip_key_and_grid_corners(self) -> None:
        # Verified against live s3://naip-visualization/tx/2022/60cm/rgb/25097/m_2509703_ne_14_060_20220605.tif
        blk, qq = latlon_to_usgs_quarter_quad(25.96875, -97.65625)
        assert blk == "25097"
        assert qq == "2509703_ne"

        # NW-most quarter-quad of 1x1 cell 32097: lat in [32.9375, 33.0), lon in [-98.0, -97.9375)
        assert latlon_to_usgs_quarter_quad(32.97, -97.97) == ("32097", "3209701_nw")
        # NE-most quarter-quad of 1x1 cell 32097: lat in [32.9375, 33.0), lon in [-97.0625, -97.0)
        assert latlon_to_usgs_quarter_quad(32.97, -97.02) == ("32097", "3209708_ne")
        # SW-most quarter-quad of 1x1 cell 32097: lat in [32.0, 32.0625), lon in [-98.0, -97.9375)
        assert latlon_to_usgs_quarter_quad(32.02, -97.97) == ("32097", "3209757_sw")
        # SE-most quarter-quad of 1x1 cell 32097: lat in [32.0, 32.0625), lon in [-97.0625, -97.0)
        assert latlon_to_usgs_quarter_quad(32.02, -97.02) == ("32097", "3209764_se")

    def test_selects_newest_acquisition_date_per_quarter_quad(self) -> None:
        qq_index = {
            "2509703_ne": [
                ("20220605", 14, "tx/2022/60cm/rgb/25097/m_2509703_ne_14_060_20220605.tif"),
                ("20220512", 14, "tx/2022/60cm/rgb/25097/m_2509703_ne_14_060_20220512.tif"),
            ]
        }
        chosen = select_best_cog_for_candidate(25.96875, -97.65625, qq_index)
        assert chosen == "tx/2022/60cm/rgb/25097/m_2509703_ne_14_060_20220605.tif"


class TestStratificationQuotas:
    def test_10k_quotas_with_20_counties_floor_sqrt_and_oversample(self) -> None:
        # 20 CAD-positive counties (2 holdout + 18 train), including one small county (<200 hexes)
        counts = {
            "comal": 1500,
            "Fortbend": 900,  # Tests DB 'Fortbend' -> canonical 'fort_bend'
            "harris": 12000,
            "dallas": 9000,
            "tarrant": 8500,
            "collin": 7000,
            "denton": 6500,
            "travis": 4000,
            "bexar": 3800,
            "montgomery": 3200,
            "williamson": 2800,
            "brazoria": 2200,
            "galveston": 1800,
            "nueces": 1600,
            "ellis": 1400,
            "johnson": 1200,
            "rockwall": 1100,
            "kaufman": 950,
            "bell": 850,
            "cameron": 140,  # Smaller than 200 floor -> must cap at 140!
        }
        assert canonical_county_slug("Fortbend", "48157") == "fort_bend"

        plan = compute_stratified_county_quotas(
            county_hex_counts=counts,
            total_target=10_000,
            val_county="comal",
            test_county="fort_bend",
            oversample_factor=1.15,
        )

        assert plan.val_county == "comal"
        assert plan.test_county == "fort_bend"
        assert plan.total_target_tiles == 10_000

        by_county = {p.county_slug: p for p in plan.county_plans}
        assert by_county["comal"].split == "val"
        assert by_county["comal"].target_quota == 1_000
        assert by_county["comal"].candidate_quota == 1_150

        assert by_county["fort_bend"].split == "test"
        assert by_county["fort_bend"].target_quota == 500
        assert by_county["fort_bend"].candidate_quota == 575

        train_plans = [p for p in plan.county_plans if p.split == "train"]
        assert len(train_plans) == 18
        assert sum(p.target_quota for p in train_plans) == 8_500
        # Zero spatial overlap between train and val/test
        train_names = {p.county_slug for p in train_plans}
        assert "comal" not in train_names
        assert "fort_bend" not in train_names

        # Floor rule: min(200, available_hexes) respected, and sqrt weighting prevents Harris from drowning smaller counties
        assert by_county["cameron"].target_quota == 140
        assert by_county["bell"].target_quota >= 200
        assert by_county["harris"].target_quota > by_county["bell"].target_quota
        assert by_county["harris"].target_quota < 1_500

    def test_holdout_fallback_when_comal_missing_in_stage(self) -> None:
        stage_counts = {
            "Collin": 294,
            "Rockwall": 568,
            "Dallas": 224,
            "Denton": 812,
            "Tarrant": 371,
            "Fortbend": 218,
            "Williamson": 578,
            "Travis": 183,
        }
        plan = compute_stratified_county_quotas(
            county_hex_counts=stage_counts,
            total_target=100,
            val_county="comal",
            test_county="fort_bend",
        )
        assert plan.test_county == "fort_bend"
        assert plan.val_county == "williamson"
        assert plan.val_fallback_note is not None
        assert plan.total_target_tiles == 100


class TestInverseAffinePixelProjectionAndExposure:
    def test_zero_drift_utm_to_pixel_projection_clipping_and_sliver_drop(self) -> None:
        # Synthetic NAIP UTM Zone 14N (EPSG:26914) 0.60 m/px source GeoTIFF around Frisco/Collin TX
        src_crs = "EPSG:26914"
        src_transform = Affine(0.6, 0.0, 700_000.0, 0.0, -0.6, 3_670_000.0)

        to_wgs84 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        # Pick centroid at pixel (1000, 1000) in source COG
        cx_utm, cy_utm = src_transform @ (1000.0, 1000.0)
        c_lon, c_lat = to_wgs84.transform(cx_utm, cy_utm)

        win = compute_centroid_tile_window(src_transform, src_crs, lat=c_lat, lon=c_lon, tile_px=640)
        assert (win.col_off, win.row_off, win.width, win.height) == (680, 680, 640, 640)

        win_transform = src_transform @ Affine.translation(win.col_off, win.row_off)

        # Parcel 1: Interior residential pool lot at exact tile pixel box [200..250, 300..360]
        p1_utm_pts = [win_transform @ pt for pt in [(200, 300), (250, 300), (250, 360), (200, 360)]]
        p1_lonlat = [to_wgs84.transform(x, y) for x, y in p1_utm_pts]
        p1_poly = Polygon(p1_lonlat)

        # Parcel 2: Boundary sliver overlapping [639.0..650.0, 100.0..102.0] -> clipped area = 1.0 * 2.0 = 2.0 sq px (< 4.0 -> dropped!)
        p2_utm_pts = [win_transform @ pt for pt in [(639.0, 100.0), (650.0, 100.0), (650.0, 102.0), (639.0, 102.0)]]
        p2_poly = Polygon([to_wgs84.transform(x, y) for x, y in p2_utm_pts])

        # Parcel 3: Straddling parcel [620..660, 400..430] -> clipped to [620..640, 400..430] = 20 * 30 = 600 sq px
        p3_utm_pts = [win_transform @ pt for pt in [(620, 400), (660, 400), (660, 430), (620, 430)]]
        p3_poly = Polygon([to_wgs84.transform(x, y) for x, y in p3_utm_pts])

        parcels_in = [
            {
                "parcel_id": "lot_pool_1",
                "cad_has_pool": True,
                "cad_has_spa": True,
                "is_residential": True,
                "normalized_use_type": "sfr",
                "geom_wkb": shapely_wkb.dumps(p1_poly),
            },
            {
                "parcel_id": "lot_sliver_2",
                "cad_has_pool": False,
                "cad_has_spa": False,
                "is_residential": True,
                "normalized_use_type": "sfr",
                "geom_wkb": shapely_wkb.dumps(p2_poly),
            },
            {
                "parcel_id": "lot_edge_3",
                "cad_has_pool": False,
                "cad_has_spa": False,
                "is_residential": True,
                "normalized_use_type": "sfr",
                "geom_wkb": shapely_wkb.dumps(p3_poly),
            },
        ]

        projected = project_parcels_to_tile_pixels(
            parcels_wkb=parcels_in,
            dst_crs=src_crs,
            win_transform=win_transform,
            tile_px=640,
            min_area_px=4.0,
        )

        ids = [p["parcel_id"] for p in projected]
        assert ids == ["lot_pool_1", "lot_edge_3"]

        p1_out = projected[0]
        assert p1_out["area_px"] == pytest.approx(50.0 * 60.0, abs=0.1)
        ring_xs = [pt[0] for pt in p1_out["pixel_rings"][0]]
        ring_ys = [pt[1] for pt in p1_out["pixel_rings"][0]]
        assert min(ring_xs) == pytest.approx(200.0, abs=0.02)
        assert max(ring_xs) == pytest.approx(250.0, abs=0.02)
        assert min(ring_ys) == pytest.approx(300.0, abs=0.02)
        assert max(ring_ys) == pytest.approx(360.0, abs=0.02)

        p3_out = projected[1]
        assert p3_out["area_px"] == pytest.approx(20.0 * 30.0, abs=0.1)
        assert max(pt[0] for pt in p3_out["pixel_rings"][0]) <= 640.0

    def test_nodata_exposure_guard_98_percent_threshold(self) -> None:
        good_chip = np.full((640, 640, 3), 110, dtype=np.uint8)
        ok, frac = is_chip_exposure_valid(good_chip)
        assert ok is True
        assert frac == 1.0

        # 5% black nodata strip fails 98% guard
        seam_chip = good_chip.copy()
        seam_chip[:, :32, :] = 0
        ok_seam, frac_seam = is_chip_exposure_valid(seam_chip)
        assert ok_seam is False
        assert frac_seam == pytest.approx(0.95, abs=0.002)
