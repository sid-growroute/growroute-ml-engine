-- ============================================================================
-- Migration: 20261115_fix_ingest_ml_detections_strict_contains.sql
-- Purpose:
--   1. Eliminate 1-to-2 lot neighbor spillover in public.ingest_ml_detections
--      by replacing ST_Expand(..., 0.00018) + ST_DWithin(..., 0.00018) (~20m
--      proximity snap across property fences) with strict point-in-polygon
--      containment: p.geom && pd.pt_geom AND ST_Contains(p.geom, pd.pt_geom).
--   2. Enforce minimum pool confidence floor >= 0.550 and preserve exact
--      ST_MakePoint(lng, lat) (X=lon, Y=lat) EPSG:4326 axis order.
--   3. Include state_code in matched_parcels join for partition pruning on
--      public.parcels.
-- ============================================================================

BEGIN;

SET LOCAL statement_timeout = '120s';

CREATE OR REPLACE FUNCTION public.ingest_ml_detections(
  p_scanned_h3_cells TEXT[],
  p_detections JSONB DEFAULT '[]'::JSONB,
  p_run_id TEXT DEFAULT NULL
)
RETURNS INT
LANGUAGE plpgsql
SECURITY DEFINER
SET statement_timeout = '120s'
AS $$
DECLARE
  v_matched_count INT := 0;
BEGIN
  UPDATE public.parcels p
  SET
    ai_last_verified_at = now(),
    ai_run_id = COALESCE(p_run_id, p.ai_run_id),
    ai_has_pool = COALESCE(p.ai_has_pool, false)
  WHERE p.h3_index_res9 = ANY(p_scanned_h3_cells)
    AND p.ai_last_verified_at IS NULL;

  IF jsonb_typeof(p_detections) = 'array' AND jsonb_array_length(p_detections) > 0 THEN
    WITH parsed_detections AS (
      SELECT
        (d->>'lat')::DOUBLE PRECISION AS lat,
        (d->>'lng')::DOUBLE PRECISION AS lng,
        COALESCE((d->>'ai_pool_confidence')::NUMERIC(5,3), 0.000) AS conf,
        COALESCE((d->>'ai_tree_canopy_pct')::NUMERIC(5,2), 0.00) AS canopy_pct,
        (d->>'ai_pool_sqft')::INT AS ai_sqft,
        COALESCE(d->>'ai_pool_condition', 'clean')::TEXT AS pool_cond,
        COALESCE((d->>'ai_has_spa')::BOOLEAN, false) AS has_spa,
        ST_SetSRID(
          ST_MakePoint((d->>'lng')::DOUBLE PRECISION, (d->>'lat')::DOUBLE PRECISION),
          4326
        ) AS pt_geom
      FROM jsonb_array_elements(p_detections) AS d
      WHERE d ? 'lat'
        AND d ? 'lng'
        AND COALESCE((d->>'ai_pool_confidence')::NUMERIC(5,3), 0.000) >= 0.550
    ),
    matched_parcels AS (
      SELECT DISTINCT ON (parcel_match.state_code, parcel_match.id)
        parcel_match.state_code,
        parcel_match.id AS parcel_id,
        pd.conf,
        pd.canopy_pct,
        pd.ai_sqft,
        pd.pool_cond,
        pd.has_spa
      FROM parsed_detections pd
      JOIN LATERAL (
        SELECT p.state_code, p.id
        FROM public.parcels p
        WHERE p.geom IS NOT NULL
          AND p.geom && pd.pt_geom
          AND ST_Contains(p.geom, pd.pt_geom)
        ORDER BY p.assessed_value DESC NULLS LAST
        LIMIT 1
      ) parcel_match ON TRUE
      ORDER BY parcel_match.state_code, parcel_match.id, pd.conf DESC
    )
    UPDATE public.parcels p
    SET
      ai_has_pool = true,
      ai_pool_confidence = GREATEST(COALESCE(p.ai_pool_confidence, 0), mp.conf),
      pool_detection_source = CASE
        WHEN p.cad_has_pool IS TRUE THEN 'both'
        ELSE 'ai'
      END,
      ai_tree_canopy_pct = mp.canopy_pct,
      ai_tree_overhang = (mp.canopy_pct >= 35.0),
      ai_pool_sqft = COALESCE(NULLIF(mp.ai_sqft, 0), p.ai_pool_sqft),
      ai_pool_condition = mp.pool_cond,
      ai_has_spa = COALESCE(p.ai_has_spa, false) OR mp.has_spa,
      ai_run_id = COALESCE(p_run_id, p.ai_run_id),
      ai_last_verified_at = now(),
      updated_at = now()
    FROM matched_parcels mp
    WHERE p.state_code = mp.state_code
      AND p.id = mp.parcel_id;

    GET DIAGNOSTICS v_matched_count = ROW_COUNT;
  END IF;

  RETURN v_matched_count;
END;
$$;

COMMIT;
