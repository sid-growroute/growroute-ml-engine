BEGIN;

SET LOCAL statement_timeout = '120s';

-- 1. Ensure all canonical ai_* columns exist on public.parcels
ALTER TABLE public.parcels
  ADD COLUMN IF NOT EXISTS ai_has_pool BOOLEAN DEFAULT false,
  ADD COLUMN IF NOT EXISTS ai_pool_confidence NUMERIC(5,3),
  ADD COLUMN IF NOT EXISTS ai_pool_condition TEXT,
  ADD COLUMN IF NOT EXISTS ai_pool_sqft INT,
  ADD COLUMN IF NOT EXISTS ai_has_spa BOOLEAN DEFAULT false,
  ADD COLUMN IF NOT EXISTS ai_tree_canopy_pct NUMERIC(5,2),
  ADD COLUMN IF NOT EXISTS ai_tree_overhang BOOLEAN DEFAULT false,
  ADD COLUMN IF NOT EXISTS ai_has_solar BOOLEAN DEFAULT false,
  ADD COLUMN IF NOT EXISTS ai_lawn_sqft INT,
  ADD COLUMN IF NOT EXISTS ai_last_verified_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS ai_run_id TEXT,
  ADD COLUMN IF NOT EXISTS pool_detection_source TEXT DEFAULT 'none';

-- 2. Backfill existing data from legacy columns before dropping them
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'parcels'
      AND column_name = 'cv_has_pool'
  ) THEN
    UPDATE public.parcels
    SET
      ai_has_pool = COALESCE(ai_has_pool, false) OR COALESCE(cv_has_pool, false),
      ai_pool_confidence = COALESCE(ai_pool_confidence, cv_confidence),
      ai_pool_condition = COALESCE(ai_pool_condition, cv_pool_condition),
      ai_pool_sqft = COALESCE(ai_pool_sqft, pool_sqft),
      ai_tree_overhang = COALESCE(ai_tree_overhang, false) OR COALESCE(tree_overhang, false),
      ai_has_solar = COALESCE(ai_has_solar, false) OR COALESCE(solar_panels, false),
      ai_lawn_sqft = COALESCE(ai_lawn_sqft, lawn_sqft),
      ai_last_verified_at = COALESCE(ai_last_verified_at, last_cv_scanned_at)
    WHERE cv_has_pool IS TRUE
       OR cv_confidence IS NOT NULL
       OR cv_pool_condition IS NOT NULL
       OR pool_sqft IS NOT NULL
       OR tree_overhang IS TRUE
       OR solar_panels IS TRUE
       OR lawn_sqft IS NOT NULL
       OR last_cv_scanned_at IS NOT NULL;
  END IF;
END $$;

-- 3. Drop the 8 duplicate/legacy columns and replace legacy index
DROP INDEX IF EXISTS public.idx_parcels_cv_leads;

ALTER TABLE public.parcels
  DROP COLUMN IF EXISTS cv_has_pool,
  DROP COLUMN IF EXISTS cv_confidence,
  DROP COLUMN IF EXISTS last_cv_scanned_at,
  DROP COLUMN IF EXISTS cv_pool_condition,
  DROP COLUMN IF EXISTS pool_sqft,
  DROP COLUMN IF EXISTS tree_overhang,
  DROP COLUMN IF EXISTS solar_panels,
  DROP COLUMN IF EXISTS lawn_sqft;

CREATE INDEX IF NOT EXISTS idx_parcels_ai_leads
  ON public.parcels (ai_has_pool, cad_has_pool, ai_tree_overhang, ai_has_solar, ai_pool_condition);

-- 4. Update public.ingest_ml_detections to write exclusively to canonical ai_* columns
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
        COALESCE((d->>'ai_pool_confidence')::NUMERIC(5,3), 0.500) AS conf,
        COALESCE((d->>'ai_tree_canopy_pct')::NUMERIC(5,2), 0.00) AS canopy_pct,
        (d->>'ai_pool_sqft')::INT AS ai_sqft,
        COALESCE(d->>'ai_pool_condition', 'clean')::TEXT AS pool_cond,
        COALESCE((d->>'ai_has_spa')::BOOLEAN, false) AS has_spa,
        ST_SetSRID(
          ST_MakePoint((d->>'lng')::DOUBLE PRECISION, (d->>'lat')::DOUBLE PRECISION),
          4326
        ) AS pt_geom
      FROM jsonb_array_elements(p_detections) AS d
      WHERE d ? 'lat' AND d ? 'lng'
    ),
    matched_parcels AS (
      SELECT DISTINCT ON (parcel_match.id)
        parcel_match.id AS parcel_id,
        pd.conf,
        pd.canopy_pct,
        pd.ai_sqft,
        pd.pool_cond,
        pd.has_spa
      FROM parsed_detections pd
      JOIN LATERAL (
        SELECT p.id
        FROM public.parcels p
        WHERE p.geom IS NOT NULL
          AND p.geom && ST_Expand(pd.pt_geom, 0.00018)
          AND ST_DWithin(p.geom, pd.pt_geom, 0.00018)
        ORDER BY
          ST_Contains(p.geom, pd.pt_geom) DESC,
          ST_Distance(p.geom, pd.pt_geom) ASC
        LIMIT 1
      ) parcel_match ON TRUE
      ORDER BY parcel_match.id, pd.conf DESC
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
    WHERE p.id = mp.parcel_id;

    GET DIAGNOSTICS v_matched_count = ROW_COUNT;
  END IF;

  RETURN v_matched_count;
END;
$$;

-- 5. Update apply_ml_batch_detections wrapper to delegate to ingest_ml_detections
CREATE OR REPLACE FUNCTION public.apply_ml_batch_detections(
  p_scanned_h3_cells TEXT[],
  p_detections JSONB DEFAULT '[]'::JSONB
)
RETURNS INT
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  RETURN public.ingest_ml_detections(p_scanned_h3_cells, p_detections, NULL);
END;
$$;

-- 6. Update get_h3_hex_clusters to use canonical ai_* columns
CREATE OR REPLACE FUNCTION public.get_h3_hex_clusters(
  min_lng DOUBLE PRECISION,
  min_lat DOUBLE PRECISION,
  max_lng DOUBLE PRECISION,
  max_lat DOUBLE PRECISION,
  p_county_fips VARCHAR(10) DEFAULT NULL
)
RETURNS TABLE (
  h3_index_res9 TEXT,
  parcel_count INTEGER,
  avg_market_value NUMERIC,
  pool_count INTEGER,
  unpermitted_pool_count INTEGER,
  tree_overhang_count INTEGER,
  center_lat DOUBLE PRECISION,
  center_lng DOUBLE PRECISION
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    p.h3_index_res9,
    COUNT(p.id)::INTEGER AS parcel_count,
    ROUND(AVG(p.market_value), 0) AS avg_market_value,
    COUNT(p.id) FILTER (WHERE COALESCE(p.cad_has_pool, false) = true OR COALESCE(p.ai_has_pool, false) = true)::INTEGER AS pool_count,
    COUNT(p.id) FILTER (WHERE COALESCE(p.ai_has_pool, false) = true AND COALESCE(p.cad_has_pool, false) = false)::INTEGER AS unpermitted_pool_count,
    COUNT(p.id) FILTER (WHERE COALESCE(p.ai_tree_overhang, false) = true OR COALESCE(p.ai_tree_canopy_pct, 0) >= 35)::INTEGER AS tree_overhang_count,
    AVG(COALESCE(p.lat, ST_Y(ST_Centroid(p.geom)))) AS center_lat,
    AVG(COALESCE(p.lng, ST_X(ST_Centroid(p.geom)))) AS center_lng
  FROM public.parcels p
  WHERE COALESCE(p.is_active, true) = true
    AND p.h3_index_res9 IS NOT NULL
    AND p.geom && ST_MakeEnvelope(min_lng, min_lat, max_lng, max_lat, 4326)
    AND (p_county_fips IS NULL OR p.county_fips = p_county_fips)
  GROUP BY p.h3_index_res9;
$$;

-- 7. Update get_parcels_in_bbox to use canonical ai_* columns
DROP FUNCTION IF EXISTS public.get_parcels_in_bbox(
  DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION,
  VARCHAR, VARCHAR, INTEGER, INTEGER, INTEGER, BOOLEAN, BOOLEAN, BOOLEAN, INTEGER
);

CREATE OR REPLACE FUNCTION public.get_parcels_in_bbox(
  min_lng DOUBLE PRECISION,
  min_lat DOUBLE PRECISION,
  max_lng DOUBLE PRECISION,
  max_lat DOUBLE PRECISION,
  p_county_fips VARCHAR(10) DEFAULT NULL,
  p_smart_campaign VARCHAR(50) DEFAULT NULL,
  p_min_gallons INTEGER DEFAULT NULL,
  p_max_canopy_pct INTEGER DEFAULT NULL,
  p_min_canopy_pct INTEGER DEFAULT NULL,
  p_has_attached_spa BOOLEAN DEFAULT NULL,
  p_absentee_only BOOLEAN DEFAULT NULL,
  p_exclude_do_not_mail BOOLEAN DEFAULT true,
  p_limit INTEGER DEFAULT 1500
)
RETURNS TABLE (
  id UUID,
  state_code VARCHAR(2),
  county_fips VARCHAR(10),
  county_name VARCHAR(100),
  apn TEXT,
  site_address TEXT,
  site_city TEXT,
  zip_code TEXT,
  mailing_address TEXT,
  mailing_city TEXT,
  mailing_state TEXT,
  mailing_zip TEXT,
  owner_entity_type TEXT,
  owner_cluster_id TEXT,
  is_owner_occupied BOOLEAN,
  is_out_of_state_owner BOOLEAN,
  total_portfolio_size INTEGER,
  do_not_mail BOOLEAN,
  market_value NUMERIC,
  year_built INTEGER,
  living_area_sqft NUMERIC,
  cad_has_pool BOOLEAN,
  cad_pool_sqft INTEGER,
  ai_has_pool BOOLEAN,
  pool_detection_source VARCHAR(30),
  gallons INTEGER,
  ai_pool_sqft INTEGER,
  ai_pool_condition VARCHAR(50),
  ai_has_spa BOOLEAN,
  cad_has_spa BOOLEAN,
  has_attached_spa BOOLEAN,
  ai_tree_canopy_pct NUMERIC,
  ai_tree_overhang BOOLEAN,
  ai_has_solar BOOLEAN,
  ai_lawn_sqft INTEGER,
  tree_canopy_coverage_pct INTEGER,
  ai_pool_confidence NUMERIC,
  confidence_score NUMERIC,
  h3_index_res9 TEXT,
  centroid_lat DOUBLE PRECISION,
  centroid_lng DOUBLE PRECISION,
  geom_geojson JSONB
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    p.id,
    p.state_code,
    p.county_fips,
    p.county_name,
    p.apn,
    p.site_address,
    p.site_city,
    p.zip_code,
    p.mailing_address,
    p.mailing_city,
    p.mailing_state,
    p.mailing_zip,
    p.owner_entity_type,
    p.owner_cluster_id,
    p.is_owner_occupied,
    p.is_out_of_state_owner,
    p.total_portfolio_size,
    p.do_not_mail,
    p.market_value,
    p.year_built,
    p.living_area_sqft,
    p.cad_has_pool,
    p.cad_pool_sqft::INTEGER AS cad_pool_sqft,
    COALESCE(p.ai_has_pool, false) AS ai_has_pool,
    p.pool_detection_source::VARCHAR(30) AS pool_detection_source,
    COALESCE(pl.gallons, COALESCE(p.ai_pool_sqft, p.cad_pool_sqft::INTEGER, 450) * 38, 18000)::INTEGER AS gallons,
    p.ai_pool_sqft,
    COALESCE(pl.pool_condition, p.ai_pool_condition, 'clean')::VARCHAR(50) AS ai_pool_condition,
    COALESCE(p.ai_has_spa, false) AS ai_has_spa,
    COALESCE(p.cad_has_spa, false) AS cad_has_spa,
    COALESCE(pl.has_attached_spa, p.ai_has_spa, p.cad_has_spa, false) AS has_attached_spa,
    p.ai_tree_canopy_pct,
    COALESCE(p.ai_tree_overhang, false) AS ai_tree_overhang,
    COALESCE(p.ai_has_solar, false) AS ai_has_solar,
    p.ai_lawn_sqft,
    COALESCE(pl.tree_canopy_coverage_pct, p.ai_tree_canopy_pct::INTEGER, CASE WHEN p.ai_tree_overhang THEN 55 ELSE 10 END)::INTEGER AS tree_canopy_coverage_pct,
    p.ai_pool_confidence,
    COALESCE(pl.confidence_score, p.ai_pool_confidence, 0.960) AS confidence_score,
    p.h3_index_res9,
    COALESCE(p.lat, ST_Y(ST_Centroid(p.geom))) AS centroid_lat,
    COALESCE(p.lng, ST_X(ST_Centroid(p.geom))) AS centroid_lng,
    ST_AsGeoJSON(p.geom)::JSONB AS geom_geojson
  FROM public.parcels p
  LEFT JOIN public.pools pl ON pl.parcel_id = p.id
  WHERE COALESCE(p.is_active, true) = true
    AND p.geom && ST_MakeEnvelope(min_lng, min_lat, max_lng, max_lat, 4326)
    AND (
      COALESCE(p.cad_has_pool, false) = true
      OR COALESCE(p.ai_has_pool, false) = true
      OR pl.id IS NOT NULL
    )
    AND (p_county_fips IS NULL OR p.county_fips = p_county_fips)
    AND (NOT p_exclude_do_not_mail OR COALESCE(p.do_not_mail, false) = false)
    AND (p_min_gallons IS NULL OR COALESCE(pl.gallons, COALESCE(p.ai_pool_sqft, p.cad_pool_sqft::INTEGER, 450) * 38, 18000) >= p_min_gallons)
    AND (p_max_canopy_pct IS NULL OR COALESCE(pl.tree_canopy_coverage_pct, p.ai_tree_canopy_pct::INTEGER, 10) <= p_max_canopy_pct)
    AND (p_min_canopy_pct IS NULL OR COALESCE(pl.tree_canopy_coverage_pct, p.ai_tree_canopy_pct::INTEGER, 10) >= p_min_canopy_pct)
    AND (p_has_attached_spa IS NULL OR COALESCE(pl.has_attached_spa, p.ai_has_spa, p.cad_has_spa, false) = p_has_attached_spa)
    AND (p_absentee_only IS NULL OR (p_absentee_only = true AND COALESCE(p.is_owner_occupied, true) = false))
    AND (
      p_smart_campaign IS NULL
      OR p_smart_campaign = 'neighbor_blitz'
      OR (p_smart_campaign = 'big_water' AND COALESCE(pl.gallons, COALESCE(p.ai_pool_sqft, p.cad_pool_sqft::INTEGER, 450) * 38, 18000) >= 20000)
      OR (p_smart_campaign = 'easy_money' AND COALESCE(pl.tree_canopy_coverage_pct, p.ai_tree_canopy_pct::INTEGER, 10) <= 20)
      OR (p_smart_campaign = 'heavy_debris' AND (COALESCE(pl.tree_canopy_coverage_pct, p.ai_tree_canopy_pct::INTEGER, 10) >= 40 OR COALESCE(p.ai_tree_overhang, false) = true))
      OR (p_smart_campaign = 'absentee_landlord' AND (COALESCE(p.is_owner_occupied, true) = false OR COALESCE(p.total_portfolio_size, 1) > 1))
      OR (p_smart_campaign = 'commercial_hoa' AND p.owner_entity_type = 'corporate')
      OR (p_smart_campaign = 'unpermitted_pools' AND COALESCE(p.ai_has_pool, false) = true AND COALESCE(p.cad_has_pool, false) = false)
    )
  LIMIT LEAST(COALESCE(p_limit, 1500), 5000);
$$;

-- 8. Update get_nearest_5_unshielded_pools to return ai_pool_sqft and cad_pool_sqft
DROP FUNCTION IF EXISTS public.get_nearest_5_unshielded_pools(UUID, DOUBLE PRECISION, DOUBLE PRECISION, INTEGER);

CREATE OR REPLACE FUNCTION public.get_nearest_5_unshielded_pools(
  p_user_id UUID,
  p_lat DOUBLE PRECISION,
  p_lng DOUBLE PRECISION,
  p_limit INTEGER DEFAULT 5
)
RETURNS TABLE (
  id TEXT,
  address TEXT,
  normalized_address TEXT,
  city TEXT,
  state TEXT,
  zip TEXT,
  lat DOUBLE PRECISION,
  lng DOUBLE PRECISION,
  ai_pool_sqft NUMERIC,
  cad_pool_sqft NUMERIC,
  market_value NUMERIC,
  distance_meters DOUBLE PRECISION
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_seed_geog GEOGRAPHY := ST_SetSRID(ST_MakePoint(p_lng, p_lat), 4326)::GEOGRAPHY;
  v_seed_geom GEOMETRY := ST_SetSRID(ST_MakePoint(p_lng, p_lat), 4326);
  v_limit INTEGER := GREATEST(1, LEAST(COALESCE(p_limit, 5), 25));
BEGIN
  IF to_regclass('public.parcels') IS NULL THEN
    RETURN;
  END IF;

  RETURN QUERY
  WITH candidate_pools AS (
    SELECT
      p.id::TEXT AS prop_id,
      p.site_address::TEXT AS prop_address,
      public.normalize_shield_address(p.site_address::TEXT) AS norm_addr,
      COALESCE(p.site_city::TEXT, 'Austin') AS prop_city,
      COALESCE(p.state_code::TEXT, 'TX') AS prop_state,
      COALESCE(p.zip_code::TEXT, '') AS prop_zip,
      COALESCE(p.lat, ST_Y(ST_Centroid(p.geom))) AS prop_lat,
      COALESCE(p.lng, ST_X(ST_Centroid(p.geom))) AS prop_lng,
      p.ai_pool_sqft::NUMERIC AS prop_ai_pool_sqft,
      COALESCE(p.cad_pool_sqft, 450)::NUMERIC AS prop_cad_pool_sqft,
      COALESCE(p.market_value, 485000)::NUMERIC AS prop_market_value,
      ST_Distance(
        ST_SetSRID(
          ST_MakePoint(
            COALESCE(p.lng, ST_X(ST_Centroid(p.geom))),
            COALESCE(p.lat, ST_Y(ST_Centroid(p.geom)))
          ),
          4326
        )::GEOGRAPHY,
        v_seed_geog
      ) AS dist_m,
      ST_SetSRID(
        ST_MakePoint(
          COALESCE(p.lng, ST_X(ST_Centroid(p.geom))),
          COALESCE(p.lat, ST_Y(ST_Centroid(p.geom)))
        ),
        4326
      ) AS pt_geom
    FROM public.parcels p
    WHERE COALESCE(p.is_active, true) = true
      AND COALESCE(p.do_not_mail, false) = false
      AND (COALESCE(p.cad_has_pool, false) = true OR COALESCE(p.ai_has_pool, false) = true)
      AND p.site_address IS NOT NULL
      AND btrim(p.site_address) <> ''
      AND ST_DWithin(
        ST_SetSRID(
          ST_MakePoint(
            COALESCE(p.lng, ST_X(ST_Centroid(p.geom))),
            COALESCE(p.lat, ST_Y(ST_Centroid(p.geom)))
          ),
          4326
        )::GEOGRAPHY,
        v_seed_geog,
        1500
      )
  )
  SELECT
    cp.prop_id AS id,
    cp.prop_address AS address,
    cp.norm_addr AS normalized_address,
    cp.prop_city AS city,
    cp.prop_state AS state,
    cp.prop_zip AS zip,
    cp.prop_lat AS lat,
    cp.prop_lng AS lng,
    COALESCE(cp.prop_ai_pool_sqft, cp.prop_cad_pool_sqft) AS ai_pool_sqft,
    cp.prop_cad_pool_sqft AS cad_pool_sqft,
    cp.prop_market_value AS market_value,
    cp.dist_m AS distance_meters
  FROM candidate_pools cp
  WHERE cp.dist_m > 12.0
    AND NOT EXISTS (
      SELECT 1
      FROM public.campaign_shield_locks csl
      WHERE csl.normalized_address = cp.norm_addr
        AND csl.expires_at > now()
    )
    AND NOT EXISTS (
      SELECT 1
      FROM public.protected_customers pc
      WHERE pc.normalized_address = cp.norm_addr
        AND pc.shield_active = true
    )
  ORDER BY cp.pt_geom <-> v_seed_geom ASC, cp.dist_m ASC
  LIMIT v_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION public.get_nearest_5_unshielded_pools(UUID, DOUBLE PRECISION, DOUBLE PRECISION, INTEGER) TO anon, authenticated, service_role;

-- 9. Update rpc_five_neighbor_blitz to use canonical ai_* columns
CREATE OR REPLACE FUNCTION public.rpc_five_neighbor_blitz(
  p_center_parcel_id UUID,
  p_profile_id UUID DEFAULT NULL,
  p_neighbors_count INTEGER DEFAULT 5
)
RETURNS TABLE (
  parcel_id UUID,
  masked_site_address TEXT,
  site_city VARCHAR,
  zip_code VARCHAR,
  lat DOUBLE PRECISION,
  lng DOUBLE PRECISION,
  distance_meters DOUBLE PRECISION,
  gallons INTEGER,
  tree_canopy_coverage_pct INTEGER
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  WITH center AS (
    SELECT id, state_code, county_fips, geom
    FROM public.parcels
    WHERE id = p_center_parcel_id
    LIMIT 1
  )
  SELECT
    p.id AS parcel_id,
    REGEXP_REPLACE(p.site_address, '^[0-9A-Za-z-]+', '***') AS masked_site_address,
    p.site_city,
    p.zip_code,
    p.lat,
    p.lng,
    ROUND(ST_Distance(p.geom::GEOGRAPHY, c.geom::GEOGRAPHY)::NUMERIC, 1)::DOUBLE PRECISION AS distance_meters,
    COALESCE(pl.gallons, COALESCE(p.ai_pool_sqft, p.cad_pool_sqft::INTEGER, 450) * 40, 18000) AS gallons,
    COALESCE(pl.tree_canopy_coverage_pct, p.ai_tree_canopy_pct::INTEGER, CASE WHEN p.ai_tree_overhang THEN 35 ELSE 10 END) AS tree_canopy_coverage_pct
  FROM center c
  INNER JOIN public.parcels p
    ON p.state_code = c.state_code
   AND p.county_fips = c.county_fips
   AND p.id != c.id
  LEFT JOIN public.pools pl ON pl.parcel_id = p.id
  WHERE COALESCE(p.is_active, true) = true
    AND COALESCE(p.do_not_mail, false) = false
    AND (COALESCE(p.cad_has_pool, false) = true OR COALESCE(p.ai_has_pool, false) = true)
    AND ST_DWithin(p.geom::GEOGRAPHY, c.geom::GEOGRAPHY, 800)
    AND NOT EXISTS (
      SELECT 1 FROM public.route_shield rs
      WHERE rs.profile_id = p_profile_id
        AND rs.parcel_id = p.id
    )
  ORDER BY p.geom <-> c.geom
  LIMIT COALESCE(p_neighbors_count, 5);
$$;

-- 10. Update rpc_search_route_command_parcels to use canonical ai_* columns
DROP FUNCTION IF EXISTS public.rpc_search_route_command_parcels(
  DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION,
  UUID, BOOLEAN, INTEGER, INTEGER, INTEGER, BOOLEAN, BOOLEAN, BOOLEAN, BOOLEAN, INTEGER
);

CREATE OR REPLACE FUNCTION public.rpc_search_route_command_parcels(
  p_min_lng DOUBLE PRECISION,
  p_min_lat DOUBLE PRECISION,
  p_max_lng DOUBLE PRECISION,
  p_max_lat DOUBLE PRECISION,
  p_profile_id UUID DEFAULT NULL,
  p_pools_only BOOLEAN DEFAULT true,
  p_min_gallons INTEGER DEFAULT 0,
  p_min_canopy_pct INTEGER DEFAULT 0,
  p_max_canopy_pct INTEGER DEFAULT 100,
  p_absentee_only BOOLEAN DEFAULT false,
  p_portfolio_only BOOLEAN DEFAULT false,
  p_unpermitted_only BOOLEAN DEFAULT false,
  p_green_algae_only BOOLEAN DEFAULT false,
  p_limit INTEGER DEFAULT 1000
)
RETURNS TABLE (
  parcel_id UUID,
  state_code VARCHAR,
  county_fips VARCHAR,
  apn VARCHAR,
  masked_site_address TEXT,
  site_city VARCHAR,
  zip_code VARCHAR,
  lat DOUBLE PRECISION,
  lng DOUBLE PRECISION,
  h3_index_res9 VARCHAR,
  is_owner_occupied BOOLEAN,
  is_out_of_state_owner BOOLEAN,
  owner_entity_type VARCHAR,
  owner_cluster_id VARCHAR,
  total_portfolio_size INTEGER,
  market_value NUMERIC,
  year_built INTEGER,
  living_area_sqft INTEGER,
  cad_has_pool BOOLEAN,
  ai_has_pool BOOLEAN,
  ai_pool_sqft INTEGER,
  cad_pool_sqft INTEGER,
  gallons INTEGER,
  ai_pool_condition VARCHAR,
  has_attached_spa BOOLEAN,
  tree_canopy_coverage_pct INTEGER,
  confidence_score NUMERIC,
  is_shielded BOOLEAN
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    p.id AS parcel_id,
    p.state_code,
    p.county_fips,
    p.apn,
    REGEXP_REPLACE(p.site_address, '^[0-9A-Za-z-]+', '***') AS masked_site_address,
    p.site_city,
    p.zip_code,
    p.lat,
    p.lng,
    p.h3_index_res9,
    p.is_owner_occupied,
    p.is_out_of_state_owner,
    p.owner_entity_type,
    p.owner_cluster_id,
    p.total_portfolio_size,
    p.market_value,
    p.year_built,
    p.living_area_sqft::INTEGER AS living_area_sqft,
    COALESCE(p.cad_has_pool, false) AS cad_has_pool,
    COALESCE(p.ai_has_pool, false) AS ai_has_pool,
    p.ai_pool_sqft,
    p.cad_pool_sqft::INTEGER AS cad_pool_sqft,
    COALESCE(pl.gallons, COALESCE(p.ai_pool_sqft, p.cad_pool_sqft::INTEGER, 450) * 40, 18000) AS gallons,
    COALESCE(pl.pool_condition, p.ai_pool_condition, 'clean')::VARCHAR AS ai_pool_condition,
    COALESCE(pl.has_attached_spa, p.ai_has_spa, p.cad_has_spa, false) AS has_attached_spa,
    COALESCE(pl.tree_canopy_coverage_pct, p.ai_tree_canopy_pct::INTEGER, CASE WHEN p.ai_tree_overhang THEN 35 ELSE 10 END) AS tree_canopy_coverage_pct,
    COALESCE(pl.confidence_score, p.ai_pool_confidence, 0.960) AS confidence_score,
    EXISTS (
      SELECT 1 FROM public.route_shield rs
      WHERE rs.profile_id = p_profile_id
        AND (rs.parcel_id = p.id OR UPPER(TRIM(rs.normalized_address)) = UPPER(TRIM(p.site_address)))
    ) AS is_shielded
  FROM public.parcels p
  LEFT JOIN public.pools pl ON pl.parcel_id = p.id
  WHERE COALESCE(p.is_active, true) = true
    AND COALESCE(p.do_not_mail, false) = false
    AND p.geom && ST_MakeEnvelope(p_min_lng, p_min_lat, p_max_lng, p_max_lat, 4326)
    AND (NOT p_pools_only OR COALESCE(p.cad_has_pool, false) = true OR COALESCE(p.ai_has_pool, false) = true)
    AND COALESCE(pl.gallons, COALESCE(p.ai_pool_sqft, p.cad_pool_sqft::INTEGER, 450) * 40, 18000) >= p_min_gallons
    AND COALESCE(pl.tree_canopy_coverage_pct, p.ai_tree_canopy_pct::INTEGER, CASE WHEN p.ai_tree_overhang THEN 35 ELSE 10 END) BETWEEN p_min_canopy_pct AND p_max_canopy_pct
    AND (NOT p_absentee_only OR COALESCE(p.is_owner_occupied, true) = false)
    AND (NOT p_portfolio_only OR COALESCE(p.total_portfolio_size, 1) > 1)
    AND (NOT p_unpermitted_only OR (COALESCE(p.ai_has_pool, false) = true AND COALESCE(p.cad_has_pool, false) = false))
    AND (NOT p_green_algae_only OR COALESCE(pl.pool_condition, p.ai_pool_condition, '') ILIKE '%green%' OR COALESCE(pl.pool_condition, p.ai_pool_condition, '') ILIKE '%algae%')
  LIMIT LEAST(COALESCE(p_limit, 1000), 5000);
$$;

NOTIFY pgrst, 'reload schema';

COMMIT;
