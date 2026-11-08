.PHONY: build-cpu run-parallel queue-dfw

MAX_WORKERS ?= 3
MAX_TILES ?= 500
NAS_ROOT ?= /Volumes/GrowrouteLLC/growroute-data

build-cpu:
	docker build -f Dockerfile.cpu -t growroute-ml:cpu .

queue-dfw:
	./.venv/bin/python3 -c 'import os; from dotenv import load_dotenv; from supabase import create_client; load_dotenv(); sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]); sb.table("county_lookup").update({"run_ml": True, "ml_status": "idle", "ml_max_tiles": $(MAX_TILES), "last_pipeline_error": None}).in_("county_fips", ["48085", "48113", "48121", "48439"]).execute(); print("Queued 4 DFW counties (max_tiles=$(MAX_TILES))")'

run-parallel:
	./.venv/bin/python3 -m src.orchestrator --executor docker-local --max-workers $(MAX_WORKERS) --nas-root $(NAS_ROOT)
