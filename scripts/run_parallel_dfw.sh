#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MAX_WORKERS="${MAX_WORKERS:-3}"
NAS_ROOT="${NAS_ROOT:-/Volumes/GrowrouteLLC/growroute-data}"

./.venv/bin/python3 -m src.orchestrator \
  --executor docker-local \
  --max-workers "${MAX_WORKERS}" \
  --nas-root "${NAS_ROOT}" \
  "$@"
