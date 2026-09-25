#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
cd "$ROOT"

pytest -q \
  tests/core/test_prefix_cache.py \
  tests/core/test_prefix_cache_runner_mixin.py \
  -k 'same_step or read_plan or prefetch or producer or publish_failure or missing_read_plan'
