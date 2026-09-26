#!/usr/bin/env bash
set -euo pipefail

# Run this on the prepared remote VM after its GPUs are attached.
ROOT=${ROOT:-/root/autodl-tmp/vllm-omni-needs}
ENV_NAME=${ENV_NAME:-vllmomni}
SRC="$ROOT/src/vllm-omni"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is missing" >&2; exit 2; }
nvidia-smi -L
GPU_COUNT=$(nvidia-smi -L | wc -l)
(( GPU_COUNT >= 2 )) || { echo "need at least 2 visible GPUs, got $GPU_COUNT" >&2; exit 2; }

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
python -m pip check

python "$SRC/tools/issue_7902/check_model_artifacts.py" \
  "$ROOT/models/qwen3-omni-int4-autoround"
python - <<'PY'
import torch
import vllm
import vllm_omni
assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() >= 2, torch.cuda.device_count()
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert vllm.__version__ == "0.30.0", vllm.__version__
print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), vllm.__version__, vllm_omni.__version__)
PY

echo "Environment ready: conda activate $ENV_NAME"
