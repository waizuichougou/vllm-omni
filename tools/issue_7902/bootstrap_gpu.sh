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
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# The repository setup selects CUDA dependencies after torch is present.
python -m pip install torch==2.13.0
python -m pip install -e "$SRC[dev]"
python -m pip install modelscope

python "$SRC/tools/issue_7902/check_model_artifacts.py" \
  "$ROOT/models/qwen3-omni-int4-autoround" \
  --sha256-manifest "$ROOT/models/qwen3-omni-int4-autoround.SHA256"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() >= 2, torch.cuda.device_count()
print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
PY

echo "Environment ready: conda activate $ENV_NAME"
