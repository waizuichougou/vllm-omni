# Issue 7902 Item 2

This directory contains the prepared deployment and measurement assets for
the prefix-cache read-plan acceptance run.

## Prepared model

The intended checkpoint is
`Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound`. It quantizes the thinker
language-model stage; talker and Code2Wav remain normal runtime stages.

The model directory is expected at:

```text
/root/autodl-tmp/vllm-omni-needs/models/qwen3-omni-int4-autoround
```

## GPU setup

The final run requires two visible GPUs on one host. The prepared remote
instance currently has no `/dev/nvidia*`, so `bootstrap_gpu.sh` intentionally
fails until a GPU-backed instance is attached.

After GPUs are attached, check the existing environment without reinstalling it:

```bash
cd /root/autodl-tmp/vllm-omni-needs/src/vllm-omni
bash tools/issue_7902/bootstrap_gpu.sh
conda activate vllmomni
```

## Start the server

```bash
MODEL=/root/autodl-tmp/vllm-omni-needs/models/qwen3-omni-int4-autoround
vllm serve "$MODEL" --omni --port 8091 \
  --deploy-config vllm_omni/deploy/qwen3_omni_moe_autoround_item2.yaml \
  --async-chunk --log-stats
```

For the cache-off control, restart the server with
`vllm_omni/deploy/qwen3_omni_moe_autoround_item2_cache_off.yaml` instead.
Both profiles keep talker and Code2Wav caching disabled.

The HTTP client labels runs for the experiment log. The label does not switch
prefetch: the current server deploy schema does not expose `prefetch_reads`.
Prefetch on/off parity is covered only by direct manager tests; a server-level
performance comparison requires a real runtime switch. Do not report the
HTTP label as a server-side A/B test. Keep separate cache-on/cache-off runs,
starting each run on a cold server, with exact commands and source SHA.

## Acceptance client

```bash
python tools/issue_7902/measure_acceptance.py \
  --base-url http://127.0.0.1:8091 \
  --model "$MODEL" \
  --prefetch-label enabled \
  --output /root/autodl-tmp/vllm-omni-needs/results/item2-prefetch-on.json
```

The client covers two sequential identical requests, one `n=2` request, and
eight identical concurrent requests. It records status, wall time, response
size, and response hashes and exits nonzero on failed requests. It does not
compare response text or measure Seed-TTS WER. For issue acceptance, run a
cache-off control on the same quantized checkpoint and collect text and audio
outputs with a separate WER evaluation. A BF16 baseline is needed before
claiming quality equivalence to the official checkpoint.

## No-GPU checks

These checks are safe before renting GPUs:

```bash
python tools/issue_7902/check_model_artifacts.py /path/to/model
python -m py_compile tools/issue_7902/*.py
pytest -q tests/core/test_prefix_cache.py tests/core/test_prefix_cache_runner_mixin.py
```
