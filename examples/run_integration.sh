#!/usr/bin/env bash
# Launch only when GPUs and the K3.1 target/draft weights have been provided.
set -euo pipefail

: "${K3_MODEL_PATH:?Set the host path to Kimi-K3.1-Pro}"
: "${K3_DRAFT_PATH:?Set the host path to Kimi-K3.1-Dflash2-Preview}"
K3_RUN_DIR="${K3_RUN_DIR:-$PWD/k31-run}"
K3_IMAGE="${K3_IMAGE:-kimi-k31-video:dev-kimi-k31-pro-plugin-0.0.2}"
K3_CONTAINER="${K3_CONTAINER:-kimi-k31-video-e2e}"
K3_PORT="${K3_PORT:-31023}"
test -f "$K3_MODEL_PATH/config.json"
test -f "$K3_DRAFT_PATH/config.json"
mkdir -p "$K3_RUN_DIR/cache" "$K3_RUN_DIR/results" "$K3_RUN_DIR/fixtures"
K3_RUN_DIR=$(cd "$K3_RUN_DIR" && pwd)

# Passing arguments replaces this reference set. SGLang owns all parameter
# validation; the plugin does not change TP/DCP/EP, memory, speculation or cache.
if (( $# == 0 )); then
  set -- \
    --trust-remote-code --tp-size=8 --dcp-size=8 --dcp-comm-backend=fi_a2a \
    --ep-size=2 --mem-fraction-static=0.85 --max-running-requests=50 \
    --max-mamba-cache-size=270 --weight-loader-prefetch-checkpoints \
    '--model-loader-extra-config={"enable_multithread_load": true}' \
    --mamba-radix-cache-strategy=extra_buffer \
    --reasoning-parser=kimi_k3 --tool-call-parser=kimi_k3 \
    --speculative-algorithm=DFLASH --speculative-draft-model-path=/draft-model \
    --speculative-dflash-block-size=5 --speculative-draft-kv-cache-dtype=fp8_e4m3 \
    --speculative-draft-attention-backend=cutedsl_mla \
    --enable-linear-replayssm-spec --tokenizer-worker-num=4
fi

docker run --init --detach \
  --name "$K3_CONTAINER" \
  --gpus all --network host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add SYS_NICE \
  --mount "type=bind,src=$K3_MODEL_PATH,dst=/model,readonly" \
  --mount "type=bind,src=$K3_DRAFT_PATH,dst=/draft-model,readonly" \
  --mount "type=bind,src=$K3_RUN_DIR/cache,dst=/root/.cache" \
  --mount "type=bind,src=$K3_RUN_DIR,dst=/integration" \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env SGLANG_DCP_PUSH_COMBINE_MAX_ROWS=1536 \
  --env SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB=32768 \
  --env "SGLANG_PLUGINS=${SGLANG_PLUGINS:+${SGLANG_PLUGINS},}kimi_k3_video" \
  --env SGLANG_HTTP_SERVER_APP_PATH=kimi_k3_video.http_app:app \
  --env KIMI_K3_VIDEO_MAX_FRAMES \
  --env KIMI_K3_VIDEO_SAMPLE_FPS \
  --env KIMI_K3_VIDEO_TEMPORAL_MERGE_KERNEL_SIZE \
  --env KIMI_K3_VIDEO_IN_PATCH_LIMIT_EACH_FRAME \
  --env KIMI_K3_VIDEO_IN_PATCH_LIMIT_VIDEO \
  --env KIMI_K3_VIDEO_PATCH_LIMIT_ON_ONE_SIDE \
  --env KIMI_K3_VIDEO_TIMESTAMP_MODE \
  --entrypoint /bin/bash \
  "$K3_IMAGE" -c '
    set -euo pipefail
    kimi-k3-video-doctor --check-hooks --model-path /model
    port=$1
    shift
    exec python3 -m sglang.launch_server \
      --model-path /model --served-model-name kimi-k3.1-pro \
      --host 127.0.0.1 --port "$port" "$@"
  ' integration "$K3_PORT" "$@"
