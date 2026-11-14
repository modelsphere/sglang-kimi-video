# sglang-kimi-video

[![version](https://img.shields.io/badge/version-0.0.2-blue)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

Video input for **Kimi-K3.1-Pro** on SGLang: Base64 videos, multiple videos,
mixed text/image/video inputs, streaming responses, and video token usage.
The plugin reuses the server's vision model and GPUs without modifying SGLang or model files.

Requires Python 3.11+ and SGLang revision `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae`.
The base image is **`lmsysorg/sglang:dev-kimi-k31-pro`**; it must provide
Torch/CUDA, torchcodec or decord, and FFmpeg.

## Get started

1. **Build** the wheel and plugin image from the repository root:

   ```sh
   python3 -m pip install build
   python3 -m build --wheel
   docker build -f examples/Dockerfile \
     -t kimi-k31-video:dev-kimi-k31-pro-plugin-0.0.2 .
   ```

2. **Launch** with local target and draft model directories:

   ```sh
   export K3_MODEL_PATH=/absolute/path/to/Kimi-K3.1-Pro
   export K3_DRAFT_PATH=/absolute/path/to/Kimi-K3.1-Dflash2-Preview
   export K3_PORT=31031
   bash examples/run_integration.sh
   ```

## Video sampling

Set overrides with `export NAME=value` before launch. Unset variables use the model defaults
below; model files remain unchanged.

| parameter | model default | environment variable | accepted values |
| --- | --- | --- | --- |
| `sample_fps` | `8` | `KIMI_K3_VIDEO_SAMPLE_FPS` | finite positive number |
| `max_num_frames_each_video` | `null` (no cap) | `KIMI_K3_VIDEO_MAX_FRAMES` | positive integer; capped by any model frame limit |
| `temporal_merge_kernel_size` | `4` | `KIMI_K3_VIDEO_TEMPORAL_MERGE_KERNEL_SIZE` | integer 1–4, within the vision model's temporal capacity |
| `in_patch_limit_each_frame` | `16384` | `KIMI_K3_VIDEO_IN_PATCH_LIMIT_EACH_FRAME` | positive integer |
| `in_patch_limit_video` | `655360` | `KIMI_K3_VIDEO_IN_PATCH_LIMIT_VIDEO` | positive integer, or `none` to disable the shared budget |
| `patch_limit_on_one_side` | `512` | `KIMI_K3_VIDEO_PATCH_LIMIT_ON_ONE_SIDE` | positive integer |
| `timestamp_mode` | `hh:mm:ss.fff` | `KIMI_K3_VIDEO_TIMESTAMP_MODE` | `hh:mm:ss.fff`, `mm:ss.fff`, or `mm:ss` |

The launch script forwards these variables to Docker. For Helm, add them to `env` in
[deployment values](examples/deployment.values.yaml). Restart the service (recreate Docker containers)
after changes; unset a variable to restore its model default. Per-request overrides are unsupported.

Frames are sampled uniformly using Python's ties-to-even `round(source_frames × min(target_fps, source_fps) / source_fps)`,
clamped to 1–source frame count before applying frame limits. Low-FPS sources are not padded with duplicate frames;
shorter final chunks are preserved. Patch budgets must fit the spatial merge grid.

GPU preprocessing batches at most 16 frames and 16 megapixels of source pixels
per call (a single larger frame is processed alone). This bounds intermediate
frame batches before resizing without changing sampling or temporal chunks.
Failed preprocessing releases its Future reference so intermediate tensors do
not wait for cyclic garbage collection.

## Deployment

Set these environment variables **before running the launch script**:

| variable | default | purpose |
| --- | --- | --- |
| `K3_MODEL_PATH` | required; no default | host directory containing the target model |
| `K3_DRAFT_PATH` | required; no default | host directory containing the draft model |
| `K3_IMAGE` | `kimi-k31-video:dev-kimi-k31-pro-plugin-0.0.2` | plugin image to run |
| `K3_CONTAINER` | `kimi-k31-video-e2e` | container name |
| `K3_PORT` | `31023` | server port; the launch example overrides it to `31031` |
| `K3_RUN_DIR` | `$PWD/k31-run` | directory for caches and run files |

Match the client `--base-url` to the server address and port. The server listens on `0.0.0.0` by default; restrict network access or configure authentication before exposing the port.

SGLang server flags are configurable in the [launch script](examples/run_integration.sh)'s
`set --` block, or by passing a complete replacement argument list to the script.
These are **reference test settings**, not plugin requirements or SGLang defaults.
For Helm, edit `extraArgs` and `env` in the [deployment values](examples/deployment.values.yaml).
To change the base image, pass `--build-arg BASE_IMAGE=<image>` to `docker build`;
its default is `lmsysorg/sglang:dev-kimi-k31-pro`. The image must satisfy the SGLang revision requirement above.

**Already running SGLang?** Install the wheel with `pip install --no-deps`, add
`kimi_k3_video` to `SGLANG_PLUGINS`, and set
`SGLANG_HTTP_SERVER_APP_PATH=kimi_k3_video.http_app:app` for multiple tokenizer workers.
Run `kimi-k3-video-doctor --check-hooks --model-path /path/to/model` before restarting.

Build from the repository root, then copy the wheel into the SGLang container:

```sh
python3 -m build --wheel

# Inside the SGLang container:
python3 -m pip install --no-deps \
  /tmp/sglang_kimi_k3_video-0.0.2-py3-none-any.whl

export SGLANG_PLUGINS="${SGLANG_PLUGINS:+${SGLANG_PLUGINS},}kimi_k3_video"
export SGLANG_HTTP_SERVER_APP_PATH=kimi_k3_video.http_app:app

# Replace /model with the model path inside the container.
kimi-k3-video-doctor --check-hooks --model-path /model

# Restart SGLang with its original command in this shell.
```

### Deployment example

After building the image in Get started, replace the model paths and launch:

```sh
export K3_MODEL_PATH=/absolute/path/to/Kimi-K3.1-Pro
export K3_DRAFT_PATH=/absolute/path/to/Kimi-K3.1-Dflash2-Preview
export K3_PORT=31031
export KIMI_K3_VIDEO_SAMPLE_FPS=4
export KIMI_K3_VIDEO_MAX_FRAMES=16
bash examples/run_integration.sh
```

This example samples at 4 fps with a 16-frame cap per video.

## Request

The client encodes local media as Base64 for `/v1/chat/completions`.
Video inputs must use `data:video/<format>;base64,<payload>`; combine `--video` and
`--image` arguments in the desired request order.

Pass these options to `python3 examples/chat_video.py`:

| option | client default | how to change |
| --- | --- | --- |
| `--base-url` | `http://localhost:30000` | set it to your server address and port |
| `--model` | `kimi-k3.1-pro` | set it to the server's served model name |
| `--prompt` | built-in media-description prompt | provide your own prompt text |
| `--max-tokens` | `256` | set the maximum number of generated tokens |
| `--n` | `1` | set the number of completions |
| `--stream` | disabled (`false`) | add the flag to enable streaming |
| `--video` / `--image` | none; at least one required | repeat these options to add media in order |

The client does not expose `temperature`, `top_p`, or `reasoning_effort` flags.
To change them, set the corresponding fields in your own Chat Completions request JSON.
When omitted, SGLang/model defaults apply; the plugin does not override them.

### Request example

Once the server is ready, replace the video path and send a streaming request:

```sh
python3 examples/chat_video.py \
  --base-url http://127.0.0.1:31031 \
  --model kimi-k3.1-pro \
  --video /absolute/path/to/video.mp4 \
  --prompt "Describe the actions in this video." \
  --max-tokens 1024 --stream
```
