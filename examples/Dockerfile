ARG BASE_IMAGE=lmsysorg/sglang:dev-kimi-k31-pro
FROM ${BASE_IMAGE}

# Build the wheel first; no SGLang source or checkpoint file is overwritten.
RUN rm -rf /tmp/kimi-k3-video
COPY dist/sglang_kimi_k3_video-*.whl /tmp/kimi-k3-video/
RUN python3 -m pip install --no-deps --force-reinstall /tmp/kimi-k3-video/*.whl

# Run doctor with --check-hooks and --model-path on the target deployment before
# starting the original server command. Decoder versions follow this base image.
