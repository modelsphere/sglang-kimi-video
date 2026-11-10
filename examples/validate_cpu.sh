#!/usr/bin/env bash
# Run inside the pinned image, without --gpus. Does not launch a model server.
set -euo pipefail
K31_SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
K31_RESULT_DIR=${1:?Pass an absolute output directory for logs and wheel artifacts}
mkdir -p "$K31_RESULT_DIR"
K31_RESULT_DIR=$(cd "$K31_RESULT_DIR" && pwd)
K31_BUILD_DIR=$(mktemp -d /tmp/k31-video-cpu.XXXXXX)
cp -R "$K31_SOURCE_DIR/." "$K31_BUILD_DIR/"
cd "$K31_BUILD_DIR"
python3 -m build --no-isolation --wheel --sdist --outdir "$K31_RESULT_DIR" 2>&1 | tee "$K31_RESULT_DIR/build.log"
python3 -m pip install --no-deps --force-reinstall "$K31_RESULT_DIR/sglang_kimi_k3_video-0.0.1-py3-none-any.whl"
# Read CPU integration source bodies from the installed image, not an overlaid
# checkout. Runtime tests run new interpreters and import the installed wheel.
KIMI_K3_VIDEO_TEST_SRT=$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("sglang").origin).parent / "srt")')
export KIMI_K3_VIDEO_TEST_SRT
python3 -m ruff check . 2>&1 | tee "$K31_RESULT_DIR/lint.log"
python3 -m ruff format --check . 2>&1 | tee "$K31_RESULT_DIR/format.log"
{
  for test_file in tests/test_*.py; do
    [[ "$test_file" == tests/test_cuda.py ]] && continue
    python3 -m pytest "$test_file" -q --tb=short
  done
} 2>&1 | tee "$K31_RESULT_DIR/pytest.log"
python3 - <<'PY' | tee "$K31_RESULT_DIR/environment.json"
import importlib.metadata
import json
import platform
import torch
from kimi_k3_video.compat import check_sources, source_root
print(json.dumps({
    'python': platform.python_version(),
    'versions': {name: importlib.metadata.version(name) for name in
        ['sglang', 'sglang-kimi-k3-video', 'torch', 'torchcodec', 'pytest', 'av']},
    'cuda_available': torch.cuda.is_available(),
    'sglang_source': str(source_root()),
    'runtime_sources': check_sources(),
}, indent=2))
PY
