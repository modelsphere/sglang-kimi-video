"""Check the pinned source interfaces without relying on nightly version labels."""

from __future__ import annotations

import hashlib
import inspect
import json
import pkgutil
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

SGLANG_REVISION = "afcd7d6b487a9cbe73e25941971b4cc107e6f9ae"


def source_root():
    spec = find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate the installed SGLang package")
    return Path(spec.origin).parent


def check_sources():
    """Validate hook dependencies even on hosts without a CUDA device.

    Hash only the source files the plugin depends on. The pinned image can use
    an SCM/dev package version and need not retain its Git checkout metadata.
    """
    manifest = json.loads(Path(__file__).with_name("runtime_sources.json").read_text())
    root = source_root()
    mismatches = []
    for name, expected in manifest.items():
        path = root / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(
            f"kimi_k3_video requires SGLang source interfaces from {SGLANG_REVISION}; "
            f"incompatible files: {', '.join(mismatches)}"
        )
    return {"revision": SGLANG_REVISION, "checked_files": len(manifest)}


def check_version():
    try:
        installed = version("sglang")
    except PackageNotFoundError:
        raise RuntimeError(
            f"Install SGLang built from {SGLANG_REVISION} before loading kimi_k3_video"
        ) from None
    check_sources()
    return installed


def check_targets(*, applied=False):
    from .plugin import HOOKS

    for target, hook, parameters in HOOKS:
        fn = pkgutil.resolve_name(target)
        actual = inspect.signature(fn).parameters
        if any(name not in actual for name in parameters):
            raise RuntimeError(f"Incompatible hook signature: {target}")
        if applied and not _contains_hook(fn, hook):
            raise RuntimeError(f"Video hook is not active: {target}")
    from sglang.srt.managers.schedule_batch import MultimodalDataItem

    if not hasattr(MultimodalDataItem, "set_identity"):
        raise RuntimeError("SGLang does not expose the required wide identity API")


def _contains_hook(fn, hook):
    seen = set()
    while callable(fn) and id(fn) not in seen:
        seen.add(id(fn))
        if inspect.isfunction(fn):
            closure = inspect.getclosurevars(fn)
            if closure.nonlocals.get("hook") is hook:
                return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def check_decoder():
    failures = []
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        return "torchcodec"
    except (ImportError, RuntimeError, OSError) as error:
        failures.append(type(error).__name__)
    try:
        from decord import VideoReader  # noqa: F401

        return "decord"
    except (ImportError, RuntimeError, OSError) as error:
        failures.append(type(error).__name__)
    raise RuntimeError(
        f"No working torchcodec/decord backend ({', '.join(failures)}); check FFmpeg"
    )
