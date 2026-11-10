"""Environment overrides must be validated and leave model configuration intact."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kimi_k3_video.config import VIDEO_ENV_VARS, VideoConfig, video_config_overrides
from kimi_k3_video.sampling import sampling_indices


@pytest.fixture(autouse=True)
def clean_video_environment(monkeypatch):
    for name in VIDEO_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_unset_overrides_preserve_model_defaults(config_values):
    assert VideoConfig.from_environment(config_values) == VideoConfig.from_mapping(
        config_values
    )
    assert video_config_overrides() == {}


def test_environment_overrides_do_not_mutate_model(config_values, monkeypatch):
    original = copy.deepcopy(config_values)
    values = {
        "KIMI_K3_VIDEO_SAMPLE_FPS": "2.5",
        "KIMI_K3_VIDEO_MAX_FRAMES": "16",
        "KIMI_K3_VIDEO_TEMPORAL_MERGE_KERNEL_SIZE": "2",
        "KIMI_K3_VIDEO_IN_PATCH_LIMIT_EACH_FRAME": "4096",
        "KIMI_K3_VIDEO_IN_PATCH_LIMIT_VIDEO": "32768",
        "KIMI_K3_VIDEO_PATCH_LIMIT_ON_ONE_SIDE": "128",
        "KIMI_K3_VIDEO_TIMESTAMP_MODE": "mm:ss",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    config = VideoConfig.from_environment(config_values)
    assert config.sample_fps == 2.5
    assert config.frame_limit == 16
    assert config.temporal_merge == 2
    assert config.frame_patch_limit == 4096
    assert config.video_patch_limit == 32768
    assert config.side_patch_limit == 128
    assert config.timestamp_mode == "mm:ss"
    assert config.per_frame_budget(16) == 2048
    assert config_values == original
    for name in values:
        monkeypatch.delenv(name)
    assert VideoConfig.from_environment(config_values) == VideoConfig.from_mapping(
        original
    )


@pytest.mark.parametrize("model_limit,expected", [(None, 16), (8, 8), (32, 16)])
def test_environment_frame_limit_preserves_model_cap(
    config_values, model_limit, expected
):
    config_values["max_num_frames_each_video"] = model_limit
    config = VideoConfig.from_environment(
        config_values, {"KIMI_K3_VIDEO_MAX_FRAMES": "16"}
    )
    assert config.frame_limit == expected


def test_environment_changes_sampling_and_chunk_configuration(config_values):
    config = VideoConfig.from_environment(
        config_values,
        {
            "KIMI_K3_VIDEO_SAMPLE_FPS": "4",
            "KIMI_K3_VIDEO_TEMPORAL_MERGE_KERNEL_SIZE": "2",
        },
    )
    indices = sampling_indices(81, 16, config)
    assert len(indices) == 20
    assert indices[0] == 0 and indices[-1] == 80
    assert config.temporal_merge == 2
    with pytest.raises(ValueError, match="capacity"):
        config.validate_vision(
            SimpleNamespace(
                patch_size=14, merge_kernel_size=(2, 2), init_pos_emb_time=1
            )
        )


def test_environment_can_disable_shared_patch_budget(config_values):
    config = VideoConfig.from_environment(
        config_values, {"KIMI_K3_VIDEO_IN_PATCH_LIMIT_VIDEO": "none"}
    )
    assert config.video_patch_limit is None
    assert config.per_frame_budget(1000) == 16384


@pytest.mark.parametrize(
    "name,value",
    [
        ("KIMI_K3_VIDEO_SAMPLE_FPS", value)
        for value in ("", "nan", "inf", "-inf", "0", "-1", "fast")
    ]
    + [
        (name, value)
        for name in (
            "KIMI_K3_VIDEO_MAX_FRAMES",
            "KIMI_K3_VIDEO_TEMPORAL_MERGE_KERNEL_SIZE",
            "KIMI_K3_VIDEO_IN_PATCH_LIMIT_EACH_FRAME",
            "KIMI_K3_VIDEO_IN_PATCH_LIMIT_VIDEO",
            "KIMI_K3_VIDEO_PATCH_LIMIT_ON_ONE_SIDE",
        )
        for value in ("", "0", "-1", "1.5")
    ]
    + [
        ("KIMI_K3_VIDEO_TEMPORAL_MERGE_KERNEL_SIZE", "5"),
        ("KIMI_K3_VIDEO_TIMESTAMP_MODE", "seconds"),
        ("KIMI_K3_VIDEO_TIMESTAMP_MODE", ""),
    ],
)
def test_invalid_overrides_report_environment_variable(config_values, name, value):
    with pytest.raises(ValueError, match=name):
        VideoConfig.from_environment(config_values, {name: value})


def test_overridden_settings_change_video_cache_identity(config_values):
    import torch

    from kimi_k3_video.identity import video_identity
    from kimi_k3_video.sampling import VisualItem

    original = VideoConfig.from_environment(config_values, {})
    override = VideoConfig.from_environment(
        config_values, {"KIMI_K3_VIDEO_SAMPLE_FPS": "4"}
    )
    item = VisualItem("video", 0, (torch.zeros(3, 4, 4, dtype=torch.uint8),), 4, 4, 16)
    assert video_identity(item, original, {}, [1, 2, 2]) != video_identity(
        item, override, {}, [1, 2, 2]
    )


@pytest.fixture
def isolated_doctor(monkeypatch):
    from kimi_k3_video import doctor

    monkeypatch.setattr(doctor, "check_version", lambda: "test")
    monkeypatch.setattr(doctor, "check_sources", lambda: {"checked_files": 19})
    monkeypatch.setattr(doctor, "check_decoder", lambda: "test")
    return doctor


def test_doctor_reports_effective_overrides_without_changing_model_files(
    isolated_doctor, monkeypatch, capsys
):
    fixture = Path(__file__).parent / "fixtures/kimi_k31"
    before = {path: path.read_bytes() for path in fixture.glob("*.json")}
    monkeypatch.setenv("KIMI_K3_VIDEO_SAMPLE_FPS", "4")
    monkeypatch.setenv("KIMI_K3_VIDEO_MAX_FRAMES", "16")
    monkeypatch.setenv("KIMI_K3_VIDEO_IN_PATCH_LIMIT_VIDEO", "none")
    assert isolated_doctor.main(["--model-path", str(fixture)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["video_overrides"]["sample_fps"] == 4
    assert result["video_config"]["sample_fps"] == 4
    assert result["video_config"]["frame_limit"] == 16
    assert result["video_config"]["video_patch_limit"] is None
    assert before == {path: path.read_bytes() for path in fixture.glob("*.json")}


def test_doctor_rejects_invalid_overrides_without_model_path(
    isolated_doctor, monkeypatch, capsys
):
    monkeypatch.setenv("KIMI_K3_VIDEO_SAMPLE_FPS", "nan")
    assert isolated_doctor.main([]) == 1
    assert "KIMI_K3_VIDEO_SAMPLE_FPS" in capsys.readouterr().err
