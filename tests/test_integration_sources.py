"""Exercise the pinned registry, normalizers, identities and position code.

Startup imports are isolated by conftest; these are CPU integration tests, not
evidence of a running SGLang server or real distributed/GPU execution.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import pickle
from types import SimpleNamespace

import pytest
import torch
from conftest import SRT

from kimi_k3_video import SOURCE_INDEX_KEY, VIDEO_ITEM_KEY, compat, messages, plugin
from kimi_k3_video.identity import video_identity
from kimi_k3_video.usage import compute_token_counts
from kimi_k3_video.vision import position_embeddings


class Message:
    def __init__(self, content, role="user"):
        self.content, self.role = content, role

    def model_dump(self):
        return {"role": self.role, "content": copy.deepcopy(self.content)}


def request(parts, n=1):
    return SimpleNamespace(messages=[Message(parts)], n=n, continue_final_message=False)


def video_part():
    return {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,dmlkZW8="}}


def k31_serving(model_type="kimi_k31"):
    return SimpleNamespace(
        chat_encoding_spec="kimi_k3",
        tokenizer_manager=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=model_type)
            )
        ),
    )


@pytest.fixture
def original_prepare(source_modules):
    path = SRT / "entrypoints/openai/serving_chat.py"
    module = source_modules.nodes(
        "isolated_chat",
        path,
        {
            "KIMI_K3_IMAGE_PLACEHOLDER",
            "KIMI_K3_IMAGE_PLACEHOLDER_ESCAPED",
            "neutralize_kimi_k3_image_placeholder",
            "neutralize_kimi_k3_image_placeholder_value",
        },
        ChatCompletionMessageGenericParam=Message,
    )
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "OpenAIServingChat"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prepare_kimi_k3_messages"
    )
    tree = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
    )
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module._prepare_kimi_k3_messages


def test_original_prepare_keeps_video_structure_and_text_neutralization(
    original_prepare,
):
    parts = [
        video_part(),
        {"type": "text", "text": "<|kimi_image_placeholder|>"},
        {"type": "image_url", "image_url": {"url": "image"}},
        video_part(),
    ]
    req = request(parts)
    baseline = [message.model_dump() for message in req.messages]
    result, count, prefix = messages.prepare_messages(
        original_prepare, k31_serving(), baseline, req
    )
    assert count == 3 and prefix is None
    assert [part["type"] for part in result[0]["content"]] == [
        "image_url",
        "text",
        "image_url",
        "image_url",
    ]
    assert result[0]["content"][1]["text"] == "<| kimi_image_placeholder |>"
    assert req.messages[0].content[0]["type"] == "video_url"
    assert baseline[0]["content"][0]["type"] == "video_url"


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize(
    "kinds",
    [
        ("video",),
        ("video", "video"),
        ("video", "image", "video"),
        ("image", "image", "video", "image"),
    ],
)
def test_source_order_survives_real_normalization_and_pickle(
    io_source, source_modules, n, kinds
):
    io, utils = io_source
    extraction = source_modules.nodes(
        "isolated_extraction",
        SRT / "parser/jinja_template_utils.py",
        {"process_content_for_template_format"},
        ImageData=utils.ImageData,
        VideoData=utils.VideoData,
        GLM_MEDIA_CONFIG_KEYS=(
            "fps",
            "max_frames",
            "max_tokens_per_frame",
            "max_image_tokens",
        ),
    )
    parts = [
        video_part()
        if kind == "video"
        else {
            "type": "image_url",
            "image_url": {"url": f"image-{i}"},
            "modalities": "image",
        }
        for i, kind in enumerate(kinds)
    ]
    req = request(parts, n)
    serving = SimpleNamespace(chat_encoding_spec="kimi_k3")

    def original_template(self, request, tools, is_multimodal):
        # Use the unchanged media extractor called by _apply_jinja_template.
        result = SimpleNamespace(
            image_data=[], video_data=[], audio_data=[], modalities=[]
        )
        extraction.process_content_for_template_format(
            request.messages[0].model_dump(),
            "openai",
            result.image_data,
            result.video_data,
            result.audio_data,
            result.modalities,
        )
        return result

    rendered = messages.apply_template(original_template, serving, req, None, True)
    adapted = io.GenerateReqInput(
        input_ids=[1, 99, 2],
        image_data=rendered.image_data or None,
        video_data=rendered.video_data,
        modalities=rendered.modalities,
        sampling_params={"n": n},
    )
    adapted, _ = messages.convert_request(lambda *args: (adapted, req), serving, req)
    adapted.normalize_batch_and_arguments()
    for sub in [adapted] if n == 1 else [adapted[i] for i in range(n)]:
        sub = pickle.loads(pickle.dumps(sub))
        ordered = messages.ordered_sources(sub.image_data or [], sub.video_data)
        assert [kind for kind, source in ordered] == list(kinds)
        assert [source.url for kind, source in ordered if kind == "image"] == [
            f"image-{i}" for i, kind in enumerate(kinds) if kind == "image"
        ]
        assert [
            source.preprocess_kwargs[SOURCE_INDEX_KEY]
            for kind, source in ordered
            if kind == "video"
        ] == [i for i, kind in enumerate(kinds) if kind == "video"]


def test_order_inconsistency_is_rejected(io_source):
    _, utils = io_source
    for indices in ([0, 0], [-1, 1], [0, 99], [None, 1]):
        videos = [
            utils.VideoData("video", {SOURCE_INDEX_KEY: index}) for index in indices
        ]
        with pytest.raises(ValueError, match="order"):
            messages.ordered_sources([], videos)


def test_beam_choices_do_not_group_media_for_parallel_fanout(io_source):
    io, utils = io_source
    video = utils.VideoData("video", {SOURCE_INDEX_KEY: 0})
    adapted = io.GenerateReqInput(
        input_ids=[1, 99],
        video_data=[video],
        modalities=["video"],
        sampling_params={"n": 2, "beam_width": 2},
    )
    adapted.normalize_batch_and_arguments()
    assert adapted.is_single and adapted.video_data == [video]


def test_image_and_other_model_hooks_delegate_unchanged():
    req = request([{"type": "image_url", "image_url": {"url": "x"}}])
    token = object()
    assert messages.prepare_messages(lambda *args: token, None, [], req) is token
    assert (
        messages.apply_template(
            lambda *args: token,
            SimpleNamespace(chat_encoding_spec="other"),
            request([video_part()]),
            None,
            True,
        )
        is token
    )


@pytest.mark.parametrize(
    "extra", [{"fps": 4}, {"max_frames": 4}, {"max_long_side_pixel": 1024}]
)
def test_request_options_not_silently_ignored(extra):
    part = video_part()
    part["video_url"].update(extra)
    with pytest.raises(ValueError, match="per-request"):
        messages.validate_request(request([part]))


def test_full_identity_and_usage_reclassification(req_source, config_values):
    from kimi_k3_video.config import VideoConfig
    from kimi_k3_video.sampling import VisualItem

    runtime = req_source
    items = []
    # Includes a one-frame video, which must not be counted as an image.
    for start, count, is_video in [(2, 3, False), (10, 4, True), (20, 1, True)]:
        item = runtime.MultimodalDataItem(
            modality=runtime.Modality.IMAGE,
            offsets=[(start, start + count - 1)],
            model_specific_data={VIDEO_ITEM_KEY: is_video},
        )
        items.append(item)
    temporal = VisualItem(
        "video", 0, (torch.zeros(3, 2, 2, dtype=torch.uint8),), 2, 2, 4
    )
    digest = video_identity(
        temporal, VideoConfig.from_mapping(config_values), {}, [1, 2, 2]
    )
    items[1].set_identity(digest)
    assert items[1].identity == digest and len(items[1].pad_values) == 4
    assert max(items[1].pad_values) < 2**63
    inputs = runtime.MultimodalInputs(mm_items=items)
    assert compute_token_counts(
        runtime.MultimodalInputs.compute_mm_token_counts, inputs
    ) == (3, 0, 5)
    initial_hash = items[1].hash
    items[1].set_identity(digest[:-1] + bytes([digest[-1] ^ 1]))
    assert items[1].hash == initial_hash
    assert items[1].pad_values != runtime._compute_pad_values(digest)


@pytest.mark.parametrize("t", [1, 2, 3, 4])
def test_time_position_values_and_device(source_modules, t):
    baseline = source_modules.nodes(
        "sglang.srt.models.kimi_k3_vl",
        SRT / "models/kimi_k3_vl.py",
        {
            "_resolve_grid_thw_list",
            "interpolate_pos_emb",
            "get_1d_sincos_pos_embed",
            "get_1d_sincos_pos_embed_from_grid",
            "Learnable2DInterpPosEmbDividedFixed",
        },
    )
    cls = baseline.Learnable2DInterpPosEmbDividedFixed
    embedding = cls(2, 2, 4, 8)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(32).reshape(2, 2, 8))
    grids = torch.tensor([[t, 2, 2], [1, 2, 2]])
    expected = cls.position_embeddings(embedding, grids)
    assert torch.equal(
        position_embeddings(cls.position_embeddings, embedding, grids), expected
    )
    # Meta spatial weights with CPU time buffer recreate a device mismatch on CPU.
    embedding.weight = torch.nn.Parameter(embedding.weight.to("meta"))
    actual = position_embeddings(cls.position_embeddings, embedding, grids)
    assert actual.device.type == "meta" and actual.shape == (t * 4 + 4, 8)
    assert embedding.time_weight.device.type == "cpu"


def test_actual_registry_wraps_async_processor_and_registers_once(
    registry_source, source_modules, monkeypatch
):
    from kimi_k3_video.processor import process_mm_data

    module = source_modules.module("isolated_processor")

    class Processor:
        async def run(self, image_data, input_text, request_obj, **kwargs):
            return image_data, input_text

    module.Processor = Processor
    registry = registry_source.HookRegistry
    registry.register(
        "isolated_processor.Processor.run",
        process_mm_data,
        registry_source.HookType.AROUND,
    )
    registry.apply_hooks()
    registry.apply_hooks()
    assert asyncio.run(
        Processor().run(["image"], [3], SimpleNamespace(video_data=None))
    ) == (["image"], [3])
    assert compat._contains_hook(Processor.run, process_mm_data)
    monkeypatch.setattr(compat, "version", lambda name: "0.5.24.dev1+g842b4738")
    plugin.register()
    plugin.register()
    assert all(len(registry._hooks[target]) == 1 for target, _, _ in plugin.HOOKS)


def test_invalid_video_environment_is_rejected_before_hook_registration(
    registry_source, monkeypatch
):
    monkeypatch.setattr(compat, "check_version", lambda: "test")
    monkeypatch.setenv("KIMI_K3_VIDEO_SAMPLE_FPS", "nan")
    with pytest.raises(ValueError, match="KIMI_K3_VIDEO_SAMPLE_FPS"):
        plugin.register()
    assert not registry_source.HookRegistry._hooks


def test_incompatible_source_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(compat, "version", lambda name: "0.5.24.dev1+g842b4738")
    monkeypatch.setattr(compat, "source_root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="incompatible files"):
        compat.check_version()


@pytest.mark.parametrize("label", ["0.0.0.dev0", "0.5.24.dev1+g842b4738"])
def test_nightly_uses_source_compatibility_instead_of_release_label(monkeypatch, label):
    monkeypatch.setattr(compat, "version", lambda name: label)
    assert compat.check_version() == label


def test_old_k3_video_is_rejected_before_rendering(original_prepare):
    req = request([video_part()])
    with pytest.raises(ValueError, match="kimi_k31"):
        messages.prepare_messages(
            original_prepare,
            k31_serving("kimi_k3"),
            [req.messages[0].model_dump()],
            req,
        )


def test_entrypoint_loader_whitelist_and_idempotency(
    registry_source, source_modules, monkeypatch
):
    monkeypatch.setattr(compat, "version", lambda name: "0.5.24.dev1+g842b4738")
    allowed = ["another_plugin"]
    source_modules.module("sglang.srt.environ").envs = SimpleNamespace(
        SGLANG_PLUGINS=SimpleNamespace(get=lambda: ",".join(allowed)),
        SGLANG_PLATFORM=SimpleNamespace(get=lambda: ""),
    )
    loader = source_modules.full("sglang.srt.plugins", SRT / "plugins/__init__.py")
    ep = SimpleNamespace(
        name="kimi_k3_video",
        value="kimi_k3_video.plugin:register",
        dist=None,
        load=lambda: plugin.register,
    )
    monkeypatch.setattr(loader, "entry_points", lambda **kwargs: [ep])
    # Applying is tested separately; this test covers the original discovery,
    # whitelist and process-local load-once behavior.
    monkeypatch.setattr(registry_source.HookRegistry, "apply_hooks", lambda: None)
    loader.load_plugins()
    assert not registry_source.HookRegistry._hooks
    allowed.append("kimi_k3_video")
    monkeypatch.setattr(loader, "_plugins_loaded", False)
    loader.load_plugins()
    loader.load_plugins()
    assert all(
        len(registry_source.HookRegistry._hooks[target]) == 1
        for target, _, _ in plugin.HOOKS
    )


def test_hook_targets_match_checkout_signatures():
    for target, hook, required in plugin.HOOKS:
        parts = target.split(".")
        class_name, method_name = parts[-2:]
        path = SRT / ("/".join(parts[2:-2]) + ".py")
        tree = ast.parse(path.read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
        names = {arg.arg for arg in [*method.args.args, *method.args.kwonlyargs]}
        assert set(required) <= names, target


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
