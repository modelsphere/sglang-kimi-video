"""Expand structural media tokens without re-tokenizing the surrounding text."""

from __future__ import annotations

from itertools import groupby


def encode_special(tokenizer, text):
    try:
        return list(tokenizer.encode(text, allowed_special="all"))
    except TypeError:
        return list(tokenizer.encode(text))


def expand_prompt(input_ids, token_id, items, configs, tokenizer):
    # Adapted from SGLang #35325, 1ee7a2ae; offsets recorded during expansion.
    if hasattr(input_ids, "flatten"):
        input_ids = input_ids.flatten().tolist()
    if len(items) != len(configs):
        raise ValueError("Each Kimi-K3 visual item needs one resize config")
    groups = [
        list(group)
        for _, group in groupby(
            zip(items, configs), key=lambda pair: pair[0].source_index
        )
    ]
    if [group[0][0].source_index for group in groups] != list(range(len(groups))):
        raise ValueError("Kimi-K3 visual items are not in source media order")
    if sum(value == token_id for value in input_ids) != len(groups):
        raise ValueError(
            "Kimi-K3 structural placeholder count does not match source media"
        )
    output, offsets = [], []
    groups = iter(groups)
    for value in input_ids:
        if value != token_id:
            output.append(int(value))
            continue
        group = next(groups)
        if group[0][0].media_type == "image" and len(group) != 1:
            raise ValueError("One image source must produce exactly one grid")
        for item, config in group:
            prefix = (
                f"image {item.width}x{item.height}"
                if item.media_type == "image"
                else "video"
            )
            output.extend(
                encode_special(
                    tokenizer,
                    f"{item.timestamp}<|media_begin|>{prefix}<|media_content|>",
                )
            )
            count = config["num_tokens"]
            if type(count) is not int or count <= 0:
                raise ValueError("Kimi-K3 visual token count must be positive")
            start = len(output)
            output.extend([token_id] * count)
            offsets.append((start, len(output) - 1))
            output.extend(encode_special(tokenizer, "<|media_end|>"))
    return output, offsets
