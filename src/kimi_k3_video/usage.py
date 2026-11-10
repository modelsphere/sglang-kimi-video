"""Preserve IMAGE execution while reporting temporal items as video tokens."""

from . import VIDEO_ITEM_KEY


def compute_token_counts(original, self):
    image, audio, video = original(self)
    temporal = sum(
        end - start + 1
        for item in self.mm_items
        if item.is_image() and item.model_specific_data.get(VIDEO_ITEM_KEY) is True
        for start, end in (item.offsets or [])
    )
    return image - temporal, audio, video + temporal
