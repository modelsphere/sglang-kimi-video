"""Minimal multi-frame position embedding correction from SGLang #35325."""


def position_embeddings(original, self, grid_thws, *, grid_thw_list=None):
    import torch
    from sglang.srt.models.kimi_k3_vl import _resolve_grid_thw_list, interpolate_pos_emb

    shapes = _resolve_grid_thw_list(grid_thws, grid_thw_list)
    if all(t == 1 for t, _, _ in shapes):
        return original(self, grid_thws, grid_thw_list=grid_thw_list)
    outputs = []
    for t, h, w in shapes:
        if not 1 <= t <= self.num_frames:
            raise ValueError(
                f"Kimi-K3 temporal grid {t} exceeds position encoding capacity"
            )
        spatial = (
            self.weight.flatten(end_dim=1)
            if (h, w) == self.weight.shape[:-1]
            else interpolate_pos_emb(self.weight, self.interpolation_mode, (h, w))
        )
        position = (
            spatial
            if t == 1
            else (
                spatial.unsqueeze(0).repeat(t, 1, 1)
                + self.time_weight[:t].to(device=spatial.device, dtype=spatial.dtype)
            )
        )
        outputs.append(position.reshape(-1, position.shape[-1]))
    return torch.cat(outputs)
