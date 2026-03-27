import torch
import torch.nn as nn

from mmengine.registry import MODELS


def _segment_offsets(sorted_values: torch.Tensor, device: torch.device) -> torch.Tensor:
    zero_tensor = torch.tensor([0], device=device)
    if sorted_values.numel() <= 1:
        return zero_tensor
    seg_flag_tensor = torch.nonzero(torch.diff(sorted_values)).flatten() + 1
    return torch.cat((zero_tensor, seg_flag_tensor))


@MODELS.register_module()
class SegmentMeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat: torch.Tensor, id_list: torch.Tensor) -> torch.Tensor:
        bag_sorted_list = torch.sort(id_list)
        offsets = _segment_offsets(bag_sorted_list.values, feat.device)
        return nn.functional.embedding_bag(
            input=bag_sorted_list.indices, weight=feat, offsets=offsets, mode="mean"
        )


@MODELS.register_module()
class SegmentMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat: torch.Tensor, id_list: torch.Tensor) -> torch.Tensor:
        bag_sorted_list = torch.sort(id_list)
        offsets = _segment_offsets(bag_sorted_list.values, feat.device)
        return nn.functional.embedding_bag(
            input=bag_sorted_list.indices, weight=feat, offsets=offsets, mode="max"
        )


@MODELS.register_module()
class PointColorPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, feat: torch.Tensor, rgbs: torch.Tensor, mode: str = "mean"
    ) -> torch.Tensor:
        _, inverse_indices = torch.unique(
            rgbs, dim=0, sorted=False, return_inverse=True
        )
        bag_sorted_list = torch.sort(inverse_indices)
        offsets = _segment_offsets(bag_sorted_list.values, feat.device)
        bag_feat = nn.functional.embedding_bag(
            input=bag_sorted_list.indices,
            weight=feat,
            offsets=offsets,
            mode=mode,
        )
        return bag_feat[inverse_indices]


@MODELS.register_module()
class MixedPooling(nn.Module):
    def __init__(self, compose):
        super().__init__()
        self.pool = nn.ModuleList([MODELS.build(component) for component in compose])

    def forward(self, feat: torch.Tensor, id_list: torch.Tensor) -> torch.Tensor:
        pooled_feats = [pool(feat, id_list) for pool in self.pool]
        return sum(pooled_feats)
