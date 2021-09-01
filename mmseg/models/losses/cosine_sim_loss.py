import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import LOSSES
from .utils import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class CosineSimLoss(nn.Module):
    """CrossEntropyLoss.
    Args:
        dataset (str): Dataset name.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 use_sigmoid=False,
                 loss_weight=1.0):
        super(CosineSimLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        # cat2vec
        self.vec = torch.tensor(np.load('./mmseg/models/losses/ade20k_ori_cat2vec.npy')).float()
        # normalize vec
        self.vec = self.vec / self.vec.norm(dim=1, keepdim=True)

        self.cls_criterion = cross_entropy
        self.logit_scale = nn.Parameter(torch.ones([]))

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # normalize features
        cls_score = cls_score / cls_score.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        cls_score = logit_scale * cls_score.permute(0, 2, 3, 1) @ self.vec.to(device=cls_score.device).t() # [N, H, W, num_cls]
        cls_score = cls_score.permute(0, 3, 1, 2).contiguous() # [N, num_cls, H, W]

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls