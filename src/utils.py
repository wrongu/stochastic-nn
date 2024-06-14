import torch
import torch.nn.functional as F
from typing import Callable


# Signature: stats_fn(output, targets, reduction) -> {"name": value}
type StatsFn = Callable[[torch.Tensor, torch.Tensor, str], dict[str, float | torch.Tensor]]


def _reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return x.mean()
    elif reduction == "sum":
        return x.sum()
    elif reduction == "none":
        return x
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def cross_entropy_stats(
    logits: torch.Tensor, targets: torch.Tensor, reduction="mean"
) -> dict[str, torch.Tensor]:
    return {"ce": F.cross_entropy(logits, targets, reduction=reduction)}


def accuracy_stats(
    logits: torch.Tensor, targets: torch.Tensor, reduction="mean"
) -> dict[str, torch.Tensor]:
    return {"acc": _reduce((logits.argmax(dim=1) == targets).float(), reduction)}
