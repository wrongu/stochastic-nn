import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def estimate_entropy_diagonal_gaussian(x: torch.Tensor) -> torch.Tensor:
    """Given n x d matrix of n samples from a d-dimensional distribution, estimate the entropy of
    the distribution by fitting a diagonal Gaussian to the data and computing the entropy of the
    Gaussian. In general this is an upper bound to the true entropy of the distribution.
    """
    d = x.size(1)
    var = torch.var(x, dim=0, unbiased=True)
    entropy = 0.5 * torch.sum(torch.log(2 * np.pi * var)) + 0.5 * d * (1 + np.log(2 * np.pi))
    return entropy


def log_det_fisher_categorical(logits: torch.Tensor) -> torch.Tensor:
    """Given batch of logits, produce a batch of log(det(FIM)) where the FIM is the Fisher
    Information Matrix for a categorical distribution parameterized by the logits.

    We allow the logits to be unnormalized, so the distribution is technically overpartameterized,
    but the result here is equivalent to (1) projecting to the d-1 dimensional parameterization and
    (2) computing the FIM for the categorical distribution in that parameterization.
    """
    return torch.sum(torch.log_softmax(logits, dim=1), dim=1)


class CategoricalKLQP(nn.Module):
    def __init__(self, num_classes: int, label_smoothing: float = 0.1, lam: float = 1.0):
        super(CategoricalKLQP, self).__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.lam = lam

    def forward(self, logits, labels):
        log_p_true = np.log(1 - self.label_smoothing)
        log_p_false = np.log(self.label_smoothing / (self.num_classes - 1))
        log_p = F.one_hot(labels, self.num_classes).float() * (log_p_true - log_p_false)
        log_q = F.log_softmax(logits, dim=1)
        kl_q_p = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)
        if self.lam > 0:
            batch_entropy_logits = estimate_entropy_diagonal_gaussian(logits)
            logdet_fisher_q = torch.mean(log_det_fisher_categorical(logits))
            mutual_information_term = batch_entropy_logits - 1 / 2 * logdet_fisher_q
        else:
            mutual_information_term = 0
        return kl_q_p - self.lam * mutual_information_term
