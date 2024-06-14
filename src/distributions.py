import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import abc


def estimate_entropy_diagonal_gaussian(x: torch.Tensor) -> torch.Tensor:
    """Given n x d matrix of n samples from a d-dimensional distribution, estimate the entropy of
    the distribution by fitting a diagonal Gaussian to the data and computing the entropy of the
    Gaussian. In general this is an upper bound to the true entropy of the distribution.
    """
    d = x.size(1)
    var = torch.var(x, dim=0, unbiased=True)
    entropy = 0.5 * torch.sum(torch.log(2 * np.pi * var)) + 0.5 * d * (1 + np.log(2 * np.pi))
    return entropy


def _reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return x.mean()
    elif reduction == "sum":
        return x.sum()
    elif reduction == "none":
        return x
    else:
        raise ValueError(f"Invalid reduction '{reduction}'.")


class DistributionLayer(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, alpha: float = 1.0):
        super(DistributionLayer, self).__init__()
        if alpha < 0.0:
            raise ValueError("Alpha must be non-negative.")
        if alpha > 1.0:
            raise ValueError("Alpha must be less than or equal to 1.0.")
        self.alpha = alpha

    @abc.abstractmethod
    def num_params(self) -> int:
        """Return dimensionality of the parameter vector 'theta' that this distribution uses."""

    @abc.abstractmethod
    def project(self, theta: torch.Tensor) -> torch.Tensor:
        """Project the parameter vector 'theta' into the parameter space of the distribution.

        If the distribution is overparameterized, this function should project the parameter
        vector into the constrained space of the distribution. If the distribution is not
        overparameterized, this function should return the parameter vector as is.
        """

    @abc.abstractmethod
    def klqp(self, theta: torch.Tensor, targets: torch.Tensor, reduction="mean") -> torch.Tensor:
        """Compute the KL divergence between the distribution parameterized by theta and the target
        distribution. Subclasses may define the target distribution in different ways.
        """

    def sample(self, theta: torch.Tensor, n: int) -> torch.Tensor:
        """Sample from the distribution parameterized by theta."""
        with torch.no_grad():
            return self.rsample(theta, n)

    def rsample(self, theta: torch.Tensor, n: int) -> torch.Tensor:
        """Sample from the distribution parameterized by theta using reparameterization trick."""
        raise NotImplementedError("Reparameterization trick not implemented for this distribution.")

    @abc.abstractmethod
    def log_det_fisher(self, theta: torch.Tensor, reduction="mean") -> torch.Tensor:
        """Compute the log determinant of the Fisher Information Matrix for the distribution
        parameterized by theta.
        """

    def mixture_kl_loss(
        self,
        theta: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Loss function estimating KL(m || p) where m is a stochastic mixture of components q_θ(z).

        Uses the decomposition KL(m || p) = E_θ[KL(q_θ || p)] - MI where θ are the parameters of
        the mixture components. MI is E_θ[KL(q_θ || m)] (mutual information between θ and z),
        and we use the lower bound MI ≥ H[θ] + 0.5 E_θ[log(det(F(θ)))] as an estimate of the
        mutual information.
        """
        b = theta.size(0)
        klqp = self.klqp(theta, targets, reduction=reduction)

        if self.alpha > 0 or return_components:
            # Use variation across the batch in the constrained parameter space as an estimate of
            # the entropy of the distribution over parameters.
            # TODO - this is a poor estimate of entropy for multiple reasons. Look into ways to
            #  improve this estimate. Ideally we'd estimate entropy per item in the batch. Any
            #  way to do that without sampling >1 times per input?
            entropy_theta = estimate_entropy_diagonal_gaussian(self.project(theta))
            if reduction == "none":
                # This (poor) estimate of entropy is naturally mean-reduced across the batch. If
                # user requests "none", we need to expand it back out to the full batch size and
                # divide it equally among each item in the batch.
                entropy_theta = entropy_theta.unsqueeze(0).expand_as(klqp) / b
            elif reduction == "sum":
                # If user requests "sum", we need to multiply the entropy by the batch size to
                # compensate for the mean reduction that was applied to the entropy.
                entropy_theta = entropy_theta * b
            log_det_fisher = self.log_det_fisher(theta, reduction=reduction)
            mutual_information_lower_bound = entropy_theta + 0.5 * log_det_fisher
        else:
            mutual_information_lower_bound = torch.zeros_like(klqp)
            entropy_theta = torch.ones_like(klqp) * float("nan")
            log_det_fisher = torch.ones_like(klqp) * float("nan")

        if return_components:
            return {
                "klqp": klqp,
                "entropy_theta": entropy_theta,
                "log_det_fisher": log_det_fisher,
                "loss": klqp - self.alpha * mutual_information_lower_bound,
            }
        else:
            return klqp - self.alpha * mutual_information_lower_bound

    # Alias for folks who prefer to think in terms of ELBOs rather than KLs
    mixture_elbo_loss = mixture_kl_loss


class Categorical(DistributionLayer):
    def __init__(self, num_classes: int, label_smoothing: float = 0.1, *args, **kwargs):
        super(Categorical, self).__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def num_params(self) -> int:
        return self.num_classes

    def project(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(theta, dim=-1)

    def klqp(self, theta: torch.Tensor, targets: torch.Tensor, reduction="mean") -> torch.Tensor:
        """Compute the KL divergence between the distribution parameterized by theta and the targets
        given as class labels.
        """
        log_p_true = np.log(1 - self.label_smoothing)
        log_p_false = np.log(self.label_smoothing) - np.log(self.num_classes - 1)
        log_p = (
            F.one_hot(targets, self.num_classes).float() * (log_p_true - log_p_false) + log_p_false
        )
        log_q = F.log_softmax(theta, dim=-1)
        # Do our own reduction because F.kl_div naturally reduces across the entire batch x
        # classes tensor of q*(log_q - log_p), but we want to reduce only across the batch. They
        # provide a "batchmean" reduction for this but no "batchnone" reduction.
        return _reduce(
            torch.sum(F.kl_div(log_q, log_p, reduction="none", log_target=True), dim=-1), reduction
        )

    def sample(self, theta: torch.Tensor, n: int) -> torch.Tensor:
        return torch.multinomial(torch.softmax(theta, dim=-1), n, replacement=True)

    def log_det_fisher(self, theta: torch.Tensor, reduction="mean") -> torch.Tensor:
        return _reduce(torch.sum(torch.log_softmax(theta, dim=-1), dim=-1), reduction)
