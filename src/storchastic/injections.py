import torch
from torch import nn
from torch.nn import functional as F


def stochastify(cls):
    """Class decorator to inject all the necessary methods for a stochastic module.

    Example: Say you want a class that acts like nn.Linear but where weights and biases are
    stochastic and the parameters of the layer are actually the mean and standard deviation of
    the weights and biases. You can create a class like this:

        @stochastify
        class StochasticLinear(nn.Linear):
            pass

    which is equivalent to:

        class StochasticLinear(nn.Linear):
            pass
        StochasticLinear = stochastify(StochasticLinear)

    but NOT equivalent to:

        StochasticLinear = stochastify(nn.Linear)  # DO NOT DO THIS

    because stochastify modifies the given class in place.

    See also the StochasticMeta metaclass in storchastic/__init__.py.
    """
    inject_init(cls)
    inject_sample_parameters(cls)
    inject_forward(cls)
    return cls


def inject_init(cls):
    """Helper function to inject a new __init__ method for a Module."""

    old_init = cls.__init__

    def _replace_parameters(module):
        to_add = {}
        to_remove = []
        for name, param in module.named_parameters():
            # Skip nested parameters - we'll handle them with recursion to child modules
            if "."  in name:
                continue

            mean = nn.Parameter(param.data)
            inv_softplus_std = nn.Parameter(torch.zeros_like(param.data) - 4.0)
            to_add[f"{name}_mean"] = mean
            to_add[f"{name}_inv_softplus_var"] = inv_softplus_std
            to_remove.append(name)

        for name in to_remove:
            delattr(module, name)

        for name, param in to_add.items():
            module.register_parameter(name, param)

        # Recurse on child modules
        for name, child in module._modules.items():
            _replace_parameters(child)

    def _init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        _replace_parameters(self)

    setattr(cls, "__init__", _init)


def inject_sample_parameters(cls):
    """Helper function to inject a new sample_parameters method for a Module.
    """

    def _sample_parameters(self):
        for name, _ in self.named_parameters():
            # Skip nested parameters - we'll handle them with recursion to child modules
            if "." in name:
                continue

            if name.endswith("_mean"):
                param_name = name[:-5]
                mean = getattr(self, f"{param_name}_mean")
                inv_softplus_std = getattr(self, f"{param_name}_inv_softplus_var")
                std = F.softplus(inv_softplus_std)
                sampled_param = mean + std * torch.randn_like(std)
                setattr(self, param_name, sampled_param)

        for name, child in self._modules.items():
            _sample_parameters(child)

    setattr(cls, "sample_parameters", _sample_parameters)


def inject_forward(cls):
    """Helper function to inject a new forward method for a Module.
    """

    old_forward = cls.forward

    def _forward(self, *args, **kwargs):
        self.sample_parameters()
        return old_forward(self, *args, **kwargs)

    setattr(cls, "forward", _forward)
