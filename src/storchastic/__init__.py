from storchastic.injections import stochastify


class StochasticMeta(type):
    """Metaclass for stochastic modules.

    Example: Say you want a class that acts like nn.Linear but where weights and biases are
    stochastic and the parameters of the layer are actually the mean and standard deviation of
    the weights and biases. You can create a class like this:

        class StochasticLinear(nn.Linear, metaclass=StochasticMeta):
            pass
    """

    def __new__(cls, name, bases, attrs):
        # Inject stochastic methods
        return stochastify(super().__new__(cls, name, bases, attrs))


__all__ = [
    "StochasticMeta",
    "stochastify",
]
