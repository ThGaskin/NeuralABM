from typing import Union

import torch

""" General utility functions """


def random_tensor(
    cfg: Union[dict, list], *, size: tuple = None, device: str = "cpu", **__
) -> torch.Tensor:
    """Generates a multi-dimensional random tensor. Each entry can be initialised separately, or a common
    initialisation configuration is used for each entry. For instance, the configuration

    .. code-block::

        cfg:
            distribution: uniform
            parameters:
                lower: 0
                upper: 1

    together with `size: (2, 2)` will initialise a 2x2 matrix with entries drawn from a uniform distribution on
    [0, 1]. The configuration

    .. code-block::

        cfg:
            - distribution: uniform
              parameters:
                 lower: 0
                 upper: 1
            - distribution: normal
              parameters:
                mean: 0.5
                std: 0.1

    will initialise a (2, 1) tensor with entries drawn from different distributions.

    :param cfg: the configuration entry containing the initialisation data
    :param size (optional): the size of the tensor, in case the configuration is not a list
    :param device: the device onto which to load the data
    :param __: additional kwargs (ignored)
    :return: the tensor of random variables
    """

    def _random_tensor_1d(
        *, distribution: str, parameters: dict, s: tuple = (1,), **__
    ) -> torch.Tensor:

        """Generates a random tensor according to a distribution.

        :param distribution: the type of distribution. Can be 'uniform' or 'normal'.
        :param parameters: the parameters relevant to the respective distribution
        :param s: the size of the random tensor
        """

        # Uniform distribution in an interval
        if distribution == "uniform":

            l, u = parameters.get("lower"), parameters.get("upper")
            if l > u:
                raise ValueError(
                    f"Upper bound must be greater or equal to lower bound; got {l} and {u}!"
                )

            return torch.tensor((u - l), dtype=torch.float) * torch.rand(
                s, dtype=torch.float, device=device
            ) + torch.tensor(l, dtype=torch.float)

        # Normal distribution
        elif distribution == "normal":
            return torch.normal(
                parameters.get("mean"),
                parameters.get("std"),
                size=s,
                device=device,
                dtype=torch.float,
            )

        else:
            raise ValueError(f"Unrecognised distribution type {distribution}!")

    if isinstance(cfg, list):
        return torch.tensor([_random_tensor_1d(**entry) for entry in cfg]).to(device)
    else:
        return _random_tensor_1d(**cfg, s=size).to(device)
