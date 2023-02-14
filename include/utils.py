import torch

""" General utility functions """


def random_tensor(
        *, distribution: str, parameters: dict, size: tuple, device: str, **__
) -> torch.Tensor:

    """ Generates a random tensor according to a distribution.

    :param distribution: the type of distribution. Can be 'uniform' or 'normal'.
    :param parameters: the parameters relevant to the respective distribution
    :param size: the size of the random tensor
    :param device: the device onto which to load the data
    :param __: additional kwargs (ignored)
    :return: the tensor of random variables
    """

    # Uniform distribution in an interval
    if distribution == "uniform":
        return (parameters.get("upper") - parameters.get("lower")) * torch.rand(
            size, dtype=torch.float, device=device
        ) + parameters.get("lower")

    # Normal distribution
    elif distribution == "normal":
        return torch.normal(
            parameters.get("mean"), parameters.get("std"), size=size, device=device
        )

    else:
        raise ValueError(f"  Unrecognised distribution type {distribution}!")
