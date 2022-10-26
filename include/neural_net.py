from typing import Any, List, Union

import torch
from torch import nn

# Pairs of activation functions and whether they are part of the torch.nn module, in which case they must be called
# via func(*args, **kwargs)(x).

ACTIVATION_FUNCS = {
    "abs": [torch.abs, False],
    "celu": [torch.nn.CELU, True],
    "cos": [torch.cos, False],
    "cosine": [torch.cos, False],
    "elu": [torch.nn.ELU, True],
    "gelu": [torch.nn.GELU, True],
    "hardshrink": [torch.nn.Hardshrink, True],
    "hardsigmoid": [torch.nn.Hardsigmoid, True],
    "hardswish": [torch.nn.Hardswish, True],
    "hardtanh": [torch.nn.Hardtanh, True],
    "leakyrelu": [torch.nn.LeakyReLU, True],
    "linear": [None, False],
    "logsigmoid": [torch.nn.LogSigmoid, True],
    "mish": [torch.nn.Mish, True],
    "none": [None, False],
    "prelu": [torch.nn.PReLU, True],
    "relu": [torch.nn.ReLU, True],
    "rrelu": [torch.nn.RReLU, True],
    "selu": [torch.nn.SELU, True],
    "sigmoid": [torch.nn.Sigmoid, True],
    "silu": [torch.nn.SiLU, True],
    "sin": [torch.sin, False],
    "sine": [torch.sin, False],
    "softplus": [torch.nn.Softplus, True],
    "softshrink": [torch.nn.Softshrink, True],
    "swish": [torch.nn.SiLU, True],
    "tanh": [torch.nn.Tanh, True],
    "tanhshrink": [torch.nn.Tanhshrink, True],
    "threshold": [torch.nn.Threshold, True]
}

# ----------------------------------------------------------------------------------------------------------------------
# -- NN utility function -----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_activation_funcs(n_layers: int, cfg: Union[str, dict] = None) -> List[Any]:
    """Extracts the activation functions from the config. The config is a dictionary, with the keys representing
    the layer number, and the entry the activation function to use. Alternatively, the config can also be a single
    string, which is then applied to the entire neural net.

    Example:
        activation_funcs: abs    # applies the absolute value to the entire neural net
    Example:
        activation_funcs:        # applies the nn.Hardtanh activation function to the entire neural net
          name: HardTanh
          args:
            - -2
            - 2
    Example:
        activation_funcs:
          0: abs
          1: relu
          2: tanh
    """

    funcs = [None] * (n_layers + 1)

    if cfg is None:
        return funcs

    elif isinstance(cfg, str):
        _f = ACTIVATION_FUNCS[cfg.lower()]
        if _f[1]:
            return [_f[0]()] * (n_layers + 1)
        else:
            return [_f[0]] * (n_layers + 1)

    elif isinstance(cfg, dict):
        if 'name' in cfg.keys():
            _f = ACTIVATION_FUNCS[cfg.get('name').lower()]
            if _f[1]:
                return [_f[0](
                    *cfg.get('args', ()), **cfg.get('kwargs', {})
                )] * (n_layers + 1)
            else:
                return [_f[0]] * (n_layers + 1)
        else:
            for idx, entry in cfg.items():

                if isinstance(entry, str):
                    _f = ACTIVATION_FUNCS[entry.lower()]
                    if _f[1]:
                        funcs[idx] = _f[0]()
                    else:
                        funcs[idx] = _f[0]
                elif isinstance(entry, dict):
                    funcs[idx] = ACTIVATION_FUNCS[entry.get('name').lower()][0](
                        *entry.get('args', ()), **entry.get('kwargs', {})
                    )

                else:
                    raise ValueError(f"Unrecognised argument {entry} in 'activation_funcs' dictionary!")
            return funcs
    else:
        raise ValueError(f"Unrecognised argument {cfg} for 'activation_funcs'!")


# -----------------------------------------------------------------------------
# -- Neural net class ---------------------------------------------------------
# -----------------------------------------------------------------------------


class NeuralNet(nn.Module):
    OPTIMIZERS = {
        "Adagrad": torch.optim.Adagrad,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SparseAdam": torch.optim.SparseAdam,
        "Adamax": torch.optim.Adamax,
        "ASGD": torch.optim.ASGD,
        "LBFGS": torch.optim.LBFGS,
        "NAdam": torch.optim.NAdam,
        "RAdam": torch.optim.RAdam,
        "RMSprop": torch.optim.RMSprop,
        "Rprop": torch.optim.Rprop,
        "SGD": torch.optim.SGD,
    }

    def __init__(
            self,
            *,
            input_size: int,
            output_size: int,
            num_layers: int,
            nodes_per_layer: int,
            activation_funcs: dict = None,
            optimizer: str = "Adam",
            learning_rate: float = 0.001,
            bias: bool = False,
            init_bias: tuple = None,
            **__,
    ):
        """

        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_layers: the number of hidden layers
        :param nodes_per_layer: the number of neurons in the hidden layers
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param learning_rate: the learning rate of the optimizer
        :param bias: whether to initialise the layers with a bias
        :param init_bias: the interval from which to uniformly initialise the bias
        :param __: Additional model parameters (ignored)
        """

        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()

        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = num_layers
        self.activation_funcs = get_activation_funcs(num_layers, activation_funcs)
        architecture = [input_size] + [nodes_per_layer] * num_layers + [output_size]

        # Add the neural net layers
        self.layers = nn.ModuleList()
        for i in range(len(architecture) - 1):
            layer = nn.Linear(architecture[i], architecture[i + 1], bias=bias)

            # Initialise the biases of the layers with a uniform distribution on init_bias
            if bias and init_bias is not None:
                torch.nn.init.uniform_(layer.bias, init_bias[0], init_bias[1])
            self.layers.append(layer)

        # Get the optimizer
        self.optimizer = self.OPTIMIZERS[optimizer](self.parameters(), lr=learning_rate)

    # ... Evaluation functions .........................................................................................

    # The model forward pass
    def forward(self, x):
        for i in range(len(self.layers)):
            if self.activation_funcs[i] is None:
                x = self.layers[i](x)
            else:
                x = self.activation_funcs[i](self.layers[i](x))
        return x
