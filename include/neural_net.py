import torch
from torch import nn
from typing import Any, List, Union

# -----------------------------------------------------------------------------
# -- NN utility functions -----------------------------------------------------
# -----------------------------------------------------------------------------


def get_activation_funcs(n_layers: int, cfg: Union[str, dict] = None) -> List[Any]:

    """Extracts the activation functions from the config"""

    def return_function(name: str):
        if name in ['Linear', 'linear', 'lin', 'None']:
            return None
        elif name in ['sigmoid', 'Sigmoid']:
            return torch.sigmoid
        elif name in ['relu', 'ReLU']:
            return torch.relu
        elif name in ['sin', 'sine']:
            return torch.sin
        elif name in ['cos', 'cosine']:
            return torch.cos
        elif name in ['tanh']:
            return torch.tanh
        elif name in ['abs']:
            return torch.abs
        else:
            raise ValueError(f"Unrecognised activation function {name}!")

    funcs = [None] * (n_layers+1)

    if cfg is None:
        return funcs
    elif isinstance(cfg, str):
        return [return_function(cfg)] * (n_layers + 1)
    elif isinstance(cfg, dict):
        for val in cfg.keys():
            if val in [0]:
                funcs[0] = return_function(cfg[0])
            elif val in [-1]:
                funcs[-1] = return_function(cfg[-1])
            else:
                funcs[val-1] = return_function(cfg[val])

        return funcs
    else:
        raise ValueError(f"Unrecognised argument {cfg} for 'activation_funcs'!")


def get_optimizer(name):

    """Returns the optimizer from the config"""

    if name == 'Adagrad':
        return torch.optim.Adagrad
    elif name == 'Adam':
        return torch.optim.Adam
    elif name == 'AdamW':
        return torch.optim.AdamW
    elif name == 'SparseAdam':
        return torch.optim.SparseAdam
    elif name == 'Adamax':
        return torch.optim.Adamax
    elif name == 'ASGD':
        return torch.optim.ASGD
    elif name == 'LBFGS':
        return torch.optim.LBFGS
    elif name == 'NAdam':
        return torch.optim.NAdam
    elif name == 'RAdam':
        return torch.optim.RAdam
    elif name == 'RMSprop':
        return torch.optim.RMSprop
    elif name == 'Rprop':
        return torch.optim.Rprop
    elif name == 'SGD':
        return torch.optim.SGD
    else:
        raise ValueError(f'Unrecognized opimiser {name}!')

# -----------------------------------------------------------------------------
# -- Neural net class ---------------------------------------------------------
# -----------------------------------------------------------------------------


class NeuralNet(nn.Module):

    def __init__(self,
                 *,
                 input_size: int,
                 output_size: int,
                 num_layers: int,
                 nodes_per_layer: int,
                 activation_funcs: dict = None,
                 optimizer: str = 'Adam',
                 learning_rate: float = 0.001,
                 bias: bool = False,
                 init_bias: tuple = None,
                 **__):
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
        self.optimizer = get_optimizer(optimizer)(self.parameters(), lr=learning_rate)

        # Initialize the loss tracker dictionary, which can be used to evaluate the training progress
        self._loss_tracker: dict = {'iteration': [], 'training_loss': [], 'parameter_loss': []}

    @property
    def loss_tracker(self) -> dict:
        return self._loss_tracker

    # Updates the loss tracker with a time value and the loss values
    def update_loss_tracker(self, it, *, training_loss, parameter_loss):
        self._loss_tracker['iteration'].append(it)
        self._loss_tracker['training_loss'].append(training_loss)
        self._loss_tracker['parameter_loss'].append(parameter_loss)

    def reset_loss_tracker(self):
        self._loss_tracker = {'iteration': [], 'training_loss': [], 'parameter_loss': []}

    # ... Evaluation functions .........................................................................................

    # The model forward pass
    def forward(self, x):
        for i in range(len(self.layers)):
            if self.activation_funcs[i] is None:
                x = self.layers[i](x)
            else:
                x = self.activation_funcs[i](self.layers[i](x))
        return x
