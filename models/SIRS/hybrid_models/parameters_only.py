import torch
from typing import Union

# Import local modules
import sys
from os.path import dirname as up
from dantro._import_tools import import_module_from_path
sys.path.append(up(up(up(__file__))))
sys.path.append(up(up(__file__)))
include = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")
SIRS = import_module_from_path(mod_path=up(up(__file__)), mod_str="SIRS")
from include import FeedForwardNN
from SIRS import SIRS_euler

## ---------------------------------------------------------------------------------------------------------------------
## Parameters-only training
## ---------------------------------------------------------------------------------------------------------------------
def get_params_NN(*,
                  input_size: int,
                  num_layers: int = 1,
                  nodes_per_layer: int = 6,
                  activation_func: str = 'softplus',
                  biases: Union[tuple, None] = None,
                  **__
                  ) -> FeedForwardNN:
    """Sets up the parameters-only neural net.

    :param input_size: input size (length of time series)
    :param num_layers: number of layers
    :param nodes_per_layer: nodes per layer
    :param activation_func: activation function to use on each layer
    :param biases: tuple of initial bias values. If None, no biases are used
    :param __: other kwargs (ignored)
    :return: the initialised neural network
    """
    return FeedForwardNN(
        input_size=input_size,  # Input entire time series
        output_size=3,  # Number of parameters
        num_layers=num_layers,
        nodes_per_layer={"default": nodes_per_layer},
        activation_funcs={"default": activation_func},
        learning_rate=0.002,
        optimizer='Adam',
        biases={"default": biases}
    )


def make_params_pred(*,
                     NN: FeedForwardNN,
                     t_span: tuple,
                     dt: Union[float, torch.Tensor],
                     y0: torch.Tensor,
                     X_input: torch.Tensor,
                     **__):
    """ Makes a prediction using the parameters-only NN

    :param NN: the neural network
    :param t_span: tuple of start and endpoints
    :param dt: time differential
    :param y0: initial state
    :param X_input: the input time series
    :param __: other kwargs (ignored)
    :return: predicted time series and predicted parameters
    """
    # Make a prediction from the time series
    pred = NN(X_input)

    # Generate a SIRS time series
    _, Y_pred = SIRS_euler(y0=y0, t_span=t_span, dt=dt, k_I=pred[0], k_R=pred[1], k_S=pred[2])

    return Y_pred, pred
