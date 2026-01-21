import torch
from typing import Union, Tuple

# Import local modules
import sys
from os.path import dirname as up
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(up(__file__))))
sys.path.append(up(up(__file__)))
include = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")
SIRS = import_module_from_path(mod_path=up(up(__file__)), mod_str="SIRS")
from include import FeedForwardNN, build_time_grid
from SIRS import euler_solver_t


## ---------------------------------------------------------------------------------------------------------------------
## Black-box model/Neural ODE
## This model is the fully black-box level of hybridisation (neural ODE)
## ---------------------------------------------------------------------------------------------------------------------
def get_bb_NN(*,
              num_layers: int = 2,
              nodes_per_layer: int = 6,
              activation_func: str = 'tanh',
              biases: Union[None, tuple] = None,
              z=0) -> FeedForwardNN:
    """ Generates a feed-forward neural network for the black-box/neural ODE experiments

    :param num_layers: number of layers
    :param nodes_per_layer: nodes per layer
    :param activation_func: activation function on each layer
    :param biases: initialisation range for the biases. If None, no biases are used
    :param z: latent dimension
    :return: feed-forward network
    """
    return FeedForwardNN(
        input_size=3 + z,  # Input current state + latent state, if given
        output_size=3,  # Output derivatives
        num_layers=num_layers,
        nodes_per_layer={"default": nodes_per_layer},
        activation_funcs={"default": activation_func},
        biases={"default": biases}
    )


def SIRS_rhs_bb(t: float,
                y: torch.Tensor,
                d_s: torch.Tensor,
                d_i: torch.Tensor,
                d_r: torch.Tensor) -> torch.Tensor:
    """Defines the RHS of the black-box SIRS model such that it can be passed to a numerical solver.

    :param t: current time (not used, required for compatibility with torchdiffeq interface)
    :param y: current state (not used, required for compatibility with torchdiffeq interface)
    :param d_s: derivative of the susceptibility state S
    :param d_i: derivative of the infection state I
    :param d_r: derivative of the recovered state R
    :return: derivative dy(t)
    """
    return torch.stack((d_s, d_i, d_r), dim=-1)


def SIRS_bb_Euler(**kwargs):
    """Euler solver for black-box SIRS model."""
    params = {k: v for k, v in kwargs.items() if k.startswith('d_')}
    return euler_solver_t(
        rhs_func=SIRS_rhs_bb,
        params=params,
        **{k: v for k, v in kwargs.items() if not k.startswith('d_')}
    )


def make_bb_pred(*,
                 NN: FeedForwardNN,
                 y0: torch.Tensor,
                 t_span: tuple,
                 dt: Union[float, torch.Tensor],
                 recursive: bool = False,
                 Y_input: Union[None, torch.Tensor] = None,
                 z: Union[None, torch.Tensor] = None,
                 **__) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prediction function for the black-box model

    :param NN: neural network to use
    :param y0: initial value
    :param t_span: tuple of time range
    :param dt: time differential
    :param recursive: whether to generate predictions by recursively inserting the state estimate into the neural network
    :param Y_input: input dataset to use if predictions are not being generated recursively
    :param z: additional identifying data, used to generalise the time-dependent component across datasets
    :param __: other kwargs (ignored)
    :return: tuple of time series and parameter predictions
    """
    # Make a prediction from the time series using the observation data as input
    if not recursive:
        pred = NN(Y_input)
        Y_pred = \
            SIRS_bb_Euler(y0=y0, t_span=t_span, dt=dt, d_s=pred[:, 0], d_i=pred[:, 1], d_r=pred[:, 2])[1]
        return Y_pred[:-1], pred

    # Neural ODE: state estimate is recursively inserted as input
    else:
        t = build_time_grid(None, t_span, dt)
        Y_pred = [y0]
        if z is None:
            pred = [NN(y0).flatten()]
        else:
            pred = [NN(torch.cat([y0, z], dim=0).flatten())]

        for t_idx, ti in enumerate(t):
            Y_pred.append(Y_pred[t_idx] + SIRS_rhs_bb(
                t=ti, y=Y_pred[-1], d_s=pred[-1][0], d_i=pred[-1][1], d_r=pred[-1][2]
            ) * dt)
            if z is None:
                pred.append(NN(Y_pred[-1]).flatten())
            else:
                pred.append(NN(torch.cat([Y_pred[-1], z], dim=0).flatten()))

        return torch.stack(Y_pred)[:-1], torch.stack(pred)[:-1]
