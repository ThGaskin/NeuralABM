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
## Hybrid model 2
## This model is the second level of hybridisation, where we give a structure for the derivative
## ---------------------------------------------------------------------------------------------------------------------
def get_hybrid_2_NN(*,
                    num_layers: int = 2,
                    nodes_per_layer: int = 6,
                    activation_func: str = 'softplus',
                    biases: Union[None, tuple] = None,
                    z=0) -> FeedForwardNN:
    """ Returns a feed-forward network for the hybrid-2 model.

    :param num_layers: number of layers
    :param nodes_per_layer: nodes per layer
    :param activation_func: activation function to use on each layer
    :param biases: initialisation range for the bias. If None, no bias is used
    :param z: latent dimension
    :return: feed-forward network
    """
    return FeedForwardNN(
        input_size=3 + z,  # Input current state
        output_size=3,  # Output derivatives
        num_layers=num_layers,
        nodes_per_layer={"default": nodes_per_layer},
        activation_funcs={"default": activation_func},
        biases={"default": biases}
    )


def SIRS_hybrid_2_rhs(t: float,
                      y: torch.Tensor,
                      k_ISI: torch.Tensor,
                      k_SR: torch.Tensor,
                      k_RI: torch.Tensor) -> torch.Tensor:
    """ RHS of the hybrid ODE.

    :param t: current time (not used, required for compatibility with torchdiffeq interface)
    :param y: current state (not used, required for compatibility with torchdiffeq interface)
    :param k_ISI: time-dependent component k_I * S * I
    :param k_SR: time-dependent component k_S * R
    :param k_RI: time-dependent component k_R * I
    :return: derivative of y at time t
    """
    dS = -k_ISI + k_SR
    dI = k_ISI - k_RI
    dR = k_RI - k_SR
    return torch.stack((dS, dI, dR), dim=-1)


def SIRS_hybrid_2_Euler(**kwargs):
    """Euler solver for hybrid-2 SIRS model."""
    params = {k: v for k, v in kwargs.items() if k.startswith('k_')}
    return euler_solver_t(
        rhs_func=SIRS_hybrid_2_rhs,
        params=params,
        **{k: v for k, v in kwargs.items() if not k.startswith('k_')}
    )


def make_hybrid_2_pred(*,
                       NN: FeedForwardNN,
                       y0: torch.Tensor,
                       t_span: tuple,
                       dt: Union[float, torch.Tensor],
                       recursive: bool = False,
                       Y_input: Union[None, torch.Tensor] = None,
                       z: Union[None, torch.Tensor] = None,
                       **__) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Generates a SIRS timeseries prediction using the hybrid-2 model. If specified, the prediction is generated recursively,
    using the estimated state as input to the neural network. Alternatively, the observation data can be passed as input.


    :param NN: neural network to use. The neural network takes the current state as input and outputs three time-dependent
        components
    :param y0: initial state
    :param t_span: tuple of the time range
    :param dt: time differential
    :param recursive: whether to generate predictions recursively, i.e. by inserting the estimated state as input to the
        neural network, or whether to use the observations as input
    :param Y_input: time series of observations to use as input, if specified.
    :param z: additional identifying data, used to generalise the time-dependent component across datasets
    :param __: other kwargs (ignored)
    :return: tuple of predicted time series and predicted parameters
    """

    # Non-recursive case: we make a prediction using the observed data as input to the neural network
    if not recursive:
        pred = NN(Y_input)
        Y_pred = SIRS_hybrid_2_Euler(y0=y0, t_span=t_span, dt=dt,
                                     k_ISI=pred[:, 0], k_SR=pred[:, 1], k_RI=pred[:, 2])[1]
        return Y_pred[:-1], pred

    # Recursive case: states are recursively inserted as input
    else:
        t = build_time_grid(None, t_span, dt)
        Y_pred = [y0]
        if z is None:
            pred = [NN(y0).flatten()]
        else:
            pred = [NN(torch.cat([y0, z], dim=0).flatten())]

        # Recursively integrate the equation
        for t_idx, ti in enumerate(t):
            Y_pred.append(Y_pred[t_idx] + SIRS_hybrid_2_rhs(
                t=ti, y=Y_pred[-1], k_ISI=pred[-1][0], k_SR=pred[-1][1], k_RI=pred[-1][2]
            ) * dt)
            if z is None:
                pred.append(NN(Y_pred[-1]).flatten())
            else:
                pred.append(NN(torch.cat([Y_pred[-1], z], dim=0).flatten()))

        return torch.stack(Y_pred)[:-1], torch.stack(pred)[:-1]
