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
## Hybrid model 1
## This model is the first level of hybridisation, where we learn the constant parameters k_R and k_S, as well as the
## time-dependent component k_I * S(t).
## ---------------------------------------------------------------------------------------------------------------------
def get_hybrid_1_NN(*,
                    input_size: int,
                    num_layers_const: int = 1,
                    num_layers_time_dep: int = 2,
                    activation_func: str = 'softplus',
                    biases: Union[None, tuple] = None,
                    nodes_per_layer: int = 6,
                    z=0) -> dict:
    """ Sets up two neural networks for the hybrid-1 model: one learns constant parameters from an entire time series,
    one learns the time-dependent k_I * S part.

    :param input_size: input size of the constant-part neural network (length of the time series)
    :param num_layers_const: number of layers of the constant-part neural network
    :param num_layers_time_dep: number of layers of the time-dependent neural network
    :param activation_func: activation function to use for both
    :param biases: tuple of bias initialisation range. If None, no bias is used
    :param nodes_per_layer: number of nodes to use on each layer
    :param z: latent dimension of the time-dependent component. This is used to pass additional identifiers for each
        dataset, allowing generalisation across datasets.
    :return: dictionary containing the two neural networks and a coupled optimizer
    """
    res = dict(
        # Learns constant parameters
        const_params=FeedForwardNN(
            input_size=input_size,  # Input entire time series
            output_size=2,  # Output parameters
            num_layers=num_layers_const,
            nodes_per_layer={"default": nodes_per_layer},
            activation_funcs={"default": activation_func},
            biases={"default": biases}
        ),
        # Learns time dependent part (k_I * S)
        time_dep_params=FeedForwardNN(
            input_size=3 + z,  # Input current state plus latent variable
            output_size=1,  # Output one derivative
            num_layers=num_layers_time_dep,
            nodes_per_layer={"default": nodes_per_layer},
            activation_funcs={"default": activation_func},
            biases={"default": biases}
        ))

    # Set up a coupled optimizer
    res['optimizer'] = torch.optim.Adam(
        list(res['const_params'].parameters()) + list(res['time_dep_params'].parameters()), lr=1e-3)
    res['optimizer'].zero_grad()

    return res


def SIRS_hybrid_1_rhs(t, y, k_IS: torch.Tensor, k_S: torch.Tensor, k_R: torch.Tensor) -> torch.Tensor:
    """ RHS of the hybrid ODE.

    :param t: current time (not used, required for compatibility with torchdiffeq interface)
    :param y: current state
    :param k_IS: time-dependent component k_I * S
    :param k_S: constant rate k_S
    :param k_R: constant rate k_R
    :return: derivative of y at time t
    """

    # Get the infection and recovery state
    I = y[..., 1]
    R = y[..., 2]

    # Calculate the derivative
    dS = -k_IS * I + k_S * R
    dI = k_IS * I - k_R * I
    dR = k_R * I - k_S * R

    return torch.stack((dS, dI, dR), dim=-1)


def SIRS_hybrid_1_Euler(**kwargs):
    """Euler solver for hybrid-1 SIRS model."""
    params = {k: v for k, v in kwargs.items() if k.startswith('k_')}
    return euler_solver_t(
        rhs_func=SIRS_hybrid_1_rhs,
        params=params,
        **{k: v for k, v in kwargs.items() if not k.startswith('k_')}
    )

def make_hybrid_1_pred(*,
                       NN: dict,
                       y0: torch.Tensor,
                       t_span: tuple,
                       dt: Union[float, torch.Tensor],
                       recursive: bool = False,
                       X_input: torch.Tensor,
                       Y_input: Union[None, torch.Tensor] = None,
                       z: Union[None, torch.Tensor] = None,
                       **__) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Generates a SIRS timeseries prediction using the hybrid-1 model. If specified, the prediction is generated recursively,
    using the estimated state as input to the neural network. Alternatively, the observation data can be passed as input.

    :param NN: dictionary containing the neural networks
    :param y0: initial state of the SIRS model
    :param t_span: tuple of time range
    :param dt: time differential
    :param recursive: whether to generate predictions recursively (neural-ODE type) or using a fixed input series
        of observations
    :param X_input: input time series for the constant-part neural network
    :param Y_input: input time series to use for the time-dependent neural network, if specified.
    :param z: additional identifying data, used to generalise the time-dependent component across datasets
    :param __: other kwargs (ignored)
    :return: tuple of estimated Y time series, constant and time-dependent parameter predictions
    """

    # Make a prediction for the constant parameters
    pred_const = NN['const_params'](X_input)

    # Generate the time series: in the non-recursive case, the observation data is used as input to the neural network.
    # In the recursive case (neural ODE-type), the state prediction itself is used as an input.
    if not recursive:
        pred_time_dep = NN['time_dep_params'](Y_input).flatten()
        _, Y_pred = SIRS_hybrid_1_Euler(
            y0=y0, t_span=t_span, dt=dt, k_IS=pred_time_dep, k_S=pred_const[0], k_R=pred_const[1])
        return Y_pred[:-1], pred_const, pred_time_dep

    # Recursively generated predictions based on neural-estimated state, i.e. the state \hat{x}(t+1) is as input to the neural network, rather
    # than the observation data.
    else:
        t = build_time_grid(None, t_span, dt)
        # List for the time series and the time-dependent parameter predictions
        Y_pred = [y0]
        if z is None:
            pred_time_dep = [NN['time_dep_params'](y0).flatten()]
        else:
            pred_time_dep = [NN['time_dep_params'](torch.cat([y0, z], dim=0).flatten())]

        # Recursively integrate
        for t_idx, ti in enumerate(t):
            Y_pred.append(
                Y_pred[-1] + SIRS_hybrid_1_rhs(
                    t=ti, y=Y_pred[-1], k_S=pred_const[0], k_IS=pred_time_dep[-1][0], k_R=pred_const[1]
                ) * dt
            )
            if z is None:
                pred_time_dep.append(NN['time_dep_params'](Y_pred[-1]).flatten())
            else:
                pred_time_dep.append(NN['time_dep_params'](torch.cat([Y_pred[-1], z], dim=0).flatten()))

        return torch.stack(Y_pred)[:-1], pred_const, torch.stack(pred_time_dep)[:-1]
