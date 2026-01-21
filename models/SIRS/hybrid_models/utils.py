""" Utility functions for hybrid experiments"""
import torch
from typing import Literal, Tuple, Union
import sys
from os.path import dirname as up
from dantro._import_tools import import_module_from_path
sys.path.append(up(up(up(__file__))))
include = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")
from include import colors
from include.neural_net import FeedForwardNN

# Hybrid model prediction functions
from .parameters_only import make_params_pred
from .hybrid_1 import make_hybrid_1_pred
from .hybrid_2 import make_hybrid_2_pred
from .black_box import make_bb_pred

## ---------------------------------------------------------------------------------------------------------------------
## General utility functions
## ---------------------------------------------------------------------------------------------------------------------

def get_input(input_data: torch.Tensor, z: Union[None, torch.Tensor] = None) -> torch.Tensor:
    """ Reshapes and stacks a latent identifier z onto the input data. If z is None, returns the input unmodified

    :param input_data: input data without latent variable
    :param z: latent variable to use
    :return: stacked input data
    """
    if z is None:
        return input_data
    return torch.hstack([input_data, z[None, :].repeat(input_data.shape[0], 1)])

def make_pred(key: Literal["params", "hybrid_1", "hybrid_2", "bb"],
              NN: Union[FeedForwardNN, dict],
              y0: torch.Tensor,
              t_span: tuple,
              dt: Union[float, torch.Tensor],
              recursive: bool = False,
              **kwargs
              ) -> torch.Tensor:
    """ Make a prediction for each model type

    :param key: which type of model to use
    :param NN: neural network or dictionary of neural networks (for hybrid-1)
    :param y0: initial state
    :param t_span: tuple of time range
    :param dt: time differential
    :param recursive: whether to generate predictions recursively
    :param kwargs: additional kwargs passed to the prediction function
    :return: the predicted time series
    :raises: ValueError if an unrecognised key is passed
    """
    if key == 'params':
        return make_params_pred(NN=NN, y0=y0, t_span=t_span, dt=dt, **kwargs)[0]
    elif key == 'hybrid_1':
        return make_hybrid_1_pred(NN=NN, y0=y0, t_span=t_span, dt=dt, recursive=recursive, **kwargs)[0]
    elif key == 'hybrid_2':
        return make_hybrid_2_pred(NN=NN, y0=y0, t_span=t_span, dt=dt, recursive=recursive, **kwargs)[0]
    elif key == 'bb':
        return make_bb_pred(NN=NN, y0=y0, t_span=t_span, dt=dt, recursive=recursive, **kwargs)[0]
    else:
        raise ValueError(f'Unknown key {key}')

def epoch(key: Literal["params", "hybrid_1", "hybrid_2", "bb"],
          *,
          NN: Union[dict, FeedForwardNN],
          Y_target: torch.Tensor,
          y0: torch.Tensor,
          dt: Union[float, torch.Tensor],
          t_span: tuple,
          loss_array: list,
          recursive: bool = False,
          X_input: Union[None, torch.Tensor] = None,
          Y_input: Union[None, torch.Tensor] = None,
          z: Union[None, torch.Tensor] = None,
          **__
          ) -> torch.Tensor:

    """ Training epoch, using an L2 loss function. The batch size is fixed to the number of training datasets (SGD).

    :param key: which model type to use
    :param NN: neural network
    :param Y_target: target data to use for the loss
    :param y0: initial state
    param dt: time differential
    param t_span: tuple of time range
    :param loss_array: list to which to append the current loss value
    :param recursive: whether to generate time-dependent components recursively (by inserting the neural predictions as
        input for the next step) or by using the observation data
    :param X_input: input data for learning constant components (optional)
    :param Y_input: input data for learning time-dependent components, if not recursively generated
    :param z: additional identifying data, used to generalise the time-dependent component across datasets
    :param __: other kwargs (ignored)
    :return: the current loss
    """

    # Expand the tensors if only one training dataset has been passed
    if Y_target.dim() == 2:
        Y_target = Y_target[None, :]
    if X_input is not None and X_input.dim() == 1:
        X_input = X_input[None, :]
    if Y_input is not None and Y_input.dim() == 2:
        Y_input = Y_input[None, :]
    if z is not None and z.dim() == 1:
        z = z[None, :]
    if y0.dim() == 1:
        y0 = y0[None, :]

    # Initialise the loss
    loss = torch.tensor(0.0, requires_grad=True)

    # Batch size = training data set size
    for idx in range(Y_target.shape[0]):

        # Make a time series prediction only on the training data
        Y_pred = make_pred(key,
                           NN=NN,
                           t_span = t_span,
                           y0 = y0[idx],
                           dt=dt,
                           recursive=recursive,
                           X_input=X_input[idx] if X_input is not None else None,
                           Y_input=Y_input[idx] if Y_input is not None else None,
                           z=z[idx] if z is not None else None)

        # Calculate the loss
        loss = loss + torch.nn.functional.mse_loss(Y_pred[:Y_target[idx].shape[0]], Y_target[idx])

    # Gradient descent step over entire batch, using the coupled optimizer for the hybrid-1 model
    loss.backward()
    if key != 'hybrid_1':
        NN.optimizer.step()
        NN.optimizer.zero_grad()
    else:
        NN['optimizer'].step()
        NN['optimizer'].zero_grad()

    loss_array.append(loss.detach())
    return loss.detach()

## ---------------------------------------------------------------------------------------------------------------------
## Plotting
## ---------------------------------------------------------------------------------------------------------------------
def mark_training_range(ax, L, T):
    """ Marks the training range on an axis

    :param ax: axis to use
    :param L: training range
    :param T: maximum time point
    """
    # Mark training and testing periods, if given
    ax.axvspan(0, L, color=colors['c_pink'], alpha=0.2, zorder=-2, lw=0)
    ax.axvspan(L, T, color=colors['c_lightblue'], alpha=0.2, zorder=-2, lw=0)

# Handy plotting function
def plot_preds_to_axs(*,
                      key: Literal["params", "hybrid_1", "hybrid_2", "bb"],
                      ax,
                      NN: Union[dict, torch.Tensor],
                      y0: torch.Tensor,
                      dt: Union[float, torch.Tensor],
                      t_span: tuple,
                      Y_target: torch.Tensor,
                      recursive: bool = False,
                      X_input: Union[None, torch.Tensor] = None,
                      Y_input: Union[None, torch.Tensor] = None,
                      z: Union[None, torch.Tensor] = None,
                      L: int = None,
                      add_labels: bool = False):
    """ Plots predictions and true target data to an axis, marking the training and testing range.

    :param key: model to use
    :param ax: axis to which to plot
    :param NN: neural network to use; in the case of hybrid-1, a dictionary containing the two neural networks
    :param y0: initial state
    :param dt: time differential
    param t_span: tuple of time range
    :param Y_target: target data
    :param recursive: whether the inputs are recursively generated
    :param X_input: input time series to any constant-component models (optional)
    :param Y_input: input time series to any time-dependent component models, if not recursive (optional)
    :param z: latent identifier (optional)
    :param L: training period; if not None, the training and test periods are marked on the axis
    :param add_labels: whether to add S, I, R labels
    """
    # Generate a predicted time series
    with torch.no_grad():
        Y_pred = make_pred(key, NN=NN, y0=y0, t_span=t_span, dt=dt, recursive=recursive,
                           X_input=X_input, Y_input=Y_input, z=z)

    # Plot the prediction to axis
    ax.plot(Y_pred, label=['S', 'I', 'R'] if add_labels else None)
    ax.plot(Y_target, ls='dotted', c=colors['c_lightgrey'])
    ax.set(ylim=(0, 1))

    # Mark training and testing periods, if given
    if L is not None:
        mark_training_range(ax, L, len(Y_target))