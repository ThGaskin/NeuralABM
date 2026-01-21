"""
This file contains numerical solvers for the SIRS model. You can add your own solvers here (write your own decorator
and add it to include.solvers if you wish).
"""
from enum import IntEnum
import torch
from typing import Union

# Import the solver module (located in `include`)
import sys
from os.path import dirname as up
from dantro._import_tools import import_module_from_path
sys.path.append(up(up(__file__)))
include = import_module_from_path(mod_path=up(up(__file__)), mod_str="include")
from include.solvers import torchdiffeq_solver

# Define the SIR compartments
class SIR(IntEnum):
    Susceptible = 0
    Infected = 1
    Recovered = 2


def SIRS_rhs(t: float,
             y: torch.Tensor,
             k_S: Union[float, torch.Tensor] = 0.3,
             k_I: Union[float, torch.Tensor] = 0.1,
             k_R: Union[float, torch.Tensor] = 0.01) -> torch.Tensor:
    """Defines the RHS of the SIRS model such that it can be passed to a numerical solver.

    :param t: time (ignored)
    :param y: initial condition
    :param k_S: susceptibility rate
    :param k_I: infection rate
    :param k_R: recovery rate
    :param **_: other kwargs (ignored)
    :return: derivative dy(t)
    """
    S, I, R = y.unbind(-1) if y.ndim > 1 and y.shape[-1] == 3 else y
    dS = -k_I * S * I + k_S * R
    dI = k_I * S * I - k_R * I
    dR = k_R * I - k_S * R
    return torch.stack((dS, dI, dR), dim=-1)

# Euler solver
@torchdiffeq_solver(method="euler", adjoint=False)
def SIRS_euler(t, y, k_S=0.01, k_I=0.3, k_R=0.1, **_):
    return SIRS_rhs(t, y, k_S=k_S, k_I=k_I, k_R=k_R)


# Dopri5 solver
@torchdiffeq_solver(method="dopri5", adjoint=False)
def SIRS_dopri5(t, y, k_S=0.01, k_I=0.3, k_R=0.1, **_):
    return SIRS_rhs(t, y, k_S=k_S, k_I=k_I, k_R=k_R)


# Runge-Kutta 4th order solver
@torchdiffeq_solver(method="rk4", adjoint=False)
def SIRS_rk4_adj(t, y, k_S=0.01, k_I=0.3, k_R=0.1, **_):
    return SIRS_rhs(t, y, k_S=k_S, k_I=k_I, k_R=k_R)


def euler_solver_t(*,
                   y0,
                   rhs_func,
                   params: dict,
                   t: torch.Tensor = None,
                   t_span: tuple = None,
                   dt: float = None,
                   device: torch.device = None,
                   dtype: torch.dtype = None,
                   **__
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generic Euler solver for time-dependent parameters.

    :param y0: initial state
    :param rhs_func: right-hand side function to integrate
    :param params: dict of parameters (can be constant or time-dependent tensors)
    :param t: time array (optional)
    :param t_span: tuple of time range (optional if t passed)
    :param dt: time differential
    :param device: torch training device to use
    :param dtype: torch datatype to use
    :param __: additional parameters (ignored)
    :return: tuple of time values and associated y-values
    """
    # Prepare initial condition
    y0 = torch.as_tensor(y0, device=device, dtype=dtype)
    device, dtype = y0.device, y0.dtype

    # Build time grid if not supplied
    t = include.solvers.build_time_grid(t, t_span, dt, device=device, dtype=dtype)

    # Euler step function
    res = [y0]
    for t_idx, ti in enumerate(t):
        # Extract time-dependent or constant parameters
        current_params = {}
        for key, val in params.items():
            if isinstance(val, torch.Tensor) and val.dim() > 0:
                current_params[key] = val[t_idx]
            else:
                current_params[key] = val

        res.append(
            res[-1] + rhs_func(t=ti, y=res[-1], **current_params) * dt
        )

    return t, torch.stack(res)

def SIRS_euler_t(**kwargs):
    """Euler solver for SIRS with time-dependent parameters."""
    params = {k: v for k, v in kwargs.items() if k.startswith('k_')}
    return euler_solver_t(
        rhs_func=SIRS_rhs,
        params=params,
        **{k: v for k, v in kwargs.items() if not k.startswith('k_')}
    )

