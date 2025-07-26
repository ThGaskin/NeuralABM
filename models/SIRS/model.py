"""
This file contains numerical solvers for the SIRS model. You can add your own solvers here (write your own decorator
and add it to include.solvers if you wish).
"""
from enum import IntEnum
import torch

# Import the solver module (located in `include`)
import sys
from os.path import dirname as up
from dantro._import_tools import imt_module_from_path
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
             k_S: float | torch.Tensor = 0.3,
             k_I: float | torch.Tensor = 0.1,
             k_R: float | torch.Tensor = 0.01) -> torch.Tensor:
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


# Euler solver for time-dependent parameters. The parameters can be time-series with each value
# corresponding to the parameter at time t. This function uses the same interface as the torchdiffeq solvers
def SIRS_euler_t(*,
                 y0,
                 k_S: torch.Tensor | float,
                 k_I: torch.Tensor | float,
                 k_R: torch.Tensor | float,
                 t: torch.Tensor = None,
                 t_span: tuple = None,
                 dt: float = None,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 **__
):
    # Prepare initial condition
    y0 = torch.as_tensor(y0, device=device, dtype=dtype)
    device, dtype = y0.device, y0.dtype

    # Build time grid if not supplied
    t = include.solvers.build_time_grid(t, t_span, dt, device=device, dtype=dtype)

    # Euler step function
    res = [y0]
    for t_idx, ti in enumerate(t):
        _k_I = k_I if k_I.dim() == 0 else k_I[t_idx]
        _k_R = k_R if k_R.dim() == 0 else k_R[t_idx]
        _k_S = k_S if k_S.dim() == 0 else k_S[t_idx]
        res.append(
            res[-1] + SIRS_rhs(t=ti, y=res[-1], k_S=_k_S, k_I=_k_I, k_R=_k_R) * dt
        )
    return t, torch.stack(res)