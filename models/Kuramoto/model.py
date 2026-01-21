"""
This file contains numerical solvers for the Kuramoto model. You can add your own solvers here (write your own decorator
and add it to include.solvers if you wish).
"""
from typing import Union
import torch

# Import the solver module (located in `include`)
import sys
from os.path import dirname as up
from dantro._import_tools import import_module_from_path
sys.path.append(up(up(__file__)))
include = import_module_from_path(mod_path=up(up(__file__)), mod_str="include")
from include.solvers import torchdiffeq_solver


def Kuramoto_rhs(
    t,
    state,
    *,
    adjacency_matrix: torch.Tensor,
    eigen_frequencies: torch.Tensor,
    kappa: Union[torch.Tensor, float],
    beta: Union[torch.Tensor, float],
    alpha: Union[torch.Tensor, float] = 0.0,
):
    """
    Compute the right-hand side (RHS) of the Kuramoto model.

    Parameters
    ----------
    t : float or torch.Tensor
        Current time (unused in the autonomous Kuramoto equations, but kept
        for compatibility with generic ODE solver decorators).
    state : torch.Tensor
        Current system state.
        - If `alpha == 0` (first-order dynamics): shape (N, 1),
          containing oscillator phases θ_i.
        - If `alpha != 0` (second-order dynamics): shape (N, 2),
          with [:, 0] = phases θ_i and [:, 1] = velocities dθ_i/dt.
    adjacency_matrix : torch.Tensor, shape (N, N)
        Coupling matrix describing network connections.
    eigen_frequencies : torch.Tensor, shape (N, 1)
        Natural frequencies ω_i of the oscillators.
    kappa : float
        Global coupling strength.
    alpha : float
        Inertia parameter. If 0, reduces to the standard first-order
        Kuramoto model. Otherwise, adds a second-order (inertial) term.
    beta : float
        Damping parameter. Used in both first- and second-order cases.
    sigma : float
        Noise strength (unused here — stochastic increments should be
        added at the solver level, not inside the deterministic RHS).
    device : torch.device
        Torch device for tensor operations.

    Returns
    -------
    torch.Tensor
        Derivatives of the state with the same shape as `state`:
        - First-order (alpha == 0): shape (N, 1), dθ/dt
        - Second-order (alpha != 0): shape (N, 2),
          [dθ/dt, d²θ/dt²]
    """
    phases = state[:, 0]
    if alpha != 0:  # second-order case
        velocities = state[:, 1]
    else:
        velocities = None

    # Pairwise phase differences
    diffs = torch.sin(phases - phases.reshape((len(phases),)))

    # Coupling contribution
    coupling = torch.matmul(kappa * adjacency_matrix, diffs).diag()

    if alpha == 0:
        # First-order Kuramoto
        dtheta = (eigen_frequencies.squeeze() + coupling) / beta
        return torch.stack([dtheta], dim=1)
    else:
        # Second-order Kuramoto
        dtheta = velocities
        dvel   = (eigen_frequencies.squeeze() + coupling - beta * velocities) / alpha
        return torch.stack([dtheta, dvel], dim=1)

# Euler solver
@torchdiffeq_solver(method="euler", adjoint=False)
def Kuramoto_euler(t, state, **params):
    return Kuramoto_rhs(t, state, **params)


# Dopri5 solver
@torchdiffeq_solver(method="dopri5", adjoint=False)
def Kuramoto_dopri5(t, state, **params):
    return Kuramoto_rhs(t, state, **params)


# Runge-Kutta 4th order solver
@torchdiffeq_solver(method="rk4", adjoint=False)
def Kuramoto_rk4_adj(t, state, **params):
    return Kuramoto_rhs(t, state, **params)

