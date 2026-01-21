from enum import IntEnum
import torch
from torchaudio.functional import convolve
from typing import Union


# Define an enum for the parameters and compartments so we are sure to always index in a consistent way
class Parameters(IntEnum):
    R = 0
    U = 1
    F = 2
    g_V = 3
    g_P = 4
    n_v = 5
    N = 6


class Compartments(IntEnum):
    V = 0
    W_v = 1
    W_p = 2
    R = 3
    P = 4


# ----------------------------------------------------------------------------------------------------------------------
# SOLVER
# ----------------------------------------------------------------------------------------------------------------------
def initial_condition(*, n_v: torch.Tensor,
                      n_s: torch.Tensor = torch.tensor(1.0),
                      k_F: torch.Tensor,
                      k_U: torch.Tensor,
                      k_R: torch.Tensor,
                      g_P: torch.Tensor = None,
                      g_V: torch.Tensor = None
                      ) -> torch.Tensor:
    """ Returns the initial condition as a function of the parameters. This is done by finding a vector from the kernel
    of the transition matrix that satisfies the constraints V + R + W_V = n_v and P + R + W_P = n_s = 1.

    :param n_v: number of vesicles
    :param n_s: number of sites (=1)
    :param k_F: constant fusion rate at time t0.
    :param k_U: separation rate at time t0
    :param k_R: reaction constant
    :param g_P: recovery constant. If k_F = 0, this is not needed
    :param g_V: recovery constant. If k_F = 0, this is not needed

    :return: the initial state of shape (5, )
    """

    init_state = torch.zeros(5)

    # Special case of k_F = 0
    if k_F == 0:
        _p = (n_s + n_v + k_U / k_R) / 2
        _q = n_v * n_s
        init_state[Compartments.R] = _p - torch.sqrt(_p ** 2 - _q)
        init_state[Compartments.V] = n_v - init_state[Compartments.R]
        init_state[Compartments.W_v] = 0
        init_state[Compartments.W_p] = 0
        init_state[Compartments.P] = n_s - init_state[Compartments.R]

    else:
        # Coefficients of the full equation
        a_1 = k_F / g_V + 1
        a_2 = k_R / (k_F + k_U)
        a_3 = k_F / g_P + 1

        _p = 1 / (a_1 * a_2 * a_3) + n_v / a_1 + n_s / a_3  # This is -p already, so no sign needed below
        _q = (n_v * n_s) / (a_1 * a_3)

        # Only the smaller root guarantees a valid solution
        lambda_1 = _p / 2 - torch.sqrt((_p / 2) ** 2 - _q)

        # Fill initial state
        init_state = torch.zeros(5)
        init_state[Compartments.R] = lambda_1
        init_state[Compartments.W_p] = (k_F / g_P) * lambda_1
        init_state[Compartments.W_v] = (k_F / g_V) * lambda_1
        init_state[Compartments.P] = n_s - init_state[Compartments.R] - init_state[Compartments.W_p]
        init_state[Compartments.V] = n_v - init_state[Compartments.R] - init_state[Compartments.W_v]

    return init_state


def solve_ODE(*, init_state: torch.Tensor,
              k_R: Union[float, torch.Tensor],
              k_F: torch.Tensor,
              k_U: torch.Tensor,
              g_P: Union[float, torch.Tensor],
              g_V: Union[float, torch.Tensor],
              num_steps: int,
              dt,
              **__
              ) -> torch.Tensor:
    """ Forward Euler solver for the neurotransmission recovery model.

    :param init_state: Initial state from which to solve the ODE
    :param k_R: constant release rate
    :param k_F: time-dependent fusion rate
    :param k_U: time-dependent unpriming rate
    :param g_P: constant recovery rate
    :param g_V: constant recovery rate
    :param num_steps: number of steps L to iterate
    :param dt: time differential
    :param __: other parameters (ignored)
    :return: a time series of the compartments, of shape (L + 1, 7)
    """

    #  Start from initial state
    data = [init_state]

    # Iterate ODE
    for t in range(num_steps):
        data.append(
            data[-1] + torch.stack([

                - k_R * data[-1][Compartments.V] * data[-1][Compartments.P] + g_V * data[-1][Compartments.W_v] + k_U[
                    t] * data[-1][Compartments.R],

                k_F[t] * data[-1][Compartments.R] - g_V * data[-1][Compartments.W_v],

                k_F[t] * data[-1][Compartments.R] - g_P * data[-1][Compartments.W_p],

                k_R * data[-1][Compartments.V] * data[-1][Compartments.P] - (k_F[t] + k_U[t]) * data[-1][
                    Compartments.R],

                - k_R * data[-1][Compartments.V] * data[-1][Compartments.P] + g_P * data[-1][Compartments.W_p] + k_U[
                    t] * data[-1][Compartments.R]
            ]) * dt
        )

    # Return the time series
    return torch.stack(data)


# ----------------------------------------------------------------------------------------------------------------------
# CURRENT CALCULATION
# ----------------------------------------------------------------------------------------------------------------------
def impulse_function(_t, *, A=-6.229e-6, B=2.7e-9, t0=0.0, tau_r=10.6928, tau_df=1.5e-3,
                     tau_ds=2.8e-3, **__) -> torch.Tensor:
    """ Impulse function in seconds, used to calculate the current

    :param _t: array of times
    :param A, B, t0, tau_r, tau_df, tau_ds: function parameters
    :param __: other parameters (ignored)
    :return:
    """
    _t = -(_t - t0)
    return A * (1 - torch.exp(_t / tau_r)) * (B * torch.exp(_t / tau_df) + (1 - B) * torch.exp(_t / tau_ds))


def calculate_current(*, R, k_F, g, N, dt, **__):
    """ Calculates the current by convolving the average rate of fusion events with the impulse function (g)

    :param R: number of release vesicles
    :param k_F: time-dependent fusion rate
    :param g: impulse function
    :param N: number of active zones/scaling factor
    :param dt: time differential
    :param __: other parameters (ignored)
    :return: the current over time
    """

    # Convolve the derivative of F with impulse function. The full convolution is calculated and then truncated to the support of R
    return convolve(k_F * R, N * g, mode='full')[:len(R)] * dt
