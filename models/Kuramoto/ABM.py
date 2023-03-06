import logging

import torch

""" The Kuramoto model of synchronised oscillation """

# --- The Kuramoto ABM ------------------------------------------------------------------------------------------
class Kuramoto_ABM:
    def __init__(
        self, *, N: int, sigma: float, dt: float, gamma: float, kappa: float, device: str, **__
    ):

        """The Kuramoto model numerical solver, for first and second-order dynamics.

        :param N: the number of nodes in the network
        :param sigma: the default noise variance
        :param dt: the time differential to use
        :param gamma: the parameter used for the second-order model
        :param kappa: the scaling value used for the network coupling
        :param **__: other kwargs (ignored)
        """

        # The number agents (nodes in the network)
        self.N = N

        # Noise variance
        self.sigma = torch.tensor(sigma, device=device, dtype=torch.float)

        # Time differential
        self.dt = torch.tensor(dt, device=device, dtype=torch.float)

        # Dampening coefficient (second order only)
        self.gamma = torch.tensor(gamma, device=device, dtype=torch.float)

        # Scaling value for network coupling
        self.kappa = torch.tensor(kappa, device=device, dtype=torch.float)

        # Training device to use
        self.device = device

    def run_single(
        self,
        *,
        current_phases: torch.tensor,
        current_velocities: torch.tensor = None,
        adjacency_matrix: torch.tensor,
        eigen_frequencies: torch.tensor,
        sigma: float = None,
        requires_grad: bool = True,
    ):
        """Runs the model for a single iteration.

        :param current_phases: the current phases of the oscillators
        :param current_velocities: the current velocities of the oscillators. If this is not None, a
            second-order Kuramoto scheme is used
        :param adjacency_matrix: the coupling network
        :param sigma: (optional) the noise to use during training. Defaults to the model default.
        :param requires_grad: whether the resulting values require differentiation
        :return: the updated values

        """

        sigma = self.sigma if sigma is None else sigma
        new_phases = current_phases.clone().detach().requires_grad_(requires_grad)

        diffs = torch.sin(current_phases - torch.reshape(current_phases, (self.N,)))

        # First-order dynamics
        if current_velocities is None:

            new_phases = (
                new_phases
                + (
                    eigen_frequencies
                    + torch.reshape(
                        torch.matmul(self.kappa * adjacency_matrix, diffs).diag(), (self.N, 1)
                    )
                )
                * self.dt
                + torch.normal(0.0, sigma, (self.N, 1), device=self.device)
            )

        # Second-order dynamics
        else:

            new_phases = (
                new_phases
                + (
                    (
                        eigen_frequencies
                        + torch.reshape(
                            torch.matmul(self.kappa * adjacency_matrix, diffs).diag(), (self.N, 1)
                        )
                        - self.gamma * current_velocities
                    )
                    * self.dt
                    + current_velocities
                )
                * self.dt
                + torch.normal(0.0, sigma, (self.N, 1))
            )

        return new_phases
