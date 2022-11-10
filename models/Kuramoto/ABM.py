import torch

""" The Kuramoto model of synchronised oscillation """

# --- The Kuramoto ABM ------------------------------------------------------------------------------------------
class Kuramoto_ABM:

    def __init__(
        self,
        *,
        N: int,
        sigma: float = 0,
        dt: float = 0.01,
        eigen_frequencies: torch.Tensor,
    ):

        """ The Kuramoto model numerical solver.

        :param N: the number of nodes in the network
        :param sigma: the default noise variance
        :param dt: the time differential to use
        :param eigen_frequencies: the eigenfrequencies of the oscillators
        """

        # The number agents (nodes in the network)
        self.N = N

        # Scalar parameter: noise variance
        self.sigma = sigma
        self.dt = dt

        # Initialise the opinions uniformly at random on the unit interval
        self.eigen_frequencies = eigen_frequencies

    def run_single(
        self,
        *,
        current_phases,
        adjacency_matrix: torch.tensor,
        sigma: float = None,
        requires_grad: bool = True
    ):

        """Runs the model for a single iteration.

        :param current_phases: the current phases of the oscillators
        :param adjacency_matrix: the coupling network
        :param sigma: (optional) the noise to use during training. Defaults to the model default.
        :param requires_grad: whether the resulting values require differentiation
        :return: the updated values

        """

        sigma = self.sigma if sigma is None else sigma

        new_phases = current_phases.clone().detach()

        diffs = torch.sin(current_phases - torch.reshape(current_phases, (self.N,)))

        new_phases = new_phases + (self.eigen_frequencies +
                     torch.reshape(torch.matmul(adjacency_matrix, diffs).diag(), (self.N, 1))) * self.dt + torch.normal(0.0, sigma, (self.N, 1))

        return new_phases
