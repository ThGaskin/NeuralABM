import networkx as nx
import numpy as np
import torch

""" The Hegselmann-Krause agent-based model of opinion dynamics """


# --- The OpinionDynamics ABM ------------------------------------------------------------------------------------------
class OpinionDynamics_ABM:
    def __init__(
        self,
        *,
        N: int,
        network: nx.Graph = None,
        epsilon: float = None,
        smoothing: float = None,
        mu: float = None,
        sigma: float = None,
        init_values: torch.Tensor = None
    ):

        # The number agents (nodes in the network)
        self.N = N

        # The network adjacency matrix
        self.network = network
        self.adj_matrix = (
            torch.from_numpy(nx.to_numpy_matrix(network)).float()
            if network is not None
            else None
        )

        # Scalar parameters
        self.epsilon = epsilon
        self.smoothing = smoothing
        self.mu = mu
        self.sigma = sigma

        # The parameters to learn. Keys are the parameter names, entries are the locations in the
        # neural net output vector
        to_learn = {}
        idx = 0
        for pname, p in {
            "epsilon": epsilon,
            "mu": mu,
            "sigma": sigma,
            "smoothing": smoothing,
            "network": network,
        }.items():
            if p is None:
                to_learn[pname] = idx
                idx += 1
        self.to_learn = to_learn

        # Initialise the opinions uniformly at random on the unit interval
        self.initial_opinions = (
            torch.rand((self.N, 1), dtype=torch.float)
            if init_values is None
            else init_values
        )
        self.current_opinions = self.initial_opinions.clone()

        # Collect the edge weight properties
        self.edge_weights = (
            np.array(
                [
                    torch.exp(-abs(self.current_opinions[j] - self.current_opinions[i]))
                    .detach()
                    .numpy()
                    for i, j in self.network.edges()
                ]
            ).flatten()
            if self.network is not None
            else None
        )

        # Track interactions
        self.interactions = np.full_like(self.edge_weights, False) if self.network is not None else None

    def reset(self):

        # Resets the ABM to the initial state
        self.current_opinions = self.initial_opinions.clone()

    def update_edge_weights(self):

        # Updates the edge weights
        if self.network is not None:
            self.edge_weights = np.array(
                [
                    torch.exp(-abs(self.current_opinions[j] - self.current_opinions[i]))
                    .detach()
                    .numpy()
                    for i, j in self.network.edges()
                ]
            ).flatten()

    def update_interactions(self, tol):

        # Updates whether a link has actually been part of an interaction
        if self.network is not None:
            self.interactions = np.array(
                [
                    (torch.abs(self.current_opinions[j] - self.current_opinions[i]) <= tol)
                    .detach()
                    .numpy()
                    for i, j in self.network.edges()
                ]
            ).flatten()

    def run_single(
        self,
        *,
        current_values,
        input_data: torch.tensor = None,
        sigma: float = None,
        requires_grad: bool = True
    ):

        """Runs the model for a single iteration.

        :param current_values: the current values which to take as initial data.
        :param input_data: the input parameters to learn
        :param sigma: (optional) the noise to use during training. Defaults to the model default.
        :param requires_grad: whether the resulting values require differentiation
        :return: the updated values

        """

        # Get the parameters
        adj_matrix = (
            self.adj_matrix
            if "network" not in self.to_learn.keys()
            else torch.reshape(
                input_data[
                    self.to_learn["network"] : self.to_learn["network"] + (self.N) ** 2
                ],
                (self.N, self.N),
            )
        )
        epsilon = (
            self.epsilon
            if "epsilon" not in self.to_learn.keys()
            else input_data[self.to_learn["epsilon"]]
        )
        mu = (
            self.mu
            if "mu" not in self.to_learn.keys()
            else input_data[self.to_learn["mu"]]
        )
        smoothing = (
            self.smoothing
            if "smoothing" not in self.to_learn.keys()
            else input_data[self.to_learn["smoothing"]]
        )

        # Neural net output overrides passed noise level, which in turn overrides model default
        # TODO: add noise
        sigma = sigma if sigma is not None else self.sigma
        sigma = (
            sigma
            if "sigma" not in self.to_learn.keys()
            else input_data[self.to_learn["sigma"]]
        )

        new_values = current_values.detach().clone()
        new_values.requires_grad = requires_grad

        # Get the opinion differences
        # [[o_1-o_1, ..., o_1 - o_N], ..., [o_N-o_1, ..., o_N-o_N]]
        diffs = current_values - torch.reshape(current_values, (self.N,))

        # Get the interaction kernel, replacing the step function with a smooth sigmoid approximation
        kernel = torch.sigmoid(smoothing * (epsilon - torch.abs(diffs)))

        # Normalisation values
        norms = torch.maximum(
            torch.reshape(
                torch.matmul(adj_matrix, kernel).diag(), (self.N, 1)
            ),
            torch.ones_like(current_values),
        )

        # Perform the opinion update
        new_values = (
            new_values
            + mu
            * torch.reshape(
                torch.matmul(adj_matrix, kernel * diffs).diag(), (self.N, 1)
            )
            / norms
        )

        # Update the current opinions
        self.current_opinions = new_values.clone().detach()

        # Update the current edge weights
        self.update_edge_weights()

        # Update interaction status of each link
        self.update_interactions(epsilon)

        return new_values
