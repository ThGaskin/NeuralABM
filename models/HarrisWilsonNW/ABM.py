import torch

""" The Harris and Wilson model numerical solver """


class HarrisWilsonABM:
    def __init__(
        self,
        *,
        N: int,
        M: int,
        alpha: float,
        beta: float,
        kappa: float,
        epsilon: float,
        sigma: float,
        dt: float,
        device: str
    ):
        """The Harris and Wilson model of economic activity.

        :param N: the number of origin zones
        :param M: the number of destination zones
        :param true_parameters: (optional) a dictionary of the true parameters
        :param epsilon: (optional) the epsilon value to use for the solver
        :param dt: (optional) the time differential to use for the solver
        :param device: the training device to use
        """

        # The origin zone sizes, number of origin zones, and number of destination zones
        self.N = N
        self.M = M

        # Model parameters
        self.alpha = torch.tensor(alpha, dtype=torch.float).to(device)
        self.beta = torch.tensor(beta, dtype=torch.float).to(device)
        self.kappa = torch.tensor(kappa, dtype=torch.float).to(device)
        self.sigma = torch.tensor(sigma, dtype=torch.float).to(device)
        self.epsilon = torch.tensor(epsilon, dtype=torch.float).to(device)
        self.dt = torch.tensor(dt, dtype=torch.float).to(device)
        self.device = device

    # ... Model run functions ..........................................................................................

    def run_single(
        self,
        *,
        curr_vals,
        adjacency_matrix: torch.tensor,
        epsilon: float = None,
        sigma: float = None,
        dt: float = None,
        origin_sizes: torch.tensor
    ):
        """Runs the model for a single iteration.

        :param curr_vals: the current values which to take as initial data.
        :param epsilon: (optional) the epsilon value to use. Defaults to the model default.
        :param dt: (optional) the time differential to use. Defaults to the model default.
        :param requires_grad: whether the resulting values require differentiation
        :return: the updated values

        """

        # Training parameters
        sigma = self.sigma if sigma is None else sigma
        epsilon = self.epsilon if epsilon is None else epsilon
        dt = self.dt if dt is None else dt

        new_sizes = curr_vals.clone().detach()

        # Calculate the weight matrix C^beta
        weights = torch.pow(adjacency_matrix, self.beta)

        # Calculate the exponential sizes W_j^alpha
        W_alpha = torch.pow(curr_vals, self.alpha)

        # Calculate the normalisations sum_k W_k^alpha exp(-beta * c_ik) (double transposition of weight matrix
        # necessary for this step)
        normalisations = torch.sum(
            torch.transpose(torch.mul(W_alpha, torch.transpose(weights, 0, 1)), 0, 1),
            dim=1,
            keepdim=True,
        )

        # Calculate the vector of demands
        demand = torch.mul(
            W_alpha,
            torch.reshape(
                torch.sum(
                    torch.div(
                        torch.mul(origin_sizes, weights),
                        torch.where(normalisations != 0, normalisations, 1),
                    ),
                    dim=0,
                    keepdim=True,
                ),
                (self.M, 1),
            ),
        )

        # Update the current values
        new_sizes = (
            new_sizes
            + torch.mul(
                curr_vals,
                epsilon * (demand - self.kappa * curr_vals)
                + sigma
                * 1
                / torch.sqrt(torch.tensor(2, dtype=torch.float) * torch.pi * dt).to(
                    self.device
                )
                * torch.normal(0, 1, size=(self.M, 1)).to(self.device),
            )
            * dt
        )

        return new_sizes

    def run(
        self,
        *,
        init_data,
        adjacency_matrix: torch.tensor,
        n_iterations: int,
        epsilon: float = None,
        dt: float = None,
        origin_sizes: torch.tensor,
        generate_time_series: bool = False
    ) -> torch.tensor:
        """Runs the model for n_iterations.

        :param init_data: the initial destination zone size values
        :param input_data: (optional) the parameters to use during training. Defaults to the model defaults.
        :param n_iterations: the number of iteration steps.
        :param epsilon: (optional) the epsilon value to use. Defaults to the model default.
        :param dt: (optional) the time differential to use. Defaults to the model default.
        :param requires_grad: (optional) whether the calculated values require differentiation
        :param generate_time_series: whether to generate a complete time series or only return the final value
        :return: the time series data

        """

        if not generate_time_series:
            sizes = init_data.clone()
            for _ in range(n_iterations):
                sizes = self.run_single(
                    curr_vals=sizes,
                    adjacency_matrix=adjacency_matrix,
                    epsilon=epsilon,
                    dt=dt,
                    origin_sizes=origin_sizes[_],
                )
                return torch.stack(sizes)

        else:
            sizes = [init_data.clone()]
            for _ in range(n_iterations):
                sizes.append(
                    self.run_single(
                        curr_vals=sizes[-1],
                        adjacency_matrix=adjacency_matrix,
                        epsilon=epsilon,
                        dt=dt,
                        origin_sizes=origin_sizes[_],
                    )
                )
            sizes = torch.stack(sizes)
            return torch.reshape(sizes, (sizes.shape[0], sizes.shape[1], 1))
