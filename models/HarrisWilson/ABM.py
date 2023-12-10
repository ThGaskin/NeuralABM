import torch

""" The Harris and Wilson model numerical solver """


class HarrisWilsonABM:
    def __init__(
        self,
        *,
        origin_sizes,
        network,
        M,
        true_parameters: dict = None,
        epsilon: float = 1.0,
        dt: float = 0.001,
        device: str,
    ):
        """The Harris and Wilson model of economic activity.

        :param origin_sizes: the origin sizes of the network
        :param network: the network adjacency matrix
        :param M: the number of destination zones
        :param true_parameters: (optional) a dictionary of the true parameters
        :param epsilon: (optional) the epsilon value to use for the solver
        :param dt: (optional) the time differential to use for the solver
        :param device: the training device to use
        """

        # The origin zone sizes, number of origin zones, and number of destination zones
        self.or_sizes = origin_sizes
        self.N = len(origin_sizes)
        self.M = M

        # The network
        self.nw = network

        # Model parameters
        self.true_parameters = true_parameters
        params_to_learn = (
            {}
            if true_parameters is not None
            else {"alpha": 0, "beta": 1, "kappa": 2, "sigma": 3}
        )
        if true_parameters is not None:
            idx = 0
            for param in ["alpha", "beta", "kappa", "sigma"]:
                if param not in true_parameters.keys():
                    params_to_learn[param] = idx
                    idx += 1
        self.parameters_to_learn = params_to_learn
        self.epsilon = torch.tensor(epsilon).to(device)
        self.dt = torch.tensor(dt).to(device)
        self.device = device

    # ... Model run functions ..........................................................................................

    def run_single(
        self,
        *,
        curr_vals,
        input_data=None,
        epsilon: float = None,
        dt: float = None,
    ):
        """Runs the model for a single iteration.

        :param curr_vals: the current values which to take as initial data.
        :param input_data: the input parameters (to learn). Defaults to the model defaults.
        :param epsilon: (optional) the epsilon value to use. Defaults to the model default.
        :param dt: (optional) the time differential to use. Defaults to the model default.
        :return: the updated values

        """

        # Parameters to learn
        alpha = (
            self.true_parameters["alpha"]
            if "alpha" not in self.parameters_to_learn.keys()
            else input_data[self.parameters_to_learn["alpha"]]
        )
        beta = (
            self.true_parameters["beta"]
            if "beta" not in self.parameters_to_learn.keys()
            else input_data[self.parameters_to_learn["beta"]]
        )
        kappa = (
            self.true_parameters["kappa"]
            if "kappa" not in self.parameters_to_learn.keys()
            else input_data[self.parameters_to_learn["kappa"]]
        )
        sigma = (
            self.true_parameters["sigma"]
            if "sigma" not in self.parameters_to_learn.keys()
            else input_data[self.parameters_to_learn["sigma"]]
        )

        # Training parameters
        epsilon = self.epsilon if epsilon is None else epsilon
        dt = self.dt if dt is None else dt

        new_sizes = curr_vals.clone()

        # Calculate the weight matrix C^beta
        weights = torch.pow(self.nw, beta)

        # Calculate the exponential sizes W_j^alpha
        W_alpha = torch.pow(curr_vals, alpha)

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
                    torch.div(torch.mul(self.or_sizes, weights), normalisations),
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
                epsilon * (demand - kappa * curr_vals)
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
        input_data=None,
        n_iterations: int,
        epsilon: float = None,
        dt: float = None,
        generate_time_series: bool = False,
    ) -> torch.tensor:
        """Runs the model for n_iterations.

        :param init_data: the initial destination zone size values
        :param input_data: (optional) the parameters to use during training. Defaults to the model defaults.
        :param n_iterations: the number of iteration steps.
        :param epsilon: (optional) the epsilon value to use. Defaults to the model default.
        :param dt: (optional) the time differential to use. Defaults to the model default.
        :param generate_time_series: whether to generate a complete time series or only return the final value
        :return: the time series data

        """
        sizes = [init_data.clone()]

        for _ in range(n_iterations):
            sizes.append(
                self.run_single(
                    curr_vals=sizes[-1],
                    input_data=input_data,
                    epsilon=epsilon,
                    dt=dt,
                )
            )
        sizes = torch.stack(sizes)
        if not generate_time_series:
            return sizes[-1]
        else:
            return sizes
