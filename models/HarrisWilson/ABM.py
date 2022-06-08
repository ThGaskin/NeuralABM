import torch

""" The Harris and Wilson agent-based model """


class HarrisWilsonABM:

    def __init__(self,
                 *,
                 origin_sizes,
                 network,
                 M,
                 true_parameters: dict = None,
                 epsilon: float = 1.0,
                 dt: float = 0.001,
                 sigma: float = 0.0):

        """ The Harris and Wilson model of economic activity.

        :param model_cfg: the model configuration
        :param init_origin: the mean and variance of the Gaussian with which to initialise the origin zone sizes
        :param init_origin: the mean and variance of the Gaussian with which to initialise the destination zone sizes
        :param init_weights: the mean and variance of the Gaussian with which to initialise the network weights

        """

        # The origin zone sizes and number of origin zones
        self.or_sizes = origin_sizes
        self._N = len(origin_sizes)

        # The initial destination zone sizes and number of destination zones
        self._M = M

        # The network
        self.nw = network

        # Model parameters
        self._true_parameters = true_parameters
        params_to_learn = {} if true_parameters is not None else {'alpha': 0, 'beta': 1, 'kappa': 2}
        if true_parameters is not None:
            idx = 0
            for param in ['alpha', 'beta', 'kappa']:
                if param not in true_parameters.keys():
                    params_to_learn[param] = idx
                    idx += 1
        self.parameters_to_learn = params_to_learn

        self._epsilon = epsilon

        # Time differential
        self._dt = dt

        # Noise variance
        self._sigma = sigma


    # ... Model run functions ..........................................................................................

    def run_single(self,
                   *,
                   curr_vals,
                   input_data,
                   sigma: float = None,
                   epsilon: float = None,
                   dt: float = None,
                   requires_grad: bool = True):

        """ Runs the model for a single iteration.

        :param curr_vals: the current values which to take as initial data.
        :param input_data: the input parameters (to learn)
        :param sigma: the noise variance to use. Defaults to the model default noise
        :param requires_grad: whether the resulting values require differentiation
        :return: the updated values

        """
        # Get the parameters
        alpha = self._true_parameters['alpha'] if 'alpha' not in self.parameters_to_learn.keys() \
            else input_data[self.parameters_to_learn['alpha']]
        beta = self._true_parameters['beta'] if 'beta' not in self.parameters_to_learn.keys() \
            else input_data[self.parameters_to_learn['beta']]
        kappa = self._true_parameters['kappa'] if 'kappa' not in self.parameters_to_learn.keys() \
            else input_data[self.parameters_to_learn['kappa']]
        epsilon = self._epsilon if epsilon is None else epsilon
        sigma = self._sigma if sigma is None else sigma
        dt = self._dt if dt is None else dt

        new_sizes = curr_vals.clone()
        new_sizes.requires_grad = requires_grad

        # Calculate the weight matrix C^beta
        weights = torch.pow(self.nw, beta)

        # Calculate the exponential sizes W_j^alpha
        W_alpha = torch.pow(curr_vals, alpha)

        # Calculate the normalisations sum_k W_k^alpha exp(-beta * c_ik) (double transposition of weight matrix
        # necessary for this step)
        normalisations = torch.sum(
            torch.transpose(torch.mul(W_alpha, torch.transpose(weights, 0, 1)), 0, 1),
            dim=1, keepdim=True)

        # Calculate the vector of demands
        demand = torch.mul(W_alpha,
                           torch.reshape(
                               torch.sum(torch.div(torch.mul(self.or_sizes, weights), normalisations), dim=0,
                                         keepdim=True), (self._M, 1)))

        # Update the current values
        new_sizes = new_sizes + \
                    + torch.mul(curr_vals, epsilon * (demand - kappa * curr_vals) * dt
                                + torch.normal(torch.zeros_like(curr_vals), sigma * torch.ones_like(curr_vals)))

        return new_sizes

    def run(self,
            *,
            init_data,
            input_data,
            n_iterations,
            epsilon: float = None,
            sigma: float=None,
            dt: float = None,
            requires_grad: bool = True,
            generate_time_series: bool = False):

        """ Runs the model for n_iterations.

        :param input_data:
        :param n_iterations:
        :param init_data:
        :param requires_grad:
        :param generate_time_series:
        :return:

        """

        if not generate_time_series:
            sizes = init_data.clone()
            for _ in range(n_iterations):
                sizes = self.run_single(curr_vals=sizes, input_data=input_data, epsilon=epsilon, sigma=sigma, dt=dt,
                                        requires_grad=requires_grad)
                return torch.stack(sizes)

        else:
            sizes = [init_data.clone()]
            for _ in range(n_iterations):
                sizes.append(
                    self.run_single(curr_vals=sizes[-1], input_data=input_data, epsilon=epsilon,
                                    sigma=sigma, dt=dt, requires_grad=requires_grad))
            sizes = torch.stack(sizes)
            return torch.reshape(sizes, (sizes.shape[0], sizes.shape[1], 1))
