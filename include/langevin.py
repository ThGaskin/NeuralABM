import copy

import h5py as h5
import torch
from torch.optim.optimizer import Optimizer


class pSGLD(Optimizer):
    """Implements pSGLD algorithm based on https://arxiv.org/pdf/1512.07666.pdf

    Built on the PyTorch RMSprop implementation
    (https://pytorch.org/docs/stable/_modules/torch/optim/rmsprop.html#RMSprop)

    Adapted from https://github.com/alisiahkoohi/Langevin-dynamics
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        beta: float = 0.99,
        Lambda: float = 1e-15,
        weight_decay: float = 0,
        centered: bool = False,
    ):
        """
        Initializes the pSGLD optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float, optional): Learning rate. Default is 1e-2.
            beta (float, optional): Exponential moving average coefficient.
                Default is 0.99.
            Lambda (float, optional): Epsilon value. Default is 1e-15.
            weight_decay (float, optional): Weight decay coefficient. Default
                is 0.
            centered (bool, optional): Whether to use centered gradients.
                Default is False.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= Lambda:
            raise ValueError(f"Invalid epsilon value: {Lambda}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(
            lr=lr,
            beta=beta,
            Lambda=Lambda,
            centered=centered,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("centered", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            float: Value of G (as defined in the algorithm) after the step.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("pSGLD does not support sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["V"] = torch.zeros_like(p.data)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(p.data)

                V = state["V"]
                beta = group["beta"]
                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                V.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(beta).add_(1 - beta, grad)
                    G = (
                        V.addcmul(grad_avg, grad_avg, value=-1)
                        .sqrt_()
                        .add_(group["Lambda"])
                    )
                else:
                    G = V.sqrt().add_(group["Lambda"])
                p.data.addcdiv_(grad, G, value=-group["lr"])

                noise_std = 2 * group["lr"] / G
                noise_std = noise_std.sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0, std=1) * noise_std
                p.data.add_(noise)

                # Only consider absolute values
                p.data.abs_()

        return G


class MetropolisAdjustedLangevin:
    """
    A class implementing the Metropolis-Adjusted Langevin algorithm. Adapted from
    https://github.com/alisiahkoohi/Langevin-dynamics

    Args:
        true_data (torch.Tensor): training data
        init_guess (torch.Tensor): Initial input tensor.
        lr (float, optional): Initial learning rate. Default is 1e-2.
        lr_final (float, optional): Final learning rate. Default is 1e-4.
        max_itr (float, optional): Maximum number of iterations. Default is
            1e4.
    """

    def __init__(
        self,
        *,
        true_data: torch.Tensor,
        init_guess: torch.Tensor,
        lr: float = 1e-2,
        lr_final: float = 1e-4,
        max_itr: float = 1e4,
        beta: float = 0.99,
        Lambda: float = 1e-15,
        centered: bool = False,
        write_start: int = 1,
        write_every: int = 1,
        batch_size: int = 1,
        h5File: h5.File,
        **__,
    ):
        super().__init__()

        # Training data
        self.true_data = true_data

        # Burn-in
        self.write_start = write_start

        # Thinning
        self.write_every = write_every

        # Batch size
        self.batch_size = batch_size

        # Create an h5 Group for the langevin estimation
        self.h5group = h5File.require_group("langevin_data")

        # Dataset for the log-likelihood
        self.dset_loss = self.h5group.create_dataset(
            "loss",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self.dset_loss.attrs["dim_names"] = ["sample"]
        self.dset_loss.attrs["coords_mode__sample"] = "trivial"

        # Track the total time required to run the samples
        self.dset_time = self.h5group.create_dataset(
            "time",
            (0,),
            maxshape=(1,),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["time"]
        self.dset_time.attrs["coords_mode__time"] = "trivial"

        # Set the initial guess
        self.x = [
            torch.zeros(init_guess.shape, device=init_guess.device, requires_grad=True),
            torch.zeros(init_guess.shape, device=init_guess.device, requires_grad=True),
        ]
        self.x[0].data = init_guess.data.clone()
        self.x[1].data = init_guess.data.clone()

        # Loss container
        self.loss = [
            torch.zeros([1], device=init_guess.device),
            torch.zeros([1], device=init_guess.device),
        ]

        # Gradient container
        self.grad = [
            torch.zeros(init_guess.shape, device=init_guess.device),
            torch.zeros(init_guess.shape, device=init_guess.device),
        ]

        # Optimizer
        self.optim = pSGLD(
            [self.x[1]],
            lr,
            weight_decay=0.0,
            beta=beta,
            Lambda=Lambda,
            centered=centered,
        )
        self.P = [
            torch.ones(init_guess.shape, device=init_guess.device, requires_grad=False),
            torch.ones(init_guess.shape, device=init_guess.device, requires_grad=False),
        ]

        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        self.lr_fn = self.decay_fn(lr=lr, lr_final=lr_final, max_itr=max_itr)
        self.time = 0

    def sample(self, *, force_accept: bool = False) -> tuple:
        """
        Perform a Metropolis-Hastings step to generate a sample. The sample can be force-accepted, e.g. if it is
        the first sample.

        Returns:
            tuple: A tuple containing the sampled input tensor and
                corresponding loss value.
        """
        accepted = False
        self.lr_decay()

        while not accepted:
            self.x[1].grad = self.grad[1].data
            self.optim.step()
            self.P[1] = self.optim.step()
            self.loss[1] = self.loss_function(self.x[1])
            self.grad[1].data = torch.autograd.grad(
                self.loss[1], [self.x[1]], create_graph=False
            )[0].data
            alpha = min([1.0, self.sample_prob()])
            if torch.rand([1]) <= alpha or force_accept:
                self.grad[0].data = self.grad[1].data
                self.loss[0].data = self.loss[1].data
                self.x[0].data = self.x[1].data
                self.P[0].data = self.P[1].data
                accepted = True

            else:
                self.x[1].data = self.x[0].data
                self.P[1].data = self.P[0].data

        self.time += 1

        return copy.deepcopy(self.x[1].data), self.loss[1].item()

    def proposal_dist(self, idx: int) -> torch.Tensor:
        """
        Calculate the proposal distribution for Metropolis-Hastings.

        Args:
            idx (int): Index of the current tensor.

        Returns:
            torch.Tensor: The proposal distribution.
        """
        return (
            -(0.25 / self.lr_fn(self.time))
            * (
                self.x[idx]
                - self.x[idx ^ 1]
                - self.lr_fn(self.time) * self.grad[idx ^ 1] / self.P[1]
            )
            * self.P[1]
            @ (
                self.x[idx]
                - self.x[idx ^ 1]
                - self.lr_fn(self.time) * self.grad[idx ^ 1] / self.P[1]
            )
        )

    def sample_prob(self) -> torch.Tensor:
        """
        Calculate the acceptance probability for Metropolis-Hastings.

        Returns:
            torch.Tensor: The acceptance probability.
        """
        return torch.exp(-self.loss[1] + self.loss[0]) * torch.exp(
            self.proposal_dist(0) - self.proposal_dist(1)
        )

    def decay_fn(
        self, lr: float = 1e-2, lr_final: float = 1e-4, max_itr: float = 1e4
    ) -> callable:
        """
        Generate a learning rate decay function.

        Args:
            lr (float, optional): Initial learning rate. Default is 1e-2.
            lr_final (float, optional): Final learning rate. Default is 1e-4.
            max_itr (float, optional): Maximum number of iterations. Default is
                1e4.

        Returns:
            callable: Learning rate decay function.
        """
        gamma = -0.55
        b = max_itr / ((lr_final / lr) ** (1 / gamma) - 1.0)
        a = lr / (b**gamma)

        def lr_fn(t: float, a: float = a, b: float = b, gamma: float = gamma) -> float:
            return a * ((b + t) ** gamma)

        return lr_fn

    def lr_decay(self):
        """
        Decay the learning rate of the optimizer.
        """
        for param_group in self.optim.param_groups:
            param_group["lr"] = self.lr_fn(self.time)

    def write_loss(self):
        """
        Write out the loss.
        """
        if self.time > self.write_start and self.time % self.write_every == 0:
            self.dset_loss.resize(self.dset_loss.shape[0] + 1, axis=0)
            self.dset_loss[-1] = self.loss[0].detach().numpy()

    def write_time(self, time):
        """
        Write out the current time
        """
        self.dset_time.resize(self.dset_time.shape[0] + 1, axis=0)
        self.dset_time[-1] = time
