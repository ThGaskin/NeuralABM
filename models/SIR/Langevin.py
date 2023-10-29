import logging

log = logging.getLogger(__name__)

import sys
from os.path import dirname as up

import h5py as h5
import torch
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")


class SIR_Langevin_sampler(base.MetropolisAdjustedLangevin):
    """
    A Metropolis-adjusted Langevin sampler that inherits from the base class
    """

    def __init__(
        self,
        *,
        true_data: torch.Tensor,
        prior: dict,
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
        to_learn: list,
        true_parameters: dict,
        **__,
    ):
        # Parameters to learn
        self.to_learn = {key: idx for idx, key in enumerate(to_learn)}
        self.true_parameters = {
            key: torch.tensor(val, dtype=torch.float)
            for key, val in true_parameters.items()
        }

        # Draw an initial guess from the prior
        init_guess = base.random_tensor(prior, size=(len(to_learn),))

        super().__init__(
            true_data=true_data,
            init_guess=init_guess,
            lr=lr,
            lr_final=lr_final,
            max_itr=max_itr,
            beta=beta,
            Lambda=Lambda,
            centered=centered,
            write_start=write_start,
            write_every=write_every,
            batch_size=batch_size,
            h5File=h5File,
        )

        # Create datasets for the predictions
        self.dset_parameters = self.h5group.create_dataset(
            "parameters",
            (0, len(self.to_learn.keys())),
            maxshape=(None, len(self.to_learn.keys())),
            chunks=True,
            compression=3,
        )
        self.dset_parameters.attrs["dim_names"] = ["sample", "parameter"]
        self.dset_parameters.attrs["coords_mode__sample"] = "trivial"
        self.dset_parameters.attrs["coords_mode__parameter"] = "values"
        self.dset_parameters.attrs["coords__parameter"] = to_learn

        # Calculate the initial values of the loss and its gradient
        self.loss[0] = self.loss_function(self.x[0])
        self.loss[1].data = self.loss[0].data

        self.grad[0].data = torch.autograd.grad(
            self.loss[0], [self.x[0]], create_graph=False
        )[0].data
        self.grad[1].data = self.grad[0].data

    def loss_function(self, input):
        """Calculates the loss (negative log-likelihood function) of a vector of parameters via simulation."""

        if self.true_data.shape[0] - self.batch_size == 1:
            start = 1
        else:
            start = torch.randint(
                1, self.true_data.shape[0] - self.batch_size, (1,)
            ).item()

        densities = [self.true_data[start - 1]]

        # Get the parameters: infection rate, recovery time, noise variance
        p = (
            input[self.to_learn["p_infect"]]
            if "p_infect" in self.to_learn.keys()
            else self.true_parameters["p_infect"]
        )
        t = (
            30 * input[self.to_learn["t_infectious"]]
            if "t_infectious" in self.to_learn.keys()
            else self.true_parameters["t_infectious"]
        )
        sigma = (
            input[self.to_learn["sigma"]]
            if "sigma" in self.to_learn.keys()
            else self.true_parameters["sigma"]
        )
        alpha = (
            input[self.to_learn["alpha"]]
            if "alpha" in self.to_learn.keys()
            else self.true_parameters["alpha"]
        )

        for s in range(start, start + self.batch_size - 1):
            # Recovery rate
            tau = 1 / t * torch.sigmoid(1000 * (s / t - 1))

            # Random noise
            w = torch.normal(torch.tensor(0.0), torch.tensor(1.0))

            # Solve the ODE
            densities.append(
                torch.clip(
                    densities[-1]
                    + torch.stack(
                        [
                            (-p * densities[-1][0] - sigma * w) * densities[-1][1]
                            + 1 / (10000 + alpha),
                            (p * densities[-1][0] + sigma * w - tau) * densities[-1][1]
                            + 1 / (10000 + alpha),
                            tau * densities[-1][1] + 1 / (10000 + alpha),
                        ]
                    ),
                    0,
                    1,
                )
            )

        densities = torch.stack(densities)

        # Calculate loss
        return torch.nn.functional.mse_loss(
            densities, self.true_data[start : start + self.batch_size], reduction="sum"
        )

    def write_parameters(self):
        if self.time > self.write_start and self.time % self.write_every == 0:
            self.dset_parameters.resize(self.dset_parameters.shape[0] + 1, axis=0)
            self.dset_parameters[-1, :] = torch.flatten(self.x[0].detach()).numpy() * [
                1,
                30,
                1,
            ]
