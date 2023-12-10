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


class Kuramoto_Langevin_sampler(base.MetropolisAdjustedLangevin):
    """
    A Metropolis-adjusted Langevin sampler that inherits from the base class
    """

    def __init__(
        self,
        *,
        true_data: torch.Tensor,
        eigen_frequencies: torch.Tensor,
        prior: dict = None,
        init_guess: torch.Tensor = None,
        N: int,
        alpha: torch.Tensor,
        Kuramoto_beta: torch.Tensor,
        kappa: torch.Tensor,
        dt: torch.Tensor,
        sigma: torch.Tensor,
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
        # Model parameters
        self.eigen_frequencies = eigen_frequencies
        self.N = N
        self.alpha = alpha
        self.Kuramoto_beta = Kuramoto_beta
        self.kappa = kappa
        self.sigma = sigma
        self.dt = dt

        # Draw an initial guess from the prior
        init_guess = (
            base.random_tensor(prior, size=(self.N**2,))
            if prior is not None
            else init_guess
        )

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
        self.dset_prediction = self.h5group.create_dataset(
            "predictions",
            (0, self.N, self.N),
            maxshape=(None, self.N, self.N),
            chunks=True,
            compression=3,
        )
        self.dset_prediction.attrs["dim_names"] = ["sample", "i", "j"]
        self.dset_prediction.attrs["coords_mode__sample"] = "trivial"
        self.dset_prediction.attrs["coords_mode__i"] = "trivial"
        self.dset_prediction.attrs["coords_mode__j"] = "trivial"

        # Calculate the initial values of the loss and its gradient
        self.loss[0] = self.loss_function(self.x[0])
        self.loss[1].data = self.loss[0].data

        self.grad[0].data = torch.autograd.grad(
            self.loss[0], [self.x[0]], create_graph=False
        )[0].data
        self.grad[1].data = self.grad[0].data

    def loss_function(self, nw):
        """Calculates the loss (negative log-likelihood) of an adjacency matrix via simulation."""

        nw = torch.reshape(nw, (self.N, self.N))
        loss = torch.tensor(0.0, requires_grad=True)

        for idx, training_dset in enumerate(self.true_data):
            ts = [training_dset[0]]
            idx_0 = 0
            if self.alpha != 0:
                ts.append(training_dset[1])
                idx_0 = 1
            for _ in range(idx_0, len(training_dset) - 1):
                diffs = torch.sin(ts[-1] - torch.reshape(ts[-1], (self.N,)))

                # First-order dynamics
                if self.alpha == 0:
                    ts.append(
                        ts[-1]
                        + 1
                        / self.Kuramoto_beta
                        * (
                            self.eigen_frequencies[idx][_]
                            + torch.reshape(
                                torch.matmul(self.kappa * nw, diffs).diag(),
                                (self.N, 1),
                            )
                        )
                        * self.dt
                        + torch.normal(0.0, self.sigma, (self.N, 1))
                    )

                # Second order dynamics
                else:
                    ts.append(
                        ts[-1]
                        + (
                            1
                            / self.alpha
                            * (
                                self.eigen_frequencies[idx][_]
                                + torch.reshape(
                                    torch.matmul(self.kappa * nw, diffs).diag(),
                                    (self.N, 1),
                                )
                                - self.Kuramoto_beta * (ts[-1] - ts[-2]) / self.dt
                            )
                            * self.dt
                            + (ts[-1] - ts[-2]) / self.dt
                        )
                        * self.dt
                        + torch.normal(0.0, self.sigma, (self.N, 1))
                    )

            loss = loss + torch.nn.functional.mse_loss(
                torch.stack(ts), training_dset, reduction="sum"
            )

        loss = (
            loss
            + torch.nn.functional.mse_loss(
                nw, torch.transpose(nw, 1, 0), reduction="sum"
            )
            + torch.trace(nw)
        )

        return loss

    def write_parameters(self):
        if self.time > self.write_start and self.time % self.write_every == 0:
            self.dset_prediction.resize(self.dset_prediction.shape[0] + 1, axis=0)
            self.dset_prediction[-1, :] = torch.reshape(
                self.x[0].detach(), (self.N, self.N)
            ).numpy()
