import logging
import time

import h5py as h5
import torch

log = logging.getLogger(__name__)

import sys
from os.path import dirname as up

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))
from dantro._import_tools import import_module_from_path

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")


class Covid_Langevin_sampler(base.MetropolisAdjustedLangevin):
    """
    A Metropolis-adjusted Langevin sampler for the Covid model that inherits from the base class
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
        dt: float,
        k_q: float = 10.25,
        Berlin_data_loss: bool = False,
        to_learn: list,
        time_dependent_parameters: dict = None,
        true_parameters: dict,
        h5File: h5.File,
        **__,
    ):
        # Parameters to learn
        self.to_learn = {key: idx for idx, key in enumerate(to_learn)}
        self.time_dependent_parameters = (
            time_dependent_parameters if time_dependent_parameters else {}
        )
        self.true_parameters = {
            key: torch.tensor(val, dtype=torch.float)
            for key, val in true_parameters.items()
        }
        self.all_parameters = set(self.to_learn.keys())
        self.all_parameters.update(self.true_parameters.keys())
        self.N = len(self.to_learn)

        # Draw an initial guess from the prior
        init_guess: torch.Tensor = base.random_tensor(
            prior,
            size=(
                len(
                    self.to_learn.keys(),
                )
            ),
        )

        # Initialise the parent class with the initial values
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

        # Covid equation parameters
        self.dt = torch.tensor(dt, dtype=torch.float)
        self.k_q = torch.tensor(k_q, dtype=torch.float)
        self.Berlin_data_loss = Berlin_data_loss

        # Drop D, CT compartments for Berlin model, combine Q compartments
        if self.Berlin_data_loss:
            alpha = torch.sum(self.true_data, dim=0).squeeze()
            alpha = torch.cat([alpha[0:7], torch.sum(alpha[8:11], 0, keepdim=True)], 0)
            self.alpha = torch.squeeze(alpha ** (-1))

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
        r"""Calculates the loss (negative log-likelihood function) of a vector of parameters via simulation.

        :param parameters: the vector of parameters
        :return: likelihood || \hat{T}(\hat{Lambda}) - T ||_2
        """

        if self.true_data.shape[0] - self.batch_size == 1:
            start = 1
        else:
            start = torch.randint(
                1, self.true_data.shape[0] - self.batch_size, (1,)
            ).item()

        densities = [self.true_data[start - 1]]

        parameters = {
            p: input[self.to_learn[p]]
            if p in self.to_learn.keys()
            else self.true_parameters[p]
            for p in self.all_parameters
        }

        for t in range(start, start + self.batch_size - 1):
            for key, ranges in self.time_dependent_parameters.items():
                for idx, r in enumerate(ranges):
                    if not r[1]:
                        r[1] = len(self.true_data) + 1
                    if r[0] <= t < r[1]:
                        parameters[key] = parameters[key + f"_{idx}"]
                        break

            k_Q = self.k_q * parameters["k_CT"] * densities[-1][-1]

            # Solve the ODE
            densities.append(
                torch.clip(
                    densities[-1]
                    + torch.stack(
                        [
                            (-parameters["k_E"] * densities[-1][2] - k_Q)
                            * densities[-1][0]
                            + parameters["k_S"] * densities[-1][8],
                            parameters["k_E"] * densities[-1][0] * densities[-1][2]
                            - (parameters["k_I"] + k_Q) * densities[-1][1],
                            parameters["k_I"] * densities[-1][1]
                            - (parameters["k_R"] + parameters["k_SY"] + k_Q)
                            * densities[-1][2],
                            parameters["k_R"]
                            * (
                                densities[-1][2]
                                + densities[-1][4]
                                + densities[-1][5]
                                + densities[-1][6]
                                + densities[-1][10]
                            ),
                            parameters["k_SY"] * (densities[-1][2] + densities[-1][10])
                            - (parameters["k_R"] + parameters["k_H"])
                            * densities[-1][4],
                            parameters["k_H"] * densities[-1][4]
                            - (parameters["k_R"] + parameters["k_C"])
                            * densities[-1][5],
                            parameters["k_C"] * densities[-1][5]
                            - (parameters["k_R"] + parameters["k_D"])
                            * densities[-1][6],
                            parameters["k_D"] * densities[-1][6],
                            -parameters["k_S"] * densities[-1][8]
                            + k_Q * densities[-1][0],
                            -parameters["k_I"] * densities[-1][9]
                            + k_Q * densities[-1][1],
                            parameters["k_I"] * densities[-1][9]
                            + k_Q * densities[-1][2]
                            - (parameters["k_SY"] + parameters["k_R"])
                            * densities[-1][10],
                            parameters["k_SY"] * densities[-1][2]
                            - self.k_q
                            * torch.sum(densities[-1][0:3])
                            * densities[-1][-1],
                        ]
                    )
                    * self.dt,
                    0,
                    1,
                )
            )

        densities = torch.stack(densities)

        if self.Berlin_data_loss:
            # Scale loss to prevent numerical underflow of the preconditioner (which is inversely proportional to the
            # gradient)
            loss = 5e4 * torch.dot(
                self.alpha,
                torch.concat(
                    [
                        torch.sum(
                            torch.pow(
                                densities[:, 0:7]
                                - self.true_data[start : start + self.batch_size, 0:7],
                                2,
                            ),
                            dim=0,
                        ),
                        torch.sum(
                            torch.pow(
                                torch.sum(densities[:, 8:11], dim=1)
                                - self.true_data[start : start + self.batch_size, 8],
                                2,
                            ),
                            dim=0,
                            keepdim=True,
                        ),
                    ],
                    0,
                ).squeeze(),
            )

        else:
            loss = torch.sum(
                torch.pow(
                    densities - self.true_data[start : start + self.batch_size], 2
                )
            )

        return loss

    def write_parameters(self):
        if self.time > self.write_start and self.time % self.write_every == 0:
            self.dset_parameters.resize(self.dset_parameters.shape[0] + 1, axis=0)
            self.dset_parameters[-1, :] = torch.flatten(self.x[0].detach()).numpy()


def perform_sampling(h5file, training_data, model_cfg: dict) -> None:
    """Runs the Covid Langevin sampler.

    :param h5file: hdf5 file to write the data to
    :param training_data: training data used to calculate the likelihood
    :param model_cfg: configuration file
    """

    # Number of samples
    n_samples = model_cfg["MCMC"].pop("n_samples")

    # Initialise the sampler
    sampler = Covid_Langevin_sampler(
        h5File=h5file,
        true_data=training_data[
            model_cfg["Data"].get("training_data_size", slice(None, None)), :, :
        ],
        to_learn=model_cfg["Training"]["to_learn"],
        time_dependent_parameters=model_cfg["Data"].get(
            "time_dependent_parameters", None
        ),
        true_parameters=model_cfg["Training"].get("true_parameters", {}),
        dt=model_cfg["Data"]["synthetic_data"]["dt"],
        k_q=model_cfg["Data"]["synthetic_data"]["k_q"],
        **model_cfg["MCMC"],
    )

    # Track the sampling time
    start_time = time.time()

    # Collect n_samples
    for i in range(n_samples):
        sampler.sample()
        sampler.write_loss()
        sampler.write_parameters()
        log.info(f"Collected {i} of {n_samples}; current loss: {sampler.loss[1]}")

    # Write out the total sampling time
    sampler.write_time(time.time() - start_time)

    log.success("   MCMC sampling complete.")
