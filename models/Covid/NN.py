import sys
from os.path import dirname as up

import h5py as h5
import numpy as np
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

Covid = import_module_from_path(mod_path=up(up(__file__)), mod_str="Covid")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Model implementation
# ----------------------------------------------------------------------------------------------------------------------


class Covid_NN:
    def __init__(
        self,
        *,
        rng: np.random.Generator,
        h5group: h5.Group,
        neural_net: base.NeuralNet,
        loss_function: dict,
        to_learn: list,
        time_dependent_parameters: dict = None,
        true_parameters: dict = {},
        dt: float,
        k_q: float = 10.25,
        Berlin_data_loss: bool = False,
        write_every: int = 1,
        write_start: int = 1,
        training_data: torch.Tensor,
        batch_size: int,
        scaling_factors: dict = {},
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        Args:
            rng (np.random.Generator): The shared RNG
            h5group (h5.Group): The output file group to write data to
            neural_net: The neural network
            loss_function (dict): the loss function to use
            to_learn: the list of parameter names to learn
            time_dependent_parameters: dictionary of time-dependent parameters and their granularity
            true_parameters: the dictionary of true parameters
            dt: time differential
            k_q: contact tracing rate
            Berlin_data_loss: whether to use the loss structure unique to the Berlin data
            write_every: write every iteration
            write_start: iteration at which to start writing
            training_data: the training data to use
            batch_size: epoch batch size: instead of calculating the entire time series,
                only a subsample of length batch_size can be used. The likelihood is then
                scaled up accordingly.
            scaling_factors: dictionary of scaling factors for the different parameters. Parameter estimates are
                multiplied by these to ensure all parameters are roughly of the same order of magnitude
        """
        self._h5group = h5group
        self._rng = rng

        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        self.dt = torch.tensor(dt, dtype=torch.float)
        self.k_q = torch.tensor(k_q, dtype=torch.float)
        self.Berlin_data_loss = Berlin_data_loss

        self.current_loss = torch.tensor(0.0)

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
        self.current_predictions = torch.zeros(len(self.to_learn), dtype=torch.float)

        # Training data
        self.training_data = training_data

        # Generate the batch ids
        batches = np.arange(0, self.training_data.shape[0], batch_size)
        if len(batches) == 1:
            batches = np.append(batches, training_data.shape[0] - 1)
        else:
            if batches[-1] != training_data.shape[0] - 1:
                batches = np.append(batches, training_data.shape[0] - 1)

        self.batches = batches

        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        self._dset_loss = self._h5group.create_dataset(
            "loss",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["batch"]
        self._dset_loss.attrs["coords_mode__batch"] = "start_and_step"
        self._dset_loss.attrs["coords__batch"] = [write_start, write_every]

        self.dset_time = self._h5group.create_dataset(
            "computation_time",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["epoch"]
        self.dset_time.attrs["coords_mode__epoch"] = "trivial"

        # Create a dataset for the parameter estimates
        self.dset_parameters = self._h5group.create_dataset(
            "parameters",
            (0, len(self.to_learn.keys())),
            maxshape=(None, len(self.to_learn.keys())),
            chunks=True,
            compression=3,
        )
        self.dset_parameters.attrs["dim_names"] = ["batch", "parameter"]
        self.dset_parameters.attrs["coords_mode__batch"] = "start_and_step"
        self.dset_parameters.attrs["coords__batch"] = [write_start, write_every]
        self.dset_parameters.attrs["coords_mode__parameter"] = "values"
        self.dset_parameters.attrs["coords__parameter"] = to_learn

        # --------------------------------------------------------------------------------------------------------------
        # Batches processed
        self._time = 0
        self._write_every = write_every
        self._write_start = write_start

        # Calculate the coefficients of each term in the loss function:
        # \alpha_i^{-1} = \int T_i(t) dt
        alpha = torch.sum(training_data, dim=0) * self.dt
        alpha = torch.where(alpha > 0, alpha, torch.tensor(1.0))
        self.alpha = (
            torch.cat([alpha[0:7], torch.sum(alpha[8:11], 0, keepdim=True)], 0)
        ) ** (-1)

        # Reduced data model
        # for idx in [0, 1, 2, 3, 7]:  # S, E, I, R, Q are dropped
        #     self.alpha[idx] = 0

        # Get all the jump points
        self.jump_points = {}
        if self.time_dependent_parameters:
            self.jump_points = set(
                np.hstack(
                    [
                        np.array(interval).flatten()
                        for _, interval in self.time_dependent_parameters.items()
                    ]
                )
            )
            if None in self.jump_points:
                self.jump_points.remove(None)

        # Get the scaling factors
        self.scaling_factors = torch.tensor(
            list(
                {
                    key: torch.tensor(scaling_factors[key], dtype=torch.float)
                    if key in scaling_factors.keys()
                    else torch.tensor(1.0, dtype=torch.float)
                    for key in self.to_learn.keys()
                }.values()
            ),
            dtype=torch.float,
        )

    def epoch(self):
        """Trains the model for a single epoch"""

        # Process the training data in batches
        for batch_no, batch_idx in enumerate(self.batches[:-1]):
            # Make a prediction
            predicted_parameters = self.neural_net(
                torch.flatten(self.training_data[batch_idx])
            )

            # Combine the predicted and true parameters into a dictionary
            parameters = {
                p: predicted_parameters[self.to_learn[p]]
                * self.scaling_factors[self.to_learn[p]]
                if p in self.to_learn.keys()
                else self.true_parameters[p]
                for p in self.all_parameters
            }

            # Get the initial values
            current_densities = self.training_data[batch_idx].clone()
            current_densities.requires_grad_(True)
            densities = [current_densities]

            # Integrate the ODE for B steps
            for ele in range(batch_idx + 1, self.batches[batch_no + 1] + 1):
                # Adjust for time-dependency
                for key, ranges in self.time_dependent_parameters.items():
                    for idx, r in enumerate(ranges):
                        if not r[1]:
                            r[1] = len(self.training_data) + 1
                        if r[0] <= ele < r[1]:
                            parameters[key] = parameters[key + f"_{idx}"]
                            break

                # Calculate the k_Q parameter from the current CT figures and k_CT estimate
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
                                parameters["k_SY"]
                                * (densities[-1][2] + densities[-1][10])
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

            # Discard the initial condition
            densities = torch.stack(densities[1:])

            if self.Berlin_data_loss:
                # For the Berlin dataset, combine the quarantine compartments and drop the deceased compartment,
                # which is not present in the ABM data
                densities = torch.cat(
                    [
                        densities[:, :7],
                        torch.sum(densities[:, 8:11], dim=1, keepdim=True),
                    ],
                    dim=1,
                )
                loss = (
                    self.alpha
                    * self.loss_function(
                        densities,
                        torch.cat(
                            [
                                self.training_data[
                                    batch_idx + 1 : self.batches[batch_no + 1] + 1, :7
                                ],
                                self.training_data[
                                    batch_idx + 1 : self.batches[batch_no + 1] + 1, [8]
                                ],
                            ],
                            1,
                        ),
                    ).sum(dim=0)
                ).sum()

            # Regular loss function
            else:
                loss = self.loss_function(
                    densities,
                    self.training_data[batch_idx + 1 : self.batches[batch_no + 1] + 1],
                ) / (self.batches[batch_no + 1] - batch_idx)

            # Perform a gradient descent step
            loss.backward()
            self.neural_net.optimizer.step()
            self.neural_net.optimizer.zero_grad()
            self.current_loss = loss.clone().detach().cpu().numpy().item()
            self.current_predictions = torch.tensor(
                [
                    predicted_parameters.clone().detach().cpu()[self.to_learn[p]]
                    * self.scaling_factors[self.to_learn[p]]
                    for p in self.to_learn.keys()
                ]
            )
            self._time += 1
            self.write_data()

    def write_data(self):
        """Write the current state (loss and parameter predictions) into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0):
            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1] = self.current_loss
            self.dset_parameters.resize(self.dset_parameters.shape[0] + 1, axis=0)
            self.dset_parameters[-1, :] = [
                self.current_predictions[self.to_learn[p]] for p in self.to_learn.keys()
            ]
