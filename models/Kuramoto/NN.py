import sys
from os.path import dirname as up
import time

import h5py as h5
import numpy as np
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

Kuramoto = import_module_from_path(mod_path=up(up(__file__)), mod_str="Kuramoto")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)

class Kuramoto_NN:
    def __init__(
        self,
        name: str,
        *,
        rng: np.random.Generator,
        output_data_group: h5.Group,
        neural_net: base.NeuralNet,
        loss_function: dict,
        ABM: Kuramoto.Kuramoto_ABM,
        true_network: torch.Tensor = None,
        training_data: torch.Tensor,
        eigen_frequencies: torch.Tensor,
        batch_size: int,
        write_every: int = 1,
        write_predictions_every: int = 1,
        write_start: int = 1,
        cut_off_time: int = 0,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        :param name (str): The name of this model instance
        :param rng (np.random.Generator): The shared RNG
        :param output_data_group (h5.Group): The output file group to write training data to
        :param neural_net: The neural network
        :param loss_function: dictionary of loss function to use
        :param ABM: The numerical solver
        :param true_network: (optional) true network; only used to track the prediction error over time.
            For the power grid example, this is the unperturbed network and is used to initially push the model
            towards the unperturbed network.
        :param training_data: data on which to train the neural network
        :param eigen_frequencies: time series of the eigenfrequencies of the nodes in the network
        :param batch_size: batch size to use during training
        :param write_every: write every iteration
        :param write_predictions_every: write out predicted parameters every iteration
        :param write_start: iteration at which to start writing
        :param cut_off_time: (optional) power grid only: batch at which to stop driving the NN towards the unperturbed
            network (TODO: calculate this automatically based on gradients).
        """
        self._name = name
        self._time = 0
        self._output_data_group = output_data_group
        self._rng = rng

        self.ABM = ABM
        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        # Store the true network
        self.true_network = true_network

        # Store the training data and node eigenfrequencies
        self.training_data = training_data
        self.eigen_frequencies = eigen_frequencies

        self._write_every = write_every
        self._write_predictions_every = write_predictions_every
        self._write_start = write_start
        self.batch_size = batch_size
        self.cut_off_time = cut_off_time
        self.num_agents = training_data.shape[2]
        self.nw_size = self.num_agents**2

        # Store the current losses: current total training loss, prediction loss on the data, symmetry loss, trace loss,
        # and current prediction error  on the adjacency matrix
        self.current_total_loss = torch.tensor(0.0, dtype=torch.float)
        self.current_prediction_loss = torch.tensor(0.0, dtype=torch.float)
        self.current_symmetry_loss = torch.tensor(0.0, dtype=torch.float)
        self.current_trace_loss = torch.tensor(0.0, dtype=torch.float)
        self.current_prediction_error = torch.tensor(0.0, dtype=torch.float)

        # Current predicted network
        self.current_adjacency_matrix = torch.zeros(self.num_agents, self.num_agents)

        # Batches: generate the batch ids
        batches = np.arange(
            0 if self.ABM.alpha == 0 else 1, training_data.shape[1], batch_size
        )
        if len(batches) == 1:
            batches = np.append(batches, training_data.shape[1] - 1)
        else:
            if batches[-1] != training_data.shape[1] - 1:
                batches = np.append(batches, training_data.shape[1] - 1)
        self.batches = batches

        # Store the losses and errors
        self._dset_loss = self._output_data_group.create_dataset(
            "Loss",
            (0, 5),
            maxshape=(None, 5),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["batch", "kind"]
        self._dset_loss.attrs["coords_mode__batch"] = "start_and_step"
        self._dset_loss.attrs["coords__batch"] = [write_start, write_every]
        self._dset_loss.attrs["coords_mode__kind"] = "values"
        self._dset_loss.attrs["coords__kind"] = [
            "Total loss",
            "Data loss",
            "Symmetry loss",
            "Trace loss",
            "L1 prediction error",
        ]

        # Store the neural net output, possibly less regularly than the loss
        self._dset_predictions = self._output_data_group.create_dataset(
            "predictions",
            (0, self.num_agents, self.num_agents),
            maxshape=(None, self.num_agents, self.num_agents),
            chunks=True,
            compression=3,
        )
        self._dset_predictions.attrs["dim_names"] = ["batch", "i", "j"]
        self._dset_predictions.attrs["coords_mode__batch"] = "start_and_step"
        self._dset_predictions.attrs["coords__batch"] = [
            write_start,
            max(self._write_predictions_every, 1),
        ]
        self._dset_predictions.attrs["coords_mode__i"] = "trivial"
        self._dset_predictions.attrs["coords_mode__j"] = "trivial"

        # Store the computation time
        self.dset_time = self._output_data_group.create_dataset(
            "computation_time",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["epoch"]
        self.dset_time.attrs["coords_mode__epoch"] = "trivial"

    def epoch(self):
        """Trains the model for a single epoch."""

        # Track the start time
        start_time = time.time()

        # Track the total number of processed time series frames
        counter = 0

        # Make an initial prediction
        predicted_adj_matrix = torch.reshape(
            self.neural_net(torch.flatten(self.training_data[0, 0])),
            (self.num_agents, self.num_agents),
        )

        data_loss = torch.tensor(0.0, requires_grad=True)

        # Process the training data in batches
        for batch_no, batch_idx in enumerate(self.batches[:-1]):
            for i, dset in enumerate(self.training_data):
                current_values = dset[batch_idx].clone()
                current_values.requires_grad_(True)

                # Calculate the current velocities
                current_velocities = (
                    dset[batch_idx].clone() - dset[batch_idx - 1].clone()
                ) / self.ABM.dt

                for ele in range(batch_idx + 1, self.batches[batch_no + 1] + 1):
                    # Solve the ODE
                    new_values = self.ABM.run_single(
                        current_phases=current_values,
                        current_velocities=current_velocities,
                        adjacency_matrix=predicted_adj_matrix,
                        eigen_frequencies=self.eigen_frequencies[i, ele - 1],
                        requires_grad=True,
                    )

                    # Calculate loss on the data
                    data_loss = data_loss + self.loss_function(
                        new_values, dset[ele]
                    ) / (self.batches[batch_no + 1] - batch_idx)

                    counter += 1

                    if counter % self.batch_size == 0:
                        # Enforce symmetry of the predicted adjacency matrix
                        symmetry_loss = self.loss_function(
                            predicted_adj_matrix,
                            torch.transpose(predicted_adj_matrix, 0, 1),
                        )

                        # Penalise the trace (which cannot be learned). Since the torch.trace function is not yet
                        # fully compatible with Apple Silicon GPUs, we must manually calculate it.
                        trace_loss = torch.sum(predicted_adj_matrix.diag())

                        # Add losses
                        loss = data_loss + symmetry_loss + trace_loss

                        # Power grid only: push the neural network towards the unperturbed adjacency matrix
                        # TODO: This cutoff should be automatically calculated
                        if self._time < self.cut_off_time:
                            loss = loss + 10 * self.loss_function(
                                predicted_adj_matrix, self.true_network
                            )

                        # Perform a gradient descent step
                        loss.backward()
                        self.neural_net.optimizer.step()
                        self.neural_net.optimizer.zero_grad()

                        # Store the losses
                        self.current_prediction_loss = data_loss.clone().detach()
                        self.current_symmetry_loss = symmetry_loss.clone().detach()
                        self.current_trace_loss = trace_loss.clone().detach()
                        self.current_total_loss = loss.clone().detach()

                        # Store the current prediction
                        self.current_adjacency_matrix = (
                            predicted_adj_matrix.clone().detach()
                        )

                        # Store the prediction error, if applicable
                        self.current_prediction_error = (
                            torch.nn.functional.l1_loss(
                                self.true_network, self.current_adjacency_matrix
                            )
                            if self.true_network is not None
                            else np.nan
                        )

                        # Write the data and the predictions
                        self._time += 1
                        self.write_data()
                        self.write_predictions()

                        # Make a new prediction
                        predicted_adj_matrix = torch.reshape(
                            self.neural_net(
                                torch.flatten(dset[self.batches[batch_no + 1]])
                            ),
                            (self.num_agents, self.num_agents),
                        )

                        # Wipe the loss
                        del data_loss
                        data_loss = torch.tensor(0.0, requires_grad=True)

                        # Update the current phases and phase velocities to the true values
                        current_velocities = dset[ele] - dset[ele - 1]
                        current_values = dset[ele]

                    else:
                        # Update the velocities
                        current_velocities = (new_values - current_values) / self.ABM.dt

                        current_values = new_values.clone().detach()

        self.dset_time.resize(self.dset_time.shape[0] + 1, axis=0)
        self.dset_time[-1] = time.time() - start_time

    def write_data(self):
        """Write the current losses into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and self._time % self._write_every == 0:
            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1, 0] = self.current_total_loss.cpu().numpy()
            self._dset_loss[-1, 1] = self.current_prediction_loss.cpu().numpy()
            self._dset_loss[-1, 2] = self.current_symmetry_loss.cpu().numpy()
            self._dset_loss[-1, 3] = self.current_trace_loss.cpu().numpy()
            self._dset_loss[-1, 4] = self.current_prediction_error.cpu().numpy()

    def write_predictions(self, *, write_final: bool = False):
        """Write the current predicted adjacency matrix into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._write_predictions_every == -1 and not write_final:
            pass

        else:
            if (
                self._time >= self._write_start
                and self._time % self._write_predictions_every == 0
            ):
                log.debug(f"    Writing prediction data ... ")
                self._dset_predictions.resize(
                    self._dset_predictions.shape[0] + 1, axis=0
                )
                self._dset_predictions[-1, :] = self.current_adjacency_matrix.cpu()