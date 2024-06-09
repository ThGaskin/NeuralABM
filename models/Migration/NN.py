import sys
from os.path import dirname as up

import coloredlogs
import h5py as h5
import numpy as np
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)


class Migration_NN:
    def __init__(
        self,
        *,
        rng: np.random.Generator,
        h5group: h5.Group,
        neural_net: base.NeuralNet,
        loss_function: dict,
        write_every: int = 1,
        write_start: int = 1,
        write_predictions_every: int = 1,
        write_predictions_start: int = 1,
        training_data: dict,
        training_sigma: float = 0,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        :param rng (np.random.Generator): The shared RNG
        :param h5group (h5.Group): The output file group to write data to
        :param neural_net: The neural network
        :param loss_function (dict): the loss function to use
        :param training_data: dictionary of stock data and net migration data
        :param write_every: write every iteration
        :param write_start: iteration at which to start writing
        :param batch_size: epoch batch size: instead of calculating the entire time series,
            only a subsample of length batch_size can be used. The likelihood is then
            scaled up accordingly.
        """
        self._h5group = h5group
        self._rng = rng

        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        self.current_loss = torch.tensor(0.0)

        self.net_migration_data = torch.from_numpy(training_data["net_migration"].data).float()

        # Number of years and countries
        self.L = self.net_migration_data.shape[0]
        self.N = self.net_migration_data.shape[1]

        stock_data = torch.from_numpy(training_data["stock_data"].data).float()
        self.stock_data = torch.diff(stock_data, dim=0).reshape(-1, self.N ** 2)
        self.total_population = torch.from_numpy(training_data["total_population"].data).float()

        # Get the mask for the stock data differences
        self.mask = ~torch.isnan(self.stock_data)

        # Current flow table
        self.current_predictions = torch.zeros(self.L, 2, self.N, self.N)

        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        # Write the loss after every batch
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

        # Write the computation time of every epoch
        self.dset_time = self._h5group.create_dataset(
            "computation_time",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["epoch"]
        self.dset_time.attrs["coords_mode__epoch"] = "trivial"

        # Write the parameter predictions after every batch
        self.dset_flow_table = self._h5group.create_dataset(
            "flow_table",
            (0, self.L, 2, self.N, self.N),
            maxshape=(None, self.L, 2, self.N, self.N),
            dtype=int,
            chunks=True,
            compression=3,
        )

        self.dset_flow_table.attrs["dim_names"] = ["batch", "Year", "Direction", "Origin ISO", "Destination ISO"]
        self.dset_flow_table.attrs["coords_mode__batch"] = "start_and_step"
        self.dset_flow_table.attrs["coords__batch"] = [write_predictions_start, write_predictions_every]
        self.dset_flow_table.attrs["coords_mode_Year"] = "values"
        self.dset_flow_table.attrs["coords__Year"] = training_data["net_migration"].coords["Year"].data
        self.dset_flow_table.attrs["coords_mode_Direction"] = "values"
        self.dset_flow_table.attrs["coords__Direction"] = ["Out", "In"]
        self.dset_flow_table.attrs["coords_mode_Origin ISO"] = "values"
        self.dset_flow_table.attrs["coords__Origin ISO"] = [str(x) for x in training_data["net_migration"].coords["Country ISO"].data]
        self.dset_flow_table.attrs["coords_mode_Destination ISO"] = "values"
        self.dset_flow_table.attrs["coords__Destination ISO"] = [str(x) for x in training_data["net_migration"].coords["Country ISO"].data]

        # Noise perturbation during training
        self.training_sigma = training_sigma

        # Batches processed
        self._time = 0
        self._write_every = write_every
        self._write_start = write_start
        self._write_predictions_every = write_predictions_every
        self._write_predictions_start = write_predictions_start

    def epoch(self):
        """
        """

        epoch_loss = []

        # Process the training data in batches
        for b in range(len(self.stock_data)):

            # Make a prediction, of shape (t: N_Years, k: 2, i: N, j: N). k=0 represents the flow of citizens of
            # i from i to j, k=1 represents the flow of citizens of i from j to i
            prediction = self.neural_net(
                self.net_migration_data[5*b:5*(b+1)]
            ).reshape(-1, 2, self.N, self.N)

            # Perturb the prediction by sigma
            if self.training_sigma > 0:
                prediction = prediction * torch.normal(1.0, self.training_sigma, prediction.shape)

            # Calculate the total predicted flow across five years (inflow minus outflow)
            predicted_stock_diff = (prediction[:, 0, :] - prediction[:, 1, :]).sum(dim=0).reshape(self.N ** 2)

            # Calculate the total outflow and inflow
            predicted_outflow = (prediction[:, 0, :] + prediction.transpose(-2, -1)[:, 1, :]).sum(dim=-1)
            predicted_inflow = (prediction.transpose(-2, -1)[:, 0, :] + prediction[:, 1, :]).sum(dim=-1)

            # Calculate the predicted net migration (inflow - outflow)
            predicted_net_migration = predicted_inflow - predicted_outflow

            # Loss = loss on stock data + loss on net migration + trace + total outflow must be less than population
            # of country
            loss = torch.nn.functional.mse_loss(
                predicted_stock_diff[self.mask[b]], self.stock_data[b][self.mask[b]]
            ) + torch.nn.functional.mse_loss(
                predicted_net_migration, self.net_migration_data[5 * b: 5*(b + 1)]
            ) + 10000 * torch.diagonal(
                prediction.sum(dim=1), dim1=-2, dim2=-1
            ).sum() + torch.relu(
                predicted_outflow - self.total_population[5 * b: 5*(b + 1)]
            ).sum() + torch.relu(
                predicted_inflow - self.total_population[5 * b: 5*(b + 1)]
            ).sum()

            # Perform gradient descent step
            loss.backward()
            self.neural_net.optimizer.step()
            self.neural_net.optimizer.zero_grad()

            epoch_loss.append(loss.clone().detach().numpy())

            # Save the current predictions
            self.current_predictions[5*b:5*(b+1)] = prediction.detach()

        # 2020 and 2021 not covered by stock -- learn separately
        prediction = self.neural_net(self.net_migration_data[-2:]).reshape(-1, 2, self.N, self.N)

        # Perturb the prediction by 1%
        prediction = prediction * torch.normal(1.0, self.training_sigma, prediction.shape)

        # Calculate the total outflow and inflow
        predicted_outflow = (prediction[:, 0, :] + prediction.transpose(-2, -1)[:, 1, :]).sum(dim=-1)
        predicted_inflow = (prediction.transpose(-2, -1)[:, 0, :] + prediction[:, 1, :]).sum(dim=-1)

        # Calculate the predicted net migration (inflow - outflow)
        predicted_net_migration = predicted_inflow - predicted_outflow

        # Loss = loss on stock data + loss on net migration + trace
        loss = (
                torch.nn.functional.mse_loss(
                    predicted_net_migration, self.net_migration_data[-2:]
                ) + 10000 * torch.diagonal(
                    prediction.sum(dim=1), dim1=-2, dim2=-1).sum() +
                torch.relu(
                    predicted_outflow - self.total_population[-2:]
                ).sum() +
                torch.relu(
                    predicted_inflow - self.total_population[-2:]
                ).sum()
        )

        loss.backward()
        self.neural_net.optimizer.step()
        self.neural_net.optimizer.zero_grad()

        epoch_loss.append(loss.clone().detach().numpy())

        # Save the current predictions
        self.current_predictions[-2:] = prediction.detach()

        self.current_loss = np.mean(epoch_loss)
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
        if self._time >= self._write_predictions_start and (self._time % self._write_predictions_every == 0):
            self.dset_flow_table.resize(self.dset_flow_table.shape[0] + 1, axis=0)
            self.dset_flow_table[-1, :] = self.current_predictions
