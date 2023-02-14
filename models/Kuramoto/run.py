#!/usr/bin/env python3
import sys
import time
from os.path import dirname as up

import coloredlogs
import h5py as h5
import networkx as nx
import numpy as np
import ruamel.yaml as yaml
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

Kuramoto = import_module_from_path(mod_path=up(up(__file__)), mod_str="Kuramoto")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)


# ----------------------------------------------------------------------------------------------------------------------
# -- Model implementation ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


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
        true_network: torch.tensor,
        num_agents: int,
        write_every: int = 1,
        write_predictions_every: int = 1,
        write_start: int = 1,
        num_steps: int = 3,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        Args:
            name (str): The name of this model instance
            rng (np.random.Generator): The shared RNG
            output_data_group (h5.Group): The output file group to write training data to
            neural_net: The neural network
            ABM: The numerical solver
            num_agents: the number of agents in the model
            write_every: write every iteration
            write_predictions_every: write out predicted parameters every iteration
            write_start: iteration at which to start writing
            num_steps: number of iterations of the ABM
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

        self.num_agents = num_agents
        self.nw_size = num_agents**2
        self.true_network = true_network

        self._write_every = write_every
        self._write_predictions_every = write_predictions_every
        self._write_start = write_start
        self._num_steps = num_steps

        # Current training loss, Frobenius error, and current predictions
        self.current_loss = torch.tensor(0.0)
        self.current_prediction_error = torch.tensor(0.0)
        self.current_adjacency_matrix = torch.zeros(self.num_agents, self.num_agents)

        # Store the neural net training loss
        self._dset_loss = self._output_data_group.create_dataset(
            "Training loss",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["time"]
        self._dset_loss.attrs["coords_mode__time"] = "start_and_step"
        self._dset_loss.attrs["coords__time"] = [write_start, write_every]

        # Store the prediction error
        self._dset_prediction_error = self._output_data_group.create_dataset(
            "Prediction error",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self._dset_prediction_error.attrs["dim_names"] = ["time"]
        self._dset_prediction_error.attrs["coords_mode__time"] = "start_and_step"
        self._dset_prediction_error.attrs["coords__time"] = [write_start, write_every]

        # Store the neural net output, possibly less regularly than the loss
        self._dset_predictions = self._output_data_group.create_dataset(
            "predictions",
            (0, self.num_agents, self.num_agents),
            maxshape=(None, self.num_agents, self.num_agents),
            chunks=True,
            compression=3,
        )
        self._dset_predictions.attrs["dim_names"] = ["time", "i", "j"]
        self._dset_predictions.attrs["coords_mode__time"] = "start_and_step"
        self._dset_predictions.attrs["coords__time"] = [
            write_start,
            max(self._write_predictions_every, 1),
        ]
        self._dset_predictions.attrs["coords_mode__i"] = "trivial"
        self._dset_predictions.attrs["coords_mode__j"] = "trivial"

        # Store the computation time
        self.dset_time = self._output_data_group.create_dataset(
            "computation_time",
            (0, ),
            maxshape=(None, ),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["epoch"]
        self.dset_time.attrs["coords_mode__epoch"] = "trivial"

    def epoch(
        self, *, training_data, eigen_frequencies, batch_size: int, second_order: bool
    ):

        """Trains the model for a single epoch.

        :param training_data: the training data to use
        :param eigen_frequencies: the time series of the nodes' eigenfrequencies
        :param batch_size: the number of training data time frames to process before updating the neural net
            parameters
        :param second_order: whether the dynamics are second order
        """

        # Track the start time
        start_time = time.time()

        # Generate the batch ids
        batches = np.arange(0 if not second_order else 1, training_data.shape[1], batch_size)
        if len(batches) == 1:
            batches = np.append(batches, training_data.shape[1] - 1)
        else:
            if batches[-1] != training_data.shape[1] - 1:
                batches = np.append(batches, training_data.shape[1] - 1)

        # Track the total number of processed time series frames
        counter = 0

        # Make an initial prediction
        predicted_adj_matrix = torch.reshape(
            self.neural_net(torch.flatten(training_data[0, 0])), (self.num_agents, self.num_agents)
        )

        loss = torch.tensor(0.0, requires_grad=True)

        # Process the training data in batches
        for batch_no, batch_idx in enumerate(batches[:-1]):

            for i, dset in enumerate(training_data):

                current_values = dset[batch_idx].clone()
                current_values.requires_grad_(True)

                # Calculate the current velocities if the dynamics are second order
                current_velocities = (dset[batch_idx].clone() - dset[
                    batch_idx - 1].clone()) / self.ABM.dt if second_order else None

                for ele in range(batch_idx + 1, batches[batch_no + 1] + 1):

                    # Solve the ODE
                    new_values = self.ABM.run_single(
                        current_phases=current_values,
                        current_velocities=current_velocities,
                        adjacency_matrix=predicted_adj_matrix,
                        eigen_frequencies=eigen_frequencies[i, ele-1],
                        requires_grad=True,
                    )

                    # Calculate loss
                    loss = loss + self.loss_function(new_values, dset[ele]) / (
                            batches[batch_no + 1] - batch_idx
                    )

                    counter += 1

                    if counter % batch_size == 0:

                        # Penalise the trace (which cannot be learned)
                        loss = loss + torch.trace(predicted_adj_matrix)

                        # Enforce symmetry of the predicted adjacency matrix
                        loss = loss + self.loss_function(
                            predicted_adj_matrix, torch.transpose(predicted_adj_matrix, 0, 1)
                        )

                        # Perform a gradient descent step
                        loss.backward()
                        self.neural_net.optimizer.step()
                        self.neural_net.optimizer.zero_grad()

                        # Write the data
                        self.current_loss = loss.clone().detach()
                        self.current_prediction_error = (
                            torch.nn.functional.l1_loss(self.true_network, predicted_adj_matrix.clone().detach()
                                                         )
                                .numpy()
                                .item()
                        )
                        self.current_adjacency_matrix = predicted_adj_matrix.clone().detach()
                        self._time += 1
                        self.write_data()
                        self.write_predictions()

                        predicted_adj_matrix = torch.reshape(
                            self.neural_net(torch.flatten(dset[batches[batch_no + 1]])),
                            (self.num_agents, self.num_agents)
                        )

                        del loss
                        loss = torch.tensor(0.0, requires_grad=True)

                        if second_order:
                            current_velocities = dset[ele] - dset[ele-1]
                        current_values = dset[ele]

                    else:

                        # Update the velocities, if required
                        if second_order:
                            current_velocities = (new_values - current_values) / self.ABM.dt

                        current_values = new_values.clone().detach()

        self.dset_time.resize(self.dset_time.shape[0] + 1, axis=0)
        self.dset_time[-1] = time.time() - start_time

    def write_data(self):

        """Write the current loss and Frobenius error into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start:

            if self._time % self._write_every == 0:
                self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
                self._dset_loss[-1] = self.current_loss.cpu().numpy()

                self._dset_prediction_error.resize(self._dset_prediction_error.shape[0] + 1, axis=0)
                self._dset_prediction_error[-1] = self.current_prediction_error

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


# ----------------------------------------------------------------------------------------------------------------------
# -- Performing the simulation run -------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    cfg_file_path = sys.argv[1]

    log.note("   Preparing model run ...")
    log.note(f"   Loading config file:\n        {cfg_file_path}")
    with open(cfg_file_path) as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.Loader)
    model_name = cfg.get("root_model_name", "Kuramoto")
    log.note(f"   Model name:  {model_name}")
    model_cfg = cfg[model_name]

    # Select the training device to use
    training_device = model_cfg["Training"].get("device", None)
    if training_device is not None:
        device = training_device
    else:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # Set number of threads to use
    num_threads = model_cfg["Training"].get("num_threads", None)
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    log.info(
        f"   Using '{device}' as training device. Number of threads: {torch.get_num_threads()}"
    )

    log.note("   Creating global RNG ...")
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    log.note(f"   Creating output file at:\n        {cfg['output_path']}")
    h5file = h5.File(cfg["output_path"], mode="w")
    training_data_group = h5file.create_group("training_data")
    output_data_group = h5file.create_group("output_data")

    second_order = model_cfg.get("second_order", False)

    # Get the training data and the network, if synthetic data is used
    log.info("   Generating training data ...")
    training_data, eigen_frequencies, network = Kuramoto.DataGeneration.get_data(
        model_cfg["Data"],
        h5file,
        training_data_group,
        seed=seed,
        device=device,
        second_order=second_order,
    )

    # Initialise the neural net
    num_agents = training_data.shape[2]
    output_size = num_agents**2

    log.info(
        f"   Initializing the neural net; input size: {num_agents}, output size: {output_size} ..."
    )

    net = base.NeuralNet(
        input_size=num_agents, output_size=output_size, **model_cfg["NeuralNet"]
    ).to(device)

    # Set the neural net to an initial state, if given
    if model_cfg["NeuralNet"].get("initial_state", None) is not None:
        net.load_state_dict(torch.load(model_cfg["NeuralNet"].get("initial_state")))
        net.eval()

    # Get the true parameters
    true_parameters = model_cfg["Training"]["true_parameters"]

    # Initialise the ABM
    ABM = Kuramoto.Kuramoto_ABM(
        N=num_agents,
        dt=model_cfg["Data"]["dt"],
        gamma=model_cfg["Data"]["gamma"],
        **true_parameters,
        device=device,
    )

    # Calculate the frequency with which to write out the model predictions
    write_predictions_every = cfg.get("write_predictions_every", cfg["write_every"])
    num_epochs = cfg["num_epochs"]
    batch_size = model_cfg["Training"]["batch_size"]

    # Initialise the model
    model = Kuramoto_NN(
        model_name,
        rng=rng,
        output_data_group=output_data_group,
        num_agents=num_agents,
        neural_net=net,
        loss_function=model_cfg["Training"]["loss_function"],
        true_network=torch.from_numpy(nx.to_numpy_matrix(network)).float(),
        ABM=ABM,
        num_steps=training_data.shape[1],
        write_every=cfg["write_every"],
        write_predictions_every=write_predictions_every,
        write_start=cfg["write_start"],
    )

    log.info(f"   Initialized model '{model_name}'. Now commencing training for {num_epochs} epochs ...")

    # Train the neural net
    for i in range(num_epochs):
        model.epoch(
            training_data=training_data.to(device),
            eigen_frequencies=eigen_frequencies.to(device),
            batch_size=batch_size,
            second_order=second_order,
        )

        log.progress(
            f"   Completed epoch {i + 1} / {num_epochs}; current loss: {model.current_loss}; "
            f"current L1 prediction error:  {model.current_prediction_error}；"
            f"epoch training time: {model.dset_time[-1]} s"
        )

        # Save neural net, if specified
        if model_cfg["NeuralNet"].get("save_to", None) is not None:
            torch.save(net.state_dict(), model_cfg["NeuralNet"].get("save_to"))

    if write_predictions_every == -1:
        model.write_predictions(write_final=True)

    log.info("   Simulation run finished.")

    # If specified, perform an OLS regression on the training data
    if cfg.get("perform_regression", False):
        log.info("   Performing regression ... ")
        Kuramoto.regression(
            training_data,
            eigen_frequencies,
            h5file,
            model_cfg["Data"]["synthetic_data"]["dt"],
            second_order=second_order,
            gamma=model_cfg["Data"]["gamma"]
        )

    log.info("   Wrapping up ...")

    h5file.close()

    log.success("   All done.")
