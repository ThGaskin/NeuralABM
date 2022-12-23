#!/usr/bin/env python3
import sys
import time
from os.path import dirname as up

import coloredlogs
import h5py as h5
import numpy as np
import ruamel.yaml as yaml
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

HW = import_module_from_path(mod_path=up(up(__file__)), mod_str="HarrisWilsonNW")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)


# -----------------------------------------------------------------------------
# -- Model implementation -----------------------------------------------------
# -----------------------------------------------------------------------------


class HarrisWilson_NN:
    def __init__(
        self,
        name: str,
        *,
        rng: np.random.Generator,
        training_data_group: h5.Group,
        neural_net: base.NeuralNet,
        loss_function: dict,
        ABM: HW.HarrisWilsonABM,
        true_network: torch.tensor = None,
        n_or: int,
        n_dest: int,
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
            training_data_group (h5.Group): The output file group to write training data to
            pred_nw_group (h5.Group): (optional) The output file group to write the network edges to
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
        self._training_group = training_data_group
        self._rng = rng

        self.ABM = ABM
        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        self.n_origin = n_or
        self.n_dest = n_dest
        self.nw_size = n_or * n_dest
        self.true_network = true_network

        self._write_every = write_every
        self._write_predictions_every = write_predictions_every
        self._write_start = write_start
        self._num_steps = num_steps

        # Current training loss, Frobenius error, and current predictions
        self.current_loss = torch.tensor(0.0)
        self.current_frob_error = torch.tensor(0.0)
        self.current_predictions = torch.zeros(self.nw_size)
        self.current_adjacency_matrix = torch.zeros(self.n_origin, self.n_dest)

        # Store the neural net training loss
        self._dset_loss = self._training_group.create_dataset(
            "training_loss",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["epoch"]
        self._dset_loss.attrs["coords_mode__epoch"] = "start_and_step"
        self._dset_loss.attrs["coords__epoch"] = [write_start, write_every]

        # Store the prediction error
        self._dset_frob_error = self._training_group.create_dataset(
            "frobenius_error",
            (0, ),
            maxshape=(None, ),
            chunks=True,
            compression=3,
        )
        self._dset_frob_error.attrs["dim_names"] = ["epoch"]
        self._dset_frob_error.attrs["coords_mode__epoch"] = "start_and_step"
        self._dset_frob_error.attrs["coords__epoch"] = [write_start, write_every]

        # Store the neural net output, possibly less regularly than the loss
        self._dset_predictions = self._training_group.create_dataset(
            "predictions",
            (0, self.n_origin, self.n_dest),
            maxshape=(None, self.n_origin, self.n_dest),
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

        self.dset_time = self._training_group.create_dataset(
            "computation_time",
            (0, 1),
            maxshape=(None, 1),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["epoch", "training_time"]
        self.dset_time.attrs["coords_mode__epoch"] = "trivial"
        self.dset_time.attrs["coords_mode__training_time"] = "trivial"

    def epoch(self, *, training_data, origin_sizes, batch_size: int, epsilon: float = None,
                dt: float = None):

        """Trains the model for a single epoch.

        :param training_data: the training data
        :param batch_size: (optional) the number of passes (batches) over the training data before conducting a gradient descent
                step
        :param epsilon: (optional) the epsilon value to use during training
        :param dt: (optional) the time differential to use during training
        """

        self.current_loss = torch.tensor(0.0)
        self.current_frob_error = torch.tensor(0.0)

        start_time = time.time()

        for i, dset in enumerate(training_data):

            batches = np.arange(0, len(dset), batch_size)
            if len(batches) == 1:
                batches = np.append(batches, len(dset) - 1)
            else:
                if batches[-1] != len(dset) - 1:
                    batches = np.append(batches, len(dset)-1)

            for batch_no, batch_idx in enumerate(batches[:-1]):

                predicted_parameters = self.neural_net(torch.flatten(dset[batch_idx]))
                pred_adj_matrix = torch.reshape(
                    predicted_parameters, (self.n_origin, self.n_dest)
                )
                current_values = dset[batch_idx].clone()
                current_values.requires_grad_(True)

                loss = torch.tensor(0.0, requires_grad=True)

                for ele in range(batch_idx + 1, batches[batch_no + 1] + 1):

                    # Solve the ODE
                    current_values = self.ABM.run_single(
                        adjacency_matrix=pred_adj_matrix,
                        origin_sizes=origin_sizes[i][ele-1],
                        curr_vals=current_values,
                        epsilon=epsilon,
                        dt=dt,
                    )

                    # Calculate the loss
                    loss = loss + self.loss_function(current_values, dset[ele])

                loss.backward()
                self.neural_net.optimizer.step()
                self.neural_net.optimizer.zero_grad()
                self.current_loss += loss.clone().detach().numpy().item()
                self.current_frob_error += torch.nn.functional.mse_loss(
                    self.true_network,
                    pred_adj_matrix
                ).clone().detach().numpy().item()
                self.current_predictions = predicted_parameters.clone().detach()
                self.current_adjacency_matrix = pred_adj_matrix.clone().detach()

        self._time += 1
        self.write_data()
        self.write_predictions()

        self.dset_time.resize(self.dset_time.shape[0] + 1, axis=0)
        self.dset_time[-1, :] = time.time() - start_time

    def write_data(self):
        """Write the current loss and predicted network size into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start:

            if self._time % self._write_every == 0:
                self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
                self._dset_loss[-1] = self.current_loss

                self._dset_frob_error.resize(self._dset_frob_error.shape[0] + 1, axis=0)
                self._dset_frob_error[-1] = self.current_frob_error

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
                self._dset_predictions[-1, :] = self.current_adjacency_matrix

# ----------------------------------------------------------------------------------------------------------------------
# -- Performing the simulation run -------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    cfg_file_path = sys.argv[1]

    log.note("   Preparing model run ...")
    log.note(f"   Loading config file:\n        {cfg_file_path}")
    with open(cfg_file_path) as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.Loader)
    model_name = cfg.get("root_model_name", "HarrisWilsonNW")
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
    neural_net_group = h5file.create_group("output_data")

    # Get the training data and the network, if synthetic data is used
    log.info("   Generating training data ...")

    # Get the datasets
    or_sizes, dest_sizes, network, time_series = HW.get_HW_data(
        model_cfg["Data"], h5file, training_data_group, device=device
    )

    N_origin = or_sizes.shape[-2]
    N_dest = dest_sizes.shape[-2]

    log.info(
        f"   Initializing the neural net; input size: {N_dest}, output size: {N_origin * N_dest} ..."
    )
    net = base.NeuralNet(
        input_size=N_dest, output_size=N_origin * N_dest, **model_cfg["NeuralNet"]
    )

    # Get the true parameters
    true_parameters = model_cfg["Training"]["true_parameters"]

    # Initialise the ABM
    ABM = HW.HarrisWilsonABM(
        N=N_origin,
        M=N_dest,
        device=device,
        **true_parameters
    )

    # Calculate the frequency with which to write out the model predictions
    write_predictions_every = cfg.get("write_predictions_every", cfg["write_every"])
    num_epochs = cfg["num_epochs"]
    batch_size = model_cfg["Training"]["batch_size"]

    # Initialise the model
    model = HarrisWilson_NN(
        model_name,
        rng=rng,
        training_data_group=neural_net_group,
        neural_net=net,
        n_or=N_origin,
        n_dest=N_dest,
        loss_function=model_cfg["Training"]["loss_function"],
        true_network=network,
        ABM=ABM,
        write_every=cfg["write_every"],
        write_predictions_every=write_predictions_every,
        write_start=cfg["write_start"],
    )

    log.info(f"   Initialized model '{model_name}'.")

    # Train the neural net
    log.info(f"   Now commencing training for {num_epochs} epochs ...")

    for i in range(num_epochs):
        model.epoch(training_data=dest_sizes, origin_sizes=or_sizes, batch_size=batch_size)

        log.progress(
            f"   Completed epoch {i + 1} / {num_epochs}; current loss: {model.current_loss}, "
            f"current Frobenius error: {model.current_frob_error}"
        )

    if write_predictions_every == -1:
        model.write_predictions(write_final=True)

    log.info("   Simulation run finished. Generating prediction ... ")

    predicted_ts = torch.flatten(ABM.run(
        init_data=time_series[0][0],
        adjacency_matrix=model.current_adjacency_matrix,
        n_iterations=time_series.shape[1]-1,
        origin_sizes=or_sizes[0],
        generate_time_series=True,
    ), start_dim=-2)

    dset_time_series = neural_net_group.create_dataset(
        "predicted_time_series",
        predicted_ts.shape,
        maxshape=predicted_ts.shape,
        chunks=True,
        compression=3,
    )
    dset_time_series.attrs["dim_names"] = ["time", "zone_id"]
    dset_time_series.attrs["coords_mode__time"] = "trivial"
    dset_time_series.attrs["coords_mode__zone_id"] = "trivial"

    dset_time_series[:, :] = predicted_ts

    log.info("   Wrapping up ...")

    h5file.close()

    log.success("   All done.")
