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
        *,
        rng: np.random.Generator,
        output_data_group: h5.Group,
        neural_net: base.NeuralNet,
        loss_function: dict,
        ABM: HW.HarrisWilsonABM,
        training_data: torch.Tensor,
        origin_sizes: torch.Tensor,
        true_network: torch.Tensor = None,
        write_every: int = 1,
        write_predictions_every: int = 1,
        write_start: int = 1,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        :param rng (np.random.Generator): The shared RNG
        :param output_data_group (h5.Group): The output file group to write training data to
        :param neural_net: The neural network
        :param loss_function: dictionary of loss function and properties
        :param ABM: The numerical solver
        :param training_data: data to use for training; shape is (training_set_size, n_dest, n_steps)
        :param origin_sizes: time series of origin sizes; shape is (training_set_size, n_origin, n_steps)
        :param true_network: (optional) true network
        :param write_every: write every iteration
        :param write_predictions_every: write out predicted parameters every iteration
        :param write_start: iteration at which to start writing
        """
        self._time = 0
        self._output_data_group = output_data_group
        self._rng = rng

        self.ABM = ABM
        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        self.n_origin = origin_sizes.shape[2]
        self.n_dest = training_data.shape[2]
        self.nw_size = self.neural_net.output_dim
        self.true_network = true_network

        self._write_every = write_every
        self._write_predictions_every = write_predictions_every
        self._write_start = write_start

        # Current training loss, Prediction error, and current predictions
        self.current_loss = torch.tensor(0.0, dtype=torch.float)
        self.current_prediction_error = torch.tensor(0.0, dtype=torch.float)
        self.current_adjacency_matrix = torch.zeros(self.n_origin, self.n_dest)

        # Store the neural net training loss and error
        self._dset_loss = self._output_data_group.create_dataset(
            "Loss",
            (0, 2),
            maxshape=(None, 2),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["batch", "kind"]
        self._dset_loss.attrs["coords_mode__batch"] = "start_and_step"
        self._dset_loss.attrs["coords__batch"] = [write_start, write_every]
        self._dset_loss.attrs["coords_mode__kind"] = "values"
        self._dset_loss.attrs["coords__kind"] = ["Training loss", "L1 prediction error"]

        # Store the neural net output, possibly less regularly than the loss
        self._dset_predictions = self._output_data_group.create_dataset(
            "predictions",
            (0, self.n_origin, self.n_dest),
            maxshape=(None, self.n_origin, self.n_dest),
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

        self.dset_time = self._output_data_group.create_dataset(
            "computation_time",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["epoch"]
        self.dset_time.attrs["coords_mode__epoch"] = "trivial"

        # Store training data and generate the batch ids
        self.training_data = training_data
        self.origin_sizes = origin_sizes
        batches = np.arange(0, self.training_data.shape[1], batch_size)
        if len(batches) == 1:
            batches = np.append(batches, training_data.shape[1] - 1)
        else:
            if batches[-1] != training_data.shape[1] - 1:
                batches = np.append(batches, training_data.shape[1] - 1)
        self.batches = batches

    def epoch(self, *, epsilon: float = None, dt: float = None, **__):
        """Trains the model for a single epoch.

        :param epsilon: (optional) the epsilon value to use during training
        :param dt: (optional) the time differential to use during training
        """

        start_time = time.time()

        # Track the number of gradient descent updates
        counter = 0

        # Make an initial prediction
        pred_adj_matrix = torch.reshape(
            self.neural_net(torch.flatten(self.training_data[0, 0])),
            (self.n_origin, self.n_dest),
        )

        loss = torch.tensor(0.0, requires_grad=True)

        for batch_no, batch_idx in enumerate(self.batches[:-1]):
            for i, dset in enumerate(self.training_data):
                current_values = dset[batch_idx].clone()
                current_values.requires_grad_(True)

                for ele in range(batch_idx + 1, self.batches[batch_no + 1] + 1):
                    # Solve the ODE
                    current_values = self.ABM.run_single(
                        adjacency_matrix=pred_adj_matrix,
                        origin_sizes=self.origin_sizes[i][ele - 1],
                        curr_vals=current_values,
                        epsilon=epsilon,
                        dt=dt,
                    )

                    # Calculate the loss
                    loss = loss + (self.loss_function(current_values, dset[ele])) / (
                        self.batches[batch_no + 1] - batch_idx + 1
                    )

                    counter += 1

                if counter % batch_size == 0:
                    # Perform a gradient descent step
                    loss.backward()
                    self.neural_net.optimizer.step()
                    self.neural_net.optimizer.zero_grad()
                    self.current_loss = loss.clone().detach()

                    # Normalise rows to make matrices comparable
                    row_sum = (
                        torch.sum(pred_adj_matrix, dim=1, keepdim=True).clone().detach()
                    )
                    self.current_prediction_error = (
                        (
                            torch.nn.functional.l1_loss(
                                self.true_network, pred_adj_matrix / row_sum
                            )
                        )
                        .clone()
                        .detach()
                    )
                    self.current_adjacency_matrix = (
                        pred_adj_matrix.clone().detach() / row_sum
                    )

                    # Make a new prediction
                    pred_adj_matrix = torch.reshape(
                        self.neural_net(
                            torch.flatten(dset[self.batches[batch_no + 1]])
                        ),
                        (self.n_origin, self.n_dest),
                    )

                    # Wipe the loss
                    del loss
                    loss = torch.tensor(0.0, requires_grad=True)

        self._time += 1
        self.write_data()
        self.write_predictions()

        self.dset_time.resize(self.dset_time.shape[0] + 1, axis=0)
        self.dset_time[-1] = time.time() - start_time

    def write_data(self):
        """Write the current loss and predicted network size into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start:
            if self._time % self._write_every == 0:
                self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
                self._dset_loss[-1, 0] = self.current_loss.cpu().numpy()
                self._dset_loss[-1, 1] = self.current_prediction_error.cpu().numpy()

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
    yamlc = yaml.YAML(typ="safe")
    with open(cfg_file_path) as cfg_file:
        cfg = yamlc.load(cfg_file)
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
    or_sizes, dest_sizes, training_or_sizes, network, time_series = HW.get_data(
        model_cfg["Data"], h5file, training_data_group, device=device
    )

    N_origin, N_dest = or_sizes.shape[2], dest_sizes.shape[2]

    log.info(
        f"   Initializing the neural net; input size: {N_dest}, output size: {N_origin * N_dest} ..."
    )
    net = base.NeuralNet(
        input_size=N_dest, output_size=N_origin * N_dest, **model_cfg["NeuralNet"]
    )

    # Set the neural net to an initial state, if given
    if model_cfg["NeuralNet"].get("initial_state", None) is not None:
        net.load_state_dict(torch.load(model_cfg["NeuralNet"].get("initial_state")))
        net.eval()

    # Get the true parameters
    true_parameters = model_cfg["Training"]["true_parameters"]

    # Initialise the ABM
    ABM = HW.HarrisWilsonABM(N=N_origin, M=N_dest, device=device, **true_parameters)

    # Calculate the frequency with which to write out the model predictions
    write_predictions_every = cfg.get("write_predictions_every", cfg["write_every"])
    num_epochs = cfg["num_epochs"]
    batch_size = model_cfg["Training"]["batch_size"]

    # Initialise the model
    model = HarrisWilson_NN(
        rng=rng,
        output_data_group=neural_net_group,
        neural_net=net,
        loss_function=model_cfg["Training"]["loss_function"],
        training_data=dest_sizes,
        origin_sizes=training_or_sizes,
        batch_size=batch_size,
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
        model.epoch(**model_cfg["Training"])

        # Print progress message
        log.progress(
            f"   Completed epoch {i + 1} / {num_epochs} in {model.dset_time[-1]} s \n"
            f"            ----------------------------------------------------------------- \n"
            f"            Loss: {model.current_loss} \n"
            f"            L1 prediction error: {model.current_prediction_error} \n"
        )

    if write_predictions_every == -1:
        model.write_predictions(write_final=True)

    log.info("   Training complete. Wrapping up ...")
    h5file.close()
    log.success("   All done.")
