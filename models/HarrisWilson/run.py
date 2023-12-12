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

HW = import_module_from_path(mod_path=up(up(__file__)), mod_str="HarrisWilson")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)

# ----------------------------------------------------------------------------------------------------------------------
# -- Model implementation ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class HarrisWilson_NN:
    def __init__(
        self,
        *,
        rng: np.random.Generator,
        h5group: h5.Group,
        neural_net: base.NeuralNet,
        loss_function: dict,
        ABM: HW.HarrisWilsonABM,
        to_learn: list,
        write_every: int = 1,
        write_start: int = 1,
        training_data: torch.Tensor,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        Args:
            rng (np.random.Generator): The shared RNG
            h5group (h5.Group): The output file group to write data to
            neural_net: The neural network
            loss_function (dict): the loss function to use
            ABM: The numerical solver
            to_learn: the list of parameter names to learn
            write_every: write every iteration
            write_start: iteration at which to start writing
            training_data: training data used to train the neural net

        """
        self._h5group = h5group
        self._rng = rng

        # The numerical solver
        self._ABM = ABM

        # Initialise neural net, loss tracker and prediction tracker
        self._neural_net = neural_net
        self._neural_net.optimizer.zero_grad()
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )
        self._current_loss = torch.tensor(0.0, requires_grad=False)
        self._current_predictions = torch.stack(
            [torch.tensor(0.0, requires_grad=False)] * len(to_learn)
        )

        # Setup chunked dataset to store the state data in
        self._dset_loss = self._h5group.create_dataset(
            "loss",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs["dim_names"] = ["epoch"]
        self._dset_loss.attrs["coords_mode__epoch"] = "start_and_step"
        self._dset_loss.attrs["coords__epoch"] = [write_start, write_every]

        self.dset_time = self._h5group.create_dataset(
            "computation_time",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self.dset_time.attrs["dim_names"] = ["epoch"]
        self.dset_time.attrs["coords_mode__epoch"] = "trivial"

        # Write the parameter predictions after every epoch
        self.dset_parameters = self._h5group.create_dataset(
            "parameters",
            (0, len(to_learn)),
            maxshape=(None, len(to_learn)),
            chunks=True,
            compression=3,
        )
        self.dset_parameters.attrs["dim_names"] = ["epoch", "parameter"]
        self.dset_parameters.attrs["coords_mode__epoch"] = "start_and_step"
        self.dset_parameters.attrs["coords__epoch"] = [write_start, write_every]
        self.dset_parameters.attrs["coords_mode__parameter"] = "values"
        self.dset_parameters.attrs["coords__parameter"] = to_learn

        # The training data
        self.training_data = training_data

        # Epochs processed
        self._time = 0
        self._write_every = write_every
        self._write_start = write_start

    def epoch(
        self,
        *,
        epsilon: float = None,
        dt: float = None,
        **__,
    ):
        """Trains the model for a single epoch.

        :param epsilon: (optional) the epsilon value to use during training
        :param dt: (optional) the time differential to use during training
        :param __: other parameters (ignored)
        """

        # Track the epoch training time
        start_time = time.time()

        predicted_parameters = self._neural_net(torch.flatten(self.training_data[0]))
        predicted_data = ABM.run(
            input_data=predicted_parameters,
            init_data=self.training_data[0],
            epsilon=epsilon,
            dt=dt,
            n_iterations=self.training_data.shape[0],
            generate_time_series=False,
        )

        loss = self.loss_function(predicted_data, self.training_data[-1])

        loss.backward()
        self._neural_net.optimizer.step()
        self._neural_net.optimizer.zero_grad()
        self._time += 1
        self._current_loss = (
            loss.clone().detach().cpu().numpy().item() / self.training_data.shape[0]
        )
        self._current_predictions = predicted_parameters.clone().detach().cpu()
        self.write_data()

        # Write the epoch training time (wall clock time)
        self.dset_time.resize(self.dset_time.shape[0] + 1, axis=0)
        self.dset_time[-1] = time.time() - start_time

    def write_data(self):
        """Write the current state into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0):
            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1] = self._current_loss

            self.dset_parameters.resize(self.dset_parameters.shape[0] + 1, axis=0)
            self.dset_parameters[-1, :] = self._current_predictions


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
    model_name = cfg.get("root_model_name", "HarrisWilson")
    log.note(f"   Model name:  {model_name}")
    model_cfg = cfg[model_name]

    # Select the training device and number of threads to use
    device = model_cfg["Training"].pop("device", None)
    if device is None:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    num_threads = model_cfg["Training"].get("num_threads", None)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    log.info(
        f"   Using '{device}' as training device. Number of threads: {torch.get_num_threads()}"
    )

    # Get the random number generator
    log.note("   Creating global RNG ...")
    rng = np.random.default_rng(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.random.manual_seed(cfg["seed"])

    log.note(f"   Creating output file at:\n        {cfg['output_path']}")
    h5file = h5.File(cfg["output_path"], mode="w")
    h5group = h5file.create_group(model_name)

    # Get the datasets
    or_sizes, dest_sizes, network = HW.get_HW_data(
        model_cfg["Data"], h5file, h5group, device=device
    )

    # Set up the neural net
    log.info("   Initializing the neural net ...")
    net = base.NeuralNet(
        input_size=dest_sizes.shape[1],
        output_size=len(model_cfg["Training"]["to_learn"]),
        **model_cfg["NeuralNet"],
    ).to(device)

    # Set up the numerical solver
    log.info("   Initializing the numerical solver ...")
    true_parameters = model_cfg["Training"].get("true_parameters", None)
    ABM = HW.HarrisWilsonABM(
        origin_sizes=or_sizes,
        network=network,
        true_parameters=true_parameters,
        M=dest_sizes.shape[1],
        device=device,
    )

    model = HarrisWilson_NN(
        rng=rng,
        h5group=h5group,
        neural_net=net,
        ABM=ABM,
        write_every=cfg["write_every"],
        write_start=cfg["write_start"],
        training_data=dest_sizes,
        **model_cfg["Training"],
    )
    log.info(f"   Initialized model '{model_name}'.")

    # Train the neural net
    num_epochs = cfg["num_epochs"]
    log.info(f"   Now commencing training for {num_epochs} epochs ...")

    for _ in range(num_epochs):
        model.epoch(**model_cfg["Training"])
        log.progress(
            f"   Completed epoch {_+1} / {num_epochs}; "
            f"   current loss: {model._current_loss}"
        )

    log.info("   Simulation run finished.")
    log.note("   Wrapping up ...")
    h5file.close()

    log.success("   All done.")
