#!/usr/bin/env python3
import sys
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

SIR = import_module_from_path(mod_path=up(up(__file__)), mod_str="SIR")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)
# -----------------------------------------------------------------------------
# -- Model implementation -----------------------------------------------------
# -----------------------------------------------------------------------------


class SIR_NN:
    def __init__(
        self,
        *,
        rng: np.random.Generator,
        h5group: h5.Group,
        neural_net: base.NeuralNet,
        loss_function: dict,
        to_learn: list,
        true_parameters: dict = {},
        write_every: int = 1,
        write_start: int = 1,
        training_data: torch.Tensor,
        batch_size: int,
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
            training_data: the training data to use
            batch_size: epoch batch size: instead of calculating the entire time series,
                only a subsample of length batch_size can be used. The likelihood is then
                scaled up accordingly.
            true_parameters: the dictionary of true parameters
            write_every: write every iteration
            write_start: iteration at which to start writing
        """
        self._h5group = h5group
        self._rng = rng

        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        self.current_loss = torch.tensor(0.0)

        self.to_learn = {key: idx for idx, key in enumerate(to_learn)}
        self.true_parameters = {
            key: torch.tensor(val, dtype=torch.float)
            for key, val in true_parameters.items()
        }
        self.current_predictions = torch.zeros(len(to_learn))

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

        # The training data and batch ids
        self.training_data = training_data

        batches = np.arange(0, training_data.shape[0], batch_size)
        if len(batches) == 1:
            batches = np.append(batches, training_data.shape[0] - 1)
        else:
            if batches[-1] != training_data.shape[0] - 1:
                batches = np.append(batches, training_data.shape[0] - 1)
        self.batches = batches

        # Batches processed
        self._time = 0
        self._write_every = write_every
        self._write_start = write_start

    def epoch(self):
        """
        An epoch is a pass over the entire dataset. The dataset is processed in batches, where B < L is the batch
        number. After each batch, the parameters of the neural network are updated. For example, if L = 100 and
        B = 50, two passes are made over the dataset -- one over the first 50 steps, and one
        over the second 50. The entire time series is processed, even if L is not divisible into equal segments of
        length B. For instance, is B is 30, the time series is processed in 3 steps of 30 and one of 10.

        """

        # Process the training data in batches
        for batch_no, batch_idx in enumerate(self.batches[:-1]):
            predicted_parameters = self.neural_net(
                torch.flatten(self.training_data[batch_idx])
            )

            # Get the parameters: infection rate, recovery time, noise variance
            p = (
                predicted_parameters[self.to_learn["p_infect"]]
                if "p_infect" in self.to_learn.keys()
                else self.true_parameters["p_infect"]
            )
            t = (
                10 * predicted_parameters[self.to_learn["t_infectious"]]
                if "t_infectious" in self.to_learn.keys()
                else self.true_parameters["t_infectious"]
            )
            sigma = (
                predicted_parameters[self.to_learn["sigma"]]
                if "sigma" in self.to_learn.keys()
                else self.true_parameters["sigma"]
            )

            current_densities = self.training_data[batch_idx].clone()
            current_densities.requires_grad_(True)

            loss = torch.tensor(0.0, requires_grad=True)

            for ele in range(batch_idx + 1, self.batches[batch_no + 1] + 1):
                # Recovery rate
                tau = 1 / t * torch.sigmoid(1000 * (ele / t - 1))

                # Random noise
                w = torch.normal(torch.tensor(0.0), torch.tensor(1.0))

                # Solve the ODE
                current_densities = torch.clip(
                    current_densities
                    + torch.stack(
                        [
                            (-p * current_densities[0] - sigma * w)
                            * current_densities[1],
                            (p * current_densities[0] + sigma * w - tau)
                            * current_densities[1],
                            tau * current_densities[1],
                        ]
                    ),
                    0.0,
                    1.0,
                )

                # Calculate loss
                loss = loss + self.loss_function(
                    current_densities, self.training_data[ele]
                ) * (
                    self.training_data.shape[0]
                    / (self.batches[batch_no + 1] - batch_idx)
                )

            loss.backward()
            self.neural_net.optimizer.step()
            self.neural_net.optimizer.zero_grad()
            self.current_loss = loss.clone().detach().cpu().numpy().item()
            self.current_predictions = predicted_parameters.clone().detach().cpu()
            if "t_infectious" in self.to_learn.keys():
                self.current_predictions[self.to_learn["t_infectious"]] *= 10
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


# -----------------------------------------------------------------------------
# -- Performing the simulation run --------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cfg_file_path = sys.argv[1]

    log.note("   Preparing model run ...")
    log.note(f"   Loading config file:\n        {cfg_file_path}")
    yamlc = yaml.YAML(typ="safe")
    with open(cfg_file_path) as cfg_file:
        cfg = yamlc.load(cfg_file)
    model_name = cfg.get("root_model_name", "SIR")
    log.note(f"   Model name:  {model_name}")
    model_cfg = cfg[model_name]

    # Select the training device and number of threads to use
    device = model_cfg["Training"].get("device", None)
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

    # Get the training data
    log.info("   Generating synthetic training data ...")
    training_data = SIR.get_SIR_data(data_cfg=model_cfg["Data"], h5group=h5group).to(
        device
    )

    # Initialise the neural net
    log.info("   Initializing the neural net ...")
    net = base.NeuralNet(
        input_size=3,
        output_size=len(model_cfg["Training"]["to_learn"]),
        **model_cfg["NeuralNet"],
    ).to(device)

    # Initialise the model
    model = SIR_NN(
        rng=rng,
        h5group=h5group,
        neural_net=net,
        write_every=cfg["write_every"],
        write_start=cfg["write_start"],
        num_steps=len(training_data),
        training_data=training_data,
        **model_cfg["Training"],
    )
    log.info(f"   Initialized model '{model_name}'.")

    num_epochs = cfg["num_epochs"]
    log.info(f"   Now commencing training for {num_epochs} epochs ...")
    for i in range(num_epochs):
        model.epoch()
        log.progress(
            f"   Completed epoch {i+1} / {num_epochs}; "
            f"   current loss: {model.current_loss}"
        )

    if model_cfg.get("MCMC", {}).pop("perform_sampling", False):
        log.info("   Performing MCMC sampling ... ")

        n_samples = model_cfg["MCMC"].pop("n_samples")

        # Initialise the sampler
        sampler = SIR.Langevin_sampler(
            h5File=h5file,
            true_data=training_data[
                model_cfg["Data"].get("training_data_size", slice(None, None)), :, :
            ],
            to_learn=model_cfg["Training"]["to_learn"],
            true_parameters=model_cfg["Training"].get("true_parameters", {}),
            **model_cfg["MCMC"],
        )

        import time

        start_time = time.time()

        # Collect n_samples
        for i in range(n_samples):
            sampler.sample()
            sampler.write_loss()
            sampler.write_parameters()
            log.info(f"Collected {i} of {n_samples}.")

        # Write out the total sampling time
        sampler.write_time(time.time() - start_time)

    log.info("   Simulation run finished.")
    log.info("   Wrapping up ...")
    h5file.close()

    log.success("   All done.")
