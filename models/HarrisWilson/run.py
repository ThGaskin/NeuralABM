#!/usr/bin/env python3
from os.path import dirname as up
import sys

import h5py as h5
import numpy as np
import ruamel.yaml as yaml
import time
import torch

from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

HW = import_module_from_path(mod_path=up(up(__file__)), mod_str='HarrisWilson')
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str='include')

# ----------------------------------------------------------------------------------------------------------------------
# -- Model implementation ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class HarrisWilson_NN:

    def __init__(
        self,
        name: str,
        *,
        rng: np.random.Generator,
        h5group: h5.Group,
        neural_net: base.NeuralNet,
        ABM: HW.HarrisWilsonABM,
        to_learn: list,
        write_every: int = 1,
        write_start: int = 1,
        write_time: bool = False,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        Args:
            name (str): The name of this model instance
            rng (np.random.Generator): The shared RNG
            h5group (h5.Group): The output file group to write data to
            neural_net: The neural network
            ABM: The numerical solver
            to_learn: the list of parameter names to learn
            write_every: write every iteration
            write_start: iteration at which to start writing
            num_steps: number of iterations of the ABM
            write_time: whether to write out the training time into a dataset
        """
        self._name = name
        self._time = 0
        self._h5group = h5group
        self._rng = rng

        self._ABM = ABM
        self._neural_net = neural_net
        self._neural_net.optimizer.zero_grad()

        self._current_loss = torch.tensor(0.0, requires_grad=False)
        self._current_predictions = torch.stack([torch.tensor(0.0, requires_grad=False)]*len(to_learn))

        # Setup chunked dataset to store the state data in
        self._dset_loss = self._h5group.create_dataset(
            "loss",
            (0, 1),
            maxshape=(None, 1),
            chunks=True,
            compression=3,
        )
        self._dset_loss.attrs['dim_names'] = ['time', 'training_loss']
        self._dset_loss.attrs["coords_mode__time"] = "start_and_step"
        self._dset_loss.attrs["coords__time"] = [write_start, write_every]

        if write_time:
            self.dset_time = self._h5group.create_dataset(
                "computation_time",
                (0, 1),
                maxshape=(None, 1),
                chunks=True,
                compression=3,
            )
            self.dset_time.attrs['dim_names'] = ['epoch', 'training_time']
            self.dset_time.attrs["coords_mode__epoch"] = "trivial"
            self.dset_time.attrs["coords_mode__training_time"] = "trivial"

        dset_predictions = []
        for p_name in to_learn:
            dset = self._h5group.create_dataset(
                p_name,
                (0, 1),
                maxshape=(None, 1),
                chunks=True,
                compression=3
            )
            dset.attrs['dim_names'] = ['time', 'parameter']
            dset.attrs["coords_mode__time"] = "start_and_step"
            dset.attrs["coords__time"] = [write_start, write_every]

            dset_predictions.append(dset)

        self._dset_predictions = dset_predictions
        self._write_every = write_every
        self._write_start = write_start
        self._write_time = write_time

    def epoch(self, *, training_data, batch_size: int = 1, epsilon: float = 1, dt: float = 0.001):

        """Trains the model for a single epoch.

        :param training_data: the training data
        :param batch_size: (optional) the number of passes (batches) over the training data before conducting a gradient descent
                step
        :param epsilon: (optional) the epsilon value to use during training
        :param dt: (optional) the time differential to use during training
        """

        start_time = time.time()

        loss = torch.tensor(0.0, requires_grad=True)

        start_time = time.time()

        for t, sample in enumerate(training_data):

            predicted_parameters = self._neural_net(torch.flatten(sample))
            predicted_data = ABM.run_single(input_data=predicted_parameters,
                                            curr_vals=sample,
                                            epsilon=epsilon,
                                            dt=dt,
                                            requires_grad=True)

            loss = loss + torch.nn.functional.mse_loss(predicted_data, sample) / batch_size

            self._current_loss = loss.clone().detach().numpy().item()
            self._current_predictions = predicted_parameters.clone().detach()
            self.write_data()

            # Update the model parameters after every batch and clear the loss
            if t % batch_size == 0 or t == len(training_data)-1:
                loss.backward()
                self._neural_net.optimizer.step()
                self._neural_net.optimizer.zero_grad()
                del loss
                loss = torch.tensor(0.0, requires_grad=True)

            self._time += 1

        # Write the training time (wall clock time)
        if self._write_time:
            self.dset_time.resize(self.dset_time.shape[0] + 1, axis=0)
            self.dset_time[-1, :] = time.time() - start_time

    def write_data(self):
        """Write the current state into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0):

            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1, :] = self._current_loss

            for idx, dset in enumerate(self._dset_predictions):
                dset.resize(dset.shape[0] + 1, axis=0)
                dset[-1] = self._current_predictions[idx]


# ----------------------------------------------------------------------------------------------------------------------
# -- Performing the simulation run -------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Select the training device to use
    try:
        # This will only work on Apple Silicon
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using '{device}' as training device")

    cfg_file_path = sys.argv[1]

    print("Preparing model run ...")
    print(f"  Loading config file:\n    {cfg_file_path}")
    with open(cfg_file_path, "r") as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.Loader)
    model_name = cfg.get("root_model_name", "HarrisWilson")
    print(f"Model name:  {model_name}")
    model_cfg = cfg[model_name]

    print("  Creating global RNG ...")
    rng = np.random.default_rng(cfg["seed"])
    np.random.seed(cfg['seed'])
    torch.random.manual_seed(cfg['seed'])

    print(f"  Creating output file at:\n    {cfg['output_path']}")
    h5file = h5.File(cfg["output_path"], mode="w")
    h5group = h5file.create_group(model_name)

    # Get the datasets
    or_sizes, dest_sizes, network = HW.get_HW_data(model_cfg['Data'], h5file, h5group)

    # Set up the neural net
    print("\nInitializing the neural net ...")
    net = base.NeuralNet(input_size=dest_sizes.shape[1], output_size=len(model_cfg['Training']['to_learn']),
                         **model_cfg['NeuralNet'])

    # Set up the numerical solver
    print("\nInitializing the numerical solver ...")
    true_parameters = model_cfg['Training'].pop('true_parameters', None)
    ABM = HW.HarrisWilsonABM(origin_sizes=or_sizes, network=network, true_parameters=true_parameters,
                             M=dest_sizes.shape[1])
    write_time = model_cfg.pop('write_time', False)

    # Initialise the neural net
    print("\nInitializing the neural net ...")
    model = HarrisWilson_NN(
        model_name, rng=rng, h5group=h5group, neural_net=net, ABM=ABM, to_learn=model_cfg['Training']['to_learn'],
        write_every=cfg['write_every'], write_start=cfg['write_start'], write_time=write_time
    )
    print(f"Initialized model '{model_name}'.")

    # Train the neural net
    num_epochs = cfg["num_epochs"]
    print(f"\nNow commencing training for {num_epochs} epochs ...")

    for _ in range(num_epochs):

        model.epoch(training_data=dest_sizes, batch_size=model_cfg['Training']['batch_size'])

        print(f"  Completed epoch {_+1} / {num_epochs}.")

    print("\nSimulation run finished.")
    print("  Wrapping up ...")
    h5file.close()

    print("  All done.")
