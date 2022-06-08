#!/usr/bin/env python3
from os.path import dirname as up
import sys

import h5py as h5
import numpy as np
import ruamel.yaml as yaml
import torch
from utopya._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

HW = import_module_from_path(mod_path=up(up(__file__)), mod_str='HarrisWilson')
core = import_module_from_path(mod_path=up(up(up(__file__))), mod_str='core')

# -----------------------------------------------------------------------------
# -- Model implementation -----------------------------------------------------
# -----------------------------------------------------------------------------


class HarrisWilson_NN:
    """ Should only receive: dataset, neural net, abm. does nothing but run and write.
    """

    def __init__(
        self,
        name: str,
        *,
        rng: np.random.Generator,
        h5group: h5.Group,
        neural_net: core.NeuralNet,
        ABM: HW.HarrisWilsonABM,
        params_to_learn,
        true_parameters: dict=None,
        write_every: int = 1,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        Args:
            name (str): The name of this model instance
            rng (np.random.Generator): The shared RNG
            h5group (h5.Group): The output file group to write data to
            state_size (int): Size of the state vector
            distribution_params (dict): Passed to the random number
                distribution
            **__: Additional model parameters (ignored)
        """
        self._name = name
        self._time = 0
        self._h5group = h5group
        self._rng = rng

        self._ABM = ABM
        self._neural_net = neural_net
        self._neural_net.optimizer.zero_grad()

        self._current_loss = torch.tensor(0.0, requires_grad=False)
        self._current_predictions = torch.stack([torch.tensor(0.0, requires_grad=False)]*len(params_to_learn))

        # Setup chunked dataset to store the state data in
        self._dset_loss = self._h5group.create_dataset(
            "loss",
            (0, 1),
            maxshape=(None, 1),
            chunks=True,
            compression=3,
        )
        self._dset_predictions = self._h5group.create_dataset(
            "predictions",
            (0, len(params_to_learn)),
            maxshape=(None, len(params_to_learn)),
            chunks=True,
            compression=3,
        )
        self._write_every = write_every

    def epoch(self, *, training_data, batch_size):

        """ Trains the model for a single epoch """

        loss = torch.tensor(0.0, requires_grad=True)

        for t, sample in enumerate(training_data):

            predicted_parameters = self._neural_net(torch.flatten(sample))
            predicted_data = ABM.run_single(input_data=predicted_parameters, curr_vals=sample,
                                       sigma=0.0, epsilon=1.0, dt=0.0001, requires_grad=True)

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

    def write_data(self):
        """Write the current state into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time == 0 or (self._time % self._write_every == 0):

            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1, :] = self._current_loss

            self._dset_predictions.resize(self._dset_predictions.shape[0] + 1, axis=0)
            self._dset_predictions[-1] = self._current_predictions


# -----------------------------------------------------------------------------
# -- Performing the simulation run --------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # This will only work on Apple Silicon
    device = "mps" if torch.backends.mps.is_available() else "cpu"

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
    or_sizes, dest_sizes, network = HW.get_HW_data(model_cfg['Data'], out_dir = cfg['output_dir'].replace('data/uni0', ''))
    print(or_sizes.shape, dest_sizes.shape, network.shape)
    print("\nInitializing the neural net ...")
    net = core.NeuralNet(input_size=dest_sizes.shape[1], output_size=len(model_cfg['Training']['to_learn']),
                         **model_cfg['NeuralNet'])

    print("\nInitializing the ABM ...")
    true_parameters = model_cfg['true_parameters'] if 'true_parameters' in model_cfg['Training'].keys() else None
    ABM = HW.HarrisWilsonABM(origin_sizes=or_sizes, network=network, true_parameters=true_parameters,
                             M=dest_sizes.shape[1])

    print("\nInitializing model ...")
    model = HarrisWilson_NN(
        model_name, rng=rng, h5group=h5group, neural_net=net, ABM=ABM, params_to_learn=model_cfg['Training']['to_learn'],
        write_every=cfg['write_every'] if 'write_every' in cfg.keys() else 1,
    )
    model.write_data()
    print(f"Initialized model '{model_name}'.")

    num_epochs = cfg["num_epochs"]
    print(f"\nNow commencing training for {num_epochs} epochs ...")
    for i in range(num_epochs):

        model.epoch(training_data=dest_sizes, batch_size=model_cfg['Training']['batch_size'])

        # TODO Interrupt handling

        print(f"  Completed epoch {i+1} / {num_epochs}.")

    print("\nSimulation run finished.")
    print("  Wrapping up ...")
    h5file.close()

    print("  All done.")
