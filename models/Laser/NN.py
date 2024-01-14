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

Laser = import_module_from_path(mod_path=up(up(__file__)), mod_str="Laser")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)


class Laser_NN:
    def __init__(
        self,
        *,
        rng: np.random.Generator,
        h5group: h5.Group,
        neural_net: base.NeuralNet,
        laser: Laser.Laser_cavity,
        batch_size: int,
        write_every: int = 1,
        write_start: int = 1,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        :param rng (np.random.Generator): The shared RNG
        :param h5group (h5.Group): The output file group to write data to
        :param neural_net: The neural network
        :param laser: the uninitialised laser
        :param batch_size (int): update the neural network parameters after a given number of round trips
        :param write_every: write every iteration
        :param write_start: iteration at which to start writing
        """
        self._h5group = h5group
        self._rng = rng

        self.neural_net = neural_net
        self.neural_net.optimizer.zero_grad()

        self.laser = laser

        # Objective to be maximised
        self.objective = torch.tensor(0.0)

        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        # Write the objective function over time
        self._dset_obj = self._h5group.create_dataset(
            "objective_function",
            (0,),
            maxshape=(None,),
            chunks=True,
            compression=3,
        )
        self._dset_obj.attrs["dim_names"] = ["time"]
        self._dset_obj.attrs["coords_mode__time"] = "start_and_step"
        self._dset_obj.attrs["coords__time"] = [write_start, write_every]

        # Write the parameter predictions after every batch
        self.dset_parameters = self._h5group.create_dataset(
            "parameters",
            (0, 4),
            maxshape=(None, 4),
            chunks=True,
            compression=3,
        )
        self.dset_parameters.attrs["dim_names"] = ["time", "parameter"]
        self.dset_parameters.attrs["coords_mode__time"] = "start_and_step"
        self.dset_parameters.attrs["coords__time"] = [write_start, write_every]
        self.dset_parameters.attrs["coords_mode__parameter"] = "values"
        self.dset_parameters.attrs["coords__parameter"] = ["alpha_1", "alpha_2", "alpha_3", "alpha_p"]

        # Perform a gradient descent step after a certain number of round trips
        self.batch_size = batch_size

        # Batches processed
        self._time = 0
        self._write_every = write_every
        self._write_start = write_start

    def initialise_laser(self, initial_condition: torch.Tensor, n_round_trips: int = 100) -> None:
        """ Initialises the laser to a given initial condition"""
        log.info("   Initialising the laser ...")

        # Set the initial state of the laser, given in real space
        self.laser.set_state(initial_condition, set_init=True)

        # Calculate the initial energy
        self.laser.set_energy()

        # Perform n round trips
        for n in range(n_round_trips):
            self.laser.round_trip()

        log.success("  Initialised the laser.")

    def set_parameters_from_NN(self, *, requires_grad: bool = True) -> None:
        """ Set the parameters of model to the output of the neural network. By default, these are tracked through the
        numerical solver and gradients can then be used to optimise the neural network."""

        # Make a prediction. The neural network must output values in [-1, 1]
        prediction = torch.pi * self.neural_net(
            torch.rand(self.neural_net.input_dim, requires_grad=requires_grad)
        ).flatten()

        # Set the transformation matrix of the laser using the current prediction
        self.laser.set_transformation_matrix(self.laser.calculate_transformation_matrix(alpha=prediction))

    def objective_function(self) -> torch.Tensor:
        """" Calculate the objective function of the laser, which is the energy divided by the fourth moment
         of the Fourier spectrum """

        return self.laser.solver.energy(self.laser.state).real #/ self.laser.solver.kurtosis(self.laser.state)

    def perform_step(self):
        """ Performs a single gradient descent step."""
        loss = - self.objective_function()
        loss.backward()
        self.neural_net.optimizer.step()
        self.neural_net.optimizer.zero_grad()

        self.objective = -loss.detach().clone()
        self._time += 1
        self.write_data()
        self.laser.clear_gradients()

    def write_data(self):
        """Write the current state (loss and parameter predictions) into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0):
            self._dset_obj.resize(self._dset_obj.shape[0] + 1, axis=0)
            self._dset_obj[-1] = self.objective
            self.dset_parameters.resize(self.dset_parameters.shape[0] + 1, axis=0)
            self.dset_parameters[-1, :] = self.laser.alpha()
