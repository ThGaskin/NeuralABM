"""This module implements the BaseModel class"""

import time

import h5py as h5
import numpy as np
import ruamel.yaml as yaml


class BaseModel:
    """The base model class"""

    def __init__(self, *, cfg_file_path: str):
        """Initialize the model instance, constructing an RNG and HDF5 group
        to write the output data to.

        Args:
            cfg_file_path (str): The path to the config file.
        """
        print(f"Loading config file:\n  {cfg_file_path}")
        with open(cfg_file_path) as cfg_file:
            self._cfg = yaml.load(cfg_file, Loader=yaml.Loader)

        self._name = self._cfg.get("root_model_name", "ExtendedModel")
        print(f"  Model name:  {self._name}")

        print("  Extracting time step information ...")
        self._time = 0
        self._num_steps = self._cfg["num_steps"]
        self._write_every = self._cfg["write_every"]
        self._write_start = self._cfg["write_start"]

        # TODO Configure logging here

        print("  Extracting monitoring information ...")
        self._last_emit = 0
        self._monitor_emit_interval = self._cfg["monitor_emit_interval"]

        print("  Creating global RNG ...")
        self._rng = np.random.default_rng(self._cfg["seed"])

        print(f"  Creating output file at:\n    {self._cfg['output_path']}")
        self._h5file = h5.File(self._cfg["output_path"], mode="w")
        self._h5group = self._h5file.create_group(self._name)

        print("\nInitializing model ...")
        self._model_cfg = self._cfg[self._name]
        self.setup(**self._model_cfg)

        if self._write_start <= 0:
            print("  Writing initial state ...")
            self.write_data()

        print(f"Initialized {type(self).__name__} named '{self._name}'.")

        self.monitor()
        self._last_emit = time.time()

    def __del__(self):
        """Takes care of tearing down the model"""
        print("Tearing down model instance ...")

        self._h5file.close()

        print("Teardown complete.")

    # .. Simulation control ...................................................

    def run(self):
        """Performs a simulation run for this model, calling the iterate
        method until the number of desired steps has been carried out.
        """
        print(f"\nCommencing model run with {self._num_steps} iterations ...")

        while self._time < self._num_steps:
            self.iterate()
            # TODO Interrupt handling

            print(f"  Finished iteration {self._time} / {self._num_steps}.")

        print("Simulation run finished.\n")

    def iterate(self):
        """Performs a single iteration: a simulation step, monitoring, and
        writing data"""
        self.perform_step()
        self._time += 1

        if self._monitor_should_emit():
            self.monitor()

        if (
            self._time > self._write_start
            and self._time % (self._write_every - self._write_start) == 0
        ):
            self.write_data()

    # .. Monitoring ...........................................................

    def _monitor_should_emit(self) -> bool:
        """Evaluates whether the monitor should emit. This method will only
        return True once after a monitor emit interval has passed.
        """
        t = time.time()
        if t > self._last_emit + self._monitor_emit_interval:
            self._last_emit = t
            return True
        return False

    def monitor(self):
        # TODO Implement monitor_emit_interval
        print("!!map { progress: " + str(self._time / self._num_steps) + " }")

    # .. Abstract (to-be-subclassed) methods ..................................

    def setup(self):
        raise NotImplementedError("setup")

    def perform_step(self):
        raise NotImplementedError("perform_step")

    def write_data(self):
        raise NotImplementedError("write_data")
