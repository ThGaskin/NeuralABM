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

Laser = import_module_from_path(mod_path=up(up(__file__)), mod_str="Laser")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)

# ----------------------------------------------------------------------------------------------------------------------
# Performing the simulation run
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    cfg_file_path = sys.argv[1]

    log.note("   Preparing model run ...")
    log.note(f"   Loading config file:\n        {cfg_file_path}")
    yamlc = yaml.YAML(typ="safe")
    with open(cfg_file_path) as cfg_file:
        cfg = yamlc.load(cfg_file)
    model_name = cfg.get("root_model_name", "Laser")
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

    # Instantiate a laser, which is later initialised in the model
    laser = Laser.Laser_cavity(
        parameters=model_cfg["Laser_Cavity"],
        t=torch.linspace(-25, +25, 256),  # TODO control this from the config
        z=torch.tensor([0, 0.5, 1]),  # TODO control this from the config
        alpha=torch.tensor(model_cfg["Laser_Cavity"]["initial_angles"]),
    )

    # Initialise the neural net
    log.info("   Initializing the neural net ...")
    net = base.NeuralNet(
        input_size=3,
        output_size=4,
        **model_cfg["NeuralNet"],
    ).to(device)

    # Initialise the model
    model = Laser.NN(
        rng=rng,
        h5group=h5group,
        laser=laser,
        neural_net=net,
        write_start=cfg["write_start"],
        **model_cfg["Training"],
    )
    model.initialise_laser(torch.stack([torch.cosh(model.laser.solver.t) ** (-1),
                                        0.2 * torch.cosh(model.laser.solver.t) ** (-1)
                                        ]).cfloat(),
                           n_round_trips=model_cfg["Laser_Cavity"]["n_initialisation"])

    # Now run the laser for n_steps
    num_epochs = cfg["num_epochs"]
    log.info(f"   Now running the laser for {num_epochs} time steps ...")

    # Set the parameters from the neural network
    model.set_parameters_from_NN()

    for i in range(num_epochs):

        # Perform a round trip
        model.laser.round_trip()

        # Vary the birefringence
        model.laser.solver.set_parameter(dict(K=
            np.clip(np.random.normal(
                model.laser.solver.get_parameter("K"),
                model.laser.solver.get_parameter("K_std")
            ), a_min=-0.3, a_max=0.3)
        ))

        if i % model.batch_size == 0:
            # Train the neural network
            model.perform_step()

            # Write the data
            model.write_data()

            # Update the angles with the new prediction
            model.set_parameters_from_NN()

            log.progress(
                f"   Iteration {i}, current objective: {model.objective}"
            )

        model.write_laser_state()

    log.info("   Simulation run finished.")
    log.info("   Wrapping up ...")
    h5file.close()

    log.success("   All done.")
