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

    # Generate the birefringence time series, or load if a dataset is given
    if isinstance(model_cfg["Laser_Cavity"]["K"], str):
        with h5.File(model_cfg["Laser_Cavity"]["K"], "r") as f:
            K = np.array(f["Laser"]["laser_birefringence"])
        model_cfg["Laser_Cavity"]["K"] = K[0]
    else:
        K = model_cfg["Laser_Cavity"]["K"] * np.ones(cfg["num_epochs"])
        for _ in range(len(K)-1):
            K[_+1] = np.clip(np.random.normal(K[_], model_cfg["Laser_Cavity"]["K_std"]), a_min=-0.3, a_max=+0.3)

    # Instantiate a laser, which is later initialised in the model
    # Initial angles can be specified in the config, or else are randomly generated
    # These are used as the initial guess of the neural network
    init_angles = model_cfg["Laser_Cavity"].get("initial_angles")
    init_angles = torch.pi * torch.rand(size=(4,)) if init_angles is None else torch.tensor(init_angles)
    laser = Laser.Laser_cavity(
        parameters=model_cfg["Laser_Cavity"],
        t=torch.linspace(-25, +25, 256),  # TODO control this from the config
        z=torch.tensor([0, 0.5, 1]),  # TODO control this from the config
        alpha=init_angles,
    )

    # Initialise the neural net. The angles of the waveplates are used as the initial guess
    log.info("   Initializing the neural net ...")
    if model_cfg["NeuralNet"].get("prior", None) is None:
        model_cfg["NeuralNet"]["prior"] = [
            dict(distribution='uniform', parameters=dict(lower=a.numpy(), upper=a.numpy()))
            for a in init_angles
        ]

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

        # Vary the birefringence
        model.laser.solver.set_parameter(dict(K=K[i]))

        # Perform a round trip
        model.laser.round_trip()

        # Perform a gradient step every n round trips
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
