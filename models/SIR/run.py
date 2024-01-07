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
    model = SIR.NN(
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
