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
import xarray as xr

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

Migration_model = import_module_from_path(mod_path=up(up(__file__)), mod_str="Migration")
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

    start, end = model_cfg["Data"]["training_range"][0], model_cfg["Data"]["training_range"][1] + 1

    # Load the UN net migration data
    log.info("   Loading data ...")
    net_migration_data = xr.load_dataarray(model_cfg["Data"]["net_migration"])
    net_migration_data = net_migration_data.sel({"Year": range(start, end)})

    # Load the UN stock data
    stock_data = xr.load_dataarray(model_cfg["Data"]["stock_data"])
    stock_data = stock_data.sel({"Year": np.arange(start, end, 5)})

    # Load the total population
    total_population_data = xr.load_dataarray(model_cfg["Data"]["total_population"])
    total_population_data = total_population_data.sel({"Year": range(start, end)})

    N = len(net_migration_data.coords["Country ISO"].data)

    # Initialise the neural net
    log.info("   Initializing the neural net ...")
    net = base.NeuralNet(
        input_size=N,
        output_size=2*N**2,
        **model_cfg["NeuralNet"],
    ).to(device)

    # Initialise the model
    model = Migration_model.NN(
        rng=rng,
        h5group=h5group,
        neural_net=net,
        write_every=cfg["write_every"],
        write_start=cfg["write_start"],
        write_predictions_every=cfg.get("write_predictions_every", cfg["write_every"]),
        write_predictions_start=cfg.get("write_predictions_start", cfg["write_start"]),
        training_data=dict(stock_data=stock_data, net_migration=net_migration_data, total_population=total_population_data),
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

    log.info("   Simulation run finished.")
    log.info("   Wrapping up ...")
    h5file.close()

    log.success("   All done.")
