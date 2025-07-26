#!/usr/bin/env python3
import h5py as h5
import numpy as np
import torch
import xarray as xr

# Import the base and model
import sys
from os.path import dirname as up
from dantro._import_tools import import_module_from_path
sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))
SIRS = import_module_from_path(mod_path=up(up(__file__)), mod_str="SIRS")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

# Initialise a logger
import coloredlogs
from dantro import logging
log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)

if __name__ == "__main__":

    # Load the configuration file
    cfg_file_path = sys.argv[1]
    cfg, model_cfg = base.load_config(cfg_file_path)
    model_name = cfg.get("root_model_name", "SIRS")

    # Set default training device. By default, this is the CPU, but you can specify a different device in the run
    # configuration
    device = base.set_default_device(cfg)
    log.info(
        f"   Using '{device}' as training device. Number of threads: {torch.get_num_threads()}"
    )

    # Fix the seed for reproducibility
    np.random.seed(cfg["seed"])
    torch.random.manual_seed(cfg["seed"])

    # Load data or generate synthetic training data
    if model_cfg['Data'].get('load_from_dir', None):
        training_data = xr.load_dataarray(model_cfg['Data']['load_from'])
        t = torch.from_numpy(training_data.coords['Time'].data).to(device)
        training_data = torch.from_numpy(training_data.data).to(device).float()
    else:
        t, training_data = SIRS.SIRS_euler(**model_cfg['Data']['synthetic_data'], device=device)

        # Add multiplicative noise
        noise = model_cfg['Data']['synthetic_data'].get('noise', 0.0)
        if noise > 0:
            w = torch.normal(0.0, 1.0, size=training_data.shape, device=device)
            training_data *= (1 + noise * w)

    # Set up the neural network
    input_data = training_data.clone().flatten()
    NN = base.FeedForwardNN(
        input_size=list(input_data.shape)[0],
        output_size=len(model_cfg["Learning"]["parameters_to_learn"]),
        **model_cfg["NeuralNet"]
    ).to(device)

    # Create an output file to which to save the data to. Data is saved in .h5 format.
    # TODO: move to .nc format
    log.note(f"   Creating output file at:\n        {cfg['output_path']}")
    h5file = h5.File(cfg["output_path"], mode="w")
    h5group = h5file.create_group(model_name)
    write_every = cfg["write_every"]
    write_start = cfg["write_start"]

    # Create a dataset for the loss, written at a given frequency
    dset_loss = h5group.create_dataset(
        name="loss", shape=(0, ), maxshape=(None,), chunks=True, compression=3,
    )
    dset_loss.attrs.update(dict(dim_names = ['epoch'], coords_mode__epoch = 'start_and_step',
                      coords__epoch = [write_start, write_every]))

    # Create a dataset for the parameter predictions
    dset_params = h5group.create_dataset(
        name="parameters", shape=(0, NN.output_dim), maxshape=(None, NN.output_dim),
        chunks=True, compression=3,
    )
    dset_params.attrs.update(
        **dict(dim_names=['epoch', 'parameter'], coords_mode__epoch = 'start_and_step',
               coords__epoch = [write_start, write_every], coords_mode__parameter = 'values',
               coords__parameter = model_cfg["Learning"]["parameters_to_learn"]
               )
    )

    # Training
    num_epochs = cfg["num_epochs"]

    # Get the loss function from the configuration
    loss_function = base.get_loss_function(model_cfg['Training']['loss_function'])

    for i in range(num_epochs):

        # Make a parameter prediction
        pred = NN(input_data)

        # Combine estimated and fixed parameters in a single dictionary
        params = dict((model_cfg['Learning']['parameters_to_learn'][idx], pred[idx]) for idx in range(len(model_cfg['Learning']['parameters_to_learn'])))
        params.update(**model_cfg['Learning']['fixed_parameters'])

        # Generate a time series
        _, Y_pred = SIRS.SIRS_euler(y0=training_data[0], t=t, **params)

        # Add multiplicative noise, if specified
        if 'noise' in params.keys():
            if params['noise'] > 0:
                w = torch.normal(1.0, 1.0, training_data.shape, device=device)
                Y_pred = Y_pred * (1 + params['noise'] * w)

        # Calculate the loss
        loss = loss_function(Y_pred, training_data)

        # Backprop
        loss.backward()
        NN.optimizer.step()
        NN.optimizer.zero_grad()

        # Write the loss and parameter predictions
        if i > write_start and i % write_every == 0:
            dset_loss.resize(dset_loss.shape[0] + 1, axis=0)
            dset_loss[-1] = loss.clone().detach()

            dset_params.resize(dset_params.shape[0] + 1, axis=0)
            dset_params[-1, :] = pred.detach()

        log.progress(
            f"   Completed epoch {i + 1} / {num_epochs}; "
            f"   current loss: {loss.detach()}"
        )

    # After training, write the training data and the last time series prediction
    dset_Y = h5group.create_dataset(
        name="Y", shape=training_data.shape, chunks=True, compression=3,
    )
    dset_Y.attrs.update(
        **dict(dim_names = ['Time', 'Compartment'], coords_mode__Time = 'values', coords__Time = t,
               coords_mode__Compartment='values', coords__Compartment=['S', 'I', 'R'])
    )
    dset_Y[:, :] = training_data

    dset_Y_pred = h5group.create_dataset(
        name="Y_pred", shape=training_data.shape, chunks=True, compression=3,
    )
    dset_Y_pred.attrs.update(
        **dict(dim_names = ['Time', 'Compartment'], coords_mode__Time = 'values', coords__Time = t,
               coords_mode__Compartment='values', coords__Compartment=['S', 'I', 'R'])
    )
    dset_Y_pred[:, :] = Y_pred.detach()
    h5file.close()

    # Done
    log.success("   All done.")
