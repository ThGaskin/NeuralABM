#!/usr/bin/env python3
import copy
import h5py as h5
import numpy as np
import torch
import xarray as xr
import sys
from os.path import dirname as up
from dantro._import_tools import import_module_from_path
import coloredlogs
from dantro import logging

# --- Path Setup ---
sys.path.extend([up(up(__file__)), up(up(up(__file__)))])
SIRS = import_module_from_path(mod_path=up(up(__file__)), mod_str="SIRS")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

# --- Logger ---
log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)

# --- Helper Functions ---
def _build_param_dict(pred, learnable_params, fixed_params) -> dict:
    """
    Construct a parameter dictionary for the SIRS model.

    :param pred: Predicted values from the neural network corresponding to learnable parameters.
    :param learnable_params: Names of parameters predicted by the network.
    :param fixed_params: Dictionary of fixed parameters that remain constant.
    :returns: A dictionary containing both learnable and fixed parameters for the model.
    """

    return fixed_params | {p: pred[i] for i, p in enumerate(learnable_params)}


def predict_and_simulate(NN, dset, t, learnable_params, fixed_params, device, add_noise=True):
    """
    Predict model parameters using the neural network and simulate SIRS dynamics.

    :param NN: Neural network used to predict the parameters.
    :param dset: Input dataset for the network (flattened per time series).
    :param t: Time steps for the simulation.
    :param learnable_params: Parameters to be estimated by the neural network.
    :param fixed_params: Parameters that remain fixed during simulation.
    :param device: Device on which to perform computation ('cpu' or 'cuda').
    :param add_noise: Whether to add noise to the simulated data based on the 'noise' parameter.
    :returns: Tuple containing predicted parameters, simulated SIRS trajectories, and the parameter dictionary.
    """
    x = dset.flatten()
    pred = NN(x)
    params = _build_param_dict(pred, learnable_params, fixed_params)
    _, Y_pred = SIRS.SIRS_euler(y0=dset[0], t=t, **params)

    if add_noise and "noise" in params and (( "noise" not in learnable_params) or (params["noise"] > 0)):
        w = torch.normal(1.0, 1.0, dset.shape, device=device)
        Y_pred = Y_pred * (1 + params["noise"] * w)

    return pred, Y_pred, params


def generate_data(model_cfg, device):
    """
    Generate or load datasets for training and testing the neural network.

    :param model_cfg: Model-specific configuration, including data paths and synthetic data settings.
    :param device: Device on which to store generated tensors.
    :returns: Tuple with time steps, training data, and test data.
    """
    if model_cfg['Data'].get('load_from', None):
        data = xr.load_dataarray(model_cfg['Data']['load_from'])
        t = torch.from_numpy(data.coords['Time'].data).to(device)
        data = torch.from_numpy(data.data).to(device).float().reshape(1, *data.shape)
        return t, data, None

    synth_cfg = model_cfg['Data']['synthetic_data']
    n_train, n_test = synth_cfg.get('n_train', 1), synth_cfg.get('n_test', 0)
    training_data, test_data = [], []

    for i in range(n_train + n_test):
        cfg_copy = copy.deepcopy(synth_cfg)
        for param in ['k_S', 'k_I', 'k_R', 'noise']:
            if isinstance(cfg_copy[param], list):
                cfg_copy[param] = cfg_copy[param][i]
            elif isinstance(cfg_copy[param], dict):
                cfg_copy[param] = base.utils.random_tensor(cfg_copy[param], size=(1, ))

        t, Y = SIRS.SIRS_euler(**cfg_copy, device=device)
        if cfg_copy['noise'] > 0:
            w = torch.normal(0.0, 1.0, size=Y.shape, device=device)
            Y *= (1 + cfg_copy['noise'] * w)

        (training_data if i < n_train else test_data).append(Y)

    return t, torch.stack(training_data), torch.stack(test_data) if test_data else None


def train_model(NN, training_data, test_data, t, learnable_params, fixed_params, device, cfg, loss_function):
    """
    Train the neural network to estimate SIRS parameters.

    :param NN: Neural network to train.
    :param training_data: Training dataset.
    :param test_data: Optional test dataset for validation during training.
    :param t: Time steps for simulation.
    :param learnable_params: Parameters to be learned by the neural network.
    :param fixed_params: Fixed parameters for the SIRS model.
    :param device: Device on which to perform computation.
    :param cfg: Configuration dictionary containing training hyperparameters (epochs, batch size, etc.).
    :param loss_function: Loss function used for training.
    :returns: Tuple containing loss time series and parameter time series.
    """
    num_epochs, batch_size, write_every, write_start = (
        cfg["num_epochs"], cfg["batch_size"], cfg["write_every"], cfg["write_start"]
    )
    loss_ts, parameter_ts = [], []

    for epoch in range(num_epochs):
        perm = torch.randperm(len(training_data), device=device)
        epoch_loss, epoch_params = [], []
        loss = torch.tensor(0.0, requires_grad=True)

        for batch_idx, idx in enumerate(perm):
            dset = training_data[idx]
            pred, Y_pred, _ = predict_and_simulate(NN, dset, t, learnable_params, fixed_params, device)
            loss = loss + loss_function(Y_pred, dset)

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(training_data) - 1:
                loss.backward()
                NN.optimizer.step()
                NN.optimizer.zero_grad()
                epoch_loss.append(loss.detach())
                loss = torch.tensor(0.0, requires_grad=True)

            if epoch >= write_start and epoch % write_every == 0:
                epoch_params.append(pred.detach())

        if epoch >= write_start and epoch % write_every == 0:
            epoch_params = [epoch_params[i] for i in torch.argsort(perm)]
            test_loss = torch.tensor(torch.nan)
            if test_data is not None:
                test_preds = []  # collect test parameter predictions
                with torch.no_grad():
                    test_loss_vals = []
                    for d in test_data:
                        params, Y_pred, _ = predict_and_simulate(
                            NN, d, t, learnable_params, fixed_params, device
                        )
                        test_loss_vals.append(loss_function(Y_pred, d))

                        # Append just the predicted parameters (in same order as training)
                        test_preds.append(torch.tensor([params[i] for i, _ in enumerate(learnable_params)]))

                test_loss = torch.mean(torch.stack(test_loss_vals))
                epoch_params.extend(test_preds)  # Add test predictions

            loss_ts.append(torch.stack((torch.mean(torch.stack(epoch_loss)), test_loss)))
            parameter_ts.append(torch.stack(epoch_params))

            log.progress(
                f"[Epoch {epoch + 1}/{num_epochs}] Training Loss: {loss_ts[-1][0].cpu():.4f}, Test Loss: {loss_ts[-1][1].cpu():.4f}"
            )
    return loss_ts, parameter_ts


def save_results_to_h5(cfg, model_name, loss_ts, parameter_ts, training_data, test_data, t, NN,
                       learnable_params, fixed_params, device):
    """
    Save training results, parameters, and predictions to an HDF5 file.

    :param cfg: Global configuration dictionary including output path.
    :param model_name: Name of the model, used as the root group in the HDF5 file.
    :param loss_ts: Loss values recorded during training.
    :param parameter_ts: Predicted parameters over training epochs.
    :param training_data: Training dataset.
    :param test_data: Optional test dataset.
    :param t: Time steps for predictions.
    :param NN: Trained neural network.
    :param learnable_params: Names of learnable parameters.
    :param fixed_params: Fixed parameters used for predictions.
    :param device: Device on which to perform predictions.
    """
    log.note(f"   Creating output file at: {cfg['output_path']}")
    with h5.File(cfg["output_path"], "w") as h5file:
        h5group = h5file.create_group(model_name)

        # Loss
        dset_loss = h5group.create_dataset("loss", data=loss_ts, compression=3)
        dset_loss.attrs.update(dict(
            dim_names=['epoch', 'kind'],
            coords_mode__epoch='start_and_step',
            coords__epoch=[cfg["write_start"], cfg["write_every"]],
            coords_mode__kind='values',
            coords__kind=['training_loss', 'test_loss']
        ))

        coords_dset_id = [f'train_{i}' for i in range(len(training_data))]
        if test_data is not None:
            coords_dset_id += [f'test_{i}' for i in range(len(test_data))]

        # Parameters
        dset_params = h5group.create_dataset("parameters", data=torch.stack(parameter_ts), compression=3)
        dset_params.attrs.update(dict(
            dim_names=['epoch', 'dset_id', 'parameter'],
            coords_mode__epoch='start_and_step',
            coords__epoch=[cfg["write_start"], cfg["write_every"]],
            coords_mode__dset_id='values',
            coords__dset_id=coords_dset_id,
            coords_mode__parameter='values',
            coords__parameter=learnable_params
        ))

        # Training data
        all_data = torch.cat([training_data, test_data]) if test_data is not None else training_data
        dset_Y = h5group.create_dataset("Y", data=all_data, compression=3)
        dset_Y.attrs.update(dict(
            dim_names=['dset_id', 'Time', 'Compartment'],
            coords_mode__dset_id='values',
            coords__dset_id=coords_dset_id,
            coords_mode__Time='values',
            coords__Time=t,
            coords_mode__Compartment='values',
            coords__Compartment=['S', 'I', 'R']
        ))

        # Predictions
        with torch.no_grad():
            Y_pred = torch.stack([
                predict_and_simulate(NN, d, t, learnable_params, fixed_params, device)[1]
                for d in all_data
            ])
        dset_Y_pred = h5group.create_dataset("Y_pred", data=Y_pred, compression=3)
        dset_Y_pred.attrs.update(dset_Y.attrs)

    log.success("   All done.")


if __name__ == "__main__":
    cfg_file_path = sys.argv[1]
    cfg, model_cfg = base.load_config(cfg_file_path)
    model_name = cfg.get("root_model_name", "SIRS")

    device = base.set_default_device(cfg)
    log.info(f"Using '{device}' device. Threads: {torch.get_num_threads()}")

    np.random.seed(cfg["seed"])
    torch.random.manual_seed(cfg["seed"])

    learnable_params = model_cfg["Learning"]["parameters_to_learn"]
    fixed_params = model_cfg["Learning"]["fixed_parameters"]

    # Data
    t, training_data, test_data = generate_data(model_cfg, device)

    # Model
    NN = base.FeedForwardNN(
        input_size=training_data[0].numel(),
        output_size=len(learnable_params),
        **model_cfg["NeuralNet"]
    ).to(device)

    loss_function = base.get_loss_function(model_cfg['Training']['loss_function'])

    # Train
    loss_ts, parameter_ts = train_model(
        NN, training_data, test_data, t, learnable_params, fixed_params, device, cfg, loss_function
    )

    # Save
    save_results_to_h5(
        cfg, model_name, loss_ts, parameter_ts, training_data, test_data, t, NN, learnable_params, fixed_params, device
    )
