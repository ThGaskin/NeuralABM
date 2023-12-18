#!/usr/bin/env python3
import sys
from os.path import dirname as up

import coloredlogs
import h5py as h5
import networkx as nx
import numpy as np
import ruamel.yaml as yaml
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

Kuramoto = import_module_from_path(mod_path=up(up(__file__)), mod_str="Kuramoto")
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
    yamlc = yaml.YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    with open(cfg_file_path) as cfg_file:
        cfg = yamlc.load(cfg_file)
    model_name = cfg.get("root_model_name", "Kuramoto")
    log.note(f"   Model name:  {model_name}")
    model_cfg = cfg[model_name]

    # Select the training device to use
    training_device = model_cfg["Training"].get("device", None)
    if training_device is not None:
        device = training_device
    else:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # Set number of threads to use
    num_threads = model_cfg["Training"].get("num_threads", None)
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    log.info(
        f"   Using '{device}' as training device. Number of threads: {torch.get_num_threads()}"
    )

    log.note("   Creating global RNG ...")
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    log.note(f"   Creating output file at:\n        {cfg['output_path']}")
    h5file = h5.File(cfg["output_path"], mode="w")
    training_data_group = h5file.create_group("training_data")
    output_data_group = h5file.create_group("output_data")

    # Get the training data and the network
    log.info("   Generating training data ...")
    training_data, eigen_frequencies, network = Kuramoto.DataGeneration.get_data(
        model_cfg["Data"],
        h5file,
        training_data_group,
        seed=seed,
        device=device,
    )

    # Initialise the neural net
    num_agents = training_data.shape[2]
    output_size = num_agents**2

    log.info(
        f"   Initializing the neural net; input size: {num_agents}, output size: {output_size} ..."
    )

    net = base.NeuralNet(
        input_size=num_agents, output_size=output_size, **model_cfg["NeuralNet"]
    ).to(device)

    # Get the true parameters
    true_parameters = model_cfg["Training"]["true_parameters"]

    # Initialise the ABM
    ABM = Kuramoto.Kuramoto_ABM(
        N=num_agents,
        **model_cfg["Data"],
        **true_parameters,
        device=device,
    )

    # Calculate the frequency with which to write out the model predictions
    write_predictions_every = cfg.get("write_predictions_every", cfg["write_every"])
    num_epochs = cfg["num_epochs"]
    batch_size = model_cfg["Training"]["batch_size"]

    # Initialise the model
    model = Kuramoto.Kuramoto_NN(
        model_name,
        rng=rng,
        output_data_group=output_data_group,
        neural_net=net,
        training_data=training_data[:,
                      model_cfg["Data"].get("training_data_times", slice(None, None)), :, :
        ],
        eigen_frequencies=eigen_frequencies[
                          :, model_cfg["Data"].get("training_data_times", slice(None, None)), :, :
        ],
        true_network=torch.from_numpy(nx.to_numpy_array(network)).float() if network is not None else None,
        ABM=ABM,
        write_every=cfg["write_every"],
        write_predictions_every=write_predictions_every,
        write_start=cfg["write_start"],
        **model_cfg["Training"],
    )

    log.info(
        f"   Initialized model '{model_name}'. Now commencing training for {num_epochs} epochs ..."
    )

    # Train the neural net
    for i in range(num_epochs):
        model.epoch()

        # Print progress message
        log.progress(
            f"   Completed epoch {i + 1} / {num_epochs} in {model.dset_time[-1]} s \n"
            f"            ----------------------------------------------------------------- \n"
            f"            Loss components: data:     {model.current_prediction_loss} \n"
            f"                             symmetry: {model.current_symmetry_loss}\n"
            f"                             trace:    {model.current_trace_loss}\n"
            f"                             total:    {model.current_total_loss}\n"
            f"            L1 prediction error: {model.current_prediction_error} \n"
        )

    if write_predictions_every == -1:
        model.write_predictions(write_final=True)

    log.info("   Simulation run finished.")

    # Generate a complete dataset using the predicted parameters
    log.progress("   Generating predicted dataset ...")
    predicted_time_series = training_data[0,
                            slice(model_cfg["Data"].get("training_data_times", slice(0, None)).start, None), :, :
                            ].clone()
    for step in range(0 if model.ABM.alpha == 0 else 1, predicted_time_series.shape[0] - 1):
        predicted_time_series[step + 1, :, :] = ABM.run_single(
            current_phases=predicted_time_series[step, :],
            current_velocities=(
                predicted_time_series[step, :, :]
                - predicted_time_series[step - 1, :, :]
            )
            / ABM.dt,
            adjacency_matrix=model.current_adjacency_matrix,
            eigen_frequencies=eigen_frequencies[0, model_cfg["Data"].get("training_data_times", slice(0, None)).start + step, :, :],
            requires_grad=False,
        )

    # Save prediction
    dset_phases = output_data_group.create_dataset(
        "predicted phases",
        predicted_time_series.shape,
        chunks=True,
        compression=3,
    )
    dset_phases.attrs["dim_names"] = [
        "time",
        "vertex_idx",
        "dim_name__0",
    ]
    dset_phases.attrs["coords_mode__time"] = "trivial"
    dset_phases.attrs["coords_mode__vertex_idx"] = "values"
    dset_phases.attrs["coords__vertex_idx"] = network.nodes()
    dset_phases[:, :] = predicted_time_series.cpu()

    # If specified, perform an OLS regression on the training data
    if cfg.get("perform_regression", False):
        log.info("   Performing regression ... ")
        Kuramoto.regression(
            training_data,
            eigen_frequencies,
            h5file,
            model_cfg["Data"]["dt"],
            alpha=model_cfg["Data"]["alpha"],
            beta=model_cfg["Data"]["beta"],
            kappa=model_cfg["Data"]["kappa"],
        )

    # If specified, calculate the ranks of the Gram matrices for each node
    if cfg.get("calculate_data_rank", False):
        log.info("   Calculating rank of training data ...")
        Kuramoto.rank(training_data, h5file, alpha=ABM.alpha)

    # If specified, run a Langevin MCMC scheme on the training data
    if model_cfg.get("MCMC", {}).get("perform_sampling", False):
        log.info("   Performing Langevin sampling ... ")

        n_samples = model_cfg["MCMC"].get("n_samples")
        accept_first_sample = model_cfg["MCMC"].get("accept_first_sample", False)

        model_params = model.ABM.__dict__
        # Rename the beta from the Kuramoto to avoid conflicting with the MCMC sampler beta
        model_params.update(dict(Kuramoto_beta=model_params.pop("beta")))

        sampler = Kuramoto.Kuramoto_Langevin_sampler(
            h5File=h5file,
            true_data=training_data,
            eigen_frequencies=eigen_frequencies,
            true_network=model.true_network,
            init_guess=torch.reshape(model.current_adjacency_matrix, (-1,)),
            **model_params,
            **model_cfg["MCMC"],
        )

        import time

        start_time = time.time()

        # Collect n_samples
        for i in range(n_samples):
            sampler.sample(force_accept=accept_first_sample and i == 0)
            sampler.write_loss()
            sampler.write_parameters()
            log.info(f"Collected {i} of {n_samples}.")

        # Write out the total sampling time
        sampler.write_time(time.time() - start_time)

    log.info("   Wrapping up ...")

    h5file.close()

    log.success("   All done.")
