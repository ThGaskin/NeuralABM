import logging
import sys
from os.path import dirname as up
from typing import Tuple

import h5py as h5
import numpy as np
import pandas as pd
import torch
from dantro._import_tools import import_module_from_path

from .ABM import HarrisWilsonABM

sys.path.append(up(up(up(__file__))))

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")


""" Load a dataset or generate synthetic data on which to train the neural net """

log = logging.getLogger(__name__)


def load_from_dir(dir) -> Tuple[torch.tensor, torch.tensor]:

    """Loads Harris-Wilson data from a directory.

    :returns the origin sizes, network, and the time series
    """

    log.note("   Loading data ...")

    # If data is to be loaded, check whether a single h5 file, a folder containing csv files, or
    # a dictionary pointing to specific csv files has been passed.
    if isinstance(dir, str):

        # If data is in h5 format
        if dir.lower().endswith(".h5"):
            with h5.File(dir, "r") as f:
                origins = np.array(f["HarrisWilson"]["origin_sizes"])[0]
                training_data = np.array(f["HarrisWilson"]["training_data"])

        # If data is a folder, load csv files
        else:
            origins = pd.read_csv(
                dir + "/origin_sizes.csv", header=0, index_col=0
            ).to_numpy()
            training_data = pd.read_csv(
                dir + "/training_data.csv", header=0, index_col=0
            ).to_numpy()

    # If a dictionary is passed, load data from individual locations
    elif isinstance(dir, dict):

        origins = pd.read_csv(dir["origin_zones"], header=0, index_col=0).to_numpy()
        training_data = pd.read_csv(
            dir["destination_zones"], header=0, index_col=0
        ).to_numpy()

    # Reshape the origin zone sizes
    or_sizes = torch.tensor(origins, dtype=torch.float)
    N_origin = len(or_sizes)
    or_sizes = torch.reshape(or_sizes, (1, N_origin, 1))

    # Reshape the time series of the destination zone sizes
    time_series = torch.tensor(training_data, dtype=torch.float)
    N_destination = training_data.shape[1]
    time_series = torch.reshape(time_series, (1, len(time_series), N_destination, 1))

    # Return all three datasets
    return or_sizes, time_series


def generate_synthetic_data(*, cfg) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

    """Generates synthetic Harris-Wilson using a numerical solver.

    :param cfg: the configuration file
    :returns the origin sizes, network, and the time series
    """

    log.note("   Generating synthetic data ...")

    # Get run configuration properties
    data_cfg = cfg["synthetic_data"]
    N_origin, N_destination = data_cfg["N_origin"], data_cfg["N_destination"]
    num_steps = data_cfg["num_steps"]

    # Generate the network
    network: torch.tensor = torch.exp(
        -1
        * torch.abs(
            base.random_tensor(
                **data_cfg.get("init_network_weights"), size=(N_origin, N_destination)
            )
        )
    )
    network = network * torch.bernoulli(network)  # what if we turn this off?

    # Normalise the rowsums
    norms = torch.sum(network, dim=1, keepdim=True)
    network /= torch.where(norms != 0, norms, 1)

    # Extract the underlying parameters from the config
    true_parameters = {
        "alpha": data_cfg["alpha"],
        "beta": data_cfg["beta"],
        "kappa": data_cfg["kappa"],
        "sigma": data_cfg["sigma"],
        "epsilon": data_cfg["epsilon"],
    }

    # Initialise the ABM
    ABM = HarrisWilsonABM(
        N=data_cfg["N_origin"],
        M=data_cfg["N_destination"],
        dt=data_cfg["dt"],
        device="cpu",
        **true_parameters,
    )
    origin_sizes, dest_sizes = [], []

    for _ in range(cfg["training_set_size"]):

        # Generate the initial destination zone sizes
        init_dest_sizes = torch.abs(
            base.random_tensor(
                **data_cfg.get("init_dest_sizes"), size=(data_cfg["N_destination"], 1)
            )
        )

        # Generate the origin sizes time series
        or_sizes = torch.abs(
            base.random_tensor(
                **data_cfg.get("init_origin_sizes"), size=(1, N_origin, 1)
            )
        )

        if data_cfg["origin_size_std"] == 0:
            or_sizes = or_sizes.repeat(num_steps, 1, 1)
        else:
            for __ in range(num_steps):
                or_sizes = torch.cat(
                    (
                        or_sizes,
                        torch.abs(
                            or_sizes[-1]
                            + torch.normal(
                                0, data_cfg["origin_size_std"], size=(1, N_origin, 1)
                            )
                        ),
                    ),
                    dim=0,
                )

        origin_sizes.append(or_sizes)

        # Run the ABM for n iterations, generating the entire time series
        dest_sizes.append(
            ABM.run(
                init_data=init_dest_sizes,
                adjacency_matrix=network,
                n_iterations=num_steps,
                origin_sizes=or_sizes,
                generate_time_series=True,
            )
        )

    dest_sizes = torch.stack(dest_sizes)
    origin_sizes = torch.stack(origin_sizes)

    # Return all three
    return origin_sizes, dest_sizes, network


def get_HW_data(cfg, h5file: h5.File, h5group: h5.Group, *, device: str):

    """Gets the data for the Harris-Wilson model. If no path to a dataset is passed, synthetic data is generated using
    the config settings

    :param cfg: the data configuration
    :param h5file: the h5 File to use. Needed to add a network group.
    :param h5group: the h5 Group to write data to
    :return: the origin zone sizes, the training data, and the network
    """

    data_dir = cfg.get("load_from_dir", None)

    # Get the origin sizes, time series, and network data
    or_sizes, dest_sizes, network = (
        load_from_dir(data_dir)
        if data_dir is not None
        else generate_synthetic_data(cfg=cfg)
    )

    N_origin = or_sizes.shape[2]
    N_destination = dest_sizes.shape[2]

    time_series = dest_sizes

    # If time series has a single frame, double it to enable visualisation.
    # This does not affect the training data
    num_training_steps = cfg.get("num_training_steps", time_series.shape[1])
    if time_series.shape[1] == 1:
        time_series = torch.stack([time_series, time_series], dim=2)

    # Set up dataset for complete synthetic time series
    dset_time_series = h5group.create_dataset(
        "time_series",
        time_series.shape,
        maxshape=time_series.shape,
        chunks=True,
        compression=3,
    )
    dset_time_series.attrs["dim_names"] = [
        "training_set",
        "time",
        "zone_id",
        "dim_name__0",
    ]
    dset_time_series.attrs["coords_mode__training_set"] = "trivial"
    dset_time_series.attrs["coords_mode__time"] = "trivial"
    dset_time_series.attrs["coords_mode__zone_id"] = "values"
    dset_time_series.attrs["coords__zone_id"] = np.arange(
        N_origin, N_origin + N_destination, 1
    )

    # Write the time series data
    dset_time_series[:, :] = time_series

    # Training time series
    # Extract the training data from the time series data and save
    training_or_sizes, training_dset_sizes = (
        or_sizes[:, -num_training_steps:],
        dest_sizes[:, -num_training_steps:],
    )
    dset_training_data = h5group.create_dataset(
        "training_data",
        training_dset_sizes.shape,
        maxshape=training_dset_sizes.shape,
        chunks=True,
        compression=3,
    )
    dset_training_data.attrs["dim_names"] = [
        "training_set",
        "time",
        "zone_id",
        "dim_name__0",
    ]
    dset_training_data.attrs["coords_mode__time"] = "trivial"
    dset_training_data.attrs["coords_mode__zone_id"] = "values"
    dset_training_data.attrs["coords__zone_id"] = np.arange(
        N_origin, N_origin + N_destination, 1
    )
    dset_training_data[:, :] = training_dset_sizes

    # Set up chunked dataset to store the state data in
    # Origin zone sizes
    dset_origin_sizes = h5group.create_dataset(
        "origin_sizes",
        or_sizes.shape,
        maxshape=or_sizes.shape,
        chunks=True,
        compression=3,
    )
    dset_origin_sizes.attrs["dim_names"] = [
        "training_set",
        "time",
        "zone_id",
        "dim_name__0",
    ]
    dset_origin_sizes.attrs["coords_mode__training_set"] = "trivial"
    dset_origin_sizes.attrs["coords_mode__time"] = "trivial"
    dset_origin_sizes.attrs["coords_mode__zone_id"] = "values"
    dset_origin_sizes.attrs["coords__zone_id"] = np.arange(0, N_origin, 1)
    dset_origin_sizes[
        :,
    ] = or_sizes

    # Create a network group
    nw_group = h5file.create_group("true_network")
    nw_group.attrs["content"] = "graph"
    nw_group.attrs["is_directed"] = True
    nw_group.attrs["allows_parallel"] = False

    # Add vertices
    vertices = nw_group.create_dataset(
        "_vertices",
        (1, N_origin + N_destination),
        maxshape=(1, N_origin + N_destination),
        chunks=True,
        compression=3,
        dtype=int,
    )
    vertices.attrs["dim_names"] = ["dim_name__0", "vertex_idx"]
    vertices.attrs["coords_mode__vertex_idx"] = "trivial"
    vertices[0, :] = np.arange(0, N_origin + N_destination, 1)
    vertices.attrs["node_type"] = [0] * N_origin + [1] * N_destination

    # Add edges. The network is a complete bipartite graph
    # TODO: allow more general network topologies?
    edges = nw_group.create_dataset(
        "_edges",
        (1, N_origin * N_destination, 2),
        maxshape=(1, N_origin * N_destination, 2),
        chunks=True,
        compression=3,
    )
    edges.attrs["dim_names"] = ["dim_name__1", "edge_idx", "vertex_idx"]
    edges.attrs["coords_mode__edge_idx"] = "trivial"
    edges.attrs["coords_mode__vertex_idx"] = "trivial"
    edges[0, :] = np.reshape(
        [
            [[i, j] for i in range(N_origin)]
            for j in range(N_origin, N_origin + N_destination)
        ],
        (N_origin * N_destination, 2),
    )

    # Edge weights
    edge_weights = nw_group.create_dataset(
        "_edge_weights",
        (1, N_origin * N_destination),
        maxshape=(1, N_origin * N_destination),
        chunks=True,
        compression=3,
    )
    edge_weights.attrs["dim_names"] = ["dim_name__1", "edge_idx"]
    edge_weights.attrs["coords_mode__edge_idx"] = "trivial"
    edge_weights[0, :] = torch.reshape(network, (N_origin * N_destination,))

    # Adjacency matrix: only written out if explicity specified
    adjacency_matrix = nw_group.create_dataset(
        "_adjacency_matrix",
        [1] + list(np.shape(network)),
        chunks=True,
        compression=3,
    )
    adjacency_matrix.attrs["dim_names"] = ["time", "i", "j"]
    adjacency_matrix.attrs["coords_mode__i"] = "trivial"
    adjacency_matrix.attrs["coords_mode__j"] = "trivial"
    adjacency_matrix[-1, :] = network

    return (
        or_sizes.to(device),
        training_dset_sizes.to(device),
        training_or_sizes.to(device),
        network.to(device),
        time_series,
    )
