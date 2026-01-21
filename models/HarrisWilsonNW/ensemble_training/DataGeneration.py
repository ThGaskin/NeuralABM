import logging
import sys
from os.path import dirname as up
from typing import Tuple

import h5py as h5
import numpy as np
import torch
from dantro._import_tools import import_module_from_path

from .ABM import HarrisWilsonABM

sys.path.append(up(up(up(__file__))))

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

""" Load a dataset or generate synthetic data on which to train the neural net """

log = logging.getLogger(__name__)


def get_data(
    cfg, h5file: h5.File, h5group: h5.Group, device: str = "cpu"
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """Generates synthetic Harris-Wilson using a numerical solver.

    :param cfg: the configuration file
    :returns the origin sizes, network, and the time series
    """

    load_from_dir: dict = cfg.get("load_from_dir", {})

    origin_sizes, network, destination_sizes = None, None, None

    if load_from_dir.get("origin_zones", None) is not None:
        log.info("   Loading origin sizes ... ")
        with h5.File(load_from_dir["origin_zones"], "r") as f:
            # Load the origin sizes
            origin_sizes = torch.from_numpy(
                np.array(f["training_data"]["origin_sizes"])
            ).float()

    if load_from_dir.get("destination_zones", None) is not None:
        log.info("   Loading destination zones ...")
        with h5.File(load_from_dir["destination_zones"], "r") as f:
            # Load the training data
            destination_sizes = torch.from_numpy(
                np.array(f["training_data"]["destination_sizes"])
            ).float()

    if load_from_dir.get("network", None) is not None:
        log.info("   Loading network ... ")
        with h5.File(load_from_dir["network"], "r") as f:
            # Load the network
            network = torch.from_numpy(
                np.array(f["true_network"]["_adjacency_matrix"])
            ).float()

    # Get run configuration properties
    data_cfg = cfg["synthetic_data"]
    if network is not None:
        data_cfg.update(dict(N_origin=network.shape[0]))
        data_cfg.update(dict(N_destination=network.shape[1]))
    N_origin, N_destination = data_cfg["N_origin"], data_cfg["N_destination"]
    num_steps = data_cfg["num_steps"]

    # If origin sizes were not loaded, generate them
    if origin_sizes is None:
        log.info("   Generating origin sizes  ...")
        num_steps: int = data_cfg.get("num_steps")
        training_set_size: int = cfg.get("training_set_size")

        origin_sizes = torch.empty(
            (training_set_size, num_steps + 1, N_origin, 1), device=device
        )

        for idx in range(training_set_size):
            origin_sizes[idx, 0, :, :] = torch.abs(
                base.random_tensor(
                    data_cfg.get("init_origin_sizes"), size=(1, N_origin, 1)
                )
            )

            for __ in range(1, num_steps + 1):
                origin_sizes[idx, __, :, :] = torch.abs(
                    origin_sizes[idx, __ - 1, :, :]
                    + torch.normal(0, data_cfg["origin_size_std"], size=(N_origin, 1))
                )

    # If network was not loaded, generate the network
    if network is None:
        # Generate the network
        network: torch.tensor = torch.exp(
            -1
            * torch.abs(
                base.random_tensor(
                    data_cfg.get("init_network_weights"),
                    size=(N_origin, N_destination),
                )
            )
        )
        network = network * torch.bernoulli(network)  # what if we turn this off?

        # Normalise the rowsums
        norms = torch.sum(network, dim=1, keepdim=True)
        network /= torch.where(norms != 0, norms, 1)

    # If time series was not loaded, generate
    if destination_sizes is None:
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

        training_set_size = cfg["training_set_size"]
        destination_sizes = torch.empty(
            (training_set_size, num_steps + 1, N_destination, 1), device=device
        )
        for idx in range(training_set_size):
            # Generate the initial destination zone sizes
            destination_sizes[idx, 0, :, :] = torch.abs(
                base.random_tensor(
                    data_cfg.get("init_dest_sizes"),
                    size=(data_cfg["N_destination"], 1),
                )
            )

            # Run the ABM for n iterations, generating the entire time series
            destination_sizes[idx, :, :, :] = ABM.run(
                init_data=destination_sizes[idx, 0, :, :],
                adjacency_matrix=network,
                n_iterations=num_steps,
                origin_sizes=origin_sizes[idx, :],
                generate_time_series=True,
            )

    # Set up dataset for complete synthetic time series
    dset_dest_zones = h5group.create_dataset(
        "destination_sizes",
        destination_sizes.shape,
        maxshape=destination_sizes.shape,
        chunks=True,
        compression=3,
    )
    dset_dest_zones.attrs["dim_names"] = [
        "training_set",
        "time",
        "zone_id",
        "dim_name__0",
    ]
    dset_dest_zones.attrs["coords_mode__training_set"] = "trivial"
    dset_dest_zones.attrs["coords_mode__time"] = "trivial"
    dset_dest_zones.attrs["coords_mode__zone_id"] = "values"
    dset_dest_zones.attrs["coords__zone_id"] = np.arange(
        N_origin, N_origin + N_destination, 1
    )

    # Write the time series data
    dset_dest_zones[:,] = destination_sizes

    # Training time series
    # Extract the training data from the time series data and save
    num_training_steps = cfg.get("num_training_steps")

    training_or_sizes, training_dset_sizes = (
        origin_sizes[:, -num_training_steps:, :, :],
        destination_sizes[:, -num_training_steps:],
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
        origin_sizes.shape,
        maxshape=origin_sizes.shape,
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
    dset_origin_sizes[:,] = origin_sizes

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
        network.shape,
        chunks=True,
        compression=3,
    )
    adjacency_matrix.attrs["dim_names"] = ["i", "j"]
    adjacency_matrix.attrs["coords_mode__i"] = "trivial"
    adjacency_matrix.attrs["coords_mode__j"] = "trivial"
    adjacency_matrix[:, :] = network

    return (
        origin_sizes.to(device),
        training_dset_sizes.to(device),
        training_or_sizes.to(device),
        network.to(device),
        destination_sizes,
    )
