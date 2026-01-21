import logging
import sys
from os.path import dirname as up
from typing import Union

import dantro.groups.graph
import h5py as h5
import networkx as nx
import numpy as np
import torch
from dantro._import_tools import import_module_from_path
from dantro.containers import XrDataContainer

sys.path.append(up(up(up(__file__))))

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

from .ABM import Kuramoto_ABM

log = logging.getLogger(__name__)


def get_data(
    cfg,
    h5file: h5.File,
    h5group: h5.Group,
    *,
    seed: int,
    device: str,
) -> (torch.Tensor, Union[nx.Graph, None]):
    """Either loads data from an external file or synthetically generates Kuramoto data (including the network)
    from a configuration file.

    :param cfg: The configuration file containing either the paths to files to be loaded or the configuration settings
        for the synthetic data generation
    :param h5file: the h5 file to which to write any data to
    :param h5group: the h5 group tow hich
    :param seed: the seed to use for the graph generation
    :param device: the training device to which to move the data
    :return: the training data and, if given, the network
    """

    load_from_dir: dict = cfg.get("load_from_dir", {})
    write_adjacency_matrix: bool = cfg.get(
        "write_adjacency_matrix", load_from_dir == {}
    )

    training_data, network, eigen_frequencies = None, None, None

    if load_from_dir.get("training_data", None) is not None:
        log.info("   Loading training data ...")
        with h5.File(load_from_dir["training_data"], "r") as f:
            # Load the training data
            training_data = torch.from_numpy(
                np.array(f["training_data"]["phases"])
            ).float()

    if load_from_dir.get("network", None) is not None:
        log.info("   Loading network ... ")
        with h5.File(load_from_dir["network"], "r") as f:
            # Load the network
            GG = dantro.groups.GraphGroup(
                name="true_network",
                attrs=dict(
                    directed=f["true_network"].attrs["is_directed"],
                    parallel=f["true_network"].attrs["allows_parallel"],
                ),
            )
            GG.new_container(
                "nodes",
                Cls=XrDataContainer,
                data=np.array(f["true_network"]["_vertices"]),
            )
            GG.new_container("edges", Cls=XrDataContainer, data=[])

            edges = np.array(f["true_network"]["_edges"])
            edge_weights = np.expand_dims(
                np.array(f["true_network"]["_edge_weights"]), -1
            )
            weighted_edges = np.squeeze(np.concatenate((edges, edge_weights), axis=-1))

            network = GG.create_graph()
            network.add_weighted_edges_from(weighted_edges, "weight")

    if load_from_dir.get("eigen_frequencies", None) is not None:
        log.info("   Loading eigenfrequencies ... ")
        with h5.File(load_from_dir["eigen_frequencies"], "r") as f:
            # Load the network
            eigen_frequencies = torch.from_numpy(
                np.array(f["training_data"]["eigen_frequencies"])
            ).float()

    # Get the config and number of agents
    dt: float = cfg.get("dt")
    alpha: float = cfg.get("alpha")
    beta: float = cfg.get("beta")
    kappa: float = cfg.get("kappa")
    data_cfg: dict = cfg.get("synthetic_data")
    nw_cfg: dict = data_cfg.pop("network", {})

    # If network was loaded, set the number of nodes to be the network size
    if network is not None:
        data_cfg.update(dict(N=network.number_of_nodes()))
    data_cfg.update(dict(dt=dt, alpha=alpha, beta=beta, kappa=kappa))
    N: int = data_cfg["N"]

    # If network was not loaded, generate the network
    if network is None:
        log.info("   Generating graph ...")
        network: nx.Graph = base.generate_graph(N=N, **nw_cfg, seed=seed)

    # If eigenfrequencies were not loaded, generate
    if eigen_frequencies is None:
        log.info("   Generating eigenfrequencies  ...")

        num_steps: int = data_cfg.get("num_steps")
        training_set_size: int = data_cfg.get("training_set_size")

        # Generate a time series of i.i.d distributed eigenfrequencies
        eigen_frequencies = base.random_tensor(
            data_cfg.get("eigen_frequencies"),
            size=(training_set_size, 1, N, 1),
            device=device,
        )
        for _ in range(num_steps):
            eigen_frequencies = torch.concat(
                (
                    eigen_frequencies,
                    eigen_frequencies[:, -1, :, :].unsqueeze(1)
                    + torch.normal(
                        0.0,
                        data_cfg.get("eigen_frequencies")["time_series_std"],
                        size=(training_set_size, 1, N, 1),
                    ),
                ),
                dim=1,
            )
    # If training data was not loaded, generate
    if training_data is None:
        log.info("   Generating training data ...")
        num_steps: int = data_cfg.get("num_steps")
        training_set_size = data_cfg.get("training_set_size")
        training_data = torch.empty(
            (training_set_size, num_steps + 1, N, 1), device=device
        )

        adj_matrix = torch.from_numpy(nx.to_numpy_array(network)).float().to(device)

        ABM = Kuramoto_ABM(**data_cfg, device=device)

        for idx in range(training_set_size):
            training_data[idx, 0, :, :] = base.random_tensor(
                data_cfg.get("init_phases"), size=(N, 1), device=device
            )

            # For the second-order dynamics, the initial velocities must also be given
            if ABM.alpha != 0:
                training_data[idx, 1, :, :] = training_data[idx, 0, :, :] + torch.rand(
                    N, 1
                )

            # Second order dynamics require an additional initial condition and so start one step later
            t_0 = 1 if ABM.alpha != 0 else 0

            # Run the ABM for n iterations and write the data
            for i in range(t_0, num_steps):
                training_data[idx, i + 1] = ABM.run_single(
                    current_phases=training_data[idx, i],
                    current_velocities=(
                        training_data[idx, i] - training_data[idx, i - 1]
                    )
                    / ABM.dt,
                    adjacency_matrix=adj_matrix,
                    eigen_frequencies=eigen_frequencies[idx, i],
                    requires_grad=False,
                )

        log.info("   Training data generated.")

    # Save the data. If data was loaded, data can be copied if specified
    if load_from_dir.get("copy_data", True):
        # Create a graph group for the network and save it and its properties
        nw_group = h5file.create_group("true_network")
        nw_group.attrs["content"] = "graph"
        nw_group.attrs["allows_parallel"] = False
        nw_group.attrs["is_directed"] = network.is_directed()
        base.save_nw(network, nw_group, write_adjacency_matrix=write_adjacency_matrix)
        log.info("   Network generated and saved.")

        # Save the eigenfrequencies
        dset_eigen_frequencies = h5group.create_dataset(
            "eigen_frequencies",
            eigen_frequencies.shape,
            chunks=True,
            compression=3,
            dtype=float,
        )
        dset_eigen_frequencies.attrs["dim_names"] = [
            "training_set",
            "time",
            "vertex_idx",
            "dim_name__0",
        ]
        dset_eigen_frequencies.attrs["coords_mode__training_set"] = "trivial"
        dset_eigen_frequencies.attrs["coords_mode__time"] = "trivial"
        dset_eigen_frequencies.attrs["coords_mode__vertex_idx"] = "values"
        dset_eigen_frequencies.attrs["coords__vertex_idx"] = network.nodes()
        dset_eigen_frequencies[:, :] = eigen_frequencies.cpu()

        # Save training data
        dset_phases = h5group.create_dataset(
            "phases",
            training_data.shape,
            chunks=True,
            compression=3,
        )
        dset_phases.attrs["dim_names"] = [
            "training_set",
            "time",
            "vertex_idx",
            "dim_name__0",
        ]
        dset_phases.attrs["coords_mode__training_set"] = "trivial"
        dset_phases.attrs["coords_mode__time"] = "trivial"
        dset_phases.attrs["coords_mode__vertex_idx"] = "values"
        dset_phases.attrs["coords__vertex_idx"] = network.nodes()

        dset_phases[:, :] = training_data.cpu()

    # Return the training data and the network
    return training_data, eigen_frequencies, network
