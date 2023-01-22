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
    cfg, h5file: h5.File, h5group: h5.Group, *, seed: int, device: str, second_order: bool,
) -> (torch.Tensor, Union[nx.Graph, None]):

    """ Either loads data from an external file or synthetically generates Kuramoto data (including the network)
    from a configuration file.

    :param cfg: The configuration file containing either the paths to files to be loaded or the configuration settings
        for the synthetic data generation
    :param h5file: the h5 file to which to write any data to
    :param h5group: the h5 group tow hich
    :param seed: the seed to use for the graph generation
    :param device: the training device to which to move the data
    :return: the training data and, if given, the network
    """
    load_from_dir = cfg.pop("load_from_dir", {})
    write_adjacency_matrix = cfg.pop("write_adjacency_matrix", load_from_dir == {})

    training_data, network, eigen_frequencies = None, None, None

    if load_from_dir.get("training_data", None) is not None:

        log.info("   Loading training data ...")
        with h5.File(load_from_dir["training_data"], "r") as f:

            # Load the training data
            training_data = torch.from_numpy(
                np.array(f["training_data"]["training_data"])
            ).float()

    if load_from_dir.get("network", None) is not None:

        log.info("   Loading network")
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

        log.info("   Loading eigenfrequencies")
        with h5.File(load_from_dir["eigen_frequencies"], "r") as f:

            # Load the network
            eigen_frequencies = torch.from_numpy(
                np.array(f["true_network"]["_eigen_frequencies"])
            ).float()
            eigen_frequencies = torch.reshape(eigen_frequencies, (-1, 1))

    # Get the config and number of agents
    cfg = cfg.get("synthetic_data")
    nw_cfg = cfg.pop("network", {})
    N: int = cfg["N"]

    # If network was not loaded, generate the network
    if network is None:

        log.info("   Generating graph ...")
        network = base.generate_graph(N=N, **nw_cfg, seed=seed)

    # If eigenfrequencies were not loaded, generate
    if eigen_frequencies is None:

        log.info("   Generating eigenfrequencies  ...")
        eigen_frequencies = 2 * torch.rand((N, 1), dtype=torch.float) + 1

    nx.set_node_attributes(
        network,
        {idx: val for idx, val in enumerate(eigen_frequencies)},
        "eigen_frequency",
    )

    # If training data was not loaded, generate
    if training_data is None:

        log.info("   Generating training data ...")
        num_steps: int = cfg.get("num_steps")
        training_set_size = cfg.get("training_set_size")
        training_data = torch.empty((training_set_size, num_steps + 1, N, 1))

        adj_matrix = torch.from_numpy(nx.to_numpy_matrix(network)).float()

        ABM = Kuramoto_ABM(**cfg, eigen_frequencies=eigen_frequencies)

        for idx in range(training_set_size):

            training_data[idx, 0, :, :] = 2 * torch.pi * torch.rand(N, 1)
            i_0 = 0

            # For the second-order dynamics, the initial velocities must also be given
            if second_order:
                training_data[idx, 1, :, :] = training_data[idx, 0, :, :] + torch.rand(N, 1)
                i_0 = 1

            # Run the ABM for n iterations and write the data
            for i in range(i_0, num_steps):
                training_data[idx, i + 1] = ABM.run_single(
                    current_phases=training_data[idx, i],
                    current_velocities=(training_data[idx, i] - training_data[idx, i-1])/ABM.dt if second_order else None,
                    adjacency_matrix=adj_matrix,
                    requires_grad=False,
                )

        log.info("   Training data generated.")

        del ABM

    # Save the data. If data was loaded, data can be copied if specified
    if load_from_dir.get("copy_data", True):

        # Create a graph group for the network and save it and its properties
        nw_group = h5file.create_group("true_network")
        nw_group.attrs["content"] = "graph"
        nw_group.attrs["allows_parallel"] = False
        nw_group.attrs["is_directed"] = network.is_directed()

        # Save the network
        base.save_nw(network, nw_group, write_adjacency_matrix)

        # Vertex properties: eigenfrequencies
        eigen_frequencies = nw_group.create_dataset(
            "_eigen_frequencies",
            (1, network.number_of_nodes()),
            chunks=True,
            compression=3,
            dtype=float,
        )
        eigen_frequencies.attrs["dim_names"] = ["time", "vertex_idx"]
        eigen_frequencies.attrs["coords_mode__vertex_idx"] = "trivial"

        # Write node properties
        eigen_frequencies[0, :] = (
            torch.stack(list(nx.get_node_attributes(network, "eigen_frequency").values()))
                .numpy()
                .flatten()
        )
        log.info("   Network generated and saved.")

        # Save training data
        dset_training_data = h5group.create_dataset(
            "training_data",
            training_data.shape,
            chunks=True,
            compression=3,
        )
        dset_training_data.attrs["dim_names"] = [
            "runs",
            "time",
            "vertex_idx",
            "dim_name__0",
        ]
        dset_training_data.attrs["coords_mode__runs"] = "trivial"
        dset_training_data.attrs["coords_mode__time"] = "start_and_step"
        dset_training_data.attrs["coords__time"] = [1, 1]
        dset_training_data.attrs["coords_mode__vertex_idx"] = "trivial"

        dset_training_data[:, :] = training_data

    # Return the training data and the network
    return training_data.to(device), network

