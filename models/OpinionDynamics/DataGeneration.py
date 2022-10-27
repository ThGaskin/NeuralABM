import logging
import sys
from os.path import dirname as up
from typing import Union

import h5py as h5
import networkx as nx
import numpy as np
import torch
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(up(__file__))))

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

from .ABM import OpinionDynamics_ABM

log = logging.getLogger(__name__)


# --- Data generation functions ------------------------------------------------------------------------------------


def get_data(
    cfg, h5file: h5.File, h5group: h5.Group, *, seed: int, device: str
) -> (torch.Tensor, Union[nx.Graph, None]):

    load_from_dir = cfg.pop("load_from_dir", None)
    if load_from_dir is not None:
        # TODO load data from dir
        # If dir contains training data and network, return both
        # If dir only contains training data, return only training data, with network = None
        pass

    else:
        cfg = cfg.pop("synthetic_data")
        N: int = cfg["N"]
        num_steps: int = cfg.pop("num_steps")
        write_start: int = 1
        write_every: int = 1

        # Create dataset for the training data
        opinions = h5group.create_dataset(
            "true_opinions",
            (num_steps + 1, N),
            maxshape=(num_steps + 1, N),
            chunks=True,
            compression=3,
        )
        opinions.attrs["dim_names"] = ["time", "vertex_idx"]
        opinions.attrs["coords_mode__time"] = "start_and_step"
        opinions.attrs["coords__time"] = [write_start, write_every]
        opinions.attrs["coords_mode__vertex_idx"] = "trivial"

        # Create a graph group for the network. Edge weights are given by the difference in opinion space
        nw_group = h5file.create_group("true_network")

        # Generate the network
        log.info("   Generating the network (step 1 of 3) ... ")
        nw_cfg = cfg.pop("network")

        network = base.generate_graph(N=N, **nw_cfg, seed=seed)

        # Save the network
        log.info("   Saving the network (step 2 of 3) ... ")

        # Vertices
        vertices = nw_group.create_dataset(
            "_vertices", (1, network.number_of_nodes()), chunks=True, compression=3, dtype=int
        )
        vertices.attrs["dim_names"] = ["dim_name__0", "vertex_idx"]
        vertices.attrs["coords_mode__vertex_idx"] = "trivial"

        # Write network nodes
        vertices[0, :] = network.nodes()

        # Edges; the network size is assumed to remain constant
        edges = nw_group.create_dataset(
            "_edges",
            (1, network.size(), 2),
            chunks=True,
            compression=3,
        )
        edges.attrs["dim_names"] = ["dim_name__1", "edge_idx", "vertex_idx"]
        edges.attrs["coords_mode__edge_idx"] = "trivial"
        edges.attrs["coords_mode__vertex_idx"] = "trivial"

        # Write network edges
        edges[0, :, :] = network.edges()

        # Edge properties
        edge_weights = nw_group.create_dataset(
            "_edge_weights",
            (num_steps + 1, network.size()),
            maxshape=(num_steps + 1, network.size()),
            chunks=True,
            compression=3,
        )
        edge_weights.attrs["dim_names"] = ["time", "edge_idx"]
        edge_weights.attrs["coords_mode__time"] = "start_and_step"
        edge_weights.attrs["coords__time"] = [write_start, write_every]
        edge_weights.attrs["coords_mode__edge_idx"] = "trivial"

        # Interaction status
        interactions = nw_group.create_dataset(
            "_interaction",
            (1, network.size()),
            maxshape=(1, network.size()),
            chunks=True,
            compression=3,
            dtype=int
        )
        interactions.attrs["dim_names"] = ["dim_name__0", "edge_idx"]
        interactions.attrs["coords_mode__edge_idx"] = "trivial"

        # Topological properties
        degree = nw_group.create_dataset(
            "_degree", (1, N), maxshape=(None, N), chunks=True, compression=3, dtype=int
        )
        degree.attrs["dim_names"] = ["dim_name__0", "vertex_idx"]
        degree.attrs["coords_mode__vertex_idx"] = "trivial"

        degree[0, :] = [network.degree(i) for i in network.nodes()]

        clustering = nw_group.create_dataset(
            "_clustering",
            (1, N),
            maxshape=(None, N),
            chunks=True,
            compression=3,
            dtype=float,
        )
        clustering.attrs["dim_names"] = ["dim_name__0", "vertex_idx"]
        clustering.attrs["coords_mode__vertex_idx"] = "trivial"

        clustering[0, :] = [val for val in nx.clustering(network).values()]

        nw_group.attrs["content"] = "graph"
        nw_group.attrs["allows_parallel"] = False
        nw_group.attrs["is_directed"] = network.is_directed()

        # Setup the ABM and generate synthetic data
        log.info("   Generating the time series data (step 3 of 3) ... ")
        ABM = OpinionDynamics_ABM(**cfg, network=network)
        training_data = torch.empty((num_steps + 1, N, 1))
        training_data[0, :, :] = ABM.initial_opinions
        opinions[0, :] = torch.flatten(training_data[0, :, :])
        edge_weights[0, :] = ABM.edge_weights

        # Run the ABM for n iterations and write the data
        for i in range(num_steps):
            training_data[i + 1] = ABM.run_single(
                current_values=training_data[i], requires_grad=False
            )

            # Write the edge properties (the opinion differences)
            edge_weights[i + 1, :] = ABM.edge_weights
            opinions[i + 1, :] = torch.flatten(training_data[i + 1, :, :]).numpy()

        interactions[0, :] = ABM.interactions
        log.info("   Done.")
        del ABM

        # Return the training data and the network
        return training_data.to(device), network
