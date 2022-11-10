import logging
import sys
from os.path import dirname as up
from typing import Union

import numpy as np
import h5py as h5
import networkx as nx
import torch
from dantro._import_tools import import_module_from_path
import dantro.groups.graph
from dantro.containers import XrDataContainer
sys.path.append(up(up(up(__file__))))

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

from .ABM import Kuramoto_ABM

log = logging.getLogger(__name__)


def save_nw(network: nx.Graph, nw_group: h5.Group, write_adjacency_matrix: bool = False):

    """ Saves a network to a h5.Group

    :param network: the network to save
    :param nw_group: the h5.Group
    :param write_adjacency_matrix: whether to write out the entire adjacency matrix
    """

    # Vertices
    vertices = nw_group.create_dataset(
        "_vertices", (1, network.number_of_nodes()), chunks=True, compression=3, dtype=int
    )
    vertices.attrs["dim_names"] = ["time", "vertex_idx"]
    vertices.attrs["coords_mode__vertex_idx"] = "trivial"

    # Vertex properties
    eigen_frequencies = nw_group.create_dataset(
        "_eigen_frequencies", (1, network.number_of_nodes()), chunks=True, compression=3, dtype=float
    )
    eigen_frequencies.attrs["dim_names"] = ["time", "vertex_idx"]
    eigen_frequencies.attrs["coords_mode__vertex_idx"] = "trivial"

    # Edges; the network size is assumed to remain constant
    edges = nw_group.create_dataset(
        "_edges",
        (1, network.size(), 2),
        chunks=True,
        compression=3,
    )
    edges.attrs["dim_names"] = ["time", "edge_idx", "vertex_idx"]
    edges.attrs["coords_mode__edge_idx"] = "trivial"
    edges.attrs["coords_mode__vertex_idx"] = "trivial"

    # Edge properties
    edge_weights = nw_group.create_dataset(
        "_edge_weights",
        (1, network.size()),
        chunks=True,
        compression=3,
    )
    edge_weights.attrs["dim_names"] = ["time", "edge_idx"]
    edge_weights.attrs["coords_mode__edge_idx"] = "trivial"

    # Topological properties
    degree = nw_group.create_dataset(
        "_degree",
        (1, network.number_of_nodes()),
        maxshape=(1, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=int
    )
    degree.attrs["dim_names"] = ["time", "vertex_idx"]
    degree.attrs["coords_mode__vertex_idx"] = "trivial"

    degree_w = nw_group.create_dataset(
        "_degree_weighted",
        (1, network.number_of_nodes()),
        maxshape=(1, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=float
    )
    degree_w.attrs["dim_names"] = ["time", "vertex_idx"]
    degree_w.attrs["coords_mode__vertex_idx"] = "trivial"

    clustering = nw_group.create_dataset(
        "_clustering",
        (1, network.number_of_nodes()),
        maxshape=(1, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=float,
    )
    clustering.attrs["dim_names"] = ["time", "vertex_idx"]
    clustering.attrs["coords_mode__vertex_idx"] = "trivial"

    clustering_w = nw_group.create_dataset(
        "_clustering_weighted",
        (1, network.number_of_nodes()),
        maxshape=(1, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=float,
    )
    clustering_w.attrs["dim_names"] = ["time", "vertex_idx"]
    clustering_w.attrs["coords_mode__vertex_idx"] = "trivial"

    # Write network properties
    vertices[0, :] = network.nodes()
    eigen_frequencies[0, :] = torch.stack(list(nx.get_node_attributes(network, "eigen_frequency").values())).numpy().flatten()
    edges[0, :, :] = network.edges()
    edge_weights[:, :] = list(nx.get_edge_attributes(network, "weight").values())
    degree[0, :] = [network.degree(i) for i in network.nodes()]
    degree_w[0, :] = [deg[1] for deg in network.degree(weight="weight")]
    clustering[0, :] = [val for val in nx.clustering(network).values()]
    clustering_w[0, :] = [val for val in nx.clustering(network, weight="weight").values()]

    if write_adjacency_matrix:

        adj_matrix = nx.to_numpy_matrix(network)

        # Adjacency matrix: only written out if explicity specified
        adjacency_matrix = nw_group.create_dataset(
            "_adjacency_matrix",
            [1] + list(np.shape(adj_matrix)),
            chunks=True,
            compression=3,
        )
        adjacency_matrix.attrs["dim_names"] = ["time", "i", "j"]
        adjacency_matrix.attrs["coords_mode__i"] = "trivial"
        adjacency_matrix.attrs["coords_mode__j"] = "trivial"
        adjacency_matrix[-1, :] = adj_matrix

# --- Data generation functions ------------------------------------------------------------------------------------


def get_data(
    cfg, h5file: h5.File, h5group: h5.Group, *, seed: int, device: str
) -> (torch.Tensor, Union[nx.Graph, None]):

    load_from_dir = cfg.pop("load_from_dir", {})
    write_adjacency_matrix = cfg.pop("write_adjacency_matrix", False)

    training_data, network, eigen_frequencies = None, None, None

    if load_from_dir.get("training_data", None) is not None:

        log.info("   Loading training data ...")
        with h5.File(load_from_dir["training_data"], "r") as f:

            # Load the training data
            training_data = torch.from_numpy(np.array(f["training_data"]["training_data"])).float()

    if load_from_dir.get("network", None) is not None:

        log.info("   Loading network")
        with h5.File(load_from_dir["network"], "r") as f:

            # Load the network
            GG = dantro.groups.GraphGroup(name="true_network",
                                          attrs=dict(directed=f["true_network"].attrs['is_directed'],
                                                     parallel=f["true_network"].attrs['allows_parallel'])) #(f["true_network"])
            GG.new_container(
                "nodes",
                Cls=XrDataContainer,
                data=np.array(f["true_network"]["_vertices"])
            )
            GG.new_container(
                "edges",
                Cls=XrDataContainer,
                data=[]
            )

            edges = np.array(f["true_network"]["_edges"])
            edge_weights = np.expand_dims(np.array(f["true_network"]["_edge_weights"]), -1)
            weighted_edges = np.squeeze(np.concatenate((edges, edge_weights), axis=-1))

            network = GG.create_graph()
            network.add_weighted_edges_from(weighted_edges, "weight")

    if load_from_dir.get("eigen_frequencies", None) is not None:

        log.info("   Loading eigenfrequencies")
        with h5.File(load_from_dir["eigen_frequencies"], "r") as f:

            # Load the network
            eigen_frequencies = torch.from_numpy(np.array(f["true_network"]["_eigen_frequencies"])).float()
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

    nx.set_node_attributes(network, {idx: val for idx, val in enumerate(eigen_frequencies)}, "eigen_frequency")

    # If training data was not loaded, generate
    if training_data is None:

        log.info("   Generating training data ...")
        num_steps: int = cfg.pop("num_steps")
        training_set_size = cfg.pop("training_set_size")

        training_data = torch.empty((training_set_size, num_steps + 1, N, 1))
        adj_matrix = torch.from_numpy(nx.to_numpy_matrix(network)).float()

        ABM = Kuramoto_ABM(**cfg, eigen_frequencies=eigen_frequencies)

        for idx in range(training_set_size):

            initial_phases = 2 * torch.pi * torch.rand(N, 1)
            training_data[idx, 0, :, :] = initial_phases

            # Run the ABM for n iterations and write the data
            for i in range(num_steps):
                training_data[idx, i + 1] = ABM.run_single(
                    current_phases=training_data[idx, i],
                    adjacency_matrix=adj_matrix,
                    requires_grad=False
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
        save_nw(network, nw_group, write_adjacency_matrix)
        log.info("   Network generated and saved.")

        # Save training data
        dset_training_data = h5group.create_dataset(
            "training_data",
            training_data.shape,
            chunks=True,
            compression=3,
        )
        dset_training_data.attrs["dim_names"] = ["runs", "time", "vertex_idx", "dim_name__0"]
        dset_training_data.attrs["coords_mode__runs"] = "trivial"
        dset_training_data.attrs["coords_mode__time"] = "start_and_step"
        dset_training_data.attrs["coords__time"] = [1, 1]
        dset_training_data.attrs["coords_mode__vertex_idx"] = "trivial"

        dset_training_data[:, :] = training_data

    # Return the training data and the network
    return training_data.to(device), network
