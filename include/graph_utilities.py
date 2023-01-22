import numpy as np
import networkx as nx
import h5py as h5


def save_nw(
    network: nx.Graph, nw_group: h5.Group, write_adjacency_matrix: bool = False
):

    """Saves a network to a h5.Group graph group

    :param network: the network to save
    :param nw_group: the h5.Group
    :param write_adjacency_matrix: whether to write out the entire adjacency matrix
    """

    # Vertices
    vertices = nw_group.create_dataset(
        "_vertices",
        (1, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=int,
    )
    vertices.attrs["dim_names"] = ["time", "vertex_idx"]
    vertices.attrs["coords_mode__vertex_idx"] = "trivial"

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
        dtype=int,
    )
    degree.attrs["dim_names"] = ["time", "vertex_idx"]
    degree.attrs["coords_mode__vertex_idx"] = "trivial"

    degree_w = nw_group.create_dataset(
        "_degree_weighted",
        (1, network.number_of_nodes()),
        maxshape=(1, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=float,
    )
    degree_w.attrs["dim_names"] = ["time", "vertex_idx"]
    degree_w.attrs["coords_mode__vertex_idx"] = "trivial"

    triangles = nw_group.create_dataset(
        "_triangles",
        (1, network.number_of_nodes()),
        maxshape=(1, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=float,
    )
    triangles.attrs["dim_names"] = ["time", "vertex_idx"]
    triangles.attrs["coords_mode__vertex_idx"] = "trivial"

    triangles_w = nw_group.create_dataset(
        "_triangles_weighted",
        (1, network.number_of_nodes()),
        maxshape=(1, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=float,
    )
    triangles_w.attrs["dim_names"] = ["time", "vertex_idx"]
    triangles_w.attrs["coords_mode__vertex_idx"] = "trivial"

    # Write network properties
    vertices[0, :] = network.nodes()
    edges[0, :, :] = network.edges()
    edge_weights[:, :] = list(nx.get_edge_attributes(network, "weight").values())
    degree[0, :] = [network.degree(i) for i in network.nodes()]
    degree_w[0, :] = [deg[1] for deg in network.degree(weight="weight")]
    triangles[0, :] = [
        val
        for val in np.diagonal(
            np.linalg.matrix_power(np.ceil(nx.to_numpy_matrix(network)), 3)
        )
    ]
    triangles_w[0, :] = [
        val
        for val in np.diagonal(np.linalg.matrix_power(nx.to_numpy_matrix(network), 3))
    ]

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
