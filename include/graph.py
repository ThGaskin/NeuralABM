import h5py as h5
import networkx as nx
import numpy as np

""" Network generation function """


def generate_graph(
    *,
    N: int,
    mean_degree: int = None,
    type: str,
    seed: int = None,
    graph_props: dict = None,
) -> nx.Graph:
    """Generates graphs of a given topology

    :param N: number of nodes
    :param mean_degree: mean degree of the graph
    :param type: graph topology kind; can be any of 'random', 'BarabasiAlbert' (scale-free undirected), 'BollobasRiordan'
        (scale-free directed), 'WattsStrogatz' (small-world)
    :param seed: the random seed to use for the graph generation (ensuring the graphs are always the same)
    :param graph_props: dictionary containing the type-specific parameters
    :return: the networkx graph object. All graphs are fully connected
    """

    def _connect_isolates(G: nx.Graph) -> nx.Graph:
        """Connects isolated vertices to main network body."""
        isolates = list(nx.isolates(G))
        N = G.number_of_nodes()
        for v in isolates:
            new_nb = np.random.randint(0, N)
            while new_nb in isolates:
                new_nb = np.random.randint(0, N)
            G.add_edge(v, new_nb)

        return G

    # Random graph
    if type.lower() == "random":
        is_directed: bool = graph_props["is_directed"]

        G = _connect_isolates(
            nx.fast_gnp_random_graph(
                N, mean_degree / (N - 1), directed=is_directed, seed=seed
            )
        )

    # Undirected scale-free graph
    elif type.lower() == "barabasialbert":
        G = _connect_isolates(nx.barabasi_albert_graph(N, mean_degree, seed))

    # Directed scale-free graph
    elif type.lower() == "bollobasriordan":
        G = nx.DiGraph(
            _connect_isolates(
                nx.scale_free_graph(N, **graph_props["BollobasRiordan"], seed=seed)
            )
        )

    # Small-world (Watts-Strogatz) graph
    elif type.lower() == "wattsstrogatz":
        p: float = graph_props["WattsStrogatz"]["p_rewire"]

        G = _connect_isolates(nx.watts_strogatz_graph(N, mean_degree, p, seed))

    # Star graph
    elif type.lower() == "star":
        G = nx.star_graph(N)

    # Regular graph
    elif type.lower() == "regular":
        G = nx.random_regular_graph(mean_degree, N, seed)

    # Raise error upon unrecognised graph type
    else:
        raise ValueError(f"Unrecognised graph type '{type}'!")

    # Add random uniform weights to the edges
    if graph_props["is_weighted"]:
        for e in G.edges():
            G[e[0]][e[1]]["weight"] = np.random.rand()

    return G


def save_nw(
    network: nx.Graph,
    nw_group: h5.Group,
    *,
    write_adjacency_matrix: bool = False,
    static: bool = True,
):
    """Saves a network to a h5.Group graph group. The network can be either dynamic of static,
    static in this case meaning that none of the datasets have a time dimension.

    :param network: the network to save
    :param nw_group: the h5.Group
    :param write_adjacency_matrix: whether to write out the entire adjacency matrix
    :param static: whether the network is dynamic or static
    """

    # Vertices
    vertices = nw_group.create_dataset(
        "_vertices",
        (network.number_of_nodes(),) if static else (1, network.number_of_nodes()),
        maxshape=(network.number_of_nodes(),)
        if static
        else (None, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=int,
    )
    vertices.attrs["dim_names"] = ["vertex_idx"] if static else ["time", "vertex_idx"]
    vertices.attrs["coords_mode__vertex_idx"] = "trivial"

    # Edges; the network size is assumed to remain constant
    edges = nw_group.create_dataset(
        "_edges",
        (network.size(), 2) if static else (1, network.size(), 2),
        maxshape=(network.size(), 2) if static else (None, network.size(), 2),
        chunks=True,
        compression=3,
    )
    edges.attrs["dim_names"] = (
        ["edge_idx", "vertex_idx"] if static else ["time", "edge_idx", "vertex_idx"]
    )
    edges.attrs["coords_mode__edge_idx"] = "trivial"
    edges.attrs["coords_mode__vertex_idx"] = "trivial"

    # Edge properties
    edge_weights = nw_group.create_dataset(
        "_edge_weights",
        (network.size(),) if static else (1, network.size()),
        maxshape=(network.size(),) if static else (None, network.size()),
        chunks=True,
        compression=3,
    )
    edge_weights.attrs["dim_names"] = ["edge_idx"] if static else ["time", "edge_idx"]
    edge_weights.attrs["coords_mode__edge_idx"] = "trivial"

    # Topological properties
    degree = nw_group.create_dataset(
        "_degree",
        (network.number_of_nodes(),) if static else (1, network.number_of_nodes()),
        maxshape=(network.number_of_nodes(),)
        if static
        else (None, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=int,
    )
    degree.attrs["dim_names"] = ["vertex_idx"] if static else ["time", "vertex_idx"]
    degree.attrs["coords_mode__vertex_idx"] = "trivial"

    degree_w = nw_group.create_dataset(
        "_degree_weighted",
        (network.number_of_nodes(),) if static else (1, network.number_of_nodes()),
        maxshape=(network.number_of_nodes(),)
        if static
        else (None, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=float,
    )
    degree_w.attrs["dim_names"] = ["vertex_idx"] if static else ["time", "vertex_idx"]
    degree_w.attrs["coords_mode__vertex_idx"] = "trivial"

    triangles = nw_group.create_dataset(
        "_triangles",
        (network.number_of_nodes(),) if static else (1, network.number_of_nodes()),
        maxshape=(network.number_of_nodes(),)
        if static
        else (None, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=float,
    )
    triangles.attrs["dim_names"] = ["vertex_idx"] if static else ["time", "vertex_idx"]
    triangles.attrs["coords_mode__vertex_idx"] = "trivial"

    triangles_w = nw_group.create_dataset(
        "_triangles_weighted",
        (network.number_of_nodes(),) if static else (1, network.number_of_nodes()),
        maxshape=(network.number_of_nodes(),)
        if static
        else (None, network.number_of_nodes()),
        chunks=True,
        compression=3,
        dtype=float,
    )
    triangles_w.attrs["dim_names"] = (
        ["vertex_idx"] if static else ["time", "vertex_idx"]
    )
    triangles_w.attrs["coords_mode__vertex_idx"] = "trivial"

    if not static:
        for dset in [
            vertices,
            edges,
            edge_weights,
            degree,
            degree_w,
            triangles,
            triangles_w,
        ]:
            dset["coords_mode__time"] = "trivial"

    # Write network properties
    if static:
        vertices[:] = network.nodes()
        edges[:, :] = network.edges()
        edge_weights[:] = list(nx.get_edge_attributes(network, "weight").values())
        degree[:] = [network.degree(i) for i in network.nodes()]
        degree_w[:] = [deg[1] for deg in network.degree(weight="weight")]
        triangles[:] = [
            val
            for val in np.diagonal(
                np.linalg.matrix_power(np.ceil(nx.to_numpy_array(network)), 3)
            )
        ]
        triangles_w[:] = [
            val
            for val in np.diagonal(
                np.linalg.matrix_power(nx.to_numpy_array(network), 3)
            )
        ]
    else:
        vertices[0, :] = network.nodes()
        edges[0, :, :] = network.edges()
        edge_weights[0, :] = list(nx.get_edge_attributes(network, "weight").values())
        degree[0, :] = [network.degree(i) for i in network.nodes()]
        degree_w[0, :] = [deg[1] for deg in network.degree(weight="weight")]
        triangles[0, :] = [
            val
            for val in np.diagonal(
                np.linalg.matrix_power(np.ceil(nx.to_numpy_array(network)), 3)
            )
        ]
        triangles_w[0, :] = [
            val
            for val in np.diagonal(
                np.linalg.matrix_power(nx.to_numpy_array(network), 3)
            )
        ]

    if write_adjacency_matrix:
        adj_matrix = nx.to_numpy_array(network)

        # Adjacency matrix: only written out if explicity specified
        adjacency_matrix = nw_group.create_dataset(
            "_adjacency_matrix",
            np.shape(adj_matrix) if static else [1] + list(np.shape(adj_matrix)),
            chunks=True,
            compression=3,
        )
        adjacency_matrix.attrs["dim_names"] = (
            ["i", "j"] if static else ["time", "i", "j"]
        )
        adjacency_matrix.attrs["coords_mode__i"] = "trivial"
        adjacency_matrix.attrs["coords_mode__j"] = "trivial"
        if not static:
            adjacency_matrix.attrs["coords_mode__time"] = "trivial"

        if static:
            adjacency_matrix[:, :] = adj_matrix
        else:
            adjacency_matrix[0, :, :] = adj_matrix
