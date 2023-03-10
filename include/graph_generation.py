import networkx as nx
import numpy as np

""" Network generation function """


def generate_graph(
        *, N: int, mean_degree: int = None, type: str, seed: int = None, graph_props: dict = None
) -> nx.Graph:

    """ Generates graphs of a given topology

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

        G = _connect_isolates(
            nx.barabasi_albert_graph(N, mean_degree, seed)
        )

    # Directed scale-free graph
    elif type.lower() == "bollobasriordan":

        G = nx.DiGraph(_connect_isolates(
            nx.scale_free_graph(N, **graph_props["BollobasRiordan"], seed=seed)
        ))

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
