import networkx as nx
import numpy as np

""" Graph generation functions """

def connect_isolates(G: nx.Graph) -> nx.Graph:
    """Connects isolated vertices to main network body."""

    isolates = list(nx.isolates(G))
    N = G.number_of_nodes()
    for v in isolates:
        new_nb = np.random.randint(0, N)
        while new_nb in isolates:
            new_nb = np.random.randint(0, N)
        G.add_edge(v, new_nb)

    return G


def generate_graph(
    *, N: int, mean_degree: int, type: str, seed: int = None, graph_props: dict = None
) -> nx.Graph:
    G = nx.empty_graph

    # Random graphs
    if type == "random":

        is_directed: bool = graph_props["is_directed"]

        G = connect_isolates(
            nx.fast_gnp_random_graph(
                N, mean_degree / (N - 1), directed=is_directed, seed=seed
            )
        )

    # Small-world (Watts-Strogatz) graph
    elif type == "WattsStrogatz":

        p: float = graph_props["WattsStrogatz"]["p_rewire"]

        G = connect_isolates(nx.watts_strogatz_graph(N, mean_degree, p, seed))

    else:
        raise ValueError(f"Unrecognised graph type '{type}'!")

    if graph_props["is_weighted"]:
        for e in G.edges():
            G[e[0]][e[1]]["weight"] = np.random.rand()

    return G
