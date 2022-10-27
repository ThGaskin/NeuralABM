import networkx as nx
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# --- Graph generation functions ---------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
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
):
    # Random graphs
    if type == "random":

        is_directed: bool = graph_props["is_directed"]

        return connect_isolates(
            nx.fast_gnp_random_graph(
                N, mean_degree / (N - 1), directed=is_directed, seed=seed
            )
        )

    # Small-world (Watts-Strogatz) graph
    elif type == "WattsStrogatz":

        p: float = graph_props["WattsStrogatz"]["p_rewire"]

        return connect_isolates(nx.watts_strogatz_graph(N, mean_degree, p, seed))

    else:
        raise ValueError(f"Unrecognised graph type '{type}'!")
