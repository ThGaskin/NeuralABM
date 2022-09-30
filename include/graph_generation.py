import networkx as nx
import numpy as np
import torch

# ----------------------------------------------------------------------------------------------------------------------
# --- Graph generation functions ---------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def generate_graph(*, N: int, mean_degree: int, type: str, seed: int = None, graph_props: dict = None):

    # Random graphs
    if type == 'random':

        is_directed: bool = graph_props['is_directed']

        return nx.fast_gnp_random_graph(N, mean_degree / (N - 1), directed=is_directed, seed=seed)

    # Small-world (Watts-Strogatz) graph
    elif type == 'WattsStrogatz':

        p: float = graph_props['WattsStrogatz']['p_rewire']

        return nx.watts_strogatz_graph(N, mean_degree, p, seed)

    else:
        raise ValueError(f"Unrecognised graph type '{type}'!")


