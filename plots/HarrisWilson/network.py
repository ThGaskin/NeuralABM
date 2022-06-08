import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import Utils

GRAPH_DEFAULTS = {'nodes': {'node_size' : 2, 'node_color': 'black'}, 'edges': {'edge_color': 'black'}}


def plot_urban_graph(weights,
                     *,
                     node_sizes_1,
                     node_sizes_2,
                     is_directed: bool = True,
                     pos: str = 'spring',
                     figsize: tuple = (Utils.textwidth, Utils.textwidth),
                     plot_kwargs: dict = GRAPH_DEFAULTS,
                     seed: int = 7):

    fig, ax = plt.subplots(figsize=figsize)

    N, M = len(node_sizes_1), len(node_sizes_2)

    edges = []
    for m in range(N, N+M):
        for n in range(N):
            edges.append((n, m))

    weights = np.resize(weights, (N*M, ))

    G = nx.DiGraph()
    G.add_edges_from(edges)

    # positions for all nodes - seed for reproducibility
    if pos == 'spring':
        pos = nx.spring_layout(G, seed=seed)
    else:
        pos = nx.random_layout(G, seed=seed)

    # draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=np.arange(0, N),
                           node_size=plot_kwargs['nodes']['node_size']*node_sizes_1,
                           node_color='#EC7070')
    nx.draw_networkx_nodes(G, pos, nodelist=np.arange(N, N+M),
                           node_color='#2f7194',
                           node_size=plot_kwargs['nodes']['node_size']*node_sizes_2)


    # draw edges
    nx.draw_networkx_edges(G, pos, arrows=is_directed, edgelist=edges, width=weights, **plot_kwargs['edges'])

    plt.axis("off")
    plt.margins(0.0)
    plt.savefig(Utils.output_dir + '/graph')
    plt.close()
