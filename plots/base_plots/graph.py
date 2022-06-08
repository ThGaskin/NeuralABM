import networkx as nx
import matplotlib.pyplot as plt

import Utils

GRAPH_DEFAULTS = {'nodes': {'node_size' : 50, 'node_color': 'black'}, 'edges': {'edge_color': 'black'}}


def plot_graph(G,
               *,
               is_directed: bool,
               pos: str = 'spring',
               plot_kwargs: dict = GRAPH_DEFAULTS,
               figsize: tuple = (Utils.textwidth, Utils.textwidth),
               seed: int = 7):

    fig, ax = plt.subplots(figsize=figsize)
    edges = [(u, v) for (u, v, d) in G.edges(data=True)]
    weights = [5*d['weight'] for (_, _, d) in G.edges(data=True)]

    # positions for all nodes - seed for reproducibility
    if pos == 'spring':
        pos = nx.spring_layout(G, seed=seed, iterations=1000)
    elif pos == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G, seed=seed)

    # draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes, **plot_kwargs['nodes'])

    # draw edges
    nx.draw_networkx_edges(G, pos, arrows=is_directed, edgelist=edges, width=weights, **plot_kwargs['edges'])

    ax.margins(0.0)
    plt.axis("off")
    plt.savefig(Utils.output_dir + '/graph')
    plt.close()