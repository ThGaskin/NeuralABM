import sys
from os.path import dirname as up
from builtins import *
import pytest

import networkx as nx
import dantro
from dantro.containers import XrDataContainer
import numpy as np

from dantro._import_tools import import_module_from_path
from pkg_resources import resource_filename
import h5py as h5
from utopya.yaml import load_yml

sys.path.insert(0, up(up(up(__file__))))

graph_gen = import_module_from_path(
    mod_path=up(up(up(__file__))), mod_str="include.graph"
)

# Load the test config
CFG_FILENAME = resource_filename("tests", "cfgs/graph_generation.yml")
test_cfg = load_yml(CFG_FILENAME)


def test_graph_generation():
    for _, config in test_cfg.items():

        _raises = config.pop("_raises", False)
        _exp_exc = Exception if not isinstance(_raises, str) else globals()[_raises]
        _warns = config.pop("_warns", False)
        _exp_warning = UserWarning if not isinstance(_warns, str) else globals()[_warns]
        _match = config.pop("_match", " ")

        nw_cfg = config.pop("network")

        if _raises:
            with pytest.raises(_exp_exc, match=_match):
                graph_gen.generate_graph(**config, **nw_cfg)
            continue

        # Generate the graph from the configuration
        G = graph_gen.generate_graph(**config, **nw_cfg)

        # check the graph is non-empty
        assert G != nx.empty_graph()

        # check the graph has no isolated vertices
        assert not list(nx.isolates(G))

        # check the edge weights were set, if specified
        if nw_cfg["graph_props"]["is_weighted"]:
            assert [0 <= e[-1]["weight"] <= 1 for e in G.edges(data=True)]
        else:
            assert [not e[-1] for e in G.edges(data=True)]

        # Check Barabasi-Albert and BollobasRiordan graphs are undirected and directed respectively
        if nw_cfg["type"] == "BarabasiAlbert":
            assert not G.is_directed()
        elif nw_cfg["type"] == "BollobasRiordan":
            assert G.is_directed()

def test_graph_saving(tmpdir):

    # Create an h5File in the temporary directory for the
    h5dir = tmpdir.mkdir("hdf5_data")
    h5file = h5.File(h5dir.join(f"test_file.h5"), "w")
    nw_group = h5file.create_group("graph")
    nw_group.attrs["content"] = "graph"
    nw_group.attrs["allows_parallel"] = False
    nw_group.attrs["is_directed"] = True

    # Test saving a static graph
    G1 = graph_gen.generate_graph(N=16, mean_degree=4, type='random', graph_props=dict(is_directed=True, is_weighted=True))

    graph_gen.save_nw(G1, nw_group, write_adjacency_matrix=True)

    # Load the network
    GG = dantro.groups.GraphGroup(
        name="true_network",
        attrs=dict(
            directed=h5file["graph"].attrs["is_directed"],
            parallel=h5file["graph"].attrs["allows_parallel"],
        ),
    )
    GG.new_container(
        "nodes",
        Cls=XrDataContainer,
        data=np.array(h5file["graph"]["_vertices"]),
    )
    GG.new_container("edges", Cls=XrDataContainer, data=[])

    edges = np.array(h5file["graph"]["_edges"])
    edge_weights = np.expand_dims(
        np.array(h5file["graph"]["_edge_weights"]), -1
    )
    weighted_edges = np.squeeze(np.concatenate((edges, edge_weights), axis=-1))

    A = np.array(h5file["graph"]["_adjacency_matrix"])

    G2 = GG.create_graph()
    G2.add_weighted_edges_from(weighted_edges, "weight")
    h5file.close()

    A1 = nx.adjacency_matrix(G1).toarray()
    A2 = nx.adjacency_matrix(G2).toarray()

    # Assert the adjacency matrices are all equal
    assert (np.all([a == pytest.approx(0, abs=1e-6) for a in A1-A2]))
    assert (np.all([a == pytest.approx(0, abs=1e-6) for a in A1-A]))


