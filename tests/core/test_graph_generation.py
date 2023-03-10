import sys
from os.path import dirname as up
import networkx as nx

from dantro._import_tools import import_module_from_path
from pkg_resources import resource_filename

from utopya.yaml import load_yml

sys.path.insert(0, up(up(up(__file__))))

graph_gen = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include.graph_generation")

# Load the test config
CFG_FILENAME = resource_filename("tests", "cfgs/graph_generation.yml")
test_cfg = load_yml(CFG_FILENAME)


def test_graph_generation():
    for _, config in test_cfg.items():

        nw_cfg = config.pop("network")
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
