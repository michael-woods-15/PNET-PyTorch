import os
import pandas as pd
import networkx as nx
import logging
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from config_path import REACTOME_PATHWAY_PATH
from gmt_loader import load_reactome_hierarchies
from reactome_utils import sanity_check_reactome_graph

REACTOME_PATHWAYS_FILE = os.path.join(REACTOME_PATHWAY_PATH, 'ReactomePathways.gmt')
HIERARCHIES_FILE = os.path.join(REACTOME_PATHWAY_PATH, 'ReactomePathwaysRelation.txt')
PATHWAYS_NAMES_FILE = os.path.join(REACTOME_PATHWAY_PATH, 'ReactomePathways.txt')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')



def create_reactome_digraph(hierarchy):
    human_hierarchy = hierarchy[hierarchy['child'].str.contains('HSA')]
    net = nx.from_pandas_edgelist(human_hierarchy, 'child', 'parent', create_using=nx.DiGraph())
    net.name = 'reactome'

    roots = [n for n, d in net.in_degree() if d == 0]
    root_node = 'root'
    edges = [(root_node, n) for n in roots]
    net.add_edges_from(edges)

    return net


def add_edges(G, node, n_levels):
    if n_levels <= 0:
        return G
    
    edges = [
            (f"{node}_copy{i}" if i > 0 else node, f"{node}_copy{i+1}")
            for i in range(n_levels)
        ]

    G.add_edges_from(edges)
    return G


def complete_network_digraph(G, n_levels=5):
    sub_graph = nx.ego_graph(G, 'root', radius=n_levels)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d==0]

    for node in terminal_nodes:
        path_length = len(nx.shortest_path(sub_graph, source='root', target=node))
        levels_to_add = n_levels - path_length + 1

        if levels_to_add > 0:
            sub_graph = add_edges(sub_graph, node, levels_to_add)

    return sub_graph


def get_nodes_at_level(net, distance):
    nodes_at_distance = set(nx.ego_graph(net, 'root', radius=distance))

    if distance > 0:
        nodes_closer = set(nx.ego_graph(net, 'root', radius=distance - 1))
        nodes_at_distance -= nodes_closer
    
    return list(nodes_at_distance)


def _strip_copy_suffix(node_name):
    if isinstance(node_name, str) and '_copy' in node_name:
        return node_name.split('_copy')[0]
    else:
        return node_name


def get_network_layers(G, n_levels=5):
    layers = []

    for level in range(n_levels):
        nodes = get_nodes_at_level(G, level)

        level_dict = {
            _strip_copy_suffix(node) : [
                _strip_copy_suffix(child) for child in G.successors(node)
            ]
            for node in nodes
        }

        layers.append(level_dict)
        logging.info(f"Layer {level} size: {len(level_dict)}")
    
    return layers


if __name__ == "__main__":
    reactome_hierarchy = load_reactome_hierarchies(HIERARCHIES_FILE)
    network = create_reactome_digraph(reactome_hierarchy)
    summary = sanity_check_reactome_graph(network)
    completed_digraph = complete_network_digraph(network)
    completed_summary = sanity_check_reactome_graph(completed_digraph)
    layers = get_network_layers(completed_digraph)