import os
import pandas as pd
import networkx as nx
import logging
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from config_path import REACTOME_PATHWAY_PATH, SELECTED_GENES_FILE_PATH

from reactome.gmt_loader import load_reactome_pathways_file, load_reactome_hierarchies
from reactome.reactome_utils import sanity_check_reactome_graph
from data_access.data_utils import load_genes_list

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


def get_nodes_at_level(G, distance):
    nodes_at_distance = set(nx.ego_graph(G, 'root', radius=distance))

    if distance > 0:
        nodes_closer = set(nx.ego_graph(G, 'root', radius=distance - 1))
        nodes_at_distance -= nodes_closer
    
    return list(nodes_at_distance)


def _strip_copy_suffix(node_name):
    if isinstance(node_name, str) and '_copy' in node_name:
        return node_name.split('_copy')[0]
    else:
        return node_name
    

def get_final_network_layer(G):
    level_dict = {}
    reactome_pathways_df = load_reactome_pathways_file(REACTOME_PATHWAYS_FILE)
    terminal_nodes = [n for n, d in G.out_degree() if d == 0]

    for pathway in terminal_nodes:
        pathway_name = _strip_copy_suffix(pathway)
        genes = reactome_pathways_df[reactome_pathways_df['group'] == pathway_name]['gene'].unique()
        level_dict[pathway_name] = genes

    return level_dict


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

    final_layer = get_final_network_layer(G)
    layers.append(final_layer)
    return layers


def get_map_from_layer_dict(layer_dict):
    pathways = list(layer_dict.keys())
    print(len(pathways))
    child_nodes = list(set(child for children in layer_dict.values() for child in children))
    
    logging.info(f"Pathways: {len(pathways)}, Genes: {len(child_nodes)}")
    
    n_pathways = len(pathways)
    n_children = len(child_nodes)
    mat = np.zeros((n_pathways, n_children), dtype=np.float32)
    
    for pathway, children in layer_dict.items():
        child_indices = [child_nodes.index(child) for child in children]
        pathway_idx = pathways.index(pathway)
        mat[pathway_idx, child_indices] = 1

    df = pd.DataFrame(mat, index=pathways, columns=child_nodes)
    return df.T


def get_layer_maps(layers, genes):
    logging.info("Converting layers to binary matrices")
    
    # Reverse layers to go from bottom up (genes -> pathways)
    reversed_layers = layers[::-1]
    
    filtering_index = genes
    maps = []
    for i, layer_dict in enumerate(reversed_layers):
        logging.info(f"Processing layer {i}")
        
        # Convert layer to binary matrix
        layer_map = get_map_from_layer_dict(layer_dict)
        
        # Filter to only include nodes from previous layer
        filter_df = pd.DataFrame(index=filtering_index)
        filtered_map = filter_df.merge(layer_map, right_index=True, 
                                      left_index=True, how='left')
        
        
        filtered_map = filtered_map.fillna(0)
        filtering_index = filtered_map.columns.tolist()
        n_edges = int(filtered_map.sum().sum())
        logging.info(f"Layer {i}: shape={filtered_map.shape}, edges={n_edges}")
        
        filtered_map = torch.FloatTensor(filtered_map.values)
        maps.append(filtered_map)
    
    return maps


def get_connectivity_maps():
    reactome_hierarchy = load_reactome_hierarchies(HIERARCHIES_FILE)

    network = create_reactome_digraph(reactome_hierarchy)
    summary = sanity_check_reactome_graph(network)

    completed_digraph = complete_network_digraph(network)
    completed_summary = sanity_check_reactome_graph(completed_digraph)

    layers = get_network_layers(completed_digraph)
    for i, layer in enumerate(layers):
        print(f"Layer {i} size - {len(layer)}")

    genes = load_genes_list(SELECTED_GENES_FILE_PATH)
    layer_maps = get_layer_maps(layers, genes)

    return layer_maps

if __name__ == "__main__":
   get_connectivity_maps()