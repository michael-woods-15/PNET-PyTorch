import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import logging
import sys
import os  
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from reactome.pathway_hierarchy import get_connectivity_maps
from data_access.data_pipeline import run_data_pipeline
from models.layers import ModalityFusionLayer


class ReactomeGNN(nn.Module):
    def __init__(self, connectivity_maps, n_genes=9229, n_modalities=3, projection_dim=32, 
                 hidden_dim=64, dropout=0.1):
        super(ReactomeGNN, self).__init__()

        self.n_genes = n_genes
        self.projection_dim = projection_dim
        
        self.fusion_layer = ModalityFusionLayer(
            n_genes=n_genes,
            n_modalities=n_modalities,
            dropout=dropout,
            gnn=True,
            projection_dim=projection_dim,
        )

        self.edge_index = self.build_edge_index(connectivity_maps)
        
        self.conv1 = GCNConv(projection_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self._batched_edge_cache = {}

        logging.info(f"Number of edges: {self.edge_index.shape[1]}")
        logging.info(f"Number of genes: {self.n_genes}")
        logging.info(f"Edge density: {self.edge_index.shape[1] / (self.n_genes ** 2):.4f}")


    def _get_batched_edge_index(self, batch_size, device):
        """Get or create batched edge index for given batch size"""
        if batch_size not in self._batched_edge_cache:
            edge_index_list = []
            for i in range(batch_size):
                offset = i * self.n_genes
                offset_edge_index = self.edge_index + offset
                edge_index_list.append(offset_edge_index)
            
            self._batched_edge_cache[batch_size] = torch.cat(edge_index_list, dim=1)
        
        return self._batched_edge_cache[batch_size].to(device)


    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, n_genes * n_modalities]
            edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            logits: [batch_size, n_classes]
        """
        batch_size = x.size(0)
        
        # Shape: [batch_size, n_genes, projection_dim]
        gene_features = self.fusion_layer.get_gnn_projected_features(x)
        
        # Shape: [batch_size * n_genes, projection_dim]
        node_features = gene_features.reshape(-1, self.projection_dim)

        batched_edge_index = self._get_batched_edge_index(batch_size, x.device)
        
        node_features = self.conv1(node_features, batched_edge_index)
        node_features = F.relu(node_features)
        node_features = self.dropout(node_features)
        
        node_features = self.conv2(node_features, batched_edge_index)
        node_features = F.relu(node_features)
        node_features = self.dropout(node_features)
        
        # Reshape back to [batch_size, n_genes, hidden_dim]
        node_features = node_features.reshape(batch_size, self.n_genes, -1)
        
        # Global mean pooling over all genes
        graph_features = node_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        logits = self.classifier(graph_features)
        return logits


    def build_edge_index(self, adjacency_matrices):
        """
        Convert list of adjacency matrices from Reactome hierarchy to PyTorch Geometric edge_index.
        
        Args:
            adjacency_matrices: List of adjacency matrices from get_connectivity_maps()
        
        Returns:
            edge_index: [2, num_edges] tensor
        """
        all_edges = []
        
        for adj_matrix in adjacency_matrices:
            edge_index = adj_matrix.nonzero(as_tuple=False).t()
            all_edges.append(edge_index)
        
        edge_index = torch.cat(all_edges, dim=1)
        edge_index = torch.unique(edge_index, dim=1)
        return edge_index