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
                 hidden_dim=64, n_classes=1, dropout=0.1):
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
        
        self.classifier = nn.Linear(hidden_dim, n_classes)
        nn.init.xavier_uniform(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


    def forward(self, x, edge_index):
        """
        Args:
            x: Input features [batch_size, n_genes * n_modalities]
            edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            logits: [batch_size, n_classes]
        """
        batch_size = x.size(0)
        
        # Get projected gene features from fusion layer
        # Shape: [batch_size, n_genes, projection_dim]
        gene_features = self.fusion_layer.get_gnn_projected_features(x)
        
        # Flatten batch and genes for graph processing
        # Shape: [batch_size * n_genes, projection_dim]
        node_features = gene_features.reshape(-1, self.projection_dim)
        
        node_features = self.conv1(node_features, edge_index)
        node_features = F.relu(node_features)
        node_features = self.dropout(node_features)
        
        node_features = self.conv2(node_features, edge_index)
        node_features = F.relu(node_features)
        node_features = self.dropout(node_features)
        
        # Reshape back to [batch_size, n_genes, hidden_dim]
        node_features = node_features.reshape(batch_size, self.n_genes, -1)
        
        # Global mean pooling over all nodes (genes)
        graph_features = node_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Classification
        logits = self.classifier(graph_features)
        
        return logits


    def build_edge_index(self, adjacency_matrices):
        """
        Convert list of adjacency matrices from Reactome hierarchy to PyTorch Geometric edge_index.
        
        Args:
            adjacency_matrices: List of adjacency matrices from get_connectivity_maps()
            n_genes: Number of gene nodes (first n_genes nodes in the graph)
        
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