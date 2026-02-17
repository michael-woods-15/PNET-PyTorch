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

from models.layers import ModalityFusionLayer


class ReactomeGNN(nn.Module):
    def __init__(self, connectivity_maps, n_genes=9229, n_modalities=3, projection_dim=32,
                hidden_dim=64, dropout_h0=0.5, dropout=0.3):
        super().__init__()
        
        self.connectivity_maps = connectivity_maps
        self.n_genes = n_genes
        self.n_modalities = n_modalities
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.dropout_h0 = dropout_h0
        
        self.node_counts = [n_genes] 
        for adj in connectivity_maps:
            self.node_counts.append(adj.shape[1]) 

        self.total_nodes = sum(self.node_counts)
        self.n_pathway_nodes = self.total_nodes - n_genes
        logging.info(f"Hierarchy structure: {self.node_counts}")
        logging.info(f"Total nodes: {self.total_nodes}")
        logging.info(f"Pathway nodes: {self.n_pathway_nodes}")
        
        self.fusion_layer = ModalityFusionLayer(
            n_genes=self.n_genes, 
            n_modalities=self.n_modalities, 
            dropout=self.dropout_h0,
            gnn=True, 
            projection_dim=projection_dim
        )
        
        self.pathway_embeddings = nn.Parameter(
            torch.zeros(self.n_pathway_nodes, projection_dim) 
        )
        
        self.edge_index = self.build_full_edge_index(connectivity_maps)

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        conv1 = GCNConv(projection_dim, hidden_dim)
        self.conv_layers.append(conv1)
        bn1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norms.append(bn1)

        for _ in range(len(connectivity_maps)):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        self._batched_edge_cache = {}

        logging.info(f"Num GCNConv Layers: {len(self.conv_layers)}")


    def build_full_edge_index(self, connectivity_maps):
        all_edges = []
        
        offsets = [0]
        for size in self.node_counts[:-1]:
            offsets.append(offsets[-1] + size)
        # offsets = [0, 9229, 10616, 11682, 12129, 12276]
        
        for level_idx, adj in enumerate(connectivity_maps):
            edges = adj.nonzero(as_tuple=False).t()
            
            source_offset = offsets[level_idx]
            target_offset = offsets[level_idx + 1]
            
            edges[0] += source_offset 
            edges[1] += target_offset 
            
            all_edges.append(edges)
        
        edge_index = torch.cat(all_edges, dim=1)
        edge_index = torch.unique(edge_index, dim=1)
        return edge_index
    

    def _get_batched_edge_index(self, batch_size, device):
        """Get or create batched edge index for given batch size"""
        if batch_size not in self._batched_edge_cache:
            edge_index_list = []
            for i in range(batch_size):
                offset = i * self.total_nodes   
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

        #print("Input std:", x.std().item())
        batch_size = x.size(0)
        
        # Shape: [batch_size, n_genes, projection_dim]
        gene_features = self.fusion_layer.get_gnn_projected_features(x)

        # Shape: [batch_size, n_pathway_nodes, projection_dim]
        pathway_features = self.pathway_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # Shape: [batch_size, total_nodes, projection_dim]
        all_node_features = torch.cat([gene_features, pathway_features], dim=1)
        
        # Shape: [batch_size * total_nodes, projection_dim]
        node_features = all_node_features.reshape(-1, self.projection_dim)
        
        batched_edge_index = self._get_batched_edge_index(batch_size, x.device)

        for i, (conv, batch_norm) in enumerate(zip(self.conv_layers, self.batch_norms)):
            node_features = conv(node_features, batched_edge_index)
            node_features = batch_norm(node_features)
            node_features = F.tanh(node_features)
            node_features = self.dropout(node_features)
        
        
        # Reshape back to [batch_size, total_nodes, hidden_dim]
        node_features = node_features.reshape(batch_size, self.total_nodes, -1)
        
        # Pooling
        start_idx = self.total_nodes - self.node_counts[-1]
        graph_features = node_features[:, start_idx:, :].mean(dim=1)
        
        logits = self.classifier(graph_features)
        #print("Gene feature std:", gene_features.std().item())
        #print("Logit std:", logits.std().item())
        return logits