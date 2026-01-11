import torch
import torch.nn as nn
import logging
import sys
import os  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data_access.genomic_data import run_genomic_data_pipeline
from reactome.pathway_hierarchy import get_connectivity_maps

class ModalityFusionLayer(nn.Module):
    """
    Fuses multiple modalities (Mutation, Deletion and Amplification) per gene into single gene representation.
    Represents Diagonal custom Layer in the original P-NET Keras implementation
    """

    def __init__(self, n_genes=9229, n_modalities=3, dropout=0.5):
        super(ModalityFusionLayer, self).__init__()

        self.n_genes = n_genes
        self.n_modalities = n_modalities
        self.n_inputs = n_genes * n_modalities
        self.weight = nn.Parameter(torch.Tensor(self.n_inputs))
        self.bias = nn.Parameter(torch.Tensor(n_genes))
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self._initialise_parameters()

    def _initialise_parameters(self):
        """
        Initialize weights using LeCun uniform initialization.
        
        LeCun uniform: U(-limit, limit) where limit = sqrt(3 / fan_in)
        Designed for symmetric activations like tanh and sigmoid.
        Reference: LeCun et al., "Efficient BackProp" (1998)
        """
        fan_in = self.n_inputs
        limit = (3.0 / fan_in) ** 0.5
        nn.init.uniform_(self.weight, -limit, limit)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_size = x.size(0)
        weighted_inputs = x * self.weight
        modality_groups = weighted_inputs.reshape(batch_size * self.n_genes, self.n_modalities)
        gene_aggregated = modality_groups.sum(dim=1)
        output = gene_aggregated.reshape(batch_size, self.n_genes)

        output = output + self.bias
        output = self.activation(output)
        if self.dropout is not None:
            output = self.dropout(output)

        return output


class SparseBiologicalLayer(nn.Module):
    """
    Implements sparse biologically informed layers of the P-NET archictecture by constructing an index-based custom 
    sparse layer using the connectivity binary matrix that represents the Reactome pathway hierarchy layers.
    """

    def __init__(self, connectivity_map, dropout=0.1):
        super(SparseBiologicalLayer, self).__init__()

        self.in_features, self.out_features = connectivity_map.shape
        edges = connectivity_map.nonzero(as_tuple=False)
        self.register_buffer('src_idx', edges[:, 0]) 
        self.register_buffer('dst_idx', edges[:, 1])

        num_edges = edges.shape[0]  # FIX #1
        self.weight = nn.Parameter(torch.Tensor(num_edges))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self._initialise_parameters()

    def _initialise_parameters(self):
        """
        Initialize weights using LeCun uniform initialization.
        
        LeCun uniform: U(-limit, limit) where limit = sqrt(3 / fan_in)
        Designed for symmetric activations like tanh and sigmoid.
        Reference: LeCun et al., "Efficient BackProp" (1998)
        """
        num_edges = len(self.src_idx)
        avg_fan_in = num_edges / self.out_features
        
        limit = (3.0 / avg_fan_in) ** 0.5
        nn.init.uniform_(self.weight, -limit, limit)
        nn.init.zeros_(self.bias)

        """
        Non-biologically aware but matches original Keras implementation

        num_edges = len(self.src_idx)
    
        # LeCun uniform with fan_in = num_edges
        limit = (3.0 / num_edges) ** 0.5
        nn.init.uniform_(self.weight, -limit, limit)
        nn.init.zeros_(self.bias)
        """

    def forward(self, x):
        gathered_inputs = x[:, self.src_idx]
        weighted_inputs = gathered_inputs * self.weight
        
        output = torch.zeros(x.shape[0], self.out_features, dtype=x.dtype, device=x.device)
        output.index_add_(1, self.dst_idx, weighted_inputs)
        
        output = output + self.bias
        output = self.activation(output)
        if self.dropout is not None:
            output = self.dropout(output)

        return output


class OutputHead(nn.Module):
    """
    Dense layer for predictions from any hidden layer
    """
    def __init__(self, input_size):
        super(OutputHead, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x)
