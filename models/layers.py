import torch
import torch.nn as nn
import logging
import sys
import os  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


from data_access.genomic_data import run_genomic_data_pipeline

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
        self.activation = torch.tanh
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize weights using Xavier/Glorot uniform initialisation
        Glorot uniform initialisation, similar to Keras 'glorot_uniform'
        """
        nn.init.xavier_uniform_(self.weight.view(self.n_inputs, 1))
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
