import torch.nn as nn
import logging
import sys
import os  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from models.layers import ModalityFusionLayer, SparseBiologicalLayer, OutputHead

class PNet(nn.Module):
    """
    Full P-NET with multiple output heads from each layer (h0-h5).
    Matches the original Keras implementation with 6 outputs:
    - o1: output from h0 (modality fusion layer)
    - o2: output from h1 (pathway layer 1)
    - o3: output from h2 (pathway layer 2)
    - o4: output from h3 (pathway layer 3)
    - o5: output from h4 (pathway layer 4)
    - o6: output from h5 (pathway layer 5)
    """
    def __init__(self, connectivity_maps, n_genes=9229, n_modalities=3, dropout_h0=0.5, dropout_h=0.1):
        super(PNet, self).__init__()

        self.layers = nn.ModuleList()

        first_layer = ModalityFusionLayer(n_genes, n_modalities, dropout_h0)
        self.layers.append(first_layer)

        for connectivity_map in connectivity_maps:
            biological_layer = SparseBiologicalLayer(connectivity_map, dropout_h)
            self.layers.append(biological_layer)

        self.output_heads = nn.ModuleList([
                OutputHead(n_genes),
                OutputHead(connectivity_maps[0].shape[1]),
                OutputHead(connectivity_maps[1].shape[1]), 
                OutputHead(connectivity_maps[2].shape[1]),  
                OutputHead(connectivity_maps[3].shape[1]), 
                OutputHead(connectivity_maps[4].shape[1]), 
            ])
        
    def forward(self, x):
        """
        Forward pass through all layers, collecting outputs from each.
        
        Returns:
            list of tensors: [o1, o2, o3, o4, o5, o6] - predictions from each layer
        """
        outputs = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            outputs.append(self.output_heads[i](x))
        
        return outputs



class SingleOutputPNet(nn.Module):
    """
    Simpler version of P-NET with only one final output layer
    Included for simplicity to check layers are working as intended.
    """
    def __init__(self, connectivity_maps, n_genes=9229, n_modalities=3, dropout_h0=0.5, dropout_h=0.1):
        super(SingleOutputPNet, self).__init__()

        self.layers = nn.ModuleList()

        first_layer = ModalityFusionLayer(n_genes, n_modalities, dropout_h0)
        self.layers.append(first_layer)

        for connectivity_map in connectivity_maps:
            biological_layer = SparseBiologicalLayer(connectivity_map, dropout_h)
            self.layers.append(biological_layer)

        final_size = connectivity_maps[-1].shape[1]
        self.output_head = OutputHead(final_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return self.output_head(x)
        