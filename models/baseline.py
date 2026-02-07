import torch.nn as nn
import logging
import sys
import os  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

class DenseNN(nn.Module):
    def __init__(self, n_genes=9229, n_modalities=3, layer_dimensions=[1387,1066,447,147,26], dropout=0.1):
        super(DenseNN, self).__init__()

        self.layers = nn.ModuleList()
        first_layer = nn.Linear(n_genes*n_modalities, n_genes)
        self.layers.append(first_layer)

        second_layer = nn.Linear(n_genes, layer_dimensions[0])
        self.layers.append(second_layer)

        for i in range(len(layer_dimensions) - 1):
            layer = nn.Linear(layer_dimensions[i], layer_dimensions[i+1])
            self.layers.append(layer)

        self.classifier = nn.Linear(layer_dimensions[-1], 1)

        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None


        for layer in self.layers:
            logging.info(f"Layer {i}: {layer.in_features} -> {layer.out_features}")
        logging.info(f"Classifer Layer: {self.classifier.in_features} -> {self.classifier.out_features}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
            x = self.dropout(x)

        x = self.classifier(x)
        return x

        
