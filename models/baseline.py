import torch.nn as nn
import logging
import sys
import os  
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

class DenseNN(nn.Module):
    def __init__(self, n_genes=9229, n_modalities=3, dropout=0.1, hidden_layers=3):
        super(DenseNN, self).__init__()

        self.layers = nn.ModuleList()
        
        hidden_dims = self.get_hidden_dims(n_genes*n_modalities, hidden_layers, 1)
        for i in range(len(hidden_dims)-1):
            self.layers.append(
                nn.Linear(hidden_dims[i], hidden_dims[i+1])
            )
        
        self.classifier = nn.Linear(hidden_dims[-1], 1)

        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None


        for layer in self.layers:
            logging.info(f"Layer : {layer.in_features} -> {layer.out_features}")
        logging.info(f"Classifer Layer: {self.classifier.in_features} -> {self.classifier.out_features}")

    def get_hidden_dims(self, num_input_features, hidden_layers, output_dim):
        dimensions = [num_input_features]
        ratio = (output_dim / num_input_features) ** (1 / (hidden_layers + 1))
        
        for i in range(1, hidden_layers + 1):
            dim = num_input_features * (ratio ** i)
            dim = 2 ** round(math.log2(dim))
            dimensions.append(int(dim))

        return dimensions
        


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
            x = self.dropout(x)

        x = self.classifier(x)
        return x

        
