import torch
import torch.nn as nn
import logging
import sys
import os  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

class WeightedBCELoss(nn.Module):
    """
    BCE with class weighting for imbalanced data
    pos_weight=2.0 chosen as original Keras implementation used class_weight = {0: 0.75, 1: 1.5}
    """
    def __init__(self, pos_weight=2.0):
        super(WeightedBCELoss, self).__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
    
    def forward(self, pred, target):
        return self.criterion(pred, target)
    

class MultiOutputLoss(nn.Module):
    """
    Weighted sum of losses from multiple outputs
    loss_weights chosen reflect those from the original Keras implementation
    """
    def __init__(self, loss_weights=[2, 7, 20, 54, 148, 400], pos_weight=2.0):
        super(MultiOutputLoss, self).__init__()
        self.register_buffer("loss_weights", torch.tensor(loss_weights, dtype=torch.float32))
        self.criterions = nn.ModuleList([
            WeightedBCELoss(pos_weight) for _ in range(len(loss_weights))
        ])
    
    def forward(self, predictions, target):
        total_loss = 0
        for pred, weight, criterion in zip(predictions, self.loss_weights, self.criterions):
            total_loss += weight * criterion(pred, target)
        return total_loss