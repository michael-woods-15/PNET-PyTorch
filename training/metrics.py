import logging
import sys
import os  
import torch
from torchmetrics.classification import (
    BinaryF1Score, BinaryAUROC, BinaryAccuracy,
    BinaryPrecision, BinaryRecall
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

class MetricsTracker:
    def __init__(self, device='cpu', threshold=0.5):
        self.device = torch.device(device)
        self.threshold = threshold
        self._init_metrics()

    def _init_metrics(self):
        self.f1 = BinaryF1Score(threshold=self.threshold).to(self.device)
        self.auc = BinaryAUROC().to(self.device) 
        self.accuracy = BinaryAccuracy(threshold=self.threshold).to(self.device)
        self.precision = BinaryPrecision(threshold=self.threshold).to(self.device)
        self.recall = BinaryRecall(threshold=self.threshold).to(self.device)

    def set_threshold(self, threshold):
        self.threshold = threshold
        self.reset()
        self._init_metrics()

    def update(self, preds, targets):
        preds = preds.detach().to(self.device)
        targets = targets.detach().to(self.device)

        self.f1.update(preds, targets)
        self.auc.update(preds, targets)
        self.accuracy.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)

    def compute(self):
        return {
            'f1': self.f1.compute().item(),
            'auc': self.auc.compute().item(),
            'accuracy': self.accuracy.compute().item(),
            'precision': self.precision.compute().item(),
            'recall': self.recall.compute().item(),
        }

    def reset(self):
        self.f1.reset()
        self.auc.reset()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()

